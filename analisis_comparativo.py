import sys
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import logging
from collections import Counter
import networkx as nx
from networkx.exception import NetworkXError

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Configuración de Ruta e importación de módulos del proyecto ---
try:
    project_root_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(project_root_path)
    from python import data_processing, pipeline
except (ImportError, FileNotFoundError) as e:
    logging.error(f"Error importando módulos: {e}. Asegúrate de que la estructura de carpetas es correcta y que 'python/pipeline.py' y 'python/data_processing.py' existen.")
    sys.exit(1)

# --- Descarga de dependencias de NLTK (stopwords) ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logging.info("Descargando 'stopwords' de NLTK...")
    nltk.download('stopwords', quiet=True)


def generar_edgelist_completo(df_original: pd.DataFrame, nombre_dataset: str, nombre_algoritmo: str):
    """
    Genera un edgelist completo del grafo para análisis de distribución de grado.
    Incluye todos los nodos independientemente de su cluster.
    """
    logging.info(f"Generando edgelist completo para: {nombre_algoritmo} en {nombre_dataset}")
    
    # Normalizar nombres de nodos
    df_edges = df_original.copy()
    df_edges['FROM_NODE_norm'] = df_edges['FROM_NODE'].str.lower().str.strip()
    df_edges['TO_NODE_norm'] = df_edges['TO_NODE'].str.lower().str.strip()
    
    # Crear edgelist con información adicional
    edgelist = []
    for _, row in df_edges.iterrows():
        edgelist.append({
            'Dataset': nombre_dataset,
            'Algoritmo': nombre_algoritmo,
            'Source': row['FROM_NODE_norm'],
            'Target': row['TO_NODE_norm'],
            'Sign': row['SIGN']
        })
    
    return edgelist


def calcular_metricas_red_completas(df_original: pd.DataFrame, nombre_dataset: str, nombre_algoritmo: str):
    """
    Calcula métricas completas de la red para determinar si es una red real.
    """
    logging.info(f"Calculando métricas de red para: {nombre_algoritmo} en {nombre_dataset}")
    
    # Normalizar nombres de nodos
    df_edges = df_original.copy()
    df_edges['FROM_NODE_norm'] = df_edges['FROM_NODE'].str.lower().str.strip()
    df_edges['TO_NODE_norm'] = df_edges['TO_NODE'].str.lower().str.strip()
    
    # Crear grafo de NetworkX
    G = nx.Graph()  # Usar Graph() para grafo no dirigido, DiGraph() para dirigido
    
    # Agregar aristas
    for _, row in df_edges.iterrows():
        source = row['FROM_NODE_norm']
        target = row['TO_NODE_norm']
        weight = 1  # Puedes modificar si tienes pesos
        
        if G.has_edge(source, target):
            G[source][target]['weight'] += weight
        else:
            G.add_edge(source, target, weight=weight)
    
    # Calcular métricas básicas
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    metricas = {
        'Dataset': nombre_dataset,
        'Algoritmo': nombre_algoritmo,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': nx.density(G) if num_nodes > 1 else 0,
        'num_components': nx.number_connected_components(G),
    }
    
    # Solo calcular métricas avanzadas si el grafo tiene suficientes nodos
    if num_nodes > 1 and nx.is_connected(G):
        try:
            # Métricas para grafos conectados
            metricas['average_shortest_path_length'] = nx.average_shortest_path_length(G)
            metricas['diameter'] = nx.diameter(G)
            metricas['radius'] = nx.radius(G)
            metricas['center_size'] = len(nx.center(G))
            metricas['periphery_size'] = len(nx.periphery(G))
        except NetworkXError as e:
            logging.warning(f"Error calculando métricas de caminos para {nombre_dataset}: {e}")
            metricas['average_shortest_path_length'] = np.nan
            metricas['diameter'] = np.nan
            metricas['radius'] = np.nan
            metricas['center_size'] = np.nan
            metricas['periphery_size'] = np.nan
    else:
        # Para grafos no conectados, calcular métricas del componente más grande
        if num_nodes > 1:
            largest_cc = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_cc)
            
            try:
                metricas['average_shortest_path_length'] = nx.average_shortest_path_length(G_largest)
                metricas['diameter'] = nx.diameter(G_largest)
                metricas['radius'] = nx.radius(G_largest)
                metricas['center_size'] = len(nx.center(G_largest))
                metricas['periphery_size'] = len(nx.periphery(G_largest))
                metricas['largest_component_size'] = len(largest_cc)
                metricas['largest_component_fraction'] = len(largest_cc) / num_nodes
            except NetworkXError as e:
                logging.warning(f"Error calculando métricas del componente más grande para {nombre_dataset}: {e}")
                metricas['average_shortest_path_length'] = np.nan
                metricas['diameter'] = np.nan
                metricas['radius'] = np.nan
                metricas['center_size'] = np.nan
                metricas['periphery_size'] = np.nan
        else:
            metricas['average_shortest_path_length'] = np.nan
            metricas['diameter'] = np.nan
            metricas['radius'] = np.nan
            metricas['center_size'] = np.nan
            metricas['periphery_size'] = np.nan
    
    # Clustering
    if num_nodes > 2:
        try:
            metricas['average_clustering'] = nx.average_clustering(G)
            metricas['transitivity'] = nx.transitivity(G)
        except:
            metricas['average_clustering'] = np.nan
            metricas['transitivity'] = np.nan
    else:
        metricas['average_clustering'] = np.nan
        metricas['transitivity'] = np.nan
    
    # Métricas de grado
    if num_nodes > 0:
        degrees = dict(G.degree())
        degree_values = list(degrees.values())
        
        metricas['average_degree'] = np.mean(degree_values)
        metricas['max_degree'] = np.max(degree_values)
        metricas['min_degree'] = np.min(degree_values)
        metricas['degree_std'] = np.std(degree_values)
        metricas['degree_variance'] = np.var(degree_values)
        
        # Grado medio por arista (2*edges/nodes para grafos no dirigidos)
        metricas['mean_degree_connectivity'] = 2 * num_edges / num_nodes if num_nodes > 0 else 0
    else:
        metricas['average_degree'] = 0
        metricas['max_degree'] = 0
        metricas['min_degree'] = 0
        metricas['degree_std'] = 0
        metricas['degree_variance'] = 0
        metricas['mean_degree_connectivity'] = 0
    
    # Métricas de centralidad (solo para grafos no muy grandes)
    if num_nodes <= 1000 and num_nodes > 0:  # Evitar cálculos costosos en grafos muy grandes
        try:
            # Centralidad de grado
            degree_centrality = nx.degree_centrality(G)
            metricas['max_degree_centrality'] = max(degree_centrality.values()) if degree_centrality else 0
            metricas['avg_degree_centrality'] = np.mean(list(degree_centrality.values())) if degree_centrality else 0
            
            # Centralidad de cercanía (solo para grafos conectados)
            if nx.is_connected(G):
                closeness_centrality = nx.closeness_centrality(G)
                metricas['max_closeness_centrality'] = max(closeness_centrality.values()) if closeness_centrality else 0
                metricas['avg_closeness_centrality'] = np.mean(list(closeness_centrality.values())) if closeness_centrality else 0
            else:
                metricas['max_closeness_centrality'] = np.nan
                metricas['avg_closeness_centrality'] = np.nan
                
            # Centralidad de intermediación (solo para grafos pequeños)
            if num_nodes <= 500:
                betweenness_centrality = nx.betweenness_centrality(G)
                metricas['max_betweenness_centrality'] = max(betweenness_centrality.values()) if betweenness_centrality else 0
                metricas['avg_betweenness_centrality'] = np.mean(list(betweenness_centrality.values())) if betweenness_centrality else 0
            else:
                metricas['max_betweenness_centrality'] = np.nan
                metricas['avg_betweenness_centrality'] = np.nan
                
        except Exception as e:
            logging.warning(f"Error calculando métricas de centralidad para {nombre_dataset}: {e}")
            metricas['max_degree_centrality'] = np.nan
            metricas['avg_degree_centrality'] = np.nan
            metricas['max_closeness_centrality'] = np.nan
            metricas['avg_closeness_centrality'] = np.nan
            metricas['max_betweenness_centrality'] = np.nan
            metricas['avg_betweenness_centrality'] = np.nan
    else:
        metricas['max_degree_centrality'] = np.nan
        metricas['avg_degree_centrality'] = np.nan
        metricas['max_closeness_centrality'] = np.nan
        metricas['avg_closeness_centrality'] = np.nan
        metricas['max_betweenness_centrality'] = np.nan
        metricas['avg_betweenness_centrality'] = np.nan
    
    # Asortatividad (mezcla homofílica)
    try:
        if num_edges > 0:
            metricas['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
        else:
            metricas['degree_assortativity'] = np.nan
    except:
        metricas['degree_assortativity'] = np.nan
    
    # Propiedades de mundo pequeño
    if num_nodes > 10 and nx.is_connected(G):
        try:
            # Comparar con grafo aleatorio equivalente
            random_G = nx.erdos_renyi_graph(num_nodes, metricas['density'])
            if nx.is_connected(random_G):
                random_clustering = nx.average_clustering(random_G)
                random_path_length = nx.average_shortest_path_length(random_G)
                
                # Coeficientes de mundo pequeño
                metricas['small_world_clustering_ratio'] = metricas['average_clustering'] / random_clustering if random_clustering > 0 else np.nan
                metricas['small_world_path_ratio'] = metricas['average_shortest_path_length'] / random_path_length if random_path_length > 0 else np.nan
                
                # Índice de mundo pequeño (sigma)
                if (metricas['small_world_clustering_ratio'] > 1 and 
                    metricas['small_world_path_ratio'] is not np.nan and 
                    metricas['small_world_path_ratio'] > 0):
                    metricas['small_world_sigma'] = metricas['small_world_clustering_ratio'] / metricas['small_world_path_ratio']
                else:
                    metricas['small_world_sigma'] = np.nan
            else:
                metricas['small_world_clustering_ratio'] = np.nan
                metricas['small_world_path_ratio'] = np.nan
                metricas['small_world_sigma'] = np.nan
        except Exception as e:
            logging.warning(f"Error calculando métricas de mundo pequeño para {nombre_dataset}: {e}")
            metricas['small_world_clustering_ratio'] = np.nan
            metricas['small_world_path_ratio'] = np.nan
            metricas['small_world_sigma'] = np.nan
    else:
        metricas['small_world_clustering_ratio'] = np.nan
        metricas['small_world_path_ratio'] = np.nan
        metricas['small_world_sigma'] = np.nan
    
    return metricas


def analizar_terminos_tfidf(df_original: pd.DataFrame, df_nodes_clusters: pd.DataFrame,
                              nombre_dataset: str, nombre_algoritmo: str, tfidf_params: dict,
                              cluster_map: dict, ngram_config: dict):
    """
    Calcula los términos más relevantes por clúster usando TF-IDF.
    Esta versión modificada selecciona la columna de texto ('BODY' o 'RESUME')
    basándose en el nombre del dataset.
    """
    ngram_range = ngram_config['range']
    ngram_type_name = ngram_config['name']
    logging.info(f"Analizando términos ({ngram_type_name}) para: {nombre_algoritmo} en {nombre_dataset}")
    
    df_original['FROM_NODE_norm'] = df_original['FROM_NODE'].str.lower().str.strip()
    df_original['TO_NODE_norm'] = df_original['TO_NODE'].str.lower().str.strip()
    df_nodes_clusters['NODE_NAME_norm'] = df_nodes_clusters['NODE_NAME'].str.lower().str.strip()
    
    # --- MODIFICACIÓN CLAVE #2 ---
    # Seleccionamos la columna de texto basándonos en el nombre del dataset.
    if nombre_dataset == 'Primarias_2025_RESUME':
        logging.info(f"Detectado dataset '{nombre_dataset}'. Usando la columna 'RESUME' para el análisis TF-IDF.")
        if 'RESUME' not in df_original.columns:
            logging.error(f"La columna 'RESUME' no existe en el dataset '{nombre_dataset}'. Abortando análisis de términos.")
            return []
        df_original['full_text'] = df_original['TITLE'].fillna('') + '. ' + df_original['RESUME'].fillna('')
    else:
        # Para 'Plebiscito_2022' y 'Primarias_2025_Body'
        logging.info(f"Dataset '{nombre_dataset}'. Usando la columna 'BODY' para el análisis TF-IDF.")
        if 'BODY' not in df_original.columns:
            logging.error(f"La columna 'BODY' no existe en el dataset '{nombre_dataset}'. Abortando análisis de términos.")
            return []
        df_original['full_text'] = df_original['TITLE'].fillna('') + '. ' + df_original['BODY'].fillna('')
    
    mapa_nodo_cluster = pd.Series(df_nodes_clusters.CLUSTER_ASSIGNMENT.values, index=df_nodes_clusters.NODE_NAME_norm).to_dict()
    df_original['FROM_CLUSTER'] = df_original['FROM_NODE_norm'].map(mapa_nodo_cluster)
    df_original['TO_CLUSTER'] = df_original['TO_NODE_norm'].map(mapa_nodo_cluster)
    
    df_aristas_internas = df_original[(df_original['FROM_CLUSTER'] == df_original['TO_CLUSTER']) & (df_original['FROM_CLUSTER'].notna())].copy()
    df_aristas_internas.rename(columns={'FROM_CLUSTER': 'CLUSTER_ID'}, inplace=True)
    
    df_docs = df_aristas_internas[df_aristas_internas['CLUSTER_ID'].isin(cluster_map.keys())]
    df_docs = df_docs.drop_duplicates(subset=['full_text']).reset_index(drop=True)
    
    if df_docs.empty or df_docs['CLUSTER_ID'].nunique() < len(cluster_map):
        logging.warning(f"No hay suficientes datos en ambos clústeres para un análisis en {nombre_algoritmo}.")
        return []
        
    corpus = df_docs['full_text'].tolist()
    labels = df_docs['CLUSTER_ID'].tolist()
    
    stop_words_es = list(nltk.corpus.stopwords.words('spanish'))
    stop_words_es.extend(['dijo', 'ser', 'si', 'solo', 'tambien', 'tras', 'chile', 'gobierno', 'presidente', 'ex', 'san', 'mil', 'años', 'país', 'además', 'aseguró'])
    
    vectorizer = TfidfVectorizer(
        stop_words=stop_words_es,
        ngram_range=ngram_range,
        max_df=tfidf_params['max_df'],
        min_df=tfidf_params.get('min_df', 5),
        max_features=tfidf_params['max_features'],
        sublinear_tf=tfidf_params.get('sublinear_tf', False)    )    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        logging.warning(f"No se pudieron extraer términos para {nombre_algoritmo} ({ngram_type_name}), corpus vacío o parámetros muy estrictos.")
        return []
        
    feature_names = np.array(vectorizer.get_feature_names_out())
    resultados_terminos = []
    
    for cluster_id, cluster_name in cluster_map.items():
        indices_cluster = [i for i, label in enumerate(labels) if label == cluster_id]
        if not indices_cluster: continue
        
        matriz_cluster = tfidf_matrix[indices_cluster]
        scores_promedio = np.array(matriz_cluster.mean(axis=0)).flatten()
        top_indices = scores_promedio.argsort()[-tfidf_params['n_top_terms']:][::-1]
        
        for rank, idx in enumerate(top_indices):
            if scores_promedio[idx] > 1e-9:
                resultados_terminos.append({'Dataset': nombre_dataset, 'Algoritmo': nombre_algoritmo, 'Tipo_Ngram': ngram_type_name, 'Cluster': cluster_name, 'Termino': feature_names[idx], 'Score_TFIDF_Avg': round(scores_promedio[idx], 5), 'Rank': rank + 1})
                
    return resultados_terminos

def analizar_nodos_centrales(df_original: pd.DataFrame, df_nodes_clusters: pd.DataFrame, nombre_dataset: str, nombre_algoritmo: str, cluster_map: dict, n_top_nodes: int):
    logging.info(f"Analizando nodos centrales para: {nombre_algoritmo} en {nombre_dataset}")
    df_nodes_clusters['NODE_NAME_norm'] = df_nodes_clusters['NODE_NAME'].str.lower().str.strip()
    mapa_nodo_cluster = pd.Series(df_nodes_clusters.CLUSTER_ASSIGNMENT.values, index=df_nodes_clusters.NODE_NAME_norm).to_dict()
    df_original['FROM_NODE_norm'] = df_original['FROM_NODE'].str.lower().str.strip()
    df_original['TO_NODE_norm'] = df_original['TO_NODE'].str.lower().str.strip()
    df_original['FROM_CLUSTER'] = df_original['FROM_NODE_norm'].map(mapa_nodo_cluster)
    df_original['TO_CLUSTER'] = df_original['TO_NODE_norm'].map(mapa_nodo_cluster)
    df_aristas_internas = df_original[(df_original['FROM_CLUSTER'] == df_original['TO_CLUSTER']) & (df_original['FROM_CLUSTER'].notna())].copy()
    df_aristas_internas.rename(columns={'FROM_CLUSTER': 'CLUSTER_ID'}, inplace=True)
    resultados_nodos = []
    for cluster_id, cluster_name in cluster_map.items():
        df_cluster = df_aristas_internas[df_aristas_internas['CLUSTER_ID'] == cluster_id]
        if df_cluster.empty: continue
        nodos_participantes = pd.concat([df_cluster['FROM_NODE_norm'], df_cluster['TO_NODE_norm']])
        grado_nodos = nodos_participantes.value_counts().reset_index()
        grado_nodos.columns = ['Nodo', 'Grado']
        top_nodos = grado_nodos.head(n_top_nodes)
        for rank, row in top_nodos.iterrows():
            resultados_nodos.append({'Dataset': nombre_dataset, 'Algoritmo': nombre_algoritmo, 'Cluster': cluster_name, 'Nodo': row['Nodo'], 'Grado': row['Grado'], 'Rank': rank + 1})
    return resultados_nodos

def calcular_metricas_adicionales(df_nodes_clusters, df_original, cluster_map):
    """
    Calcula métricas adicionales para mejor comparación
    """
    metricas_extra = {}
    
    # Balance de clusters (qué tan equilibrados están)
    cluster_sizes = df_nodes_clusters['CLUSTER_ASSIGNMENT'].value_counts()
    total_nodes = len(df_nodes_clusters)
    
    if len(cluster_sizes) >= 2:
        # Ratio del cluster más grande vs más pequeño
        metricas_extra['cluster_size_ratio'] = cluster_sizes.max() / cluster_sizes.min()
        
        # Índice de balance (0 = perfectamente balanceado, 1 = completamente desbalanceado)
        expected_size = total_nodes / len(cluster_sizes)
        balance_index = sum(abs(size - expected_size) for size in cluster_sizes) / (2 * total_nodes)
        metricas_extra['balance_index'] = balance_index
        
        # Entropía de distribución de clusters
        probs = cluster_sizes / total_nodes
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        metricas_extra['cluster_entropy'] = entropy
    
    # Métricas de conectividad
    df_original['FROM_NODE_norm'] = df_original['FROM_NODE'].str.lower().str.strip()
    df_original['TO_NODE_norm'] = df_original['TO_NODE'].str.lower().str.strip()
    df_nodes_clusters['NODE_NAME_norm'] = df_nodes_clusters['NODE_NAME'].str.lower().str.strip()
    
    mapa_nodo_cluster = pd.Series(
        df_nodes_clusters.CLUSTER_ASSIGNMENT.values, 
        index=df_nodes_clusters.NODE_NAME_norm
    ).to_dict()
    
    df_original['FROM_CLUSTER'] = df_original['FROM_NODE_norm'].map(mapa_nodo_cluster)
    df_original['TO_CLUSTER'] = df_original['TO_NODE_norm'].map(mapa_nodo_cluster)
    
    # Ratio de aristas internas vs externas
    aristas_internas = len(df_original[
        (df_original['FROM_CLUSTER'] == df_original['TO_CLUSTER']) & 
        (df_original['FROM_CLUSTER'].notna())
    ])
    aristas_externas = len(df_original[
        (df_original['FROM_CLUSTER'] != df_original['TO_CLUSTER']) & 
        (df_original['FROM_CLUSTER'].notna()) & 
        (df_original['TO_CLUSTER'].notna())
    ])
    
    if aristas_externas > 0:
        metricas_extra['internal_external_ratio'] = aristas_internas / aristas_externas
    else:
        metricas_extra['internal_external_ratio'] = float('inf')
    
    # Densidad intra-cluster promedio
    densidades_intra = []
    for cluster_id in cluster_map.keys():
        nodos_cluster = df_nodes_clusters[
            df_nodes_clusters['CLUSTER_ASSIGNMENT'] == cluster_id
        ]['NODE_NAME_norm'].values
        
        if len(nodos_cluster) > 1:
            aristas_cluster = df_original[
                (df_original['FROM_NODE_norm'].isin(nodos_cluster)) &
                (df_original['TO_NODE_norm'].isin(nodos_cluster))
            ]
            
            max_aristas = len(nodos_cluster) * (len(nodos_cluster) - 1)
            if max_aristas > 0:
                densidad = len(aristas_cluster) / max_aristas
                densidades_intra.append(densidad)
    
    metricas_extra['avg_intra_cluster_density'] = np.mean(densidades_intra) if densidades_intra else 0
    
    return metricas_extra

def ejecutar_experimento_individual(nombre_dataset, df_dataset, config_base, default_cols, tfidf_params, cluster_map, ngram_configs, n_top_nodes):
    """
    Ejecuta un análisis completo: polarización, análisis de términos y análisis de nodos.
    """
    logging.info(f"Iniciando experimento completo para: {config_base['name']} en {nombre_dataset}")
    
    df_nodes_results, _, paper_metrics, _, _ = pipeline.ejecutar_analisis_polarizacion(
        df_input=df_dataset.copy(), 
        config=config_base.copy(), 
        default_cols=default_cols, 
        calculate_intra_cluster_cc=True
    )
    
    todos_los_terminos = []
    todos_los_nodos = []

    if df_nodes_results is not None and not df_nodes_results.empty:
        for ngram_config in ngram_configs:
            terminos_encontrados = analizar_terminos_tfidf(
                df_original=df_dataset.copy(), df_nodes_clusters=df_nodes_results.copy(),
                nombre_dataset=nombre_dataset, nombre_algoritmo=config_base['name'],
                tfidf_params=tfidf_params.copy(), cluster_map=cluster_map,
                ngram_config=ngram_config
            )
            if terminos_encontrados:
                todos_los_terminos.extend(terminos_encontrados)
        
        nodos_encontrados = analizar_nodos_centrales(
            df_original=df_dataset.copy(), df_nodes_clusters=df_nodes_results.copy(),
            nombre_dataset=nombre_dataset, nombre_algoritmo=config_base['name'],
            cluster_map=cluster_map, n_top_nodes=n_top_nodes
        )
        if nodos_encontrados:
            todos_los_nodos.extend(nodos_encontrados)

    paper_metrics['Dataset'] = nombre_dataset
    if df_nodes_results is not None:
        df_nodes_results['Dataset'] = nombre_dataset
        df_nodes_results['Algoritmo'] = config_base['name']
        
    logging.info(f"Finalizado: {config_base['name']} en {nombre_dataset}")
    
    return (paper_metrics, todos_los_terminos, todos_los_nodos, df_nodes_results)

def ejecutar_experimento_individual_mejorado(nombre_dataset, df_dataset, config_base, 
                                           default_cols, tfidf_params, cluster_map, 
                                           ngram_configs, n_top_nodes):
    """
    Versión mejorada que incluye métricas adicionales y análisis de redes
    """
    logging.info(f"Iniciando experimento completo MEJORADO para: {config_base['name']} en {nombre_dataset}")
    
    df_nodes_results, _, paper_metrics, _, _ = pipeline.ejecutar_analisis_polarizacion(
        df_input=df_dataset.copy(), 
        config=config_base.copy(), 
        default_cols=default_cols, 
        calculate_intra_cluster_cc=True
    )
    
    # Calcular métricas adicionales
    if df_nodes_results is not None and not df_nodes_results.empty:
        metricas_extra = calcular_metricas_adicionales(
            df_nodes_results.copy(), df_dataset.copy(), cluster_map
        )
        paper_metrics.update(metricas_extra)
    
    # Generar edgelist completo
    edgelist_completo = generar_edgelist_completo(
        df_dataset.copy(), nombre_dataset, config_base['name']
    )
    
# Calcular métricas de red completas (TEMPORALMENTE DESACTIVADO)
    logging.warning("El cálculo de métricas de red completas está desactivado.")
    metricas_red = {} # Se asigna un diccionario vacío como placeholder
    # metricas_red = calcular_metricas_red_completas(
    #     df_dataset.copy(), nombre_dataset, config_base['name']
    # )
    todos_los_terminos = []
    todos_los_nodos = []

    if df_nodes_results is not None and not df_nodes_results.empty:
        for ngram_config in ngram_configs:
            terminos_encontrados = analizar_terminos_tfidf(
                df_original=df_dataset.copy(), df_nodes_clusters=df_nodes_results.copy(),
                nombre_dataset=nombre_dataset, nombre_algoritmo=config_base['name'],
                tfidf_params=tfidf_params.copy(), cluster_map=cluster_map,
                ngram_config=ngram_config
            )
            if terminos_encontrados:
                todos_los_terminos.extend(terminos_encontrados)
        
        nodos_encontrados = analizar_nodos_centrales(
            df_original=df_dataset.copy(), df_nodes_clusters=df_nodes_results.copy(),
            nombre_dataset=nombre_dataset, nombre_algoritmo=config_base['name'],
            cluster_map=cluster_map, n_top_nodes=n_top_nodes
        )
        if nodos_encontrados:
            todos_los_nodos.extend(nodos_encontrados)

    paper_metrics['Dataset'] = nombre_dataset
    if df_nodes_results is not None:
        df_nodes_results['Dataset'] = nombre_dataset
        df_nodes_results['Algoritmo'] = config_base['name']
        
    logging.info(f"Finalizado MEJORADO: {config_base['name']} en {nombre_dataset}")
    
    return (paper_metrics, todos_los_terminos, todos_los_nodos, df_nodes_results, edgelist_completo, metricas_red)

def run_comparative_analysis():
    logging.info("--- INICIANDO ANÁLISIS COMPARATIVO MEJORADO ---")
    TFIDF_PARAMS = {"max_features": 50000, "max_df": 0.85, "min_df": 30, "sublinear_tf": True, "n_top_terms": 30}

    NGRAM_CONFIGS = [{"name": "Unigramas", "range": (1, 1)}, {"name": "Bigramas", "range": (2, 2)}, {"name": "Uni_y_Bigramas", "range": (1, 2)}]
    DEFAULT_COLS = {'from_node_col': 'FROM_NODE', 'to_node_col': 'TO_NODE', 'sign_col': 'SIGN'}
    K = 2
    CLUSTER_MAP = {1: 'S1', -1: 'S2'}
    N_TOP_NODES = 5
    LISTA_EXPERIMENTOS = [{"name": "Local Search (b=0.01)", "algorithm_type": "local_search_paper_k2", "k": K, "ls_beta": 0.01, "ls_max_iter": 200}]
    
    logging.info("--- Cargando Datos ---")
    try:
        ruta_plebiscito = os.path.join(project_root_path, 'News', 'output', 'df_plebiscito_2022.csv')
        df_plebiscito = pd.read_csv(ruta_plebiscito, sep=',', on_bad_lines='skip')
        ruta_primarias = os.path.join(project_root_path, 'News', 'output', 'primarias2025.csv')
        df_primarias = pd.read_csv(ruta_primarias, sep=',', on_bad_lines='skip')
        
        # --- MODIFICACIÓN CLAVE #1 ---
        # Definimos 3 experimentos distintos. Los dos últimos usan el mismo DataFrame
        # pero se procesarán de forma diferente en la función analizar_terminos_tfidf
        datasets = {
            "Plebiscito_2022": df_plebiscito, 
            "Primarias_2025": df_primarias.copy(),    # Experimento estándar con la columna BODY
            # "Primarias_2025_RESUME": df_primarias.copy()  # Nuevo experimento con la columna RESUME
        }
        
        logging.info("Datos cargados exitosamente.")
        logging.info(f"Noticias en Plebiscito_2022: {len(df_plebiscito)}")
        logging.info(f"Noticias en Primarias_2025: {len(df_primarias)}")

    except FileNotFoundError as e:
        logging.error(f"No se encontró un archivo de datos: {e}. Revisa la ruta en `News/output/`. Abortando.")
        return

    tareas = [(nombre, df, config, DEFAULT_COLS, TFIDF_PARAMS, CLUSTER_MAP, NGRAM_CONFIGS, N_TOP_NODES) for nombre, df in datasets.items() for config in LISTA_EXPERIMENTOS if not df.empty]
    logging.info(f"Se ejecutarán {len(tareas)} experimentos base en paralelo.")
    
    # Usar la función mejorada
    resultados_paralelos = Parallel(n_jobs=32, backend="loky")(delayed(ejecutar_experimento_individual_mejorado)(*task) for task in tareas)
    
    resultados_core = [res[0] for res in resultados_paralelos if res]
    todos_terminos = [item for sublist in [res[1] for res in resultados_paralelos if res and res[1]] for item in sublist]
    todos_nodos = [item for sublist in [res[2] for res in resultados_paralelos if res and res[2]] for item in sublist]
    nodos_con_clusters_list = [res[3] for res in resultados_paralelos if res and res[3] is not None]
    
    # Nuevos datos: edgelists y métricas de red
    todos_edgelists = [item for sublist in [res[4] for res in resultados_paralelos if res and res[4]] for item in sublist]
    todas_metricas_red = [res[5] for res in resultados_paralelos if res and res[5]]
    
    if not nodos_con_clusters_list:
        logging.warning("No se generaron resultados de asignación de clústeres.")
        df_nodos_clusters = pd.DataFrame()
    else:
        df_nodos_clusters = pd.concat(nodos_con_clusters_list, ignore_index=True)
        
    if not resultados_core:
        logging.error("No se generaron resultados de métricas. Revisa los logs en busca de errores.")
        return
        
    df_resultados_core = pd.DataFrame(resultados_core)
    df_resultados_terminos = pd.DataFrame(todos_terminos)
    df_resultados_nodos = pd.DataFrame(todos_nodos)
    df_edgelists = pd.DataFrame(todos_edgelists)
    df_metricas_red = pd.DataFrame(todas_metricas_red)

    results_dir = os.path.join(project_root_path, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    path_metricas = os.path.join(results_dir, 'exp_metricas_polarizacion.csv')
    path_terminos = os.path.join(results_dir, 'exp_analisis_terminos.csv')
    path_nodos_centrales = os.path.join(results_dir, 'exp_analisis_nodos_centrales.csv')
    path_asignacion_clusters = os.path.join(results_dir, 'exp_asignacion_clusters.csv')
    
    # Nuevos archivos CSV
    path_edgelist = os.path.join(results_dir, 'exp_edgelist_completo.csv')
    path_metricas_red = os.path.join(results_dir, 'exp_metricas_red_completas.csv')
    
    df_resultados_core.to_csv(path_metricas, index=False, sep=';', encoding='utf-8-sig')
    if not df_resultados_terminos.empty: df_resultados_terminos.to_csv(path_terminos, index=False, sep=';', encoding='utf-8-sig')
    if not df_resultados_nodos.empty: df_resultados_nodos.to_csv(path_nodos_centrales, index=False, sep=';', encoding='utf-8-sig')
    if not df_nodos_clusters.empty: df_nodos_clusters.to_csv(path_asignacion_clusters, index=False, sep=';', encoding='utf-8-sig')
    
    # Guardar nuevos archivos
    if not df_edgelists.empty: 
        df_edgelists.to_csv(path_edgelist, index=False, sep=';', encoding='utf-8-sig')
        logging.info(f"Edgelist completo: {path_edgelist}")
    
    if not df_metricas_red.empty: 
        df_metricas_red.to_csv(path_metricas_red, index=False, sep=';', encoding='utf-8-sig')
        logging.info(f"Métricas de red completas: {path_metricas_red}")

    logging.info("--- Resultados guardados exitosamente ---")
    logging.info(f"Métricas de polarización: {path_metricas}")
    logging.info(f"Análisis de Términos (TF-IDF): {path_terminos}")
    logging.info(f"Análisis de Nodos Centrales: {path_nodos_centrales}")
    logging.info(f"Asignación de Nodos a Clústeres: {path_asignacion_clusters}")
    
    # Devolvemos el path del archivo de términos para la impresión
    return path_terminos

def imprimir_resumen_terminos(path_archivo_terminos):
    """
    Carga el archivo de resultados de términos y muestra un resumen en la consola.
    """
    try:
        df_terminos = pd.read_csv(path_archivo_terminos, sep=';')
        
        print("\n===============================================")
        print(" 💬 TÉRMINOS MÁS IMPORTANTES POR CLÚSTER (Top 10)")
        print("===============================================")

        # Agrupamos por tipo de N-grama primero para un orden lógico
        for ngram_type, ngram_group in df_terminos.groupby('Tipo_Ngram'):
            print(f"\n################## TIPO: {ngram_type} ##################")
            for dataset_name, group in ngram_group.groupby('Dataset'):
                print(f"\n--- Dataset: {dataset_name} ---")
                
                # Agrupamos por clúster para este dataset
                for cluster_name, cluster_group in group.groupby('Cluster'):
                    # Ordenamos por ranking y tomamos los 10 primeros
                    top_terms = cluster_group.sort_values(by='Rank').head(10)
                    print(f"\n   Clúster {cluster_name}:")
                    print(f"   {top_terms['Termino'].to_list()}")
        print("\n===============================================")

    except FileNotFoundError:
        logging.error(f"El archivo de términos '{path_archivo_terminos}' no fue encontrado para imprimir el resumen.")
    except Exception as e:
        logging.error(f"Ocurrió un error al imprimir el resumen de términos: {e}")

def imprimir_resumen_metricas_red():
    """
    Imprime un resumen de las métricas de red calculadas.
    """
    try:
        results_dir = os.path.join(project_root_path, 'results')
        path_metricas_red = os.path.join(results_dir, 'exp_metricas_red_completas.csv')
        
        df_metricas = pd.read_csv(path_metricas_red, sep=';')
        
        print("\n===============================================")
        print(" 📊 RESUMEN DE MÉTRICAS DE RED")
        print("===============================================")
        
        for _, row in df_metricas.iterrows():
            print(f"\nDataset: {row['Dataset']} | Algoritmo: {row['Algoritmo']}")
            print(f"  📍 Nodos: {row['num_nodes']:,} | Enlaces: {row['num_edges']:,}")
            print(f"  📊 Densidad: {row['density']:.6f}")
            print(f"  🔗 Componentes conectados: {row['num_components']}")
            print(f"  📏 Grado promedio: {row['average_degree']:.2f} (max: {row['max_degree']}, min: {row['min_degree']})")
            
            if not pd.isna(row['average_clustering']):
                print(f"  🔄 Clustering promedio: {row['average_clustering']:.4f}")
            
            if not pd.isna(row['average_shortest_path_length']):
                print(f"  🛣️  Longitud promedio de camino: {row['average_shortest_path_length']:.2f}")
            
            if not pd.isna(row['small_world_sigma']):
                print(f"  🌍 Índice mundo pequeño (σ): {row['small_world_sigma']:.2f}")
                if row['small_world_sigma'] > 1:
                    print("    ✅ Características de mundo pequeño detectadas")
                else:
                    print("    ❌ No presenta mundo pequeño")
            
            print("  " + "-" * 50)
        
        print("\n===============================================")
        
    except FileNotFoundError:
        logging.warning("No se encontró el archivo de métricas de red para mostrar el resumen.")
    except Exception as e:
        logging.error(f"Error al mostrar resumen de métricas de red: {e}")


if __name__ == '__main__':
    path_terminos_generado = run_comparative_analysis()
    
    # Si el análisis se completó y generó un archivo de términos, lo imprimimos
    if path_terminos_generado:
        imprimir_resumen_terminos(path_terminos_generado)
        imprimir_resumen_metricas_red()