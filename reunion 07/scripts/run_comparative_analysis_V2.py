import sys
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import logging
from collections import Counter

# --- Configuración del Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Configuración de Ruta e importación de módulos del proyecto ---
try:
    # AJUSTE DE RUTA: Sube dos niveles desde /reunion 07/scripts/ para llegar a la raíz del proyecto
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root_path not in sys.path:
        sys.path.append(project_root_path)
    from python import data_processing, pipeline
except (ImportError, FileNotFoundError):
    logging.error("Error importando módulos. Asegúrate de que la estructura de carpetas es correcta y ejecutas desde `reunion 07/scripts/`.")
    sys.exit(1)

# --- Descarga de dependencias de NLTK (stopwords) ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logging.info("Descargando 'stopwords' de NLTK...")
    nltk.download('stopwords', quiet=True)


def analizar_terminos_tfidf(df_original: pd.DataFrame, df_nodes_clusters: pd.DataFrame,
                             nombre_dataset: str, nombre_algoritmo: str, tfidf_params: dict,
                             cluster_map: dict, ngram_config: dict):
    """
    Analiza los términos más relevantes por clúster promediando scores de TF-IDF.
    """
    ngram_range = ngram_config['range']
    ngram_type_name = ngram_config['name']
    
    logging.info(f"Analizando términos ({ngram_type_name}) para: {nombre_algoritmo} en {nombre_dataset}")

    # 1. PREPARACIÓN DE DATOS
    df_original['FROM_NODE_norm'] = df_original['FROM_NODE'].str.lower().str.strip()
    df_original['TO_NODE_norm'] = df_original['TO_NODE'].str.lower().str.strip()
    df_nodes_clusters['NODE_NAME_norm'] = df_nodes_clusters['NODE_NAME'].str.lower().str.strip()
    df_original['full_text'] = df_original['TITLE'].fillna('') + '. ' + df_original['BODY'].fillna('')
    
    mapa_nodo_cluster = pd.Series(df_nodes_clusters.CLUSTER_ASSIGNMENT.values, index=df_nodes_clusters.NODE_NAME_norm).to_dict()
    df_original['FROM_CLUSTER'] = df_original['FROM_NODE_norm'].map(mapa_nodo_cluster)
    df_original['TO_CLUSTER'] = df_original['TO_NODE_norm'].map(mapa_nodo_cluster)
    
    df_aristas_internas = df_original[
        (df_original['FROM_CLUSTER'] == df_original['TO_CLUSTER']) &
        (df_original['FROM_CLUSTER'].notna())
    ].copy()
    
    df_aristas_internas.rename(columns={'FROM_CLUSTER': 'CLUSTER_ID'}, inplace=True)
    
    df_docs = df_aristas_internas[df_aristas_internas['CLUSTER_ID'].isin(cluster_map.keys())]
    df_docs = df_docs.drop_duplicates(subset=['full_text']).reset_index(drop=True)

    if df_docs.empty or df_docs['CLUSTER_ID'].nunique() < len(cluster_map):
        logging.warning(f"No hay suficientes datos en ambos clústeres para un análisis en {nombre_algoritmo}.")
        return []

    # 2. CONSTRUCCIÓN DEL CORPUS Y ETIQUETAS
    corpus = df_docs['full_text'].tolist()
    labels = df_docs['CLUSTER_ID'].tolist()

    # 3. CONFIGURACIÓN Y EJECUCIÓN DE TF-IDF
    stop_words_es = list(nltk.corpus.stopwords.words('spanish'))
    stop_words_es.extend(['dijo', 'ser', 'si', 'solo', 'tambien', 'tras', 'chile', 'gobierno', 'presidente', 'ex', 'san', 'mil', 'años', 'país', 'además', 'aseguró'])
    
    vectorizer = TfidfVectorizer(
        stop_words=stop_words_es,
        ngram_range=ngram_range,
        max_df=tfidf_params['max_df'],
        min_df=tfidf_params.get('min_df', 5), 
        max_features=tfidf_params['max_features'],
        sublinear_tf=tfidf_params.get('sublinear_tf', False) 
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        logging.warning(f"No se pudieron extraer términos para {nombre_algoritmo} ({ngram_type_name}), corpus vacío o parámetros muy estrictos.")
        return []

    feature_names = np.array(vectorizer.get_feature_names_out())
    resultados_terminos = []
    
    # 4. CÁLCULO DE SCORES PROMEDIO POR CLÚSTER
    for cluster_id, cluster_name in cluster_map.items():
        indices_cluster = [i for i, label in enumerate(labels) if label == cluster_id]
        
        if not indices_cluster:
            continue
            
        matriz_cluster = tfidf_matrix[indices_cluster]
        scores_promedio = np.array(matriz_cluster.mean(axis=0)).flatten()
        
        top_indices = scores_promedio.argsort()[-tfidf_params['n_top_terms']:][::-1]

        for rank, idx in enumerate(top_indices):
            if scores_promedio[idx] > 1e-9:
                resultados_terminos.append({
                    'Dataset': nombre_dataset,
                    'Algoritmo': nombre_algoritmo,
                    'Tipo_Ngram': ngram_type_name,
                    'Cluster': cluster_name,
                    'Termino': feature_names[idx],
                    'Score_TFIDF_Avg': round(scores_promedio[idx], 5),
                    'Rank': rank + 1
                })

    return resultados_terminos

def analizar_nodos_centrales(df_original: pd.DataFrame, df_nodes_clusters: pd.DataFrame,
                              nombre_dataset: str, nombre_algoritmo: str, cluster_map: dict, n_top_nodes: int):
    """
    Identifica los nodos con mayor grado (más conexiones) dentro de cada clúster.
    """
    logging.info(f"Analizando nodos centrales para: {nombre_algoritmo} en {nombre_dataset}")
    
    df_nodes_clusters['NODE_NAME_norm'] = df_nodes_clusters['NODE_NAME'].str.lower().str.strip()
    
    mapa_nodo_cluster = pd.Series(df_nodes_clusters.CLUSTER_ASSIGNMENT.values, index=df_nodes_clusters.NODE_NAME_norm).to_dict()
    df_original['FROM_NODE_norm'] = df_original['FROM_NODE'].str.lower().str.strip()
    df_original['TO_NODE_norm'] = df_original['TO_NODE'].str.lower().str.strip()
    
    df_original['FROM_CLUSTER'] = df_original['FROM_NODE_norm'].map(mapa_nodo_cluster)
    df_original['TO_CLUSTER'] = df_original['TO_NODE_norm'].map(mapa_nodo_cluster)
    
    df_aristas_internas = df_original[
        (df_original['FROM_CLUSTER'] == df_original['TO_CLUSTER']) &
        (df_original['FROM_CLUSTER'].notna())
    ].copy()
    df_aristas_internas.rename(columns={'FROM_CLUSTER': 'CLUSTER_ID'}, inplace=True)

    resultados_nodos = []

    for cluster_id, cluster_name in cluster_map.items():
        df_cluster = df_aristas_internas[df_aristas_internas['CLUSTER_ID'] == cluster_id]
        
        if df_cluster.empty:
            continue
            
        nodos_participantes = pd.concat([df_cluster['FROM_NODE_norm'], df_cluster['TO_NODE_norm']])
        grado_nodos = nodos_participantes.value_counts().reset_index()
        grado_nodos.columns = ['Nodo', 'Grado']
        
        top_nodos = grado_nodos.head(n_top_nodes)
        
        for rank, row in top_nodos.iterrows():
            resultados_nodos.append({
                'Dataset': nombre_dataset,
                'Algoritmo': nombre_algoritmo,
                'Cluster': cluster_name,
                'Nodo': row['Nodo'],
                'Grado': row['Grado'],
                'Rank': rank + 1
            })
            
    return resultados_nodos

def analizar_fronteras_y_puentes(df_original: pd.DataFrame, df_nodes_clusters: pd.DataFrame,
                                 nombre_dataset: str, nombre_algoritmo: str, cluster_map: dict):
    """
    Analiza las fronteras de los clústeres y el rol de los nodos neutrales como puentes.
    """
    logging.info(f"Analizando fronteras y puentes para: {nombre_algoritmo} en {nombre_dataset}")

    df_nodes_clusters['NODE_NAME_norm'] = df_nodes_clusters['NODE_NAME'].str.lower().str.strip()
    mapa_nodo_cluster = pd.Series(df_nodes_clusters.CLUSTER_ASSIGNMENT.values, index=df_nodes_clusters.NODE_NAME_norm).to_dict()

    df_edges = df_original[['FROM_NODE', 'TO_NODE']].copy()
    df_edges['FROM_NODE_norm'] = df_edges['FROM_NODE'].str.lower().str.strip()
    df_edges['TO_NODE_norm'] = df_edges['TO_NODE'].str.lower().str.strip()
    
    df_edges['FROM_CLUSTER'] = df_edges['FROM_NODE_norm'].map(mapa_nodo_cluster).fillna(0)
    df_edges['TO_CLUSTER'] = df_edges['TO_NODE_norm'].map(mapa_nodo_cluster).fillna(0)

    nodos_no_neutrales = df_nodes_clusters['NODE_NAME_norm'].unique()
    todos_los_nodos = set(df_edges['FROM_NODE_norm']) | set(df_edges['TO_NODE_norm'])
    nodos_neutrales = todos_los_nodos - set(nodos_no_neutrales)

    resultados_fronteras = []
    
    for cluster_id, cluster_name in cluster_map.items():
        nodos_del_cluster = set(df_nodes_clusters[df_nodes_clusters['CLUSTER_ASSIGNMENT'] == cluster_id]['NODE_NAME_norm'])
        
        if not nodos_del_cluster:
            continue
            
        frontera = set()
        conexiones_externas = Counter()
        
        df_salientes = df_edges[df_edges['FROM_NODE_norm'].isin(nodos_del_cluster) & ~df_edges['TO_NODE_norm'].isin(nodos_del_cluster)]
        df_entrantes = df_edges[df_edges['TO_NODE_norm'].isin(nodos_del_cluster) & ~df_edges['FROM_NODE_norm'].isin(nodos_del_cluster)]

        for _, row in df_salientes.iterrows():
            frontera.add(row['FROM_NODE_norm'])
            conexiones_externas[row['FROM_NODE_norm']] += 1
        for _, row in df_entrantes.iterrows():
            frontera.add(row['TO_NODE_norm'])
            conexiones_externas[row['TO_NODE_norm']] += 1

        permeabilidad = len(frontera) / len(nodos_del_cluster) if len(nodos_del_cluster) > 0 else 0
        
        resultados_fronteras.append({
            'Dataset': nombre_dataset,
            'Algoritmo': nombre_algoritmo,
            'Tipo_Analisis': 'Cluster_Permeability',
            'Cluster': cluster_name,
            'Metrica': 'Permeabilidad',
            'Valor': round(permeabilidad, 4)
        })

        for nodo, n_conexiones in conexiones_externas.most_common(5):
             resultados_fronteras.append({
                'Dataset': nombre_dataset,
                'Algoritmo': nombre_algoritmo,
                'Tipo_Analisis': 'Frontera_Nodo_Clave',
                'Cluster': cluster_name,
                'Nodo': nodo,
                'Conexiones_Externas': n_conexiones
            })

    for nodo_neutral in nodos_neutrales:
        conexiones_a_clusters = Counter()
        
        df_neutral_participa = df_edges[(df_edges['FROM_NODE_norm'] == nodo_neutral) | (df_edges['TO_NODE_norm'] == nodo_neutral)]
        
        for _, row in df_neutral_participa.iterrows():
            if row['FROM_NODE_norm'] == nodo_neutral and row['TO_CLUSTER'] in cluster_map:
                conexiones_a_clusters[row['TO_CLUSTER']] += 1
            if row['TO_NODE_norm'] == nodo_neutral and row['FROM_CLUSTER'] in cluster_map:
                conexiones_a_clusters[row['FROM_CLUSTER']] += 1
        
        if 1 in conexiones_a_clusters and -1 in conexiones_a_clusters:
            total_conexiones = sum(conexiones_a_clusters.values())
            bias = (conexiones_a_clusters[1] - conexiones_a_clusters[-1]) / total_conexiones if total_conexiones > 0 else 0
            
            resultados_fronteras.append({
                'Dataset': nombre_dataset,
                'Algoritmo': nombre_algoritmo,
                'Tipo_Analisis': 'Neutral_Puente',
                'Nodo': nodo_neutral,
                'Conexiones_S1': conexiones_a_clusters.get(1, 0),
                'Conexiones_S2': conexiones_a_clusters.get(-1, 0),
                'Sesgo_Inclinacion': round(bias, 4)
            })

    return resultados_fronteras

def ejecutar_experimento_individual(nombre_dataset, df_dataset, config_base, default_cols, tfidf_params, cluster_map, ngram_configs, n_top_nodes):
    logging.info(f"Iniciando experimento completo para: {config_base['name']} en {nombre_dataset}")
    
    df_nodes_results, metricas_core, _, _, _ = pipeline.ejecutar_analisis_polarizacion(
        df_input=df_dataset.copy(), config=config_base.copy(), default_cols=default_cols, calculate_intra_cluster_cc=True
    )
    
    todos_los_terminos = []
    todos_los_nodos = []
    todos_los_fronteras = []

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

        fronteras_encontradas = analizar_fronteras_y_puentes(
            df_original=df_dataset.copy(), df_nodes_clusters=df_nodes_results.copy(),
            nombre_dataset=nombre_dataset, nombre_algoritmo=config_base['name'],
            cluster_map=cluster_map
        )
        if fronteras_encontradas:
            todos_los_fronteras.extend(fronteras_encontradas)

    metricas_core['Dataset'] = nombre_dataset
    logging.info(f"Finalizado: {config_base['name']} en {nombre_dataset}")
    
    return (metricas_core, todos_los_terminos, todos_los_nodos, todos_los_fronteras)

def run_comparative_analysis():
    logging.info("--- INICIANDO EXPERIMENTO COMPARATIVO (CON ANÁLISIS DE FRONTERAS) ---")

    # ==================================================================
    # === 🚀 CONFIGURACIÓN CENTRAL DEL EXPERIMENTO 🚀 ===
    # ==================================================================
    TFIDF_PARAMS = {
        "max_features": 50000, "max_df": 0.85, "min_df": 30,
        "sublinear_tf": True, "n_top_terms": 30
    }
    
    NGRAM_CONFIGS = [
        {"name": "Unigramas", "range": (1, 1)},
        {"name": "Bigramas", "range": (2, 2)},
        {"name": "Uni_y_Bigramas", "range": (1, 2)}
    ]
    
    DEFAULT_COLS = {'from_node_col': 'FROM_NODE', 'to_node_col': 'TO_NODE', 'sign_col': 'SIGN'}
    K = 2
    CLUSTER_MAP = {1: 'S1', -1: 'S2'}
    N_TOP_NODES = 5

    LISTA_EXPERIMENTOS = [
        {"name": "Local Search (b=0.01)", "algorithm_type": "local_search_paper_k2", "k": K, "ls_beta": 0.01, "ls_max_iter": 20},
    ]
    # ==================================================================

    logging.info("--- Cargando Datos ---")
    try:
        # AJUSTE DE RUTA: Usa la ruta raíz del proyecto para encontrar los datos
        ruta_plebiscito_2022 = os.path.join(project_root_path, 'News', 'output', 'df_plebiscito_2022.csv')
        df_pre, df_post = data_processing.cargar_datos_2022_pre_post(ruta_plebiscito_2022)
        datasets = {"Plebiscito_PRE": df_pre.head(5000), "Plebiscito_POST": df_post.head(5000)}
        logging.info(f"Datos cargados. Usando una muestra de {len(df_pre.head(5000))} noticias por dataset.")
    except FileNotFoundError:
        logging.error(f"No se encontró el archivo de datos '{ruta_plebiscito_2022}'. Abortando.")
        return

    tareas = [(nombre, df, config, DEFAULT_COLS, TFIDF_PARAMS, CLUSTER_MAP, NGRAM_CONFIGS, N_TOP_NODES) 
              for nombre, df in datasets.items() 
              for config in LISTA_EXPERIMENTOS if not df.empty]

    logging.info(f"Se ejecutarán {len(tareas)} experimentos base en paralelo.")
    
    resultados_paralelos = Parallel(n_jobs=-1, backend="loky")(
        delayed(ejecutar_experimento_individual)(*task) for task in tareas
    )
    
    # Procesamiento de resultados
    resultados_core = [res[0] for res in resultados_paralelos if res and res[0]]
    resultados_terminos = [item for res in resultados_paralelos if res and res[1] for item in res[1]]
    resultados_nodos = [item for res in resultados_paralelos if res and res[2] for item in res[2]]
    resultados_fronteras = [item for res in resultados_paralelos if res and res[3] for item in res[3]]

    df_resultados_core = pd.DataFrame(resultados_core)
    df_resultados_terminos = pd.DataFrame(resultados_terminos)
    df_resultados_nodos = pd.DataFrame(resultados_nodos)
    df_resultados_fronteras = pd.DataFrame(resultados_fronteras)

    # AJUSTE DE RUTA: Guarda los resultados en la carpeta /reunion 07/resultados/
    results_dir = os.path.join(project_root_path, 'reunion 07', 'resultados')
    os.makedirs(results_dir, exist_ok=True)
    
    path_metricas = os.path.join(results_dir, 'exp_metricas_polarizacion.csv')
    path_terminos = os.path.join(results_dir, 'exp_analisis_terminos.csv')
    path_nodos = os.path.join(results_dir, 'exp_analisis_nodos.csv')
    path_fronteras = os.path.join(results_dir, 'exp_analisis_fronteras.csv')
    
    df_resultados_core.to_csv(path_metricas, index=False, sep=';', encoding='utf-8-sig')
    df_resultados_terminos.to_csv(path_terminos, index=False, sep=';', encoding='utf-8-sig')
    df_resultados_nodos.to_csv(path_nodos, index=False, sep=';', encoding='utf-8-sig')
    df_resultados_fronteras.to_csv(path_fronteras, index=False, sep=';', encoding='utf-8-sig')
    
    logging.info("--- Resultados guardados exitosamente ---")
    logging.info(f"Métricas de polarización: {path_metricas}")
    logging.info(f"Análisis de Términos (TF-IDF): {path_terminos}")
    logging.info(f"Análisis de Nodos Centrales: {path_nodos}")
    logging.info(f"Análisis de Fronteras y Puentes: {path_fronteras}")

if __name__ == '__main__':
    run_comparative_analysis()