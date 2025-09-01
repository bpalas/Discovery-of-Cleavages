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
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root_path not in sys.path:
        sys.path.append(project_root_path)
    from python import data_processing, pipeline
except (ImportError, FileNotFoundError):
    logging.error("Error importando módulos. Asegúrate de que la estructura de carpetas es correcta.")
    sys.exit(1)

# --- Descarga de dependencias de NLTK (stopwords) ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logging.info("Descargando 'stopwords' de NLTK...")
    nltk.download('stopwords', quiet=True)


def analizar_terminos_tfidf(df_original: pd.DataFrame, df_nodes_clusters: pd.DataFrame,
                             nombre_dataset: str, nombre_algoritmo: str, tfidf_params: dict,
                             cluster_map: dict, ngram_config: dict): # <-- Añadido ngram_config
    """
    Analiza los términos más relevantes por clúster promediando scores de TF-IDF.
    """
    # Esta función ahora recibe el ngram_config para saber qué n-grams usar y cómo etiquetarlos.
    ngram_range = ngram_config['range']
    ngram_type_name = ngram_config['name']
    
    logging.info(f"Analizando términos ({ngram_type_name}) para: {nombre_algoritmo} en {nombre_dataset}")

    # 1. PREPARACIÓN DE DATOS (sin cambios)
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

    # 2. CONSTRUCCIÓN DEL CORPUS Y ETIQUETAS (sin cambios)
    corpus = df_docs['full_text'].tolist()
    labels = df_docs['CLUSTER_ID'].tolist()

    # 3. CONFIGURACIÓN Y EJECUCIÓN DE TF-IDF
    stop_words_es = list(nltk.corpus.stopwords.words('spanish'))
    stop_words_es.extend(['dijo', 'ser', 'si', 'solo', 'tambien', 'tras', 'chile', 'gobierno', 'presidente', 'ex', 'san', 'mil', 'años', 'país', 'además', 'aseguró'])
    
    vectorizer = TfidfVectorizer(
        stop_words=stop_words_es,
        ngram_range=ngram_range, # <-- Usa el rango del config actual
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
                    'Tipo_Ngram': ngram_type_name, # <-- NUEVA COLUMNA
                    'Cluster': cluster_name,
                    'Termino': feature_names[idx],
                    'Score_TFIDF_Avg': round(scores_promedio[idx], 5),
                    'Rank': rank + 1
                })

    return resultados_terminos

# --- NUEVA FUNCIÓN ---
# --- NUEVA FUNCIÓN (CORREGIDA) ---
def analizar_nodos_centrales(df_original: pd.DataFrame, df_nodes_clusters: pd.DataFrame,
                             nombre_dataset: str, nombre_algoritmo: str, cluster_map: dict, n_top_nodes: int):
    """
    Identifica los nodos con mayor grado (más conexiones) dentro de cada clúster.
    """
    logging.info(f"Analizando nodos centrales para: {nombre_algoritmo} en {nombre_dataset}")
    
    # --- LÍNEA AÑADIDA PARA SOLUCIONAR EL ERROR ---
    # Se crea la columna normalizada antes de usarla.
    df_nodes_clusters['NODE_NAME_norm'] = df_nodes_clusters['NODE_NAME'].str.lower().str.strip()
    
    # Mapeo de nodos a clústeres
    mapa_nodo_cluster = pd.Series(df_nodes_clusters.CLUSTER_ASSIGNMENT.values, index=df_nodes_clusters.NODE_NAME_norm).to_dict()
    df_original['FROM_NODE_norm'] = df_original['FROM_NODE'].str.lower().str.strip() # Normalizamos para el mapeo
    df_original['TO_NODE_norm'] = df_original['TO_NODE'].str.lower().str.strip()   # Normalizamos para el mapeo
    
    df_original['FROM_CLUSTER'] = df_original['FROM_NODE_norm'].map(mapa_nodo_cluster)
    df_original['TO_CLUSTER'] = df_original['TO_NODE_norm'].map(mapa_nodo_cluster)
    
    # Filtrar aristas intra-clúster
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
            
        # Contar la frecuencia de cada nodo en las aristas del clúster
        # Usamos los nodos normalizados para el conteo de grado
        nodos_participantes = pd.concat([df_cluster['FROM_NODE_norm'], df_cluster['TO_NODE_norm']])
        grado_nodos = nodos_participantes.value_counts().reset_index()
        grado_nodos.columns = ['Nodo', 'Grado']
        
        # Obtener los N nodos con mayor grado
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

# --- FUNCIÓN DE EJECUCIÓN MODIFICADA ---
def ejecutar_experimento_individual(nombre_dataset, df_dataset, config_base, default_cols, tfidf_params, cluster_map, ngram_configs, n_top_nodes):
    """
    Ejecuta un análisis completo: polarización, análisis de términos para cada config de n-gram, y análisis de nodos.
    """
    logging.info(f"Iniciando experimento completo para: {config_base['name']} en {nombre_dataset}")
    
    # 1. Ejecutar análisis de polarización (solo una vez por experimento)
    df_nodes_results, metricas_core, _, _, _ = pipeline.ejecutar_analisis_polarizacion(
        df_input=df_dataset.copy(), config=config_base.copy(), default_cols=default_cols, calculate_intra_cluster_cc=True
    )
    
    # Listas para almacenar resultados de este experimento
    todos_los_terminos = []
    todos_los_nodos = []

    if df_nodes_results is not None and not df_nodes_results.empty:
        # 2. Iterar sobre las configuraciones de n-grams para el análisis de términos
        for ngram_config in ngram_configs:
            # Crear una copia de los parámetros para no modificarlos globalmente
            params_actuales = tfidf_params.copy()
            
            terminos_encontrados = analizar_terminos_tfidf(
                df_original=df_dataset.copy(), df_nodes_clusters=df_nodes_results.copy(),
                nombre_dataset=nombre_dataset, nombre_algoritmo=config_base['name'],
                tfidf_params=params_actuales, cluster_map=cluster_map,
                ngram_config=ngram_config # Pasa la configuración específica
            )
            if terminos_encontrados:
                todos_los_terminos.extend(terminos_encontrados)
        
        # 3. Analizar nodos centrales (solo una vez por experimento)
        nodos_encontrados = analizar_nodos_centrales(
            df_original=df_dataset.copy(), df_nodes_clusters=df_nodes_results.copy(),
            nombre_dataset=nombre_dataset, nombre_algoritmo=config_base['name'],
            cluster_map=cluster_map, n_top_nodes=n_top_nodes
        )
        if nodos_encontrados:
            todos_los_nodos.extend(nodos_encontrados)

    metricas_core['Dataset'] = nombre_dataset
    logging.info(f"Finalizado: {config_base['name']} en {nombre_dataset}")
    
    # Devuelve las métricas, la lista de términos de todos los n-grams, y la lista de nodos
    return (metricas_core, todos_los_terminos, todos_los_nodos)

def run_comparative_analysis():
    """Función principal que orquesta todo el análisis."""
    logging.info("--- INICIANDO EXPERIMENTO COMPARATIVO (Múltiples N-Grams y Análisis de Nodos) ---")

    # ==================================================================
    # === 🚀 CONFIGURACIÓN CENTRAL DEL EXPERIMENTO 🚀 ===
    # ==================================================================
    
    TFIDF_PARAMS = {
        "max_features": 50000,
        "max_df": 0.85,
        "min_df": 30,
        "sublinear_tf": True,
        "n_top_terms": 30
    }
    
    # --- NUEVA CONFIGURACIÓN PARA LOS N-GRAMS ---
    NGRAM_CONFIGS = [
        {"name": "Unigramas", "range": (1, 1)},
        {"name": "Bigramas", "range": (2, 2)},
        {"name": "Uni_y_Bigramas", "range": (1, 2)}
    ]
    
    DEFAULT_COLS = {'from_node_col': 'FROM_NODE', 'to_node_col': 'TO_NODE', 'sign_col': 'SIGN'}
    K = 2
    CLUSTER_MAP = {1: 'S1', -1: 'S2'}
    N_TOP_NODES = 5 # Número de nodos centrales a extraer

    LISTA_EXPERIMENTOS = [
        {"name": "Eigensign", "algorithm_type": "eigensign", "eigen_solver": "scipy"},
        {"name": "Random Eigensign", "algorithm_type": "random_eigensign", "num_runs": 50, "eigen_solver": "scipy"},
        {"name": "Local Search (b=0.01)", "algorithm_type": "local_search_paper_k2", "k": K, "ls_beta": 0.01, "ls_max_iter": 20},
        {"name": "SCG (max_obj)", "algorithm_type": "scg", "K": K, "rounding_strategy": "max_obj"},
    ]
    
    # ==================================================================

    logging.info("--- Cargando Datos ---")
    try:
        ruta_plebiscito_2022 = os.path.join(project_root_path, 'News', 'output', 'primarias2025.csv')
        df_pre, df_post = data_processing.cargar_datos_2022_pre_post(ruta_plebiscito_2022)
        datasets = {"Plebiscito_PRE": df_pre.head(5000), "Plebiscito_POST": df_post.head(5000)}
        logging.info(f"Datos cargados. Usando una muestra de {len(df_pre.head(5000))} noticias por dataset.")
    except FileNotFoundError:
        logging.error(f"No se encontró el archivo de datos '{ruta_plebiscito_2022}'. Abortando.")
        return

    # Se modifican las tareas para pasar los nuevos parámetros
    tareas = [(nombre, df, config, DEFAULT_COLS, TFIDF_PARAMS, CLUSTER_MAP, NGRAM_CONFIGS, N_TOP_NODES) 
              for nombre, df in datasets.items() 
              for config in LISTA_EXPERIMENTOS if not df.empty]

    logging.info(f"Se ejecutarán {len(LISTA_EXPERIMENTOS) * len(datasets)} experimentos base en paralelo.")
    
    resultados_paralelos = Parallel(n_jobs=-1, backend="loky")(
        delayed(ejecutar_experimento_individual)(*task) for task in tareas
    )
    
    # Procesamiento de resultados
    resultados_core = [res[0] for res in resultados_paralelos if res]
    resultados_terminos_list = [res[1] for res in resultados_paralelos if res and res[1]]
    resultados_nodos_list = [res[2] for res in resultados_paralelos if res and res[2]]

    # Aplanar las listas de listas
    resultados_terminos = [item for sublist in resultados_terminos_list for item in sublist]
    resultados_nodos = [item for sublist in resultados_nodos_list for item in sublist]

    df_resultados_core = pd.DataFrame(resultados_core)
    df_resultados_terminos = pd.DataFrame(resultados_terminos)
    df_resultados_nodos = pd.DataFrame(resultados_nodos) # <-- Nuevo DataFrame

    # Guardado de resultados
    results_dir = os.path.join(project_root_path, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Nombres de archivo actualizados
    path_metricas = os.path.join(results_dir, 'exp_metricas_polarizacion.csv')
    path_terminos = os.path.join(results_dir, 'exp_analisis_terminos.csv')
    path_nodos = os.path.join(results_dir, 'exp_analisis_nodos.csv') # <-- Nuevo archivo
    
    df_resultados_core.to_csv(path_metricas, index=False, sep=';', encoding='utf-8-sig')
    df_resultados_terminos.to_csv(path_terminos, index=False, sep=';', encoding='utf-8-sig')
    df_resultados_nodos.to_csv(path_nodos, index=False, sep=';', encoding='utf-8-sig') # <-- Guardar nodos
    
    logging.info("--- Resultados guardados exitosamente ---")
    logging.info(f"Métricas de polarización: {path_metricas}")
    logging.info(f"Análisis de Términos (TF-IDF): {path_terminos}")
    logging.info(f"Análisis de Nodos Centrales: {path_nodos}") # <-- Log del nuevo archivo

if __name__ == '__main__':
    run_comparative_analysis()