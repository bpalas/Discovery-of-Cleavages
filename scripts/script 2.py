import sys
import os
import pandas as pd
# --- MODIFICADO: Cambiamos CountVectorizer por TfidfVectorizer ---
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np

# --- Configuración de Ruta ---
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root_path not in sys.path:
    sys.path.append(project_root_path)

from python import data_processing, pipeline

# --- FUNCIÓN MODIFICADA: Ahora usa TF-IDF + Bigramas ---
def analizar_texto_clusters(df_original: pd.DataFrame, df_nodes_clusters: pd.DataFrame, nombre_dataset: str, nombre_algoritmo: str):
    """
    Analiza el texto usando TF-IDF y N-gramas (1 y 2) para extraer los
    términos más distintivos de cada cluster.
    """
    print(f"--- Analizando texto con TF-IDF para: {nombre_algoritmo} ---")
    
    try:
        stop_words_es = stopwords.words('spanish')
    except LookupError:
        print("Error: Stopwords de NLTK no descargadas. Usando lista básica.")
        stop_words_es = ['de', 'la', 'que', 'el', 'en', 'y', 'a', 'los']
    
    df_original['full_text'] = df_original['TITLE'].fillna('') + ' ' + df_original['BODY'].fillna('')
    df_original['FROM_NODE_norm'] = df_original['FROM_NODE'].str.lower().str.strip()
    df_original['TO_NODE_norm'] = df_original['TO_NODE'].str.lower().str.strip()
    
    # Paso 1: Preparar un corpus por cada cluster
    corpus_por_cluster = {}
    for cluster_id, cluster_name in zip([1, -1], ['S1', 'S2']):
        nodos_cluster = df_nodes_clusters[df_nodes_clusters['CLUSTER_ASSIGNMENT'] == cluster_id]['NODE_NAME'].tolist()
        if not nodos_cluster:
            continue
        
        df_textos_cluster = df_original[
            df_original['FROM_NODE_norm'].isin(nodos_cluster) | 
            df_original['TO_NODE_norm'].isin(nodos_cluster)
        ]
        corpus_por_cluster[cluster_name] = " ".join(df_textos_cluster['full_text'].tolist())

    # Si no tenemos texto en al menos un cluster, no podemos comparar
    if len(corpus_por_cluster) < 1:
        print("Advertencia: No hay suficiente texto en los clusters para el análisis TF-IDF.")
        return []

    # Paso 2: Aplicar TF-IDF a los corpus de todos los clusters a la vez
    # ngram_range=(1, 2) incluye palabras sueltas y pares de palabras (bigramas)
    vectorizer = TfidfVectorizer(
    stop_words=stop_words_es,
    ngram_range=(1, 2), # Palabras sueltas y bigramas
    max_df=0.9,         # Ignora términos que aparecen en más del 90% de los documentos
    min_df=3            # Ignora términos que aparecen menos de 3 veces en todo el corpus
)
    
    # Obtenemos los nombres de los clusters y sus textos
    cluster_names = list(corpus_por_cluster.keys())
    corpus_list = list(corpus_por_cluster.values())
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus_list)
    except ValueError:
        print("Error al procesar texto (puede estar vacío después de quitar stopwords).")
        return []

    feature_names = np.array(vectorizer.get_feature_names_out())
    resultados_terminos = []

    # Paso 3: Extraer los top 10 términos para cada cluster
    for i, cluster_name in enumerate(cluster_names):
        # Obtiene la fila de la matriz correspondiente a este cluster
        scores = tfidf_matrix[i, :].toarray().flatten()
        # Obtiene los índices de los 10 scores más altos
        top_indices = scores.argsort()[-10:][::-1]
        
        for idx in top_indices:
            resultados_terminos.append({
                'Dataset': nombre_dataset,
                'Algoritmo': nombre_algoritmo,
                'Cluster': cluster_name,
                'Termino': feature_names[idx],
                'Score_TFIDF': scores[idx]
            })

    return resultados_terminos


def run_comparative_analysis():
    """
    Función principal para ejecutar y comparar múltiples algoritmos.
    """
    print("--- INICIANDO EXPERIMENTO COMPARATIVO DE ALGORITMOS ---")

    # --- Definir configuraciones (sin cambios) ---
    default_cols = {
        'from_node_col': 'FROM_NODE', 'to_node_col': 'TO_NODE', 'sign_col': 'SIGN',
        "node_norm_strategy": "lower_unidecode_strip", "weighting_strategy": "binary_sum_signs_actual"
    }
    k = 2
    lista_experimentos = [
        {"name": "Eigensign (Determinista)", "algorithm_type": "eigensign", "eigen_solver": "scipy"},
        {"name": "Random Eigensign (Probabilístico)", "algorithm_type": "random_eigensign", "num_runs": 50, "eigen_solver": "scipy"},
        {"name": "Local Search (beta=0.01)", "algorithm_type": "local_search_paper_k2", "k": k, "ls_beta": 0.01, "ls_max_iter": 20},
        {"name": "Local Search (beta=0.005)", "algorithm_type": "local_search_paper_k2", "k": k, "ls_beta": 0.005, "ls_max_iter": 20},
        {"name": "SCG (max_obj)", "algorithm_type": "scg", "K": 2, "rounding_strategy": "max_obj"},
        {"name": "SCG (randomized)", "algorithm_type": "scg", "K": 2, "rounding_strategy": "randomized"},
        {"name": "SCG (bansal)", "algorithm_type": "scg", "K": 2, "rounding_strategy": "bansal"}
    ]
    
    # --- Cargar datos (sin cambios) ---
    print("\n--- Cargando Datos ---")
    try:
        ruta_plebiscito_2022 = os.path.join(project_root_path, 'News', 'output', 'df_plebiscito_2022.csv')
        df_pre, df_post = data_processing.cargar_datos_2022_pre_post(ruta_plebiscito_2022)
        datasets_a_procesar = {"Plebiscito PRE": df_pre, "Plebiscito POST": df_post}
        print("Datos cargados correctamente.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de datos en la ruta '{ruta_plebiscito_2022}'.")
        return

    # --- Ejecutar experimentos ---
    resultados_core = []
    resultados_paper = []
    # --- MODIFICADO: El nombre de la lista de resultados para mayor claridad ---
    resultados_tfidf = []

    for nombre_dataset, df_dataset in datasets_a_procesar.items():
        print(f"\n========================================================")
        print(f"--- PROCESANDO DATASET: {nombre_dataset} ---")
        print(f"========================================================")
        if df_dataset.empty:
            print(f"El dataset '{nombre_dataset}' está vacío. Saltando...")
            continue
        for config_base in lista_experimentos:
            print(f"\n--- Ejecutando algoritmo: {config_base['name']} ---")
            config_actual = config_base.copy()
            df_nodes_results, metricas_core, metricas_paper, _, _ = pipeline.ejecutar_analisis_polarizacion(
                df_input=df_dataset, config=config_actual,
                default_cols=default_cols, calculate_intra_cluster_cc=True
            )
            if df_nodes_results is not None and not df_nodes_results.empty:
                top_terminos = analizar_texto_clusters(
                    df_original=df_dataset, df_nodes_clusters=df_nodes_results,
                    nombre_dataset=nombre_dataset, nombre_algoritmo=config_base['name']
                )
                resultados_tfidf.extend(top_terminos)
            metricas_core['Dataset'] = nombre_dataset
            metricas_paper['Dataset'] = nombre_dataset
            resultados_core.append(metricas_core)
            resultados_paper.append(metricas_paper)

    # --- Consolidar y mostrar resultados (sin cambios en la lógica) ---
    if not resultados_core:
        print("No se generaron resultados. Finalizando experimento.")
        return
    df_resultados_core = pd.DataFrame(resultados_core)
    df_resultados_paper = pd.DataFrame(resultados_paper)
    df_resultados_tfidf = pd.DataFrame(resultados_tfidf)

    # ... (código para mostrar las tablas 1 y 2 sin cambios) ...

    # --- Guardar resultados en CSV ---
    try:
        results_dir = os.path.join(project_root_path, 'results')
        os.makedirs(results_dir, exist_ok=True)
        # ... (código para guardar metricas_core y metricas_paper sin cambios) ...
        
        # --- MODIFICADO: Guardar el nuevo CSV con los resultados de TF-IDF ---
        if not df_resultados_tfidf.empty:
            ruta_tfidf = os.path.join(results_dir, 'comparacion_terminos_tfidf.csv')
            df_resultados_tfidf.to_csv(ruta_tfidf, index=False)
            print(f"Resultados de términos TF-IDF guardados en: {ruta_tfidf}")
    except Exception as e:
        print(f"\nError al guardar los resultados: {e}")

if __name__ == '__main__':
    run_comparative_analysis()

    