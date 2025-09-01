# scripts/ejecutar_analisis_temas.py

import sys
import os
import pandas as pd
import logging
import nltk 
from sklearn.feature_extraction.text import CountVectorizer 

# --- Dependencias para Topic Modeling ---
# Se encapsulan para que el script principal no falle si no están instaladas.
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("="*80)
    print("ERROR: Faltan librerías para el Topic Modeling.")
    print("Por favor, instálalas ejecutando el siguiente comando en tu terminal:")
    print("pip install bertopic'[visualization]' sentence-transformers")
    print("="*80)
    sys.exit(1)

# --- Configuración del Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# --- Configuración de Ruta e importación de módulos del proyecto ---
try:
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root_path not in sys.path:
        sys.path.append(project_root_path)
    from python import data_processing
except (ImportError, FileNotFoundError):
    logging.error("Error importando módulos. Asegúrate de que la estructura de carpetas es correcta.")
    sys.exit(1)


# --- FUNCIONES AUXILIARES ---

def crear_corpus(df_noticias_periodo, df_nodos_periodo, cluster_name):
    """Prepara la lista de textos (corpus) para un clúster y período específicos."""
    mapa_nodo_cluster = pd.Series(df_nodos_periodo['Cluster'].values, index=df_nodos_periodo['Nodo'].str.lower()).to_dict()
    
    df_noticias_periodo['FROM_CLUSTER'] = df_noticias_periodo['FROM_NODE'].str.lower().map(mapa_nodo_cluster)
    df_noticias_periodo['TO_CLUSTER'] = df_noticias_periodo['TO_NODE'].str.lower().map(mapa_nodo_cluster)
    
    df_aristas_internas = df_noticias_periodo[
        (df_noticias_periodo['FROM_CLUSTER'] == cluster_name) & 
        (df_noticias_periodo['TO_CLUSTER'] == cluster_name)
    ].copy()
    
    # Prepara el texto completo y devuelve una lista de documentos únicos.
    df_aristas_internas['full_text'] = df_aristas_internas['TITLE'].fillna('') + '. ' + df_aristas_internas['BODY'].fillna('')
    return df_aristas_internas['full_text'].drop_duplicates().tolist()

def analizar_temas_bertopic(corpus, metadata, topic_model_config):
    """Ejecuta BERTopic sobre un corpus y devuelve un DataFrame con los resultados."""
    if not corpus or len(corpus) < topic_model_config['min_topic_size']:
        logging.warning(f"No hay suficientes documentos en '{metadata['Dataset']} - {metadata['Cluster']}' para analizar. Se omite.")
        return pd.DataFrame()

    logging.info(f"Analizando temas para: {metadata['Dataset']} - {metadata['Cluster']} ({len(corpus)} documentos)")
    
    # Instancia el modelo con la configuración dada.
    topic_model = BERTopic(**topic_model_config)
    
    # Entrena el modelo y obtiene los temas.
    topics, _ = topic_model.fit_transform(corpus)
    
    # Obtiene la tabla de resultados.
    df_temas = topic_model.get_topic_info()
    
    # Enriquece la tabla con metadatos para poder identificar los resultados.
    df_temas['Dataset'] = metadata['Dataset']
    df_temas['Cluster'] = metadata['Cluster']
    
    return df_temas


# --- FUNCIÓN PRINCIPAL ---
# Reemplaza la función original con esta
def run_topic_modeling_analysis():
    """Orquesta el análisis de Topic Modeling para el algoritmo ganador."""
    
    # ==================================================================
    # === 🚀 CONFIGURACIÓN CENTRAL DEL ANÁLISIS DE TEMAS 🚀 ===
    # ==================================================================
    ALGORITMO_GANADOR = 'Local Search (b=0.01)'
    CLUSTER_S1_NAME = 'S1_'
    CLUSTER_S2_NAME = 'S2'

    # --- ¡NUEVO! Configuración del Vectorizer para eliminar Stop Words ---
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logging.info("Descargando 'stopwords' de NLTK...")
        nltk.download('stopwords', quiet=True)
    
    stop_words_es = list(nltk.corpus.stopwords.words('spanish'))
    # Añadimos palabras muy comunes en el contexto de noticias que no aportan valor temático
    stop_words_es.extend(['dijo', 'ser', 'si', 'solo', 'tambien', 'tras', 'chile', 'gobierno', 'presidente', 'ex', 'san', 'mil', 'años', 'país', 'además', 'aseguró', 'hizo', 'van', 'mil', 'millones'])
    
    # Creamos el modelo que contará las palabras, ignorando las stop words
    vectorizer = CountVectorizer(stop_words=stop_words_es)

    # --- Configuración de BERTopic (actualizada) ---
    TOPIC_MODEL_CONFIG = {
        "embedding_model": SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2"),
        "vectorizer_model": vectorizer, # <--- ¡AQUÍ ESTÁ LA MAGIA!
        "language": "multilingual",
        "min_topic_size": 20, 
        "verbose": True
    }
    
    # Rutas a los archivos de entrada y salida
    results_dir = os.path.join(project_root_path, 'results')
    path_nodos_input = os.path.join(results_dir, 'exp_analisis_nodos.csv')
    path_noticias_input = os.path.join(project_root_path, 'News', 'output', 'df_plebiscito_2022.csv')
    path_temas_output = os.path.join(results_dir, 'analisis_topic_modeling.csv')
    # ==================================================================

    logging.info(f"--- INICIANDO ANÁLISIS DE TOPIC MODELING PARA: '{ALGORITMO_GANADOR}' (con filtro de Stop Words) ---")

    # 1. Cargar datos necesarios
    try:
        logging.info("Cargando datos de noticias y resultados de clustering...")
        df_nodos_total = pd.read_csv(path_nodos_input, sep=';')
        df_pre, df_post = data_processing.cargar_datos_2022_pre_post(path_noticias_input)
    except FileNotFoundError as e:
        logging.error(f"Error: No se encontró un archivo de entrada necesario: {e}. Asegúrate de haber ejecutado el script anterior primero.")
        return

    # 2. Filtrar los resultados de nodos para el algoritmo ganador
    df_nodos_filtrados = df_nodos_total[df_nodos_total['Algoritmo'] == ALGORITMO_GANADOR]
    nodos_pre = df_nodos_filtrados[df_nodos_filtrados['Dataset'] == 'Plebiscito_PRE']
    nodos_post = df_nodos_filtrados[df_nodos_filtrados['Dataset'] == 'Plebiscito_POST']

    # 3. Preparar los 4 corpus
    logging.info("Preparando los 4 corpus de texto (PRE/POST y S1/S2)...")
    corpus_pre_s1 = crear_corpus(df_pre, nodos_pre, CLUSTER_S1_NAME)
    corpus_pre_s2 = crear_corpus(df_pre, nodos_pre, CLUSTER_S2_NAME)
    corpus_post_s1 = crear_corpus(df_post, nodos_post, CLUSTER_S1_NAME)
    corpus_post_s2 = crear_corpus(df_post, nodos_post, CLUSTER_S2_NAME)

    # Lista de tareas a ejecutar
    tareas = [
        (corpus_pre_s1, {'Dataset': 'Plebiscito_PRE', 'Cluster': CLUSTER_S1_NAME}),
        (corpus_pre_s2, {'Dataset': 'Plebiscito_PRE', 'Cluster': CLUSTER_S2_NAME}),
        (corpus_post_s1, {'Dataset': 'Plebiscito_POST', 'Cluster': CLUSTER_S1_NAME}),
        (corpus_post_s2, {'Dataset': 'Plebiscito_POST', 'Cluster': CLUSTER_S2_NAME})
    ]

    # 4. Ejecutar análisis y recolectar resultados
    todos_los_temas = []
    for corpus, metadata in tareas:
        df_resultado_temas = analizar_temas_bertopic(corpus, metadata, TOPIC_MODEL_CONFIG)
        if not df_resultado_temas.empty:
            todos_los_temas.append(df_resultado_temas)
            
    # 5. Guardar resultados consolidados
    if not todos_los_temas:
        logging.warning("No se generaron resultados de Topic Modeling para guardar.")
        return

    df_resultados_finales = pd.concat(todos_los_temas, ignore_index=True)
    df_resultados_finales.to_csv(path_temas_output, index=False, sep=';', encoding='utf-8-sig')
    
    logging.info("--- ✅ Análisis de Topic Modeling finalizado exitosamente ---")
    logging.info(f"Resultados guardados en: {path_temas_output}")
if __name__ == '__main__':
    run_topic_modeling_analysis()