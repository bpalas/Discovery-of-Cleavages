# run_comparative_analysis_V4.py

import sys
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import logging
from collections import Counter

# --- Configuración del Logging y Rutas (Asegúrate que tus módulos 'pipeline' etc. estén accesibles) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
try:
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root_path not in sys.path:
        sys.path.append(project_root_path)
    from python import data_processing, pipeline
except (ImportError, FileNotFoundError):
    logging.error("Error importando módulos. Asegúrate de que la estructura de carpetas es correcta.")
    sys.exit(1)

# ----------------------------------------------------------------------------------
# --- NUEVA VERSIÓN V4 DE LA FUNCIÓN DE ANÁLISIS DE FRONTERAS ---
# ----------------------------------------------------------------------------------
def analizar_fronteras_y_puentes_V4(df_original: pd.DataFrame, df_nodes_clusters: pd.DataFrame,
                                     nombre_dataset: str, nombre_algoritmo: str, cluster_map: dict):
    logging.info(f"Analizando fronteras y puentes (V4 con detalle de destino) para: {nombre_algoritmo} en {nombre_dataset}")

    df_nodes_clusters['NODE_NAME_norm'] = df_nodes_clusters['NODE_NAME'].str.lower().str.strip()
    mapa_nodo_cluster = pd.Series(df_nodes_clusters.CLUSTER_ASSIGNMENT.values, index=df_nodes_clusters.NODE_NAME_norm).to_dict()

    df_edges = df_original[['FROM_NODE', 'TO_NODE', 'SIGN']].copy()
    df_edges['FROM_NODE_norm'] = df_edges['FROM_NODE'].str.lower().str.strip()
    df_edges['TO_NODE_norm'] = df_edges['TO_NODE'].str.lower().str.strip()
    df_edges['FROM_CLUSTER'] = df_edges['FROM_NODE_norm'].map(mapa_nodo_cluster).fillna(0)
    df_edges['TO_CLUSTER'] = df_edges['TO_NODE_norm'].map(mapa_nodo_cluster).fillna(0)

    nodos_no_neutrales = df_nodes_clusters['NODE_NAME_norm'].unique()
    todos_los_nodos = set(df_edges['FROM_NODE_norm']) | set(df_edges['TO_NODE_norm'])
    nodos_neutrales = todos_los_nodos - set(nodos_no_neutrales)
    
    resultados_analisis = []

    # === 1) ANÁLISIS DE FRONTERAS DE CLÚSTER ===
    for cluster_id, cluster_name in cluster_map.items():
        nodos_del_cluster = set(df_nodes_clusters[df_nodes_clusters['CLUSTER_ASSIGNMENT'] == cluster_id]['NODE_NAME_norm'])
        otro_cluster_id = -1 if cluster_id == 1 else 1

        if not nodos_del_cluster: continue
            
        frontera = set()
        # V4: Contadores más específicos para el destino de la conexión
        conex_vs_neutral = Counter()
        conex_vs_otro_cluster = Counter()
        
        aristas_externas = df_edges[
            (df_edges['FROM_CLUSTER'] == cluster_id) & (df_edges['TO_CLUSTER'] != cluster_id) |
            (df_edges['TO_CLUSTER'] == cluster_id) & (df_edges['FROM_CLUSTER'] != cluster_id)
        ]

        for _, row in aristas_externas.iterrows():
            if row['FROM_CLUSTER'] == cluster_id:
                nodo_frontera = row['FROM_NODE_norm']
                frontera.add(nodo_frontera)
                if row['TO_CLUSTER'] == 0: # Conexión hacia un Neutral
                    conex_vs_neutral[nodo_frontera] += 1
                elif row['TO_CLUSTER'] == otro_cluster_id: # Conexión hacia el Otro Cluster
                    conex_vs_otro_cluster[nodo_frontera] += 1
            
            elif row['TO_CLUSTER'] == cluster_id:
                nodo_frontera = row['TO_NODE_norm']
                frontera.add(nodo_frontera)
                if row['FROM_CLUSTER'] == 0:
                    conex_vs_neutral[nodo_frontera] += 1
                elif row['FROM_CLUSTER'] == otro_cluster_id:
                    conex_vs_otro_cluster[nodo_frontera] += 1
        
        # 1.1) Fracción de nodos en la frontera (Permeabilidad)
        permeabilidad = len(frontera) / len(nodos_del_cluster) if len(nodos_del_cluster) > 0 else 0
        resultados_analisis.append({
            'Dataset': nombre_dataset, 'Algoritmo': nombre_algoritmo, 'Tipo_Analisis': 'Cluster_Permeability',
            'Cluster': cluster_name, 'Metrica': 'Permeabilidad', 'Valor': round(permeabilidad, 4)
        })

        # 1.2) Nodos con más conexiones a NEUTRALES
        for nodo, n_conexiones in conex_vs_neutral.most_common(5):
            resultados_analisis.append({
                'Dataset': nombre_dataset, 'Algoritmo': nombre_algoritmo, 'Tipo_Analisis': 'Frontera_vs_Neutral',
                'Cluster': cluster_name, 'Nodo': nodo, 'Conexiones': n_conexiones
            })
            
        # 1.3) Nodos con más conexiones al OTRO CLUSTER
        for nodo, n_conexiones in conex_vs_otro_cluster.most_common(5):
            resultados_analisis.append({
                'Dataset': nombre_dataset, 'Algoritmo': nombre_algoritmo, 'Tipo_Analisis': 'Frontera_vs_OtroCluster',
                'Cluster': cluster_name, 'Nodo': nodo, 'Conexiones': n_conexiones
            })

    # === 2) ANÁLISIS DE NODOS PUENTE NEUTRALES ===
    for nodo_neutral in nodos_neutrales:
        conexiones_a_clusters = Counter()
        
        df_neutral_participa = df_edges[(df_edges['FROM_NODE_norm'] == nodo_neutral) | (df_edges['TO_NODE_norm'] == nodo_neutral)]
        
        for _, row in df_neutral_participa.iterrows():
            if row['FROM_NODE_norm'] == nodo_neutral and row['TO_CLUSTER'] in cluster_map:
                conexiones_a_clusters[row['TO_CLUSTER']] += 1
            if row['TO_NODE_norm'] == nodo_neutral and row['FROM_CLUSTER'] in cluster_map:
                conexiones_a_clusters[row['FROM_CLUSTER']] += 1
        
        # Si tiene vecinos en AMBOS clusters
        if 1 in conexiones_a_clusters and -1 in conexiones_a_clusters:
            total_conexiones = sum(conexiones_a_clusters.values())
            
            resultados_analisis.append({
                'Dataset': nombre_dataset, 'Algoritmo': nombre_algoritmo, 'Tipo_Analisis': 'Neutral_Puente',
                'Nodo': nodo_neutral,
                'Conexiones_S1': conexiones_a_clusters.get(1, 0),
                'Conexiones_S2': conexiones_a_clusters.get(-1, 0),
                'Conexiones_Totales': total_conexiones,
                'Sesgo_Equilibrio': (conexiones_a_clusters.get(1, 0) - conexiones_a_clusters.get(-1, 0)) / total_conexiones
            })

    return resultados_analisis

def ejecutar_experimento_individual(nombre_dataset, df_dataset, config_base, default_cols, cluster_map):
    logging.info(f"Iniciando experimento completo para: {config_base['name']} en {nombre_dataset}")
    
    # Ejecutar el pipeline de polarización principal
    df_nodes_results, metricas_core, _, _, _ = pipeline.ejecutar_analisis_polarizacion(
        df_input=df_dataset.copy(), config=config_base.copy(), default_cols=default_cols, calculate_intra_cluster_cc=True
    )
    
    resultados_fronteras = []
    if df_nodes_results is not None and not df_nodes_results.empty:
        # LLAMADA A LA NUEVA FUNCIÓN V3
        fronteras_encontradas = analizar_fronteras_y_puentes_V3(
            df_original=df_dataset.copy(), df_nodes_clusters=df_nodes_results.copy(),
            nombre_dataset=nombre_dataset, nombre_algoritmo=config_base['name'],
            cluster_map=cluster_map
        )
        if fronteras_encontradas:
            resultados_fronteras.extend(fronteras_encontradas)

    metricas_core['Dataset'] = nombre_dataset
    logging.info(f"Finalizado: {config_base['name']} en {nombre_dataset}")
    
    return (metricas_core, resultados_fronteras)

def run_comparative_analysis():
    logging.info("--- INICIANDO EXPERIMENTO COMPARATIVO (V3 CON ANÁLISIS DE SIGNO) ---")

    # === CONFIGURACIÓN ===
    DEFAULT_COLS = {'from_node_col': 'FROM_NODE', 'to_node_col': 'TO_NODE', 'sign_col': 'SIGN'}
    CLUSTER_MAP = {1: 'S1', -1: 'S2'}
    LISTA_EXPERIMENTOS = [
        {"name": "Local Search (b=0.01)", "algorithm_type": "local_search_paper_k2", "k": 2, "ls_beta": 0.01, "ls_max_iter": 20},
    ]
    # =====================

    logging.info("--- Cargando Datos ---")
    try:
        ruta_plebiscito_2022 = os.path.join(project_root_path, 'News', 'output', 'df_plebiscito_2022.csv')
        df_pre, df_post = data_processing.cargar_datos_2022_pre_post(ruta_plebiscito_2022)
        datasets = {"Plebiscito_PRE": df_pre.head(5000), "Plebiscito_POST": df_post.head(5000)}
        logging.info(f"Datos cargados. Usando una muestra de {len(df_pre.head(5000))} noticias por dataset.")
    except FileNotFoundError:
        logging.error(f"No se encontró el archivo de datos '{ruta_plebiscito_2022}'. Abortando.")
        return

    tareas = [(nombre, df, config, DEFAULT_COLS, CLUSTER_MAP) 
              for nombre, df in datasets.items() 
              for config in LISTA_EXPERIMENTOS if not df.empty]

    logging.info(f"Se ejecutarán {len(tareas)} experimentos base en paralelo.")
    
    resultados_paralelos = Parallel(n_jobs=-1, backend="loky")(
        delayed(ejecutar_experimento_individual)(*task) for task in tareas
    )
    
    # Procesamiento de resultados
    resultados_core = [res[0] for res in resultados_paralelos if res and res[0]]
    resultados_fronteras = [item for res in resultados_paralelos if res and res[1] for item in res[1]]

    df_resultados_core = pd.DataFrame(resultados_core)
    df_resultados_fronteras = pd.DataFrame(resultados_fronteras)

    # Guarda los resultados en la carpeta /reunion 07/resultados/
    results_dir = os.path.join(project_root_path, 'reunion 07', 'resultados')
    os.makedirs(results_dir, exist_ok=True)
    
    # CAMBIO: Nuevos nombres de archivo para V3
    path_metricas = os.path.join(results_dir, 'exp_metricas_polarizacion_V3.csv')
    path_fronteras = os.path.join(results_dir, 'exp_analisis_fronteras_V3.csv')
    
    df_resultados_core.to_csv(path_metricas, index=False, sep=';', encoding='utf-8-sig')
    df_resultados_fronteras.to_csv(path_fronteras, index=False, sep=';', encoding='utf-8-sig')
    
    logging.info("--- Resultados V3 guardados exitosamente ---")
    logging.info(f"Métricas de polarización: {path_metricas}")
    logging.info(f"Análisis de Fronteras y Puentes (con signo): {path_fronteras}")

if __name__ == '__main__':
    run_comparative_analysis()