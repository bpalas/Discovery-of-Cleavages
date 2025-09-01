import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

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
    from python import pipeline
except (ImportError, FileNotFoundError) as e:
    logging.error(f"Error importando módulos: {e}. Asegúrate de que 'python/pipeline.py' existe.")
    sys.exit(1)


def ejecutar_clustering_individual(dataset_path: str, dataset_name: str, config_clustering: dict):
    """
    Ejecuta clustering en un dataset individual y guarda los resultados.
    
    Args:
        dataset_path (str): Ruta al archivo CSV del dataset
        dataset_name (str): Nombre identificador del dataset
        config_clustering (dict): Configuración del algoritmo de clustering
    
    Returns:
        dict: Métricas del análisis
    """
    logging.info(f"Iniciando análisis individual para: {dataset_name}")
    
    # Cargar dataset
    try:
        df_dataset = pd.read_csv(dataset_path, sep=',', on_bad_lines='skip')
        logging.info(f"Dataset cargado: {len(df_dataset)} registros")
    except FileNotFoundError:
        logging.error(f"No se encontró el archivo: {dataset_path}")
        return None
    except Exception as e:
        logging.error(f"Error cargando dataset: {e}")
        return None
    
    # Validar columnas requeridas
    required_cols = ['FROM_NODE', 'TO_NODE', 'SIGN']
    missing_cols = [col for col in required_cols if col not in df_dataset.columns]
    if missing_cols:
        logging.error(f"Faltan columnas requeridas: {missing_cols}")
        return None
    
    # Configuración por defecto para el pipeline
    default_cols = {
        'from_node_col': 'FROM_NODE',
        'to_node_col': 'TO_NODE', 
        'sign_col': 'SIGN'
    }
    
    # Ejecutar análisis de clustering
    try:
        df_nodes_results, _, paper_metrics, _, _ = pipeline.ejecutar_analisis_polarizacion(
            df_input=df_dataset.copy(), 
            config=config_clustering.copy(), 
            default_cols=default_cols, 
            calculate_intra_cluster_cc=True
        )
        
        if df_nodes_results is None or df_nodes_results.empty:
            logging.error("No se generaron resultados del clustering")
            return None
            
    except Exception as e:
        logging.error(f"Error en el análisis de clustering: {e}")
        return None
    
    # Preparar dataset original con clusters
    df_original_con_clusters = df_dataset.copy()
    
    # Normalizar nombres de nodos para el mapeo
    df_original_con_clusters['FROM_NODE_norm'] = df_original_con_clusters['FROM_NODE'].str.lower().str.strip()
    df_original_con_clusters['TO_NODE_norm'] = df_original_con_clusters['TO_NODE'].str.lower().str.strip()
    df_nodes_results['NODE_NAME_norm'] = df_nodes_results['NODE_NAME'].str.lower().str.strip()
    
    # Crear mapeo de nodos a clusters
    mapa_nodo_cluster = pd.Series(
        df_nodes_results.CLUSTER_ASSIGNMENT.values, 
        index=df_nodes_results.NODE_NAME_norm
    ).to_dict()
    
    # Asignar clusters a los nodos FROM y TO
    df_original_con_clusters['FROM_CLUSTER'] = df_original_con_clusters['FROM_NODE_norm'].map(mapa_nodo_cluster).fillna(0)
    df_original_con_clusters['TO_CLUSTER'] = df_original_con_clusters['TO_NODE_norm'].map(mapa_nodo_cluster).fillna(0)
    
    # Para simplificar, usamos el cluster del nodo FROM como cluster principal de la arista
    # Si ambos nodos tienen cluster asignado y son diferentes, podríamos usar otra lógica
    df_original_con_clusters['CLUSTER'] = df_original_con_clusters['FROM_CLUSTER']
    
    # Si el nodo FROM no tiene cluster pero TO sí, usar el de TO
    mask_from_sin_cluster = (df_original_con_clusters['FROM_CLUSTER'] == 0) & (df_original_con_clusters['TO_CLUSTER'] != 0)
    df_original_con_clusters.loc[mask_from_sin_cluster, 'CLUSTER'] = df_original_con_clusters.loc[mask_from_sin_cluster, 'TO_CLUSTER']
    
    # Mantener solo las columnas requeridas en el orden especificado
    columnas_finales = ['FROM_NODE', 'TO_NODE', 'SIGN', 'CLUSTER']
    df_resultado_final = df_original_con_clusters[columnas_finales].copy()
    
    # Convertir CLUSTER a int
    df_resultado_final['CLUSTER'] = df_resultado_final['CLUSTER'].astype(int)
    
    # Crear directorio de salida
    output_dir = os.path.join(project_root_path, 'data', 'datasets_cluster')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar timestamp para archivos únicos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar dataset con clusters
    output_dataset_path = os.path.join(output_dir, f"{dataset_name}_clustered_{timestamp}.csv")
    df_resultado_final.to_csv(output_dataset_path, index=False, sep=',', encoding='utf-8-sig')
    logging.info(f"Dataset con clusters guardado: {output_dataset_path}")
    
    # Preparar y guardar métricas
    metricas_con_info = paper_metrics.copy()
    metricas_con_info['Dataset'] = dataset_name
    metricas_con_info['Algoritmo'] = config_clustering['name']
    metricas_con_info['Timestamp'] = timestamp
    metricas_con_info['Total_Edges'] = len(df_resultado_final)
    metricas_con_info['Edges_Cluster_1'] = len(df_resultado_final[df_resultado_final['CLUSTER'] == 1])
    metricas_con_info['Edges_Cluster_-1'] = len(df_resultado_final[df_resultado_final['CLUSTER'] == -1])
    metricas_con_info['Edges_Cluster_0'] = len(df_resultado_final[df_resultado_final['CLUSTER'] == 0])
    
    # Guardar métricas
    output_metrics_path = os.path.join(output_dir, f"{dataset_name}_metrics_{timestamp}.csv")
    df_metricas = pd.DataFrame([metricas_con_info])
    df_metricas.to_csv(output_metrics_path, index=False, sep=',', encoding='utf-8-sig')
    logging.info(f"Métricas guardadas: {output_metrics_path}")
    
    # Guardar información de nodos con clusters (opcional, para referencia)
    output_nodes_path = os.path.join(output_dir, f"{dataset_name}_nodes_clusters_{timestamp}.csv")
    df_nodes_final = df_nodes_results[['NODE_NAME', 'CLUSTER_ASSIGNMENT']].copy()
    df_nodes_final.rename(columns={'CLUSTER_ASSIGNMENT': 'CLUSTER'}, inplace=True)
    df_nodes_final.to_csv(output_nodes_path, index=False, sep=',', encoding='utf-8-sig')
    logging.info(f"Asignación de nodos guardada: {output_nodes_path}")
    
    # Resumen en consola
    print(f"\n{'='*60}")
    print(f"RESUMEN DEL ANÁLISIS: {dataset_name}")
    print(f"{'='*60}")
    print(f"Algoritmo utilizado: {config_clustering['name']}")
    print(f"Total de aristas: {len(df_resultado_final):,}")
    print(f"Aristas en Cluster 1: {metricas_con_info['Edges_Cluster_1']:,}")
    print(f"Aristas en Cluster -1: {metricas_con_info['Edges_Cluster_-1']:,}")
    print(f"Aristas neutrales (Cluster 0): {metricas_con_info['Edges_Cluster_0']:,}")
    print(f"Total de nodos únicos: {len(df_nodes_final):,}")
    print(f"Nodos en Cluster 1: {len(df_nodes_final[df_nodes_final['CLUSTER'] == 1]):,}")
    print(f"Nodos en Cluster -1: {len(df_nodes_final[df_nodes_final['CLUSTER'] == -1]):,}")
    print(f"Archivo principal: {os.path.basename(output_dataset_path)}")
    print(f"{'='*60}")
    
    return {
        'dataset_path': output_dataset_path,
        'metrics_path': output_metrics_path,
        'nodes_path': output_nodes_path,
        'metrics': metricas_con_info
    }


def main():
    """
    Función principal que permite ejecutar el análisis desde línea de comandos
    o modificando directamente las rutas aquí.
    """
    
    # --- CONFIGURACIÓN ---
    # Modifica estas rutas según tus datos
    DATASET_CONFIGS = [
        {
            'path': os.path.join(project_root_path, 'News', 'output', 'df_plebiscito_2022.csv'),
            'name': 'Plebiscito_2022'
        },
        {
            'path': os.path.join(project_root_path, 'News', 'output', 'primarias2025.csv'),
            'name': 'Primarias_2025'
        }
    ]
    
    # Configuración del algoritmo de clustering
    CLUSTERING_CONFIG = {
        "name": "Local Search (b=0.01)",
        "algorithm_type": "local_search_paper_k2", 
        "k": 2,
        "ls_beta": 0.005,
        "ls_max_iter": 2000
    }
    
    # --- EJECUCIÓN ---
    logging.info("=== INICIANDO ANÁLISIS INDIVIDUAL DE CLUSTERING ===")
    
    resultados = []
    
    for dataset_config in DATASET_CONFIGS:
        if os.path.exists(dataset_config['path']):
            logging.info(f"Procesando: {dataset_config['name']}")
            resultado = ejecutar_clustering_individual(
                dataset_path=dataset_config['path'],
                dataset_name=dataset_config['name'],
                config_clustering=CLUSTERING_CONFIG
            )
            
            if resultado:
                resultados.append(resultado)
                logging.info(f"✅ Completado: {dataset_config['name']}")
            else:
                logging.error(f"❌ Error procesando: {dataset_config['name']}")
        else:
            logging.warning(f"⚠️ Archivo no encontrado: {dataset_config['path']}")
    
    # Resumen final
    if resultados:
        print(f"\n{'='*80}")
        print("🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")
        print(f"{'='*80}")
        print(f"Datasets procesados: {len(resultados)}")
        print("Archivos generados en: data/datasets_cluster/")
        print(f"{'='*80}")
    else:
        print("\n❌ No se procesaron datasets. Verifica las rutas y configuración.")


if __name__ == '__main__':
    main()