import sys
import os
import pandas as pd

# --- Configuración de Ruta para importar la librería ---
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root_path not in sys.path:
    sys.path.append(project_root_path)

from python import data_processing, pipeline

def run_comparative_analysis():
    """
    Función principal para ejecutar y comparar múltiples algoritmos
    en diferentes conjuntos de datos.
    """
    print("--- INICIANDO EXPERIMENTO COMPARATIVO DE ALGORITMOS ---")

    # --- 1. Definir las configuraciones de cada experimento ---
    default_cols = {
        'from_node_col': 'FROM_NODE', 
        'to_node_col': 'TO_NODE', 
        'sign_col': 'SIGN',
        "node_norm_strategy": "lower_unidecode_strip",
        "weighting_strategy": "binary_sum_signs_actual"
    }
    k = 2
    
    lista_experimentos = [
        {
            "name": "Eigensign (Determinista)", 
            "algorithm_type": "eigensign",
            "eigen_solver": "scipy"
        },
        {
            "name": "Random Eigensign (Probabilístico)",
            "algorithm_type": "random_eigensign",
            "num_runs": 50,
            "eigen_solver": "scipy"
        },
        {
            "name": "Local Search (beta=0.01)",
            "algorithm_type": "local_search_paper_k2",
            "k": k, "ls_beta": 0.01, "ls_max_iter": 20
        },
        {
            "name": "Local Search (beta=0.005)",
            "algorithm_type": "local_search_paper_k2",
            "k": k, "ls_beta": 0.005, "ls_max_iter": 20
        },
        {
            "name": "SCG (max_obj)", "algorithm_type": "scg",
            "K": 2, "rounding_strategy": "max_obj"
        },
        {
            "name": "SCG (randomized)", "algorithm_type": "scg",
            "K": 2, "rounding_strategy": "randomized"
        },
        {
            "name": "SCG (bansal)", "algorithm_type": "scg",
            "K": 2, "rounding_strategy": "bansal"
        }
    ]
    
    # --- 2. Cargar y preparar los conjuntos de datos ---
    print("\n--- Cargando Datos ---")
    try:
        ruta_plebiscito_2022 = os.path.join(project_root_path, 'News', 'output', 'df_plebiscito_2022.csv')
        df_pre, df_post = data_processing.cargar_datos_2022_pre_post(ruta_plebiscito_2022)
        
        datasets_a_procesar = {
            "Plebiscito PRE": df_pre,
            "Plebiscito POST": df_post
        }
        print("Datos cargados correctamente.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo de datos en la ruta '{ruta_plebiscito_2022}'.")
        return

    # --- 3. Ejecutar los experimentos de forma iterativa ---
    resultados_core = []
    resultados_paper = []

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
            
            # Ejecutamos el pipeline
            _, metricas_core, metricas_paper, _, _ = pipeline.ejecutar_analisis_polarizacion(
            df_input=df_dataset,
            config=config_actual,
            default_cols=default_cols,
            calculate_intra_cluster_cc=True  # <--- ¡ESTA ES LA LÍNEA CLAVE!
        )
            


            # Añadimos información contextual
            metricas_core['Dataset'] = nombre_dataset
            metricas_paper['Dataset'] = nombre_dataset
            
            resultados_core.append(metricas_core)
            resultados_paper.append(metricas_paper)

    # --- 4. Consolidar y mostrar los resultados finales ---
    if not resultados_core:
        print("No se generaron resultados. Finalizando experimento.")
        return

    df_resultados_core = pd.DataFrame(resultados_core)
    df_resultados_paper = pd.DataFrame(resultados_paper)

    print("\n\n========================================================")
    print("--- TABLA 1: MÉTRICAS BÁSICAS DE POLARIZACIÓN ---")
    print("========================================================")
    cols_core_ordenadas = ['Dataset', 'Algoritmo', 'Polaridad', 'N_S1', 'N_S2', 'N_S0', 'N_Total']
    otras_cols_core = [col for col in df_resultados_core.columns if col not in cols_core_ordenadas]
    print(df_resultados_core[cols_core_ordenadas + otras_cols_core].to_string())

    # --- INICIO DE LA MODIFICACIÓN ---
    print("\n\n========================================================")
    print("--- TABLA 2: MÉTRICAS DETALLADAS (SEGÚN PAPER) ---")
    print("========================================================")
    
    # Se eliminan las columnas MAC, MAO, DENS, K y se añaden las de Coeficiente de Clustering Intra-Cluster
    # Asumimos que tu pipeline ahora las devuelve como 'CC_intra_S1' y 'CC_intra_S2'
    cols_paper_ordenadas = [
        "Dataset", "Algoritmo", "POL", "BA-POL", "BAL", "SIZE", 
        "ISO", "CC+", "CC-", "CC_intra_S1", "CC_intra_S2"
    ]
    
    # Nos aseguramos de que solo se usen las columnas que existen en el DataFrame para evitar errores
    cols_a_mostrar = [col for col in cols_paper_ordenadas if col in df_resultados_paper.columns]
    
    print(df_resultados_paper[cols_a_mostrar].to_string())
    # --- FIN DE LA MODIFICACIÓN ---


    # --- 5. Guardar los resultados en archivos CSV separados ---
    try:
        results_dir = os.path.join(project_root_path, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        ruta_core = os.path.join(results_dir, 'comparacion_metricas_core.csv')
        df_resultados_core.to_csv(ruta_core, index=False)
        print(f"\nResultados de métricas básicas guardados en: {ruta_core}")
        
        ruta_paper = os.path.join(results_dir, 'comparacion_metricas_paper.csv')
        # Guardamos el DataFrame con las columnas ya seleccionadas y ordenadas
        df_resultados_paper[cols_a_mostrar].to_csv(ruta_paper, index=False)
        print(f"Resultados de métricas del paper guardados en: {ruta_paper}")

    except Exception as e:
        print(f"\nError al guardar los resultados: {e}")

if __name__ == '__main__':
    run_comparative_analysis()
    