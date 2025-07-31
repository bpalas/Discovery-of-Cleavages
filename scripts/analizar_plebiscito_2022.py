
import sys
import os
import pandas as pd

# --- Configuración de Ruta para importar la librería ---
# Añadimos la raíz del proyecto para que encuentre la carpeta 'polarization_analyzer'
project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root_path not in sys.path:
    sys.path.append(project_root_path)

from python import data_processing, pipeline, analysis,  visualization
config_analisis = {
    "name": "Ejemplo_Social_Polarization",
    "algorithm_type": "eigensign",
    "node_norm_strategy": "lower_unidecode_strip",
    "weighting_strategy": "binary_sum_signs_actual",

}
default_cols = {'FROM_NODE': 'FROM_NODE', 'TO_NODE': 'TO_NODE', 'SIGN': 'SIGN'}

k = 2
beta_elegido = 0.005
alpha_recomendado = 1 / (k - 1) if k > 1 else 1.0

config_local_search = {
    "name": f"Análisis Local Search (β={beta_elegido})",
    "algorithm_type": "local_search_paper_k2",
    "k": k,
    "ls_beta": beta_elegido,
    "ls_alpha": alpha_recomendado,
    "ls_max_iter": 30
}


# --- 2. FUNCIÓN PARA ENCAPSULAR EL ANÁLISIS ---
def procesar_periodo(df_input, config, cols):
    """
    Ejecuta el análisis completo de polarización y reducción para un DataFrame.
    """
    print(f"--- Procesando DataFrame... ---")
    
    # PASO A: Ejecutar el análisis de polarización principal
    df_nodos, metricas, A_s, n_to_idx = pipeline.ejecutar_analisis_polarizacion(
        df_input=df_input,
        config=config_local_search,
        default_cols=cols
    )
    
    # PASO B: Ejecutar la reducción dimensional
    df_resultado, _ = analysis.reduccion_dimensional(
        A_s_original=A_s,
        df_nodos_original=df_nodos,
        node_to_idx_original=n_to_idx,
        eigen_solver_config=config_local_search
    )
    visualization.clivaje_recta(df_resultado) 
    
    print("--- Proceso completado. ---")
    return df_resultado, metricas


def run_analysis():
    """Función principal para este experimento específico."""
    print("--- INICIANDO EXPERIMENTO: ANÁLISIS PLEBISCITO 2022 ---")

    # 1. Definir configuraciones
    default_cols = {'FROM_NODE': 'FROM_NODE', 'TO_NODE': 'TO_NODE', 'SIGN': 'SIGN'}
    config_local_search = {
        "name": "Análisis Local Search",
        "algorithm_type": "local_search_paper_k2",
        "k": 2, "ls_beta": 0.005, "ls_alpha": 1.0, "ls_max_iter": 30,
        'eigen_solver': 'numpy_robust'
    }

    # 2. Cargar datos
    ruta_plebiscito_2022 = 'News/output/df_plebiscito_2022.csv' # Ruta relativa desde la carpeta scripts
    df_pre, df_post = data_processing.cargar_datos_2022_pre_post(ruta_plebiscito_2022)

    # 3. Procesar período PRE
    print("\n--- Procesando Período PRE ---")
    config_pre = config_local_search.copy()
    config_pre['name'] = "Plebiscito 2022 - PRE"
    df_resultado_pre, metricas_pre = procesar_periodo(df_post, config_local_search, default_cols)

    print("\n--- Procesando Período POST ---")
    config_post = config_local_search.copy()
    config_post['name'] = "Plebiscito 2022 - POST"
    df_resultado_post, metricas_post = procesar_periodo(df_post, config_local_search, default_cols)

    # 5. Guardar o mostrar resultados comparativos
    metricas_pre['Periodo'] = 'Pre'
    metricas_post['Periodo'] = 'Post'
    df_comparativo = pd.DataFrame([metricas_pre, metricas_post])
    print("\n--- RESULTADOS COMPARATIVOS ---")
    print(df_comparativo)


if __name__ == '__main__':
    run_analysis()