import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .algorithms import _get_principal_eigenvector




def clivaje_recta(df_resultado: pd.DataFrame, num_labels: int = 15):
    """
    Crea una visualización de línea recta del espectro de polarización,
    etiquetando los 'num_labels' nodos más importantes.

    Args:
        df_resultado (pd.DataFrame): DataFrame resultante de la función reduccion_dimensional.
                                     Debe contener 'NODE_NAME', 'GRADO', 'V1_REPROYECTADO'
                                     y 'CLUSTER_ASSIGNMENT'.
        num_labels (int): Número de nodos a etiquetar en la visualización, seleccionados por
                          mayor grado.
    """
    if df_resultado is None or df_resultado.empty:
        print("DataFrame vacío, no se puede visualizar.")
        return
    required_cols = ['GRADO', 'V1_REPROYECTADO', 'NODE_NAME', 'CLUSTER_ASSIGNMENT']
    if not all(col in df_resultado.columns for col in required_cols):
        print(f"Error: El DataFrame debe contener las columnas {required_cols}.")
        return

    print(f"--- Creando visualización de clivaje para el Top {num_labels} de nodos ---")
    
    # Seleccionar los nodos a etiquetar basados en el grado más alto
    df_etiquetar = df_resultado.sort_values(by='GRADO', ascending=False).head(num_labels)
    nodos_a_etiquetar = set(df_etiquetar['NODE_NAME'])
    
    # Mapeo de colores para los clusters
    color_map = {-1: '#21908dff', 1: '#fde725ff', 0: 'grey'}
    node_colors = df_resultado['CLUSTER_ASSIGNMENT'].map(color_map).fillna('grey')

    # Configuración del gráfico
    plt.figure(figsize=(20, 6))
    
    # Dibuja la línea horizontal central
    plt.axhline(0, color='grey', linestyle='-', linewidth=1, zorder=1)
    
    # Dibuja los nodos como puntos en la línea
    plt.scatter(df_resultado['V1_REPROYECTADO'], np.zeros(len(df_resultado)),
                c=node_colors, s=55, alpha=0.8, zorder=2, edgecolors='black', linewidth=0.5)

    # Lógica para etiquetar los nodos seleccionados de forma alterna (arriba/abajo)
    y_offset = 0.015  # Distancia vertical de la etiqueta a la línea
    pos_alterna = 1   # 1 para arriba, -1 para abajo
    
    # Iterar solo sobre los nodos que se deben etiquetar, ordenados por su posición en el eje x
    df_a_etiquetar = df_resultado[df_resultado['NODE_NAME'].isin(nodos_a_etiquetar)].sort_values("V1_REPROYECTADO")
    
    for _, row in df_a_etiquetar.iterrows():
        y_pos = y_offset * pos_alterna
        va = 'bottom' if pos_alterna == 1 else 'top' # Alineación vertical
        
        # Línea punteada desde el punto hasta la etiqueta
        plt.plot([row['V1_REPROYECTADO'], row['V1_REPROYECTADO']], [0, y_pos], color='grey', ls='--', lw=0.8)
        
        # Texto de la etiqueta
        plt.text(row['V1_REPROYECTADO'], y_pos, row['NODE_NAME'], ha='center', va=va, fontsize=9, rotation=45)
        
        pos_alterna *= -1 # Alternar posición para la siguiente etiqueta

    # Estilo final del gráfico
    plt.yticks([], []) # Ocultar el eje Y
    for spine in ['left', 'right', 'top']:
        plt.gca().spines[spine].set_visible(False)
    
    # Títulos y etiquetas
    plt.xlabel(r"Vector de Clivaje: $\vec{d}_{\text{clivaje}} \in \mathbb{R}^m$", fontsize=12)
    plt.title(f"Visualización del Clivaje Político (Top {num_labels} Nodos por Grado)", fontsize=14, weight='bold')
    
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.ylim(-y_offset * 2.5, y_offset * 2.5) # Ajustar límites del eje Y para dar espacio
    plt.tight_layout()
    plt.show()

    
def plot_eigenvector_spectrum(df_resultado: pd.DataFrame):
    """
    Crea una visualización del espectro del eigenvector, similar a la imagen de ejemplo.
    
    El DataFrame de entrada debe estar ordenado por la columna 'V1_REPROYECTADO'.
    El eje X será el índice del vértice/nodo en el DataFrame ordenado.
    El eje Y será el valor de 'V1_REPROYECTADO'.
    """
    if df_resultado is None or df_resultado.empty:
        print("DataFrame de resultado está vacío. No se puede generar el gráfico.")
        return
        
    if 'V1_REPROYECTADO' not in df_resultado.columns:
        print("Error: La columna 'V1_REPROYECTADO' no se encuentra en el DataFrame.")
        return

    # El eje Y son los valores del eigenvector.
    y_values = df_resultado['V1_REPROYECTADO']
    
    # El eje X es simplemente el índice de los nodos una vez ordenados (de 0 a N-1).
    x_values = np.arange(len(df_resultado))

    # Creación del gráfico
    plt.style.use('ggplot') # Usamos un estilo similar al de tu imagen
    plt.figure(figsize=(12, 7))
    
    plt.scatter(x_values, y_values, s=15, color='black')
    
    # Añadir títulos y etiquetas
    plt.title("Espectro del Eigenvector Principal del Núcleo de Conflicto", fontsize=16, weight='bold')
    plt.xlabel("Vértice (ordenado por componente del eigenvector)", fontsize=12)
    plt.ylabel("Componente del Eigenvector (V1_REPROYECTADO)", fontsize=12)
    
    # Ajustes finales
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
