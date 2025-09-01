"""
Análisis de Polarización en Votaciones - Versión Corregida
Comparación entre Primarias y Plebiscito Constitucional
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Configuración visual global
plt.style.use('seaborn-v0_8-whitegrid')

class AnalizadorPolarizacion:
    """Clase para analizar y visualizar métricas de polarización en votaciones"""
    
    def __init__(self, base_path='../results/'):
        self.base_path = Path(base_path)
        self.df_metricas = None
        self.df_terminos = None
        self.df_nodos_centrales = None
        self.df_clusters = None
        
        # Paleta de colores mejorada
        self.colores_cluster = {
            'S1': '#2563eb',  # Azul vibrante
            'S2': '#dc2626'   # Rojo vibrante
        }
        self.colores_dataset = {
            'Primarias': '#10b981',   # Verde esmeralda
            'Plebiscito': '#f59e0b'   # Naranja/Ámbar
        }
        
    def cargar_datos(self):
        """Carga todos los archivos de resultados"""
        try:
            self.df_metricas = pd.read_csv('results/exp_metricas_polarizacion.csv', sep=';')
            self.df_terminos = pd.read_csv('results/exp_analisis_terminos.csv', sep=';')
            self.df_nodos_centrales = pd.read_csv('results/exp_analisis_nodos_centrales.csv', sep=';')
            self.df_clusters = pd.read_csv('results/exp_asignacion_clusters.csv', sep=';')
            print("✅ Datos cargados exitosamente")
            return True
        except FileNotFoundError as e:
            print(f"❌ Error al cargar archivos: {e}")
            return False
    
    def visualizar_composicion_clusters_solo_nodos(self, top_nodos=20, figsize=(16, 8)):
        """Visualiza solo la composición de clusters basándose en los datos de nodos disponibles"""
        if self.df_nodos_centrales is None:
            print("❌ No hay datos de nodos disponibles")
            return None
            
        datasets = self.df_nodos_centrales['Dataset'].unique()
        n_datasets = len(datasets)
        fig = plt.figure(figsize=figsize)
        
        for i, dataset in enumerate(datasets):
            # 1. Nodos centrales
            ax1 = plt.subplot(n_datasets, 3, i*3 + 1)
            self._plot_nodos_centrales_mejorado(dataset, ax1, top_nodos)
            
            # 2. Distribución de grados (boxplot mejorado)
            ax2 = plt.subplot(n_datasets, 3, i*3 + 2)
            self._plot_distribucion_grados_mejorado(dataset, ax2)
            
            # 3. Distribución log-log de grados
            ax3 = plt.subplot(n_datasets, 3, i*3 + 3)
            self._plot_distribucion_log_log(dataset, ax3)
        
        plt.suptitle('Composición Detallada de Clusters por Votación', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def _plot_nodos_centrales_mejorado(self, dataset, ax, top_n):
        """Grafica los nodos más centrales por cluster"""
        df_dataset = self.df_nodos_centrales[self.df_nodos_centrales['Dataset'] == dataset]
        
        # Tomar top N nodos por cluster
        df_top = df_dataset.groupby('Cluster').apply(
            lambda x: x.nsmallest(top_n, 'Rank')
        ).reset_index(drop=True)
        
        # Crear gráfico de dispersión
        clusters = df_top['Cluster'].unique()
        for cluster in clusters:
            df_cluster = df_top[df_top['Cluster'] == cluster]
            x = range(len(df_cluster))
            ax.scatter(x, df_cluster['Grado'], 
                      label=f'Cluster {cluster}',
                      color=self.colores_cluster.get(cluster, '#95a5a6'),
                      s=100, alpha=0.7)
            
            # Añadir etiquetas para los top 3
            for i, (idx, row) in enumerate(df_cluster.head(3).iterrows()):
                ax.annotate(row['Nodo'], 
                           (i, row['Grado']),
                           xytext=(5, 5), 
                           textcoords='offset points',
                           fontsize=8, alpha=0.8)
        
        ax.set_xlabel('Ranking')
        ax.set_ylabel('Grado (Conexiones)')
        ax.set_title(f'Nodos Centrales - {dataset}', fontsize=12, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_distribucion_grados_mejorado(self, dataset, ax):
        """Grafica la distribución de grados con colores vibrantes"""
        df_dataset = self.df_nodos_centrales[self.df_nodos_centrales['Dataset'] == dataset]
        
        clusters = sorted(df_dataset['Cluster'].unique())
        data_to_plot = []
        positions = []
        colors = []
        
        for i, cluster in enumerate(clusters):
            df_cluster = df_dataset[df_dataset['Cluster'] == cluster]
            data_to_plot.append(df_cluster['Grado'].values)
            positions.append(i + 1)
            colors.append(self.colores_cluster.get(cluster, '#95a5a6'))
        
        # Crear boxplot
        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                        patch_artist=True, 
                        boxprops=dict(linewidth=1.5, edgecolor='black'),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2, color='gold'))
        
        # Colorear las cajas
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        
        # Añadir puntos para outliers con colores
        for i, (data, pos, color) in enumerate(zip(data_to_plot, positions, colors)):
            y = data
            x = np.random.normal(pos, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.3, s=20, color=color, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Cluster', fontsize=10, fontweight='bold')
        ax.set_ylabel('Grado', fontsize=10, fontweight='bold')
        ax.set_title(f'Distribución de Centralidad - {dataset}', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels([f'Cluster {c}' for c in clusters])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Añadir estadísticas
        for i, (data, pos) in enumerate(zip(data_to_plot, positions)):
            stats_text = f'μ={np.mean(data):.1f}\nσ={np.std(data):.1f}'
            ax.text(pos, ax.get_ylim()[1]*0.95, stats_text,
                   ha='center', va='top', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', alpha=0.8, edgecolor='gray'))

    def _plot_distribucion_log_log(self, dataset, ax):
        """Grafica la distribución de grados en escala log-log para todos los nodos y los más centrales."""
        df_dataset = self.df_nodos_centrales[self.df_nodos_centrales['Dataset'] == dataset]

        # --- Análisis para TODOS los nodos ---
        grados_todos = df_dataset['Grado'].values
        grados_unicos_todos, conteos_todos = np.unique(grados_todos, return_counts=True)
        probabilidades_todos = conteos_todos / len(grados_todos)
        
        ax.scatter(grados_unicos_todos, probabilidades_todos,
                   label='Todos los Nodos',
                   color='royalblue',
                   alpha=0.8, s=50, edgecolor='black', linewidth=1)

        # Ajuste de línea de tendencia para todos los nodos
        if len(grados_unicos_todos) > 2:
            mask = (grados_unicos_todos > 0) & (probabilidades_todos > 0)
            if np.sum(mask) > 2:
                log_x_todos = np.log10(grados_unicos_todos[mask])
                log_y_todos = np.log10(probabilidades_todos[mask])
                coef_todos = np.polyfit(log_x_todos, log_y_todos, 1)
                poly_func_todos = np.poly1d(coef_todos)
                x_fit_todos = np.logspace(np.log10(grados_unicos_todos[mask].min()),
                                         np.log10(grados_unicos_todos[mask].max()), 50)
                y_fit_todos = 10**(poly_func_todos(np.log10(x_fit_todos)))
                ax.plot(x_fit_todos, y_fit_todos, '--',
                        color='red',
                        alpha=0.7, linewidth=2,
                        label=f'Ajuste Todos (γ={-coef_todos[0]:.2f})')
        
        # --- Análisis para los NODOS CENTRALES (Top 10%) ---
        num_centrales = max(1, int(len(df_dataset) * 0.1))  # Al menos 1 nodo
        df_centrales = df_dataset.nsmallest(num_centrales, 'Rank')
        
        grados_centrales = df_centrales['Grado'].values
        grados_unicos_centrales, conteos_centrales = np.unique(grados_centrales, return_counts=True)
        probabilidades_centrales = conteos_centrales / len(grados_centrales)
        
        ax.scatter(grados_unicos_centrales, probabilidades_centrales,
                   label=f'Top {num_centrales} Nodos Centrales',
                   color='darkorange',
                   alpha=0.8, s=80, marker='D', edgecolor='black', linewidth=1.5)

        # --- Configuración del plot ---
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Grado (k)', fontsize=10, fontweight='bold')
        ax.set_ylabel('P(k)', fontsize=10, fontweight='bold')
        ax.set_title(f'Distribución Log-Log de Grados - {dataset}',
                     fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax.grid(True, which="both", ls="-", alpha=0.3)

def generar_datos_nodos_desde_csv(ruta_csv, nombre_dataset):
    """
    Lee un CSV de aristas, construye una red y calcula el grado y ranking de cada nodo.
    
    Args:
        ruta_csv (str): La ruta al archivo CSV.
        nombre_dataset (str): El nombre para identificar este dataset (ej. 'Plebiscito').
        
    Returns:
        pd.DataFrame: Un DataFrame con columnas ['Nodo', 'Grado', 'Rank', 'Dataset', 'Cluster'].
    """
    print(f"🔄 Procesando el archivo: {ruta_csv}...")
    
    try:
        # 1. Cargar los datos
        df_edges = pd.read_csv(ruta_csv)
        print(f"📄 Archivo cargado. Columnas encontradas: {list(df_edges.columns)}")
        print(f"📊 {len(df_edges)} aristas encontradas")
        
        # 2. Verificar que las columnas necesarias existan
        if 'FROM_NODE' not in df_edges.columns or 'TO_NODE' not in df_edges.columns:
            print(f"⚠️ Columnas esperadas: FROM_NODE, TO_NODE")
            print(f"⚠️ Columnas encontradas: {list(df_edges.columns)}")
            # Intentar mapear columnas comunes
            if 'source' in df_edges.columns and 'target' in df_edges.columns:
                df_edges = df_edges.rename(columns={'source': 'FROM_NODE', 'target': 'TO_NODE'})
                print("🔄 Mapeado: source -> FROM_NODE, target -> TO_NODE")
            elif len(df_edges.columns) >= 2:
                df_edges.columns = ['FROM_NODE', 'TO_NODE'] + list(df_edges.columns[2:])
                print("🔄 Se asignaron los nombres FROM_NODE y TO_NODE a las primeras dos columnas")
            else:
                raise ValueError("No se pueden identificar las columnas de aristas")
        
        # 3. Crear el grafo desde la lista de aristas
        G = nx.from_pandas_edgelist(df_edges, 'FROM_NODE', 'TO_NODE')
        print(f"🌐 Grafo creado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
        
        # 4. Calcular el grado de cada nodo
        grados = dict(G.degree())
        if not grados:
            print(f"⚠️ El grafo para {nombre_dataset} está vacío. No se generarán datos de nodos.")
            return pd.DataFrame()
            
        df_nodos = pd.DataFrame(grados.items(), columns=['Nodo', 'Grado'])
        
        # 5. Añadir columnas requeridas por la clase de análisis
        df_nodos['Dataset'] = nombre_dataset
        df_nodos['Rank'] = df_nodos['Grado'].rank(ascending=False, method='first').astype(int)
        
        # Asignar clusters basándose en el grado (simulación simple)
        # Los nodos con grado mayor a la mediana van a S1, el resto a S2
        mediana_grado = df_nodos['Grado'].median()
        df_nodos['Cluster'] = df_nodos['Grado'].apply(lambda x: 'S1' if x >= mediana_grado else 'S2')
        
        print(f"✅ Procesamiento de {nombre_dataset} completado. {len(df_nodos)} nodos encontrados.")
        print(f"📈 Grado promedio: {df_nodos['Grado'].mean():.2f}")
        print(f"📊 Distribución de clusters: {df_nodos['Cluster'].value_counts().to_dict()}")
        return df_nodos
        
    except Exception as e:
        print(f"❌ Error procesando {ruta_csv}: {str(e)}")
        return pd.DataFrame()

def main_con_preprocesamiento():
    """
    Función principal que primero pre-procesa los CSV y luego ejecuta el análisis.
    """
    
    # 1. Define las rutas a tus archivos CSV originales
    # ¡¡¡ ATENCIÓN: AJUSTA ESTAS RUTAS !!!
    ruta_plebiscito = r'News\output\df_plebiscito_2022.csv'
    ruta_primarias = r'News\output\primarias2025.csv'

    print("🚀 Iniciando análisis de polarización desde archivos CSV...")
    
    # 2. Generar los DataFrames de nodos usando la nueva función
    df_nodos_plebiscito = generar_datos_nodos_desde_csv(ruta_plebiscito, 'Plebiscito')
    df_nodos_primarias = generar_datos_nodos_desde_csv(ruta_primarias, 'Primarias')
    
    # 3. Verificar que se hayan cargado datos
    if df_nodos_plebiscito.empty and df_nodos_primarias.empty:
        print("❌ No se pudieron cargar datos de ningún archivo")
        return
    
    # 4. Combinar ambos DataFrames en uno solo
    dataframes_validos = [df for df in [df_nodos_plebiscito, df_nodos_primarias] if not df.empty]
    df_nodos_completo = pd.concat(dataframes_validos, ignore_index=True)
    
    print(f"📋 Datos combinados: {len(df_nodos_completo)} nodos totales")
    print(f"📊 Datasets: {df_nodos_completo['Dataset'].value_counts().to_dict()}")
    
    # 5. Crear la instancia del analizador
    analizador = AnalizadorPolarizacion()
    
    # 6. Inyectar el DataFrame generado en la instancia
    analizador.df_nodos_centrales = df_nodos_completo
    
    # 7. Ejecutar la visualización usando el método corregido
    print("\n🔍 Analizando composición de clusters y distribución de grados...")
    fig = analizador.visualizar_composicion_clusters_solo_nodos()
    
    if fig is not None:
        # Guardar la figura
        fig.savefig('composicion_clusters_desde_bruto.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n✅ Análisis de distribución log-log completado exitosamente.")
    else:
        print("❌ No se pudo generar la visualización")

if __name__ == "__main__":
    main_con_preprocesamiento()