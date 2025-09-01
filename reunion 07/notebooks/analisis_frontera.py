"""
Módulo de Análisis de Nodos Frontera
=====================================
Análisis y visualización de nodos en la frontera entre clusters políticos
Identifica actores puente y su comportamiento de conectividad cruzada
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

def calcular_metricas_frontera(df_analisis):
    """
    Calcula métricas adicionales para caracterizar mejor la frontera
    """
    df = df_analisis.copy()
    
    # Asegurar columnas necesarias
    if 'Cluster_Original' not in df.columns:
        df['Cluster_Original'] = df['Sesgo_Equilibrio'].apply(
            lambda x: 'S1' if x > 0 else 'S2'
        )
    
    # Calcular conexiones anómalas si no existe
    if 'Conexiones_Anomalas' not in df.columns:
        # Método 1: Basado en el sesgo de equilibrio
        # Si sesgo = 0, 50% son anómalas
        # Si sesgo = 1 o -1, 0% son anómalas
        df['Conexiones_Anomalas'] = (
            df['Conexiones_Totales'] * (1 - abs(df['Sesgo_Equilibrio'])) / 2
        ).astype(int)
    
    # Calcular métricas de frontera
    df['Proporcion_Anomala'] = df['Conexiones_Anomalas'] / (df['Conexiones_Totales'] + 1e-8)
    df['Indice_Frontera'] = df['Proporcion_Anomala'] * np.log1p(df['Conexiones_Anomalas'])
    df['Peso_Frontera'] = df['Conexiones_Anomalas'] * (1 - abs(df['Sesgo_Equilibrio']))
    
    return df

def analizar_nodos_frontera(df_analisis, top_n=10, metrica_size='grado_ponderado', figsize=(20, 10)):
    """
    Análisis de Nodos en la Frontera entre Clusters
    
    Visualiza nodos según su posición en la frontera:
    - Eje X: Número de conexiones anómalas (hacia el cluster opuesto)
    - Eje Y: Sesgo de equilibrio (balance entre clusters)
    - Color: Cluster original
    - Tamaño: Métrica de influencia seleccionada
    
    Parámetros:
    -----------
    df_analisis : DataFrame
        DataFrame con los datos de análisis
    top_n : int
        Número de nodos top a mostrar (default: 10)
    metrica_size : str
        Métrica para el tamaño de los nodos:
        - 'grado_total': Conexiones totales
        - 'grado_ponderado': Conexiones totales * proporción anómala
        - 'indice_frontera': Índice compuesto de frontera
    figsize : tuple
        Tamaño de la figura
    """
    
    # Filtrar y preparar datos
    df_puentes = df_analisis[df_analisis['Tipo_Analisis'] == 'Neutral_Puente'].copy()
    
    if df_puentes.empty:
        print("⚠️ No se encontraron nodos neutrales frontera en los datos")
        return None
    
    # Calcular métricas de frontera
    df_puentes = calcular_metricas_frontera(df_puentes)
    
    # Configuración de la visualización
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Configuración de períodos
    periodos = [
        ('Plebiscito_PRE', 'PRE-Plebiscito'),
        ('Plebiscito_POST', 'POST-Plebiscito')
    ]
    
    # Colores para clusters
    colores_cluster = {
        'S1': '#E63946',  # Rojo para Oficialismo
        'S2': '#1E88E5'   # Azul para Oposición
    }
    
    for i, (period, title) in enumerate(periodos):
        ax = axes[i]
        
        # Filtrar datos del período
        data_period = df_puentes[df_puentes['Dataset'] == period].copy()
        
        if data_period.empty:
            ax.text(0.5, 0.5, f'No hay datos para\n{title}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, alpha=0.7)
            continue
        
        # Seleccionar top nodos por índice de frontera
        data = data_period.nlargest(top_n, 'Indice_Frontera')
        
        # Calcular tamaños según métrica seleccionada
        if metrica_size == 'grado_ponderado':
            size_values = data['Conexiones_Totales'] * data['Proporcion_Anomala']
        elif metrica_size == 'indice_frontera':
            size_values = data['Indice_Frontera']
        else:  # grado_total
            size_values = data['Conexiones_Totales']
        
        # Normalizar tamaños
        min_size, max_size = 200, 1500
        if size_values.max() > size_values.min():
            sizes = min_size + (size_values - size_values.min()) / \
                    (size_values.max() - size_values.min()) * (max_size - min_size)
        else:
            sizes = [min_size] * len(size_values)
        
        # Colores según cluster original
        colors = [colores_cluster.get(cluster, '#808080') for cluster in data['Cluster_Original']]
        
        # === ZONA DE FRONTERA (fondo) ===
        # Definir zonas según el sesgo
        ax.axhspan(-0.3, 0.3, alpha=0.08, color='green', zorder=1)  # Zona frontera
        ax.axhspan(0.3, 1.1, alpha=0.05, color='red', zorder=1)    # Zona S1
        ax.axhspan(-1.1, -0.3, alpha=0.05, color='blue', zorder=1)  # Zona S2
        
        # Líneas de referencia
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=2, zorder=2)
        ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.3, linewidth=1, zorder=2)
        ax.axhline(y=-0.3, color='gray', linestyle='--', alpha=0.3, linewidth=1, zorder=2)
        
        # === SCATTER PLOT PRINCIPAL ===
        scatter = ax.scatter(
            data['Conexiones_Anomalas'], 
            data['Sesgo_Equilibrio'],
            s=sizes,
            c=colors,
            alpha=0.7,
            edgecolors='white',
            linewidths=2,
            zorder=5
        )
        
        # === ETIQUETAS INTELIGENTES ===
        # Agrupar nodos cercanos para evitar superposición
        for idx, (_, row) in enumerate(data.iterrows()):
            # Determinar cuadrante para posicionamiento
            if row['Sesgo_Equilibrio'] > 0:
                offset_y = 20 if idx % 2 == 0 else 35
            else:
                offset_y = -20 if idx % 2 == 0 else -35
            
            # Color del borde según proximidad a la frontera
            if abs(row['Sesgo_Equilibrio']) < 0.3:
                edge_color = 'green'  # En la frontera
                edge_width = 2
            else:
                edge_color = colores_cluster.get(row['Cluster_Original'], '#808080')
                edge_width = 1.5
            
            # Etiqueta con información
            label = f"{row['Nodo'].title()[:15]}\n({row['Proporcion_Anomala']:.0%})"
            
            ax.annotate(
                label,
                (row['Conexiones_Anomalas'], row['Sesgo_Equilibrio']),
                xytext=(10 if idx % 3 != 0 else -10, offset_y), 
                textcoords='offset points', 
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor='white', 
                         alpha=0.9, 
                         edgecolor=edge_color,
                         linewidth=edge_width),
                arrowprops=dict(arrowstyle='-', 
                              connectionstyle='arc3,rad=0.2',
                              color=edge_color,
                              alpha=0.4,
                              lw=1)
            )
        
        # === CONFIGURACIÓN DE EJES ===
        ax.set_xlabel('Influencia en la Frontera\n(N° Conexiones Anómalas)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Sesgo de Equilibrio\n← S2 (Oposición) | S1 (Oficialismo) →', 
                     fontsize=12, fontweight='bold')
        ax.set_title(f'Mapa de la Frontera Inter-Cluster\n{title}', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Grid mejorado
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)
        
        # Límites dinámicos
        ax.set_ylim(-1.1, 1.1)
        if not data.empty:
            x_max = data['Conexiones_Anomalas'].max() * 1.15
            ax.set_xlim(-x_max * 0.02, x_max)
        
        # === PANEL DE ESTADÍSTICAS ===
        # Calcular métricas por zona
        en_frontera = data[abs(data['Sesgo_Equilibrio']) <= 0.3]
        sesgo_s1 = data[data['Sesgo_Equilibrio'] > 0.3]
        sesgo_s2 = data[data['Sesgo_Equilibrio'] < -0.3]
        
        stats_text = "📊 CARACTERIZACIÓN DE LA FRONTERA\n"
        stats_text += "─" * 25 + "\n"
        stats_text += f"🟢 En Frontera: {len(en_frontera)} nodos\n"
        if len(en_frontera) > 0:
            stats_text += f"   Promedio anómalas: {en_frontera['Conexiones_Anomalas'].mean():.0f}\n"
        stats_text += f"🔴 Sesgo S1: {len(sesgo_s1)} nodos\n"
        stats_text += f"🔵 Sesgo S2: {len(sesgo_s2)} nodos\n"
        stats_text += "─" * 25 + "\n"
        stats_text += f"📈 Nodo más fronterizo:\n"
        if not en_frontera.empty:
            top_frontera = en_frontera.nlargest(1, 'Conexiones_Anomalas').iloc[0]
            stats_text += f"   {top_frontera['Nodo'].title()[:12]}\n"
            stats_text += f"   ({top_frontera['Proporcion_Anomala']:.0%} anómalas)"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.5', 
                        facecolor='white', 
                        edgecolor='gray',
                        linewidth=1,
                        alpha=0.95))
    
    # === LEYENDA PERSONALIZADA ===
    legend_elements = [
        mpatches.Circle((0, 0), 1, fc=colores_cluster['S1'], 
                       ec='white', linewidth=2, alpha=0.7,
                       label='Origen: Oficialismo (S1)'),
        mpatches.Circle((0, 0), 1, fc=colores_cluster['S2'], 
                       ec='white', linewidth=2, alpha=0.7,
                       label='Origen: Oposición (S2)'),
        mpatches.Rectangle((0, 0), 1, 1, fc='green', 
                          alpha=0.2, label='Zona Frontera (|sesgo| < 0.3)'),
        plt.Line2D([0], [0], marker='o', color='w', 
                  markerfacecolor='gray', markersize=12,
                  markeredgecolor='white', markeredgewidth=2,
                  label=f'Tamaño = {metrica_size.replace("_", " ").title()}')
    ]
    
    fig.legend(handles=legend_elements, 
              loc='upper center', 
              bbox_to_anchor=(0.5, -0.02),
              ncol=4, 
              frameon=True,
              fancybox=True,
              shadow=True,
              fontsize=10)
    
    # Título general
    fig.suptitle('🔍 CARACTERIZACIÓN DE LA FRONTERA POLÍTICA\n'
                'Identificación de Nodos Puente y su Comportamiento de Conectividad Cruzada', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.12)
    
    print("\n" + "="*70)
    print("📊 INTERPRETACIÓN DE LA VISUALIZACIÓN")
    print("="*70)
    print("📍 EJE X: Número absoluto de conexiones hacia el cluster opuesto")
    print("📍 EJE Y: Sesgo de equilibrio (qué tan balanceado está el nodo)")
    print("🟢 ZONA VERDE: Frontera real (sesgo equilibrado entre -0.3 y 0.3)")
    print("🔴 NODOS ROJOS: Originalmente del cluster Oficialismo")
    print("🔵 NODOS AZULES: Originalmente del cluster Oposición")
    print("⭕ TAMAÑO: Influencia ponderada en la frontera")
    print("="*70)
    
    return fig


def visualizar_frontera_avanzada(df_analisis, top_n=12, figsize=(16, 10)):
    """
    Visualización avanzada de la frontera con métricas compuestas
    
    Crea una visualización que pondera múltiples factores para identificar
    los verdaderos nodos frontera según su comportamiento de conectividad.
    """
    # Preparar datos
    df_puentes = df_analisis[df_analisis['Tipo_Analisis'] == 'Neutral_Puente'].copy()
    if df_puentes.empty:
        print("No hay datos de nodos frontera")
        return None
    
    df_puentes = calcular_metricas_frontera(df_puentes)
    
    # Crear figura con subplot único para comparación
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    
    # Colores y marcadores por período y cluster
    colores_cluster = {'S1': '#E63946', 'S2': '#1E88E5'}
    marcadores_periodo = {'Plebiscito_PRE': 'o', 'Plebiscito_POST': 's'}
    
    # Plot para cada combinación período-cluster
    for period in df_puentes['Dataset'].unique():
        data_period = df_puentes[df_puentes['Dataset'] == period]
        
        # Seleccionar top nodos
        data_top = data_period.nlargest(top_n, 'Peso_Frontera')
        
        for cluster in ['S1', 'S2']:
            data_cluster = data_top[data_top['Cluster_Original'] == cluster]
            
            if not data_cluster.empty:
                # Calcular tamaños (peso frontera normalizado)
                sizes = 100 + 500 * (data_cluster['Peso_Frontera'] / 
                                     df_puentes['Peso_Frontera'].max())
                
                # Plot
                scatter = ax.scatter(
                    data_cluster['Proporcion_Anomala'] * 100,  # Convertir a porcentaje
                    data_cluster['Conexiones_Anomalas'],
                    s=sizes,
                    c=[colores_cluster[cluster]] * len(data_cluster),
                    marker=marcadores_periodo[period],
                    alpha=0.6,
                    edgecolors='white',
                    linewidths=2,
                    label=f'{cluster} - {period.replace("Plebiscito_", "")}'
                )
                
                # Etiquetas para los más importantes
                for _, row in data_cluster.nlargest(3, 'Peso_Frontera').iterrows():
                    ax.annotate(
                        row['Nodo'].title()[:10],
                        (row['Proporcion_Anomala'] * 100, row['Conexiones_Anomalas']),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        alpha=0.8
                    )
    
    # Zonas de interés
    ax.axvspan(30, 50, alpha=0.1, color='green', label='Zona Frontera Ideal (30-50%)')
    ax.axhline(y=ax.get_ylim()[1] * 0.5, color='gray', linestyle='--', alpha=0.3)
    
    # Configuración
    ax.set_xlabel('Proporción de Conexiones Anómalas (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Número Absoluto de Conexiones Anómalas', fontsize=12, fontweight='bold')
    ax.set_title('Mapa de Frontera: Proporción vs Volumen de Conectividad Cruzada', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='upper left', fontsize=9)
    
    plt.tight_layout()
    return fig


def analizar_metricas_frontera(df_analisis, exportar=False):
    """
    Análisis detallado de métricas de frontera con recomendaciones
    
    Retorna un DataFrame con todas las métricas calculadas y recomendaciones
    sobre qué nodos son verdaderamente fronterizos.
    """
    df = df_analisis[df_analisis['Tipo_Analisis'] == 'Neutral_Puente'].copy()
    if df.empty:
        print("No hay datos para analizar")
        return None
    
    df = calcular_metricas_frontera(df)
    
    # Calcular score compuesto de frontera
    # Combina: proporción anómala, volumen de conexiones, y equilibrio de sesgo
    df['Score_Frontera'] = (
        df['Proporcion_Anomala'] * 0.4 +  # 40% peso a la proporción
        (df['Conexiones_Anomalas'] / df['Conexiones_Anomalas'].max()) * 0.3 +  # 30% al volumen
        (1 - abs(df['Sesgo_Equilibrio'])) * 0.3  # 30% al equilibrio
    )
    
    # Clasificar nodos
    df['Tipo_Frontera'] = pd.cut(
        df['Score_Frontera'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Periférico', 'Moderado', 'Fronterizo', 'Puente_Clave']
    )
    
    print("\n" + "="*70)
    print("📊 ANÁLISIS DE MÉTRICAS DE FRONTERA")
    print("="*70)
    
    for period in df['Dataset'].unique():
        data_period = df[df['Dataset'] == period]
        
        print(f"\n🕐 {period.replace('_', ' ').upper()}")
        print("-" * 40)
        
        # Top 5 nodos frontera por score compuesto
        top_frontera = data_period.nlargest(5, 'Score_Frontera')
        
        print("\n🏆 TOP 5 NODOS FRONTERA (Score Compuesto):")
        for i, (_, row) in enumerate(top_frontera.iterrows(), 1):
            cluster_emoji = "🔴" if row['Cluster_Original'] == 'S1' else "🔵"
            print(f"\n  {i}. {cluster_emoji} {row['Nodo'].title()}")
            print(f"     • Score Frontera: {row['Score_Frontera']:.3f}")
            print(f"     • Tipo: {row['Tipo_Frontera']}")
            print(f"     • Proporción Anómala: {row['Proporcion_Anomala']:.1%}")
            print(f"     • Conexiones Anómalas: {row['Conexiones_Anomalas']:.0f}")
            print(f"     • Sesgo: {row['Sesgo_Equilibrio']:+.3f}")
        
        # Distribución por tipo
        print("\n📈 DISTRIBUCIÓN POR TIPO DE FRONTERA:")
        for tipo in ['Puente_Clave', 'Fronterizo', 'Moderado', 'Periférico']:
            count = len(data_period[data_period['Tipo_Frontera'] == tipo])
            if count > 0:
                print(f"   • {tipo}: {count} nodos")
    
    print("\n" + "="*70)
    print("💡 RECOMENDACIONES")
    print("="*70)
    print("1. Los nodos 'Puente_Clave' son los verdaderos mediadores")
    print("2. Proporción anómala > 30% indica comportamiento fronterizo real")
    print("3. Score > 0.5 sugiere rol importante en la comunicación inter-cluster")
    print("4. Combinar volumen Y proporción da la mejor caracterización")
    print("="*70)
    
    if exportar:
        filename = 'metricas_frontera_detalladas.csv'
        df.to_csv(filename, index=False)
        print(f"\n✅ Métricas exportadas a: {filename}")
    
    return df


def comparar_periodos_frontera(df_analisis, figsize=(14, 8)):
    """
    Compara la evolución de la frontera entre períodos PRE y POST
    """
    df = df_analisis[df_analisis['Tipo_Analisis'] == 'Neutral_Puente'].copy()
    if df.empty:
        return None
    
    df = calcular_metricas_frontera(df)
    
    # Crear visualización comparativa
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Colores
    colores = {'Plebiscito_PRE': '#2E86AB', 'Plebiscito_POST': '#A23B72'}
    
    # 1. Distribución de proporción anómala
    ax1 = axes[0, 0]
    for period in df['Dataset'].unique():
        data = df[df['Dataset'] == period]['Proporcion_Anomala'] * 100
        ax1.hist(data, bins=15, alpha=0.6, label=period.replace('Plebiscito_', ''),
                color=colores[period], edgecolor='black')
    ax1.set_xlabel('Proporción de Conexiones Anómalas (%)')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución de Comportamiento Fronterizo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Boxplot de conexiones anómalas por cluster
    ax2 = axes[0, 1]
    data_box = []
    labels = []
    colors_box = []
    for period in df['Dataset'].unique():
        for cluster in ['S1', 'S2']:
            mask = (df['Dataset'] == period) & (df['Cluster_Original'] == cluster)
            if mask.any():
                data_box.append(df[mask]['Conexiones_Anomalas'])
                labels.append(f"{cluster}\n{period.replace('Plebiscito_', '')}")
                colors_box.append(colores[period])
    
    bp = ax2.boxplot(data_box, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_ylabel('N° Conexiones Anómalas')
    ax2.set_title('Volumen de Conectividad Cruzada por Cluster')
    ax2.grid(True, alpha=0.3)
    
    # 3. Evolución del índice de frontera
    ax3 = axes[1, 0]
    for cluster in ['S1', 'S2']:
        valores_pre = df[(df['Dataset'] == 'Plebiscito_PRE') & 
                        (df['Cluster_Original'] == cluster)]['Indice_Frontera'].mean()
        valores_post = df[(df['Dataset'] == 'Plebiscito_POST') & 
                         (df['Cluster_Original'] == cluster)]['Indice_Frontera'].mean()
        
        color = '#E63946' if cluster == 'S1' else '#1E88E5'
        ax3.plot([0, 1], [valores_pre, valores_post], marker='o', 
                markersize=10, linewidth=2, label=f'{cluster}', color=color)
    
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['PRE', 'POST'])
    ax3.set_ylabel('Índice de Frontera Promedio')
    ax3.set_title('Evolución del Comportamiento Fronterizo')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Scatter de cambio
    ax4 = axes[1, 1]
    # Encontrar nodos comunes
    nodos_pre = set(df[df['Dataset'] == 'Plebiscito_PRE']['Nodo'])
    nodos_post = set(df[df['Dataset'] == 'Plebiscito_POST']['Nodo'])
    nodos_comunes = list(nodos_pre.intersection(nodos_post))
    
    if nodos_comunes:
        cambios_x = []
        cambios_y = []
        colors_scatter = []
        
        for nodo in nodos_comunes[:20]:  # Limitar a 20 para claridad
            pre = df[(df['Dataset'] == 'Plebiscito_PRE') & (df['Nodo'] == nodo)]
            post = df[(df['Dataset'] == 'Plebiscito_POST') & (df['Nodo'] == nodo)]
            
            if not pre.empty and not post.empty:
                cambios_x.append(pre['Proporcion_Anomala'].values[0] * 100)
                cambios_y.append(post['Proporcion_Anomala'].values[0] * 100)
                cluster = pre['Cluster_Original'].values[0]
                colors_scatter.append('#E63946' if cluster == 'S1' else '#1E88E5')
        
        ax4.scatter(cambios_x, cambios_y, c=colors_scatter, alpha=0.6, s=100, edgecolor='white', linewidth=1)
        ax4.plot([0, 100], [0, 100], 'k--', alpha=0.3)  # Línea diagonal
        ax4.set_xlabel('Proporción Anómala PRE (%)')
        ax4.set_ylabel('Proporción Anómala POST (%)')
        ax4.set_title('Cambio Individual PRE → POST')
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Análisis Comparativo de la Frontera: PRE vs POST Plebiscito', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


# Función principal de exportación con todas las métricas
def exportar_analisis_completo(df_analisis, prefijo='frontera'):
    """
    Exporta un análisis completo de la frontera con todas las métricas
    """
    df = df_analisis[df_analisis['Tipo_Analisis'] == 'Neutral_Puente'].copy()
    if df.empty:
        print("No hay datos para exportar")
        return None
    
    df = calcular_metricas_frontera(df)
    
    # Calcular score compuesto
    df['Score_Frontera'] = (
        df['Proporcion_Anomala'] * 0.4 +
        (df['Conexiones_Anomalas'] / df['Conexiones_Anomalas'].max()) * 0.3 +
        (1 - abs(df['Sesgo_Equilibrio'])) * 0.3
    )
    
    # Clasificar
    df['Tipo_Frontera'] = pd.cut(
        df['Score_Frontera'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Periférico', 'Moderado', 'Fronterizo', 'Puente_Clave']
    )
    
    # Ordenar por score
    df = df.sort_values(['Dataset', 'Score_Frontera'], ascending=[True, False])
    
    # Exportar
    filename = f'{prefijo}_analisis_completo.csv'
    df.to_csv(filename, index=False)
    
    print(f"✅ Análisis completo exportado a: {filename}")
    print(f"   Total registros: {len(df)}")
    print(f"   Columnas: {', '.join(df.columns)}")
    
    return df