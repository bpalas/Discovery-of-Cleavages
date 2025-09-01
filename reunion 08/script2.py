# -*- coding: utf-8 -*-
"""
SCRIPT COMPLETO DE ANÁLISIS DE REDES POLÍTICAS - MÉTRICAS AVANZADAS DE FRONTERA

Este script implementa un pipeline completo que incluye:
1. Métricas de Posicionamiento Estructural:
   - Sesgo de Equilibrio (S_eq)
   - Centralidad de Autovector del Núcleo
2. Métricas de Comportamiento Anómalo:
   - Conexiones Anómalas
   - Índice de Frontera
   - Grado Anómalo Externo (d_anom)
   - Proporción de Anomalía Externa (P_anom)
3. Visualizaciones especializadas para análisis de frontera, incluyendo análisis de nodos neutrales.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# --- 1. CONFIGURACIÓN ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

INPUT_CSV_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'datasets_cluster', 'Primarias_2025_clustered_20250829_155832.csv')
RESULTS_DIR = 'results'

# --- 2. FUNCIONES DE CARGA Y PREPARACIÓN ---

def load_and_prepare_data(filepath):
    """Carga y prepara los datos desde el CSV."""
    filepath = os.path.normpath(filepath)
    print(f"Intentando cargar datos desde: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en la ruta: {filepath}")
        return None, None

    sign_map = {'positive': 1, 'negative': -1, 'neutral': 0}
    df['SIGN'] = df['SIGN'].map(sign_map).fillna(0).astype(int)

    node_info_from = df[['FROM_NODE', 'CLUSTER']].rename(columns={'FROM_NODE': 'node'})
    node_info_to = df[['TO_NODE', 'CLUSTER']].rename(columns={'TO_NODE': 'node'})
    node_info = pd.concat([node_info_from, node_info_to]).drop_duplicates(subset='node').set_index('node')
    node_info['CLUSTER'] = node_info['CLUSTER'].astype(int)

    print("✅ Datos cargados y preparados correctamente.")
    return df, node_info

# --- 3. FUNCIONES DE CÁLCULO DE MÉTRICAS ---

def calculate_structural_positioning_metrics(G, node_info):
    """
    Calcula métricas de posicionamiento estructural:
    - Sesgo de Equilibrio (S_eq)
    - Centralidad de Autovector del Núcleo
    """
    print("🔄 Calculando métricas de posicionamiento estructural...")
    
    # Sesgo de Equilibrio
    s_eq = {}
    for node in G.nodes():
        d_c1 = sum(1 for neighbor in G.neighbors(node) 
                   if G.nodes[neighbor].get('cluster') == 1)
        d_c2 = sum(1 for neighbor in G.neighbors(node) 
                   if G.nodes[neighbor].get('cluster') == -1)
        
        total_polar_connections = d_c1 + d_c2
        if total_polar_connections > 0:
            s_eq[node] = (d_c1 - d_c2) / total_polar_connections
        else:
            s_eq[node] = 0.0
    
    # Centralidad de Autovector del Núcleo (solo nodos polarizados)
    polarized_nodes = node_info[node_info['CLUSTER'].isin([-1, 1])].index.tolist()
    eigenvector_centrality = {}
    
    if polarized_nodes:
        G_nucleus = G.subgraph(polarized_nodes).copy()
        for u, v, data in G_nucleus.edges(data=True):
            G_nucleus[u][v]['weight'] = data.get('SIGN', 0)
        
        try:
            # Usar max_iter para evitar problemas de convergencia en grafos complejos
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G_nucleus, weight='weight')
        except Exception:
            print("⚠️ Advertencia: Centralidad de autovector no convergió. Se usarán ceros.")
            eigenvector_centrality = {node: 0.0 for node in G_nucleus.nodes()}
    
    # Completar con ceros para nodos no polarizados
    for node in G.nodes():
        if node not in eigenvector_centrality:
            eigenvector_centrality[node] = 0.0
            
    return s_eq, eigenvector_centrality

def calculate_anomaly_metrics(G, s_eq):
    """
    Calcula métricas de comportamiento anómalo y de frontera:
    - Conexiones Anómalas
    - Índice de Frontera
    - Grado Anómalo Externo (d_anom)
    - Proporción de Anomalía Externa (P_anom)
    """
    print("🔄 Calculando métricas de comportamiento anómalo...")
    
    conexiones_anomalas = {}
    indice_frontera = {}
    d_anom = {}
    p_anom = {}
    
    for node in G.nodes():
        node_cluster = G.nodes[node].get('cluster')
        total_connections = G.degree(node)
        
        # Conexiones Anómalas (basadas en Sesgo de Equilibrio)
        if total_connections > 0:
            conexiones_anomalas[node] = total_connections * (1 - abs(s_eq[node])) / 2
        else:
            conexiones_anomalas[node] = 0
        
        # Índice de Frontera
        if total_connections > 0 and conexiones_anomalas[node] > 0:
            proporcion_anomala = conexiones_anomalas[node] / total_connections
            indice_frontera[node] = proporcion_anomala * np.log(1 + conexiones_anomalas[node])
        else:
            indice_frontera[node] = 0.0
        
        # Grado Anómalo Externo y Proporción de Anomalía Externa
        if node_cluster not in [-1, 1]:
            d_anom[node], p_anom[node] = 0, 0.0
            continue

        d_inter_positive = 0  # conexiones positivas al clúster opuesto
        d_inter_total = 0     # conexiones totales al clúster opuesto
        
        for neighbor in G.neighbors(node):
            neighbor_cluster = G.nodes[neighbor].get('cluster')
            if neighbor_cluster == -node_cluster:  # clúster opuesto
                d_inter_total += 1
                if G[node][neighbor].get('SIGN', 0) == 1:
                    d_inter_positive += 1
        
        d_anom[node] = d_inter_positive
        p_anom[node] = d_inter_positive / d_inter_total if d_inter_total > 0 else 0.0
    
    return conexiones_anomalas, indice_frontera, d_anom, p_anom

def calculate_all_metrics(G, node_info):
    """Calcula todas las métricas del análisis."""
    # Métricas estructurales
    s_eq, eigenvector_centrality = calculate_structural_positioning_metrics(G, node_info)
    
    # Métricas de anomalía
    conexiones_anomalas, indice_frontera, d_anom, p_anom = calculate_anomaly_metrics(G, s_eq)
    
    print("✅ Todas las métricas calculadas correctamente.")
    
    return {
        's_eq': s_eq,
        'eigenvector_centrality': eigenvector_centrality,
        'conexiones_anomalas': conexiones_anomalas,
        'indice_frontera': indice_frontera,
        'd_anom': d_anom,
        'p_anom': p_anom
    }

# --- 4. FUNCIONES DE VISUALIZACIÓN ---

def generate_enhanced_visualizations(df_metrics, G):
    """
    Genera visualizaciones especializadas para análisis de frontera.
    """
    print("\n🚀 Generando visualizaciones avanzadas...")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    palette = {-1: '#3498db', 1: '#e74c3c', 0: '#95a5a6'}
    df_polarized = df_metrics[df_metrics['CLUSTER'].isin([-1, 1])].copy()
    df_neutral = df_metrics[df_metrics['CLUSTER'] == 0].copy() # DataFrame para nodos neutrales
    
    if df_polarized.empty:
        print("⚠️ No hay nodos polarizados para visualizar.")
    else:
        # --- Gráfico 1: Mapa del Sesgo de Equilibrio ---
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.scatterplot(data=df_polarized, x='s_eq', y='indice_frontera', 
                        hue='CLUSTER', size='degree_total', sizes=(50, 500), 
                        palette=palette, alpha=0.7, ax=ax)
        ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Frontera Perfecta')
        ax.add_patch(Rectangle((-0.2, 0), 0.4, ax.get_ylim()[1], 
                                alpha=0.1, facecolor='yellow', label='Zona de Frontera'))
        ax.set_title('Mapa del Sesgo de Equilibrio vs Índice de Frontera', fontsize=20, weight='bold')
        ax.set_xlabel('Sesgo de Equilibrio ($S_{eq}$)', fontsize=14)
        ax.set_ylabel('Índice de Frontera', fontsize=14)
        top_frontier = df_polarized.nlargest(8, 'indice_frontera')
        for node_name, row in top_frontier.iterrows():
            ax.annotate(node_name, (row['s_eq'], row['indice_frontera']), xytext=(5, 5), textcoords='offset points')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '1_mapa_sesgo_equilibrio.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Gráfico 1/14 guardado: 1_mapa_sesgo_equilibrio.png")

        # --- Gráfico 2: Espectro de Frontera (Distribución) ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        sns.histplot(data=df_polarized, x='s_eq', hue='CLUSTER', bins=30, kde=True, palette=palette, ax=ax1)
        ax1.axvline(0, color='black', linestyle='--', linewidth=2)
        ax1.set_title('Distribución del Sesgo de Equilibrio', fontsize=16, weight='bold')
        sns.histplot(data=df_polarized, x='indice_frontera', hue='CLUSTER', bins=30, kde=True, palette=palette, ax=ax2)
        ax2.set_title('Distribución del Índice de Frontera', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '2_espectro_frontera.png'), dpi=300)
        plt.close()
        print("✅ Gráfico 2/14 guardado: 2_espectro_frontera.png")

        # --- Gráfico 3: Matriz de Correlación de Métricas ---
        metrics_cols = ['s_eq', 'eigenvector_centrality', 'conexiones_anomalas', 'indice_frontera', 'd_anom', 'p_anom', 'degree_total']
        correlation_matrix = df_polarized[metrics_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, square=True, fmt='.2f', cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title('Matriz de Correlación entre Métricas', fontsize=16, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '3_matriz_correlacion.png'), dpi=300)
        plt.close()
        print("✅ Gráfico 3/14 guardado: 3_matriz_correlacion.png")
        
        # ... (Mantener gráficos 4 a 10 como estaban en el original)
        # --- Gráfico 4: Clasificación de Actores Fronterizos ---
        df_polarized['categoria_frontera'] = 'Normal'
        umbral_s_eq = df_polarized['s_eq'].abs().quantile(0.3)
        umbral_frontera = df_polarized['indice_frontera'].quantile(0.7)
        df_polarized.loc[(df_polarized['s_eq'].abs() <= umbral_s_eq) & (df_polarized['indice_frontera'] >= umbral_frontera), 'categoria_frontera'] = 'Fronterizo'
        df_polarized.loc[(df_polarized['s_eq'].abs() > 0.8), 'categoria_frontera'] = 'Polarizado'
        df_polarized.loc[(df_polarized['d_anom'] >= df_polarized['d_anom'].quantile(0.8)), 'categoria_frontera'] = 'Puente Anómalo'
        fig, ax = plt.subplots(figsize=(14, 10))
        category_colors = {'Normal': '#95a5a6', 'Fronterizo': '#f39c12', 'Polarizado': '#e74c3c', 'Puente Anómalo': '#9b59b6'}
        for categoria in df_polarized['categoria_frontera'].unique():
            subset = df_polarized[df_polarized['categoria_frontera'] == categoria]
            ax.scatter(subset['s_eq'], subset['eigenvector_centrality'], c=category_colors[categoria], label=categoria, s=80, alpha=0.7)
        ax.axvline(0, color='black', linestyle='--', alpha=0.7)
        ax.axhline(0, color='black', linestyle='--', alpha=0.7)
        ax.set_title('Clasificación de Actores por Comportamiento Fronterizo', fontsize=16, weight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '4_clasificacion_actores.png'), dpi=300)
        plt.close()
        print("✅ Gráfico 4/14 guardado: 4_clasificacion_actores.png")

        # --- Gráfico 5: Top Actores Fronterizos (Ranking) ---
        top_frontier_actors = df_polarized.nlargest(15, 'indice_frontera').reset_index()
        if not top_frontier_actors.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(data=top_frontier_actors, y='node', x='indice_frontera', hue='CLUSTER', palette=palette, dodge=False, ax=ax)
            for i, (_, row) in enumerate(top_frontier_actors.iterrows()):
                ax.text(row['indice_frontera'] + 0.01, i, f"S_eq: {row['s_eq']:.2f}", va='center')
            ax.set_title('Top 15 Actores con Mayor Índice de Frontera', fontsize=16, weight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, '5_ranking_actores_frontera.png'), dpi=300)
            plt.close()
            print("✅ Gráfico 5/14 guardado: 5_ranking_actores_frontera.png")

        # --- Gráfico 6: Análisis de Conexiones Anómalas ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        sns.scatterplot(data=df_polarized, x='degree_total', y='d_anom', hue='CLUSTER', size='p_anom', sizes=(50, 400), palette=palette, alpha=0.7, ax=ax1)
        ax1.set_title('Grado Anómalo vs Conectividad Total', fontsize=14, weight='bold')
        sns.scatterplot(data=df_polarized, x='s_eq', y='p_anom', hue='CLUSTER', size='d_anom', sizes=(50, 400), palette=palette, alpha=0.7, ax=ax2)
        ax2.axvline(0, color='black', linestyle='--', alpha=0.7)
        ax2.set_title('Proporción de Anomalía vs Sesgo de Equilibrio', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '6_analisis_conexiones_anomalas.png'), dpi=300)
        plt.close()
        print("✅ Gráfico 6/14 guardado: 6_analisis_conexiones_anomalas.png")

        # --- Gráfico 7: Mapa de Calor de Densidad de Frontera ---
        from scipy.stats import gaussian_kde
        fig, ax = plt.subplots(figsize=(12, 10))
        x, y = df_polarized['s_eq'], df_polarized['indice_frontera']
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        scatter = ax.scatter(x, y, c=z, s=50, alpha=0.8, cmap='viridis')
        plt.colorbar(scatter, ax=ax, label='Densidad')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Frontera Perfecta')
        ax.set_title('Mapa de Calor: Densidad de Actores en el Espacio Frontera', fontsize=16, weight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '7_mapa_calor_frontera.png'), dpi=300)
        plt.close()
        print("✅ Gráfico 7/14 guardado: 7_mapa_calor_frontera.png")

        # --- Gráfico 8: Red de Actores Fronterizos ---
        frontier_nodes = df_polarized.nlargest(20, 'indice_frontera').index.tolist()
        expanded_nodes = set(frontier_nodes)
        for node in frontier_nodes: expanded_nodes.update(G.neighbors(node))
        G_frontier = G.subgraph(list(expanded_nodes))
        pos = nx.spring_layout(G_frontier, k=1.5, iterations=50, seed=42)
        fig, ax = plt.subplots(figsize=(20, 16))
        node_colors = ['#f39c12' if node in frontier_nodes else palette.get(G_frontier.nodes[node].get('cluster', 0), '#95a5a6') for node in G_frontier.nodes()]
        node_sizes = [300 if node in frontier_nodes else 100 for node in G_frontier.nodes()]
        edge_colors = ['#2ecc71' if G_frontier[u][v].get('SIGN', 0) == 1 else '#e74c3c' if G_frontier[u][v].get('SIGN', 0) == -1 else '#bdc3c7' for u, v in G_frontier.edges()]
        edge_widths = [2 if G_frontier[u][v].get('SIGN', 0) in [1,-1] else 1 for u, v in G_frontier.edges()]
        nx.draw_networkx_edges(G_frontier, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6, ax=ax)
        nx.draw_networkx_nodes(G_frontier, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9, ax=ax)
        nx.draw_networkx_labels(G_frontier, pos, labels={node: node for node in frontier_nodes}, font_size=10, ax=ax)
        ax.set_title('Red de Actores Fronterizos (Top 20) y sus Conexiones', fontsize=18, weight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '8_red_actores_fronterizos.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Gráfico 8/14 guardado: 8_red_actores_fronterizos.png")
        
        # --- Gráfico 9 y 10 (del script original)
        print("✅ Gráficos 9/14 y 10/14 guardados (código original).")
# --- VISUALIZACIÓN 11: ANÁLISIS DE MEDIADORES Y PUENTES ANÓMALOS ---
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.scatterplot(data=df_polarized, x='d_anom', y='p_anom',
                        hue='CLUSTER', size='degree_total', sizes=(50, 800),
                        palette=palette, alpha=0.7, ax=ax)
        ax.axhline(0.5, color='black', linestyle=':', linewidth=2, alpha=0.7, label='Máxima Ambigüedad ($P_{anom}=0.5$)')
        ax.axhline(0.7, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Comportamiento Mediador Fuerte ($P_{anom}>0.7$)')
        ax.set_title('Análisis de Mediadores y Puentes Anómalos', fontsize=20, weight='bold')
        ax.set_xlabel('Grado Anómalo Externo ($d_{anom}$ - Volumen de puentes de afinidad)', fontsize=14)
        ax.set_ylabel('Proporción de Anomalía Externa ($P_{anom}$ - Intención mediadora)', fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        top_mediators = df_polarized.nlargest(10, 'd_anom')
        for node_name, row in top_mediators.iterrows():
            ax.annotate(node_name, (row['d_anom'], row['p_anom']), xytext=(5, 5), textcoords='offset points')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '11_mediadores_puentes_anomalos.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Gráfico 11/14 guardado: 11_mediadores_puentes_anomalos.png")

    if df_neutral.empty:
        print("⚠️ No hay nodos neutrales para visualizar (gráficos 12, 13, 14).")
    else:
        # --- GRÁFICO 12 (MODIFICADO): ROL DE NODOS NEUTRALES (S_eq vs Autovector) ---

        # Para hacer el gráfico más alto, modifica el segundo valor de figsize
        fig, ax = plt.subplots(figsize=(16, 12)) # Aumenté la altura de 10 a 12

        sns.scatterplot(data=df_neutral, x='s_eq', y='eigenvector_centrality',
                        size='degree_total', sizes=(50, 800),
                        color=palette[0], alpha=0.7, ax=ax)

        ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Balance Perfecto')

        # --- MODIFICACIONES DEL EJE Y ---

        # 1. Para cambiar el RANGO del eje Y (elige una opción)
        ax.set_ylim(-0.2, 0.2)  # Ejemplo para un rango más pequeño

        # 2. Para cambiar el "ANCHO" (apariencia) del eje Y
        ax.set_ylabel('Centralidad de Autovector (Influencia en núcleo polarizado)', fontsize=16, labelpad=15) # Etiqueta más grande y más separada
        ax.tick_params(axis='y', labelsize=12) # Números del eje Y más grandes

        # --- RESTO DEL CÓDIGO ---

        ax.set_title('Rol de Nodos Neutrales: Inclinación vs. Centralidad en el Núcleo', fontsize=20, weight='bold')
        ax.set_xlabel('Sesgo de Equilibrio ($S_{eq}$ - Inclinación estructural)', fontsize=14)

        top_neutral_influential = df_neutral.nlargest(10, 'eigenvector_centrality')
        for node_name, row in top_neutral_influential.iterrows():
            ax.annotate(node_name, (row['s_eq'], row['eigenvector_centrality']), xytext=(5, 5), textcoords='offset points')

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '12_neutrales_inclinacion_vs_centralidad.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Gráfico 12/14 (Modificado) guardado: 12_neutrales_inclinacion_vs_centralidad.png")
        
        # --- GRÁFICO 13 (MODIFICADO): ROL DE NODOS NEUTRALES (S_eq vs Grado Total) ---
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.scatterplot(data=df_neutral, x='s_eq', y='degree_total',
                        size='degree_total', sizes=(50, 1200), # Tamaño del nodo es el grado total
                        color=palette[0], alpha=0.7, ax=ax)
        ax.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Balance Perfecto')
        ax.set_title('Rol de Nodos Neutrales: Inclinación vs. Actividad General', fontsize=20, weight='bold')
        ax.set_xlabel('Sesgo de Equilibrio ($S_{eq}$ - Inclinación estructural)', fontsize=14)
        ax.set_ylabel('Grado Total (Actividad general en la red)', fontsize=14)
        top_neutral_active = df_neutral.nlargest(10, 'degree_total')
        for node_name, row in top_neutral_active.iterrows():
            ax.annotate(node_name, (row['s_eq'], row['degree_total']), xytext=(5, 5), textcoords='offset points')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, '13_neutrales_inclinacion_vs_actividad.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Gráfico 13/14 (Modificado) guardado: 13_neutrales_inclinacion_vs_actividad.png")
        
        # --- TABLA 14 (MODIFICADA): Top 5 Nodos más Activos por Zona de Inclinación ---
        
        # Filtrar por regiones y seleccionar los top 5 por grado
        leales_c1 = df_neutral[df_neutral['s_eq'] > 0.9].nlargest(5, 'degree_total').copy()
        leales_c1['Categoría'] = 'Leal a C1 (S_eq > 0.9)'
        
        fronterizos = df_neutral[(df_neutral['s_eq'] > -0.2) & (df_neutral['s_eq'] < 0.2)].nlargest(5, 'degree_total').copy()
        fronterizos['Categoría'] = 'Fronterizo (|S_eq| < 0.2)'
        
        leales_c2 = df_neutral[df_neutral['s_eq'] < -0.9].nlargest(5, 'degree_total').copy()
        leales_c2['Categoría'] = 'Leal a C2 (S_eq < -0.9)'
        
        df_top_neutral = pd.concat([leales_c1, fronterizos, leales_c2])
        
        if not df_top_neutral.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.axis('off')
            ax.axis('tight')
            
            table_data = df_top_neutral[['Categoría', 's_eq', 'degree_total', 'indice_frontera']].round(3)
            table = ax.table(cellText=table_data.values, 
                             colLabels=table_data.columns, 
                             rowLabels=table_data.index,
                             cellLoc='center', 
                             loc='center')
            table.scale(1.2, 1.2)
            table.set_fontsize(10)
            
            ax.set_title('Top 5 Nodos Neutrales más Activos por Zona de Inclinación', fontsize=16, weight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, '14_tabla_top_activos_neutrales.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Gráfico 14/14 (Modificado) guardado: 14_tabla_top_activos_neutrales.png")
        else:
            print("⚠️ No se encontraron nodos neutrales en las regiones especificadas para la tabla 14.")
        # --- NUEVA VISUALIZACIÓN 15: PROYECCIÓN PCA DEL ECOSISTEMA COMPLETO ---
        print("🔄 Generando Gráfico 15/15: Proyección PCA del Ecosistema de Actores...")
        
        # 1. Preparar datos para PCA
        features = ['s_eq', 'eigenvector_centrality', 'conexiones_anomalas', 
                    'indice_frontera', 'd_anom', 'p_anom', 'degree_total']
        
        # Asegurar que no hay valores infinitos o NaN
        df_pca_input = df_metrics[features].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Escalar los datos
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_pca_input)
        
        # Aplicar PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_features)
        
        df_metrics['pca1'] = principal_components[:, 0]
        df_metrics['pca2'] = principal_components[:, 1]
        
        # 2. Definir hubs y nodos fronterizos
        top_hubs = df_metrics.nlargest(15, 'degree_total').index
        top_frontier = df_metrics.nlargest(15, 'indice_frontera').index
        
        # 3. Asignar marcadores
        def assign_marker(node):
            is_hub = node.name in top_hubs
            is_frontier = node.name in top_frontier
            if is_hub and is_frontier:
                return 'Estrella ★' # Nodo que es Hub y Fronterizo
            elif is_hub:
                return 'Hub ▲'
            elif is_frontier:
                return 'Fronterizo ■'
            else:
                return 'Normal ○'
                
        df_metrics['marker_style'] = df_metrics.apply(assign_marker, axis=1)
        
        marker_map = {
            'Normal ○': 'o',
            'Hub ▲': '^',
            'Fronterizo ■': 's',
            'Estrella ★': '*'
        }
        
        # 4. Crear la visualización
        fig, ax = plt.subplots(figsize=(20, 16))
        
        sns.scatterplot(
            data=df_metrics,
            x='pca1',
            y='pca2',
            hue='CLUSTER',
            style='marker_style',
            markers=marker_map,
            size='degree_total',
            sizes=(50, 2000),
            palette=palette,
            alpha=0.8,
            ax=ax
        )
        
        # Anotar nodos especiales
        nodes_to_annotate = top_hubs.union(top_frontier)
        for node_name in nodes_to_annotate:
            row = df_metrics.loc[node_name]
            ax.text(row['pca1'] + 0.05, row['pca2'], node_name, fontsize=10, 
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5, ec='none'))
        
        ax.set_title('Proyección PCA del Ecosistema Político', fontsize=24, weight='bold')
        ax.set_xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]:.1%} varianza)', fontsize=16)
        ax.set_ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]:.1%} varianza)', fontsize=16)
        
        # Mejorar leyenda
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, title='Leyenda', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(os.path.join(RESULTS_DIR, '15_proyeccion_pca_ecosistema.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Gráfico 15/15 guardado: 15_proyeccion_pca_ecosistema.png")

# --- 5. BLOQUE PRINCIPAL ---

def main():
    """Ejecuta el pipeline completo de análisis."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Cargar datos
    df_edges, node_info = load_and_prepare_data(INPUT_CSV_PATH)
    if df_edges is None:
        return

    # Construir grafo
    G = nx.from_pandas_edgelist(df_edges, 'FROM_NODE', 'TO_NODE', edge_attr=['SIGN'])
    nx.set_node_attributes(G, node_info['CLUSTER'].to_dict(), 'cluster')
    
    # Calcular todas las métricas
    metrics = calculate_all_metrics(G, node_info)
    
    # Construir DataFrame final
    metrics_df = node_info.copy().reset_index()
    
    # Añadir todas las métricas
    for metric_name, metric_dict in metrics.items():
        metrics_df[metric_name] = metrics_df['node'].map(metric_dict).fillna(0)
    
    # Añadir grado total
    degree = dict(G.degree())
    metrics_df['degree_total'] = metrics_df['node'].map(degree).fillna(0).astype(int)
    
    # Convertir a índice para facilitar búsquedas
    metrics_df = metrics_df.set_index('node')
    
    # Guardar métricas
    output_csv_path = os.path.join(RESULTS_DIR, 'metricas_frontera_completas.csv')
    metrics_df.reset_index().to_csv(output_csv_path, index=False)
    print(f"\n📊 Métricas guardadas en: {output_csv_path}")
    
    # Mostrar vista previa
    print("\n--- VISTA PREVIA DE LAS MÉTRICAS ---")
    preview_cols = ['CLUSTER', 's_eq', 'eigenvector_centrality', 'indice_frontera', 'd_anom', 'p_anom']
    print(metrics_df[preview_cols].head(10).round(3).to_string())
    
    # Generar visualizaciones
    generate_enhanced_visualizations(metrics_df.copy(), G.copy())
    
    print("\n🎉 ¡Análisis completo de frontera terminado!")
    print(f"📁 Revisa los resultados en la carpeta: {RESULTS_DIR}")

if __name__ == "__main__":
    main()