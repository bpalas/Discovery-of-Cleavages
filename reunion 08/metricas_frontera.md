# Métricas para el Análisis de Polarización en Redes Signadas


## 1. Métricas de Posicionamiento Estructural

Estas métricas caracterizan la ubicación, inclinación e influencia de cada nodo dentro de la arquitectura global del conflicto, revelando el rol estructural que cada actor desempeña en la dinámica de polarización.

### 1.1. Sesgo de Equilibrio ($S_{eq}$)

**Definición:** Cuantifica la inclinación o lealtad estructural de un nodo hacia uno de los dos polos, basándose en el balance total de sus conexiones.

**Fórmula:**
$$S_{eq}(v) = \frac{d_{C1}(v) - d_{C2}(v)}{d_{C1}(v) + d_{C2}(v)}$$

donde $d_{Ci}(v)$ representa el grado total (conexiones) del nodo $v$ hacia el clúster $i$.

**Interpretación:**
- **$S_{eq}(v) = +1$**: Lealtad estructural total al Clúster 1
- **$S_{eq}(v) = -1$**: Lealtad estructural total al Clúster 2  
- **$S_{eq}(v) = 0$**: Nodo perfectamente balanceado o fronterizo
- **$|S_{eq}(v)| < 0.2$**: Zona de alta ambigüedad estructural

### 1.2. Centralidad de Autovector del Núcleo

**Definición:** Mide la prominencia e influencia de un actor dentro del núcleo polarizado del conflicto (excluyendo actores neutrales). Captura el principio de que un actor es central si está fuertemente conectado a otros actores centrales.

**Contexto del Vector de Asignación:** El análisis parte de un vector indicatriz $c \in \{-1, 0, 1\}^n$ que asigna cada nodo a su respectivo clúster:
- $c_i = +1$: Nodo pertenece al Clúster 1
- $c_i = -1$: Nodo pertenece al Clúster 2  
- $c_i = 0$: Nodo neutral (excluido del núcleo polarizado)

**Fórmula (Optimización del Cociente de Rayleigh):**
$x_1 = \arg\max_{x \neq 0} \frac{x^T A_p x}{x^T x}$

donde $A_p \in \mathbb{R}^{m \times m}$ es la submatriz de adyacencia signada que contiene únicamente las interacciones entre los $m = |V_p|$ nodos del núcleo polarizado (aquellos con $c_i \neq 0$).

**Interpretación:**
- **Signo $(+/-)$**: Indica la polaridad del actor en la dinámica del conflicto
- **Magnitud $|x_1(v)|$**: Cuantifica el nivel de influencia y cohesión dentro de su polo
- **Valores cercanos a cero**: Actores estructuralmente ambiguos en la frontera del clivaje
- **Relación con $c$**: El signo de $x_1(v)$ tiende a coincidir con $c_v$, pero su magnitud revela la centralidad real dentro de la polarización

---

## 2. Métricas de Comportamiento Anómalo y Agencia de Frontera

Estas métricas se enfocan en detectar y cuantificar comportamientos que desafían la división estricta de la red, identificando actores con potencial transformador o de mediación.

### 2.1. Conexiones Anómalas (Cálculo Auxiliar)

**Definición:** Estimación del número de conexiones dirigidas hacia el clúster minoritario para un nodo dado. Constituye la base para las métricas de frontera subsecuentes.

**Fórmula:**
$$\text{Conexiones\_Anómalas}(v) = d_{total}(v) \times \frac{1 - |S_{eq}(v)|}{2}$$

**Interpretación:**
- Valor máximo cuando $S_{eq}(v) = 0$ (nodo perfectamente fronterizo)
- Valor mínimo (0) cuando $|S_{eq}(v)| = 1$ (lealtad total a un polo)

### 2.2. Índice de Frontera

**Definición:** Métrica compuesta que combina la proporción y el volumen absoluto de conexiones anómalas para identificar puentes estructuralmente significativos entre clústeres.

**Fórmula:**
$$\text{Índice\_Frontera}(v) = \left( \frac{\text{Conexiones\_Anómalas}(v)}{d_{total}(v)} \right) \times \ln(1 + \text{Conexiones\_Anómalas}(v))$$

**Interpretación:**
- **Valores altos**: Candidatos robustos a elementos de frontera con agencia transformadora
- **Componente proporcional**: Normaliza por la actividad total del nodo
- **Componente logarítmica**: Penaliza nodos con pocas conexiones anómalas absolutas

### 2.3. Grado Anómalo Externo ($d_{anom}$)

**Definición:** Cuenta directa del número de **conexiones positivas** (de afinidad/apoyo) que un actor mantiene con miembros del clúster opuesto.

**Fórmula:**
$$d_{anom}(v) = d_{inter}^{+}(v)$$

donde $d_{inter}^{+}(v)$ son las aristas positivas inter-clúster del nodo $v$.

**Interpretación:**
- Medida absoluta de "puentes de afinidad" hacia el bando contrario
- Cuantifica directamente el comportamiento no polarizado
- Valor 0 indica polarización perfecta en relaciones externas

### 2.4. Proporción de Anomalía Externa ($P_{anom}$)

**Definición:** Normalización de la métrica anterior que calcula el porcentaje de conexiones inter-clúster que son de naturaleza positiva.

**Fórmula:**
$$P_{anom}(v) = \frac{d_{inter}^{+}(v)}{d_{inter}(v)}$$

**Interpretación:**
- **$P_{anom}(v) = 1$**: Anomalía total (todas las conexiones externas son de afinidad)
- **$P_{anom}(v) = 0$**: Polarización perfecta (todas las conexiones externas son de conflicto)  
- **$P_{anom}(v) = 0.5$**: Máxima ambigüedad relacional externa
- **$P_{anom}(v) > 0.7$**: Comportamiento fuertemente mediador

---

## 3. Síntesis: Tipología de Actores

La combinación de estas métricas permite identificar arquetipos fundamentales:

**Actores Nucleares:** Alto $|x_1|$, $|S_{eq}| > 0.8$, $P_{anom} < 0.2$  
**Actores de Frontera:** $|S_{eq}| < 0.3$, Alto Índice de Frontera  
**Mediadores Activos:** $d_{anom} > 0$, $P_{anom} > 0.5$  
**Puentes Estructurales:** Alto $d_{ext}$, Moderado $S_{eq}$

Esta tipología facilita la identificación de actores clave para entender tanto la reproducción como la transformación potencial de la estructura de clivaje.