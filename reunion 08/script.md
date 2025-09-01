Claro, aquí tienes un archivo en formato Markdown (`.md`) que documenta el script, sus métricas y los parámetros de las fórmulas. Puedes guardar este texto en un archivo con la extensión `.md` (por ejemplo, `DOCUMENTACION.md`).

---

# Documentación: Script de Análisis de Redes Políticas

## 1. Objetivo del Script

Este script tiene como objetivo analizar una red de interacciones políticas previamente clusterizada. Su función principal es calcular un conjunto de métricas diseñadas para medir tanto la **estructura general de la polarización** como la **agencia individual** de los actores dentro de ella.

A partir de un archivo `.csv` con las interacciones y la asignación de cada actor a un cluster, el script genera un nuevo archivo `.csv` que enriquece a cada actor con las métricas calculadas, permitiendo un análisis más profundo de su rol en el conflicto.

## 2. Requisitos

Para ejecutar el script, es necesario tener instaladas las siguientes librerías de Python:

-   **pandas**: Para la manipulación de datos y la lectura/escritura de archivos CSV.
-   **numpy**: Para operaciones numéricas eficientes, utilizado por `networkx`.
-   **networkx**: Para la creación, manipulación y análisis de la estructura de la red.

## 3. Formato de los Datos de Entrada

El script espera un archivo `.csv` con la siguiente estructura de columnas:

| Columna | Descripción | Valores de Ejemplo |
| :--- | :--- | :--- |
| **FROM\_NODE** | El actor que origina la interacción. | `Gabriel Boric` |
| **TO\_NODE** | El actor que recibe la interacción. | `José Antonio Kast` |
| **SIGN** | La naturaleza de la interacción. | `positive`, `negative`, `neutral` |
| **CLUSTER** | El polo al que pertenece el actor. | `-1`, `1`, `0` (Neutral) |

---

## 4. Métricas Clave Calculadas

El script se centra en tres métricas fundamentales que capturan diferentes facetas de la polarización.

### 4.1. Centralidad de Autovector del Núcleo

Esta métrica mide la **prominencia o cohesión** de un actor dentro del **núcleo del conflicto**, es decir, considerando únicamente a los actores que no son neutrales (clusters -1 y 1). Un actor es central si está conectado a otros actores centrales.

#### Fórmula (Cociente de Rayleigh)

El puntaje de centralidad ($x$) se obtiene al encontrar el autovector principal ($x_1$) de la matriz de adyacencia del núcleo ($A_p$), que es el vector que maximiza la siguiente ecuación:

$$x_{1} = \arg\max_{x\ne0}\frac{x^{T}A_{p}x}{x^{T}x} = \frac{\sum_{i,j}(A_{p})_{ij}x_{i}x_{j}}{\sum_{i}x_{i}^{2}}$$

#### Desglose de Parámetros

* **$A_p$**: La **matriz de adyacencia del núcleo**. Es una sub-matriz que contiene solo las interacciones entre los nodos no neutrales.
* **$x$**: El **vector de centralidad**, donde cada componente corresponde a un actor del núcleo.
* **$x_i$**, **$x_j$**: El **puntaje de centralidad** del actor *i* y el actor *j*, respectivamente.
* **$(A_p)_{ij}$**: El valor de la **interacción** entre el actor *i* y el actor *j* en la matriz del núcleo (1 para positiva, -1 para negativa).

#### Interpretación

El resultado es un puntaje para cada actor del núcleo.
* El **signo** (+ o -) indica el polo al que tiende el actor dentro de la dinámica del conflicto.
* La **magnitud** (valor absoluto) indica su nivel de centralidad o influencia en esa dinámica. Es una medida del "volumen de consistencia" de un actor.

### 4.2. Grado Anómalo Externo ($d_{anom}$)

Esta métrica cuenta el número de **conexiones positivas** que un actor tiene con miembros del **cluster opuesto**. Es una medida directa de comportamiento que desafía la polarización.

#### Fórmula

Para un nodo $v$ que pertenece a un cluster $S_i$, su grado anómalo es:

$$d_{anom}(v) = d_{inter}^{+}(v)$$

#### Desglose de Parámetros

* **$v$**: El **actor (nodo)** que se está analizando.
* **$d_{inter}^{+}(v)$**: El **conteo de aristas positivas** desde el actor $v$ hacia todos los actores que pertenecen al cluster opuesto.

#### Interpretación

Un valor de $d_{anom}(v) > 0$ indica que un actor, a pesar de estar en un polo, mantiene lazos de afinidad con el bando contrario. Cuanto más alto el valor, más fuerte es este comportamiento anómalo.

### 4.3. Proporción de Anomalía Externa ($P_{anom}$)

Esta métrica refina la anterior, calculando qué **porcentaje** de las conexiones de un actor hacia el bando contrario son positivas. Ofrece una medida normalizada de la anomalía.

#### Fórmula

Para un nodo $v$ con al menos una conexión hacia el cluster opuesto:

$$P_{anom}(v) = \frac{d_{inter}^{+}(v)}{d_{inter}(v)} = \frac{d_{anom}(v)}{d_{inter}(v)}$$

#### Desglose de Parámetros

* **$d_{inter}^{+}(v)$**: El conteo de aristas **positivas** hacia el cluster opuesto (el Grado Anómalo).
* **$d_{inter}(v)$**: El conteo **total** de aristas (positivas y negativas) hacia el cluster opuesto.

#### Interpretación

El valor resultante está en un rango de 0 a 1:
* **$P_{anom}(v) = 0$**: **Polarización perfecta**. Todas las conexiones del actor con el bando opuesto son de conflicto (-1).
* **$P_{anom}(v) = 1$**: **Anomalía total**. Todas las conexiones del actor con el bando opuesto son de afinidad (+1).
* **$P_{anom}(v) \approx 0.5$**: **Ambigüedad**. El actor mantiene una mezcla equilibrada de relaciones positivas y negativas con el otro polo.

---

## 5. Ejecución y Salida

### Cómo ejecutar el script

1.  Asegúrate de que el archivo de datos CSV esté en la ubicación correcta, según la configuración de la variable `INPUT_CSV_PATH` en el script.
2.  Abre una terminal o línea de comandos.
3.  Navega hasta la carpeta que contiene el script.
4.  Ejecuta el comando: `python nombre_del_script.py`

### Archivo de Salida

El script generará un nuevo archivo llamado `node_metrics_with_formulas.csv` en la carpeta `results`. Este archivo contendrá las siguientes columnas:

| Columna | Descripción |
| :--- | :--- |
| **NODE** | El nombre del actor. |
| **CLUSTER** | El cluster original del actor (-1, 1, o 0). |
| **eigenvector\_centrality** | El puntaje de centralidad de autovector. Será 0 para nodos neutrales. |
| **d\_anom** | El Grado Anómalo Externo. |
| **p\_anom** | La Proporción de Anomalía Externa. |