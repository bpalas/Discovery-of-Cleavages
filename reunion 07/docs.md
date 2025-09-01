# Análisis de Fronteras de Clusters en Redes Signadas

## Marco Conceptual

### Supuestos del Modelo
1. **Cohesión Interna vs. Conflicto Externo (Comportamiento Esperado)**: 
   - **Links positivos dentro de un cluster** → Cohesión ideológica
   - **Links negativos entre clusters** → Polarización y conflicto
   
2. **Anomalías de Interés Analítico**:
   - **Links positivos entre clusters** → Posibles alianzas, diálogo, moderación
   - **Links negativos dentro de un cluster** → Disidencia, conflicto interno, fragmentación

---

## 1. Análisis de Permeabilidad de Clusters

### ¿Qué fracción de cada cluster pertenece a la frontera?

La **permeabilidad** ($P(C_i)$) cuantifica qué tan "expuesto" está un cluster a influencias externas. Es la proporción de nodos que mantienen conexiones fuera de su grupo de pertenencia.

#### Definición Matemática

$$P(C_i) = \frac{|F(C_i)|}{|C_i|}$$

Donde:
- $F(C_i)$ = Conjunto de nodos frontera del cluster $C_i$
- $|C_i|$ = Tamaño total del cluster $C_i$
- Un nodo está en la frontera si tiene al menos una conexión con neutrales o el cluster opuesto

#### Interpretación de Valores

| Rango de P(Ci) | Interpretación | Implicaciones Estratégicas |
|----------------|----------------|---------------------------|
| $P(C_i) < 0.3$ | **Baja Permeabilidad** | Cluster cohesivo, resistente a influencia externa |
| $0.3 ≤ P(C_i) < 0.6$ | **Permeabilidad Media** | Balance entre cohesión y apertura |
| $P(C_i) ≥ 0.6$ | **Alta Permeabilidad** | Cluster muy expuesto, susceptible a cambios |

---

## 2. Caracterización de Nodos Frontera

### Tipos de Nodos Frontera

#### 2.1 Puentes hacia Neutrales
**Definición**: Nodos del cluster que mantienen conexiones con medios neutrales.

**Relevancia**: 
- Potenciales canales de moderación
- Puntos de entrada para información balanceada
- Indicadores de apertura al diálogo

#### 2.2 Puentes hacia el Cluster Opuesto  
**Definición**: Nodos que mantienen conexiones (especialmente positivas) con el cluster rival.

**Tipos de Conexiones**:
- **Positivas**: Colaboración, acuerdo puntual, moderación
- **Negativas**: Conflicto directo, crítica, confrontación

---

## 3. Análisis de Nodos Neutrales

### 3.1 Neutrales como Mediadores
**Métrica**: Fracción de neutrales conectados simultáneamente a ambos clusters.

$$M_{neutrales} = \frac{|\{n \in N : deg(n, C_1) > 0 \land deg(n, C_2) > 0\}|}{|N|}$$

### 3.2 Balance de Conexiones Neutrales
Para cada nodo neutral $n$, calculamos su **índice de balance**:

$$Balance(n) = \frac{|deg(n, C_1) - deg(n, C_2)|}{deg(n, C_1) + deg(n, C_2)}$$

- $Balance(n) = 0$: Perfectamente equilibrado
- $Balance(n) = 1$: Completamente sesgado hacia un cluster

---

## 4. Métricas de Análisis Avanzado

### 4.1 Asimetría de Permeabilidad
$$A_{perm} = |P(C_1) - P(C_2)|$$

Mide diferencias en la exposición entre clusters.

### 4.2 Índice de Polarización de Frontera
$$IPF = \frac{Links_{negativos\_entre\_clusters}}{Links_{totales\_entre\_clusters}}$$

Cuantifica el grado de antagonismo en las fronteras.

### 4.3 Coeficiente de Mediación Neutral
$$CMN = \frac{Links_{neutrales\_bidireccionales}}{Links_{neutrales\_totales}}$$

Mide la capacidad mediadora del conjunto neutral.

---

## 5. Interpretación Estratégica

### Escenarios por Permeabilidad

#### Escenario A: Ambos Clusters Baja Permeabilidad
- **Situación**: Polarización extrema con cámaras de eco
- **Dinámicas**: Refuerzo de sesgos, radicalización
- **Rol de neutrales**: Crítico para mantener diálogo

#### Escenario B: Permeabilidad Asimétrica
- **Situación**: Un cluster más abierto que otro
- **Dinámicas**: Posible ventaja estratégica del cluster más permeable
- **Análisis**: ¿Apertura = vulnerabilidad o fortaleza?

#### Escenario C: Alta Permeabilidad Mutua
- **Situación**: Fronteras porosas, alto intercambio
- **Dinámicas**: Mayor probabilidad de convergencia o moderación
- **Riesgos**: Posible inestabilidad de coaliciones

### Indicadores de Alerta

1. **Fragmentación Interna**: Links negativos dentro del cluster
2. **Alianzas Emergentes**: Links positivos entre clusters  
3. **Captura de Neutrales**: Sesgo extremo en conexiones neutrales
4. **Aislamiento Progresivo**: Disminución de permeabilidad en el tiempo

---

## 6. Consideraciones Metodológicas

### Limitaciones del Análisis
- **Temporalidad**: Los links pueden cambiar de signo
- **Intensidad**: No todos los links tienen igual peso
- **Contexto**: Eventos externos pueden alterar dinámicas

### Validación de Resultados
- Comparar con eventos históricos conocidos
- Analizar consistencia temporal
- Contrastar con otras métricas de red

### Extensiones Futuras
- Análisis dinámico temporal
- Incorporación de pesos en links
- Análisis de contenido de las conexiones