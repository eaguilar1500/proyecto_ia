# Sistema difuso Mamdani para la eficiencia de combustible con reglas encontradas por un algoritmo genético

Proyecto universitario de Inteligencia Artificial. Dataset **Auto MPG** (UCI).

Flujo del proyecto:

```
398 registros → eliminar 6 sin horsepower → 392 registros (datasheet)
→ análisis estadístico → mín / Q1 / mediana / Q3 / máx → funciones de pertenencia
→ algoritmo genético → reglas encontradas → support / confidence / lift
→ sistema Mamdani (ctrl.Rule) → salida MPG (centroide)
```

Todos los números de este documento provienen de `python main.py` y coinciden
con las tablas de `resultados/tablas/`.

---

## Índice

1. [Problemática y objetivo](#1-problemática-y-objetivo)
2. [Dataset Auto MPG y limpieza](#2-dataset-auto-mpg-y-limpieza)
3. [Variables utilizadas](#3-variables-utilizadas)
4. [Análisis del datasheet](#4-análisis-del-datasheet)
5. [Rangos de fuzzificación: mín, Q1, mediana, Q3, máx](#5-rangos-de-fuzzificación-mín-q1-mediana-q3-máx)
6. [Funciones de pertenencia](#6-funciones-de-pertenencia)
7. [Por qué no se normaliza](#7-por-qué-no-se-normaliza)
8. [Lógica difusa multimodal y fuzzificación](#8-lógica-difusa-multimodal-y-fuzzificación)
9. [Sistema Mamdani](#9-sistema-mamdani)
10. [Algoritmo genético](#10-algoritmo-genético)
11. [Reglas encontradas](#11-reglas-encontradas)
12. [Support, confidence y lift](#12-support-confidence-y-lift)
13. [Ejemplo completo del sistema](#13-ejemplo-completo-del-sistema)
14. [Limitaciones](#14-limitaciones)
15. [Estructura del código y mapa para la defensa](#15-estructura-del-código-y-mapa-para-la-defensa)
16. [Referencias](#16-referencias)

---

## 1. Problemática y objetivo

El consumo de combustible de un automóvil depende de características técnicas
(potencia, peso, desempeño del motor) que se relacionan con él de forma **no
lineal** y que normalmente se describen con palabras imprecisas: "un auto
pesado y potente consume mucho". La lógica difusa permite trabajar con esos
conceptos graduales y expresar el conocimiento como reglas SI … ENTONCES ….

**Objetivo.**
1. Obtener los rangos fuzzy directamente del datasheet.
2. Construir un sistema difuso Mamdani.
3. Usar un algoritmo genético para **definir la base de reglas**: cada
   individuo es una posible base de reglas y el algoritmo busca **minimizar la
   heurística de error promedio penalizado** del sistema Mamdani completo.
4. Después, analizar individualmente las reglas encontradas con support,
   confidence y lift (estas métricas **no participan** en la búsqueda genética).
5. Mostrar cómo esas reglas producen una salida mediante fuzzificación,
   inferencia, agregación y defuzzificación por centroide.

## 2. Dataset Auto MPG y limpieza

**Auto MPG** — UCI Machine Learning Repository (procede de StatLib, Carnegie
Mellon University; usado por Quinlan, 1993). Archivo: `data/auto-mpg.data`;
descripción: `data/auto-mpg.names`.

**¿Por qué Auto MPG?** Son datos reales y documentados de 398 automóviles, con
variables técnicas numéricas y continuas (potencia, peso, aceleración) cuya
relación con el consumo es intuitiva pero no lineal, y con una salida numérica
continua (mpg), adecuada para un consecuente Mamdani que se defuzzifica a un
número. Solo hay 6 valores faltantes.

**Limpieza** ([src/carga_datos.py](src/carga_datos.py)):
1. Lectura con los nombres de columna de `auto-mpg.names` (archivo sin
   encabezado, separado por espacios).
2. Detección de los `?` en `horsepower`: **6 registros**.
3. Eliminación **solo** de esos 6 registros: **398 → 392**. No se imputa, para
   no inventar valores que después moverían los cuartiles.
4. `horsepower` se convierte a número (cambia el tipo, no los valores).

Los **392 automóviles completos** son el datasheet de todo el proyecto: de ellos
salen los rangos fuzzy, sobre ellos se evalúa el algoritmo genético y sobre ellos
se calculan support, confidence y lift.

## 3. Variables utilizadas

| Rol | Variable | Unidad | Etiquetas |
|---|---|---|---|
| Entrada | `horsepower` | hp | baja, media, alta |
| Entrada | `weight` | lb | ligero, medio, pesado |
| Entrada | `acceleration` | s (0 a 60 mph) | baja, media, alta |
| Salida | `mpg` | millas por galón | eficiencia_baja, eficiencia_media, eficiencia_alta |

> `acceleration` son **segundos de 0 a 60 mph**: "aceleración baja" = pocos
> segundos = auto **rápido**; "aceleración alta" = auto **lento**.

**¿Por qué horsepower, weight y acceleration?**
([correlaciones.csv](resultados/tablas/correlaciones.csv))
- Son continuas y físicas: admiten etiquetas naturales ("potencia alta",
  "auto pesado").
- `weight` (r = −0.83) y `horsepower` (r = −0.78) son las más relacionadas con
  `mpg`; `acceleration` (r = +0.42) aporta información distinta que puede ser
  útil combinada en una regla.
- 3 entradas × 3 etiquetas = **27 reglas posibles**, un número que el algoritmo
  genético puede explorar y que se puede explicar regla por regla.

**Variables descartadas:** `displacement` (r = 0.93 con `weight`, redundante),
`cylinders` (discreta y redundante: r = 0.90 con `weight`), `model year`
(indica la época, no una característica física; r = 0.58 con `mpg`), `origin`
(categórica, sin orden) y `car name` (identificador).

## 4. Análisis del datasheet

Implementado en [src/exploracion.py](src/exploracion.py). Tabla:
[estadisticas_descriptivas.csv](resultados/tablas/estadisticas_descriptivas.csv).
Figura: [distribuciones.png](resultados/figuras/distribuciones.png).

| Variable | Mín | Q1 | Mediana | Media | Q3 | Máx | Desv. est. | Asimetría | Atípicos (1.5·IQR) |
|---|---|---|---|---|---|---|---|---|---|
| `horsepower` | 46 | 75 | 93.5 | 104.47 | 126 | 230 | 38.49 | 1.09 | 10 |
| `weight` | 1613 | 2225.25 | 2803.5 | 2977.58 | 3614.75 | 5140 | 849.40 | 0.52 | 0 |
| `acceleration` | 8.0 | 13.775 | 15.5 | 15.54 | 17.025 | 24.8 | 2.76 | 0.29 | 11 |
| `mpg` | 9.0 | 17 | 22.75 | 23.45 | 29 | 46.6 | 7.81 | 0.46 | 0 |

- `horsepower` y `weight` son **asimétricas a la derecha**: se concentran en
  valores bajos y tienen colas largas (hasta 230 hp y 5 140 lb). `horsepower`
  además es bimodal (un grupo principal en 70–110 hp y otro en torno a 150 hp).
- `acceleration` es casi simétrica, con atípicos en ambos extremos.
- La relación de `horsepower` y `weight` con `mpg` es monótona pero no lineal.

Estas características justifican los cuartiles y la forma de las funciones
(secciones 5 y 6).

## 5. Rangos de fuzzificación: mín, Q1, mediana, Q3, máx

**¿De dónde salieron los rangos?** De los 392 registros, en
`puntos_referencia` ([src/variables_difusas.py](src/variables_difusas.py),
**PASO 1**). Tabla para la defensa:
[estadisticas_rangos.csv](resultados/tablas/estadisticas_rangos.csv)
(`variable | mínimo | Q1 | mediana | Q3 | máximo | conjunto | función | puntos`).

| Variable | Mín | Q1 | Mediana | Q3 | Máx | Bajo: trapmf | Medio: trimf | Alto: trapmf |
|---|---|---|---|---|---|---|---|---|
| `horsepower` | 46 | 75 | 93.5 | 126 | 230 | baja [46, 46, 75, 93.5] | media [75, 93.5, 126] | alta [93.5, 126, 230, 230] |
| `weight` | 1613 | 2225.25 | 2803.5 | 3614.75 | 5140 | ligero [1613, 1613, 2225.25, 2803.5] | medio [2225.25, 2803.5, 3614.75] | pesado [2803.5, 3614.75, 5140, 5140] |
| `acceleration` | 8 | 13.775 | 15.5 | 17.025 | 24.8 | baja [8, 8, 13.775, 15.5] | media [13.775, 15.5, 17.025] | alta [15.5, 17.025, 24.8, 24.8] |
| `mpg` | 9 | 17 | 22.75 | 29 | 46.6 | eficiencia_baja [9, 9, 17, 22.75] | eficiencia_media [17, 22.75, 29] | eficiencia_alta [22.75, 29, 46.6, 46.6] |

Criterio (igual en las cuatro variables):

```
BAJO  = trapmf([mínimo, mínimo, Q1, mediana])
MEDIO = trimf([Q1, mediana, Q3])
ALTO  = trapmf([mediana, Q3, máximo, máximo])
```

**Ningún valor fue elegido arbitrariamente**: cada punto es un estadístico de la
misma fila de la tabla.

**¿Por qué cuartiles?** Porque dividen los datos en cuartos y se adaptan a la
distribución real. Con intervalos iguales entre mínimo y máximo, en
`horsepower` y `weight` (concentradas en valores bajos) los conjuntos altos
quedarían casi vacíos. Con cuartiles:
- el 25 % más bajo de los autos (mín → Q1) es totalmente "bajo" (μ = 1);
- el 25 % más alto (Q3 → máx) es totalmente "alto";
- el 50 % central (Q1 → Q3) es la zona de transición;
- la mediana es el valor más típico de "medio".

La forma también sigue la asimetría: en `horsepower` la mediana está a 18.5 hp
de Q1 y a 32.5 hp de Q3, así que el triángulo "media" es asimétrico, igual que
los datos.

## 6. Funciones de pertenencia

Creadas en [src/variables_difusas.py](src/variables_difusas.py) (**PASO 2** y
**PASO 3**) con `fuzz.trapmf` / `fuzz.trimf`, como `ctrl.Antecedent` (entradas)
y `ctrl.Consequent` (salida). Figura:
[funciones_pertenencia.png](resultados/figuras/funciones_pertenencia.png).

**¿Por qué trapmf en los extremos?** Porque representan conceptos que **se
saturan**: 230 hp no es "más alto" que 150 hp en sentido lingüístico, ambos son
"potencia alta". La meseta (μ = 1) cubre además las colas largas y los atípicos
sin parámetros extra.

**¿Por qué trimf en el centro?** Porque "medio" tiene **un valor central
representativo**, la mediana (μ = 1), y la pertenencia baja al alejarse de él.
Solo necesita tres puntos, y los tres salen del dataset.

**¿Por qué no gaussianas?** La forma del histograma describe la frecuencia de los
datos, no el significado del concepto. Una gaussiana nunca llega a 0 y
necesitaría un σ: con la desviación de `acceleration` (2.76 s), la campana
"media" tendría μ > 0.1 desde ≈ 9.6 hasta ≈ 21.4 s, solapándose con los
extremos, y cualquier otro σ sería un número elegido a mano.

**Universo de discurso** = [mínimo, máximo] de cada variable, con paso igual a
la resolución de los datos (1 para `horsepower` y `weight`, 0.1 para
`acceleration` y `mpg`). En cada punto del universo los tres grados **suman 1**
(se verifica en `main.py`): ningún valor queda sin cubrir.

## 7. Por qué no se normaliza

**No se normaliza ni se estandariza.** Cada variable conserva sus **unidades
originales** y tiene su propio universo de discurso. La función de pertenencia
produce un grado **μ(x) entre 0 y 1**.

**Normalización y fuzzificación NO son lo mismo:**
- la normalización transforma el **valor** a otra escala (sigue siendo un valor
  de la variable);
- la fuzzificación no cambia el valor: calcula **cuánto pertenece** a cada
  conjunto (2575 lb → μ_ligero = 0.395, μ_medio = 0.605).

Como cada variable tiene su propio universo, la diferencia de escalas (5 140 lb
frente a 24.8 s) no interviene en el sistema, y las reglas se leen en unidades
reales ("pesado a partir de 2803.5 lb").

## 8. Lógica difusa multimodal y fuzzificación

### 8.1 Conjunto difuso y μ(x)

En un conjunto clásico un valor pertenece o no ("potencia alta = más de
126 hp"). En un **conjunto difuso** pertenece **en cierto grado**. **μ(x)** es el
**grado de pertenencia** del valor x al conjunto: μ = 1 pertenece totalmente,
μ = 0 no pertenece, 0 < μ < 1 pertenece parcialmente. No es una probabilidad:
mide cuánto encaja un valor conocido en un concepto impreciso.

### 8.2 Lógica difusa multimodal (terminología de clase)

En este curso, **"lógica difusa multimodal"** se usa para referirse a un sistema
difuso con **varias variables de entrada**; en ese sentido se usa aquí: tres
entradas (`horsepower`, `weight`, `acceleration`) y una salida (`mpg`).
Técnicamente también se describe como sistema **multivariable** o **MISO**
(*Multiple Input, Single Output*).

### 8.3 Fuzzificación: cómo se calcula un grado de pertenencia

**Fuzzificación** = convertir un valor numérico de entrada en sus grados de
pertenencia a cada conjunto. Las **entradas** (`horsepower`, `weight`,
`acceleration`) se fuzzifican. Código: `grado_pertenencia` y `fuzzificar` en
[src/proceso_mamdani.py](src/proceso_mamdani.py) (encabezados **CÁLCULO DE
GRADO DE PERTENENCIA** y **PASO 1 - FUZZIFICACIÓN**).

| Función | Tramo | μ(x) |
|---|---|---|
| trapmf [a, b, c, d] | b ≤ x ≤ c | 1 (meseta) |
| | a < x < b | (x − a) / (b − a) |
| | c < x < d | (d − x) / (d − c) |
| trimf [a, b, c] | x = b | 1 (pico) |
| | a < x < b | (x − a) / (b − a) |
| | b < x < c | (c − x) / (c − b) |

Por ejemplo, si x está entre la mediana y Q3:
μ_medio(x) = (Q3 − x) / (Q3 − mediana) y μ_alto(x) = (x − mediana) / (Q3 − mediana).

**Ejemplo** (*pontiac j2000 se hatchback*; tabla
[ejemplo_fuzzificacion.csv](resultados/tablas/ejemplo_fuzzificacion.csv),
figura [ejemplo_fuzzificacion.png](resultados/figuras/ejemplo_fuzzificacion.png)):

| Variable | Bajo | Medio | Alto |
|---|---|---|---|
| horsepower = 85 | (93.5 − 85)/(93.5 − 75) = **0.459** | (85 − 75)/(93.5 − 75) = **0.541** | 0 |
| weight = 2575 | (2803.5 − 2575)/(2803.5 − 2225.25) = **0.395** | (2575 − 2225.25)/(2803.5 − 2225.25) = **0.605** | 0 |
| acceleration = 16.2 | 0 | (17.025 − 16.2)/(17.025 − 15.5) = **0.541** | (16.2 − 15.5)/(17.025 − 15.5) = **0.459** |

La tabla CSV incluye una columna que confirma que las fórmulas manuales dan el
mismo valor que skfuzzy (`fuzz.interp_membership`).

**Criterio de elección del auto** (no depende de las reglas): el auto cuyos tres
valores caen más "entre dos conjuntos" (mayor suma del segundo grado de
pertenencia en las tres entradas), para que el ejemplo muestre grados parciales
en las tres variables.

## 9. Sistema Mamdani

Código: [src/proceso_mamdani.py](src/proceso_mamdani.py) (PASO 1 a PASO 5) y
[src/sistema_mamdani.py](src/sistema_mamdani.py).

### 9.1 Cómo funciona una regla `ctrl.Rule`

```python
regla = ctrl.Rule(
    potencia['baja']
    & peso['ligero']
    & aceleracion['media'],
    eficiencia['eficiencia_alta']
)
```

Significa: SI potencia es baja **Y** peso es ligero **Y** aceleración es media
ENTONCES eficiencia es alta. El operador `&` es el **AND difuso**. Las reglas se
reúnen en `ctrl.ControlSystem(reglas)` y se ejecutan con
`ctrl.ControlSystemSimulation`.

### 9.2 Inferencia: AND mediante mínimo y activación alpha

La **activación alpha** de una regla es cuánto se cumple su antecedente:

$$\alpha = \min(\mu_{potencia}(x_1),\ \mu_{peso}(x_2),\ \mu_{aceleración}(x_3))$$

Se usa el **mínimo** porque una conjunción solo es tan cierta como su condición
**menos** cierta; es el AND del modelo Mamdani y coincide con el AND clásico
cuando los grados son 0 o 1. Si algún antecedente vale 0, la regla no se activa.

### 9.3 Qué ocurre con el consecuente: recorte y función de masa

El consecuente de la regla se **recorta** a la altura alpha (implicación de
Mamdani):

$$\mu'_C(y) = \min(\alpha,\ \mu_C(y))$$

La regla no produce un número sino un **conjunto difuso de salida**: "la
eficiencia es media, con fuerza alpha". **Siguiendo la terminología utilizada en
clase, este consecuente recortado se relaciona con la denominada función de
masa.** Si varias reglas proponen el mismo consecuente, cuenta la de mayor
alpha.

### 9.4 Agregación mediante máximo

Los consecuentes recortados de todas las reglas activas se unen en un solo
conjunto con el **máximo** punto a punto:

$$\mu_{agr}(y) = \max_r \mu'_r(y)$$

Se usa el máximo porque las conclusiones de las reglas son **alternativas** (una
unión, "o"): cada valor de mpg conserva el apoyo de la regla que más lo
respalda. Las reglas del proyecto no usan OR en sus antecedentes; el máximo
aparece en la agregación.

### 9.5 Defuzzificación por centroide

**Defuzzificación** = convertir el conjunto difuso agregado en un valor
numérico. **¿Por qué mpg?** Porque es la **salida** del sistema: las entradas se
fuzzifican al principio y la salida se defuzzifica al final para obtener un MPG
numérico.

$$y^* = \frac{\int y\,\mu_{agr}(y)\,dy}{\int \mu_{agr}(y)\,dy}$$

**¿Por qué centroide?** Es el método habitual de Mamdani (y el de skfuzzy): usa
**toda la forma** del conjunto agregado, no solo su punto más alto, así que
todas las reglas activas influyen en el resultado según su fuerza. Código:
encabezado **PASO 5 - DEFUZZIFICACIÓN POR CENTROIDE** en
[src/proceso_mamdani.py](src/proceso_mamdani.py) y `centroide_lineal` en
[src/sistema_mamdani.py](src/sistema_mamdani.py).

**Rango de salida.** El centroide nunca sale del intervalo entre el centroide del
conjunto más bajo y el del más alto: el sistema produce valores entre
**14.56 y 36.16 mpg** (eficiencia_media: 22.92).

## 10. Algoritmo genético

Código: [src/algoritmo_genetico.py](src/algoritmo_genetico.py), escrito y
comentado en **15 fases** (`# FASE 1 - …` a `# FASE 15 - …`). Es el
procedimiento sencillo visto en clase.

**Papel del algoritmo:** **DEFINE la base de reglas fuzzy**. Cada individuo es
una posible base de reglas; cada gen decide si existe una regla y cuál es su
consecuente. El algoritmo busca **minimizar la heurística de error promedio
penalizado**, que evalúa globalmente el sistema Mamdani completo. No modifica
las funciones de pertenencia ni el sistema Mamdani, y **support, confidence y
lift no participan en la búsqueda** (se calculan después, sección 12).

### 10.1 Representación del cromosoma

- **Individuo** = una posible base de reglas = un **cromosoma de 27 genes**.
- **Gen** = uno de los antecedentes posibles. **¿Por qué 27?** 3 niveles de
  potencia × 3 de peso × 3 de aceleración = 27 combinaciones, en orden fijo:
  G0 = (baja, ligero, baja), G1 = (baja, ligero, media), G2 = (baja, ligero,
  alta), …, G26 = (alta, pesado, alta).
- **Valor del gen:**

| Valor | Significado |
|---|---|
| 0 | no existe regla |
| 1 | eficiencia baja |
| 2 | eficiencia media |
| 3 | eficiencia alta |

Ejemplo: `[3, 0, 2, ...]` → G0: existe regla con consecuente eficiencia alta;
G1: no existe regla; G2: existe regla con consecuente eficiencia media. Cada gen
se convierte directamente en un `ctrl.Rule` (encabezado **CONVERSIÓN DEL
CROMOSOMA DEL ALGORITMO GENÉTICO A ctrl.Rule** en
[src/sistema_mamdani.py](src/sistema_mamdani.py)).

### 10.2 Heurística, aptitud y probabilidad

> **El objetivo del algoritmo genético es minimizar la heurística de error
> promedio penalizado.**
> **Menor heurística = mejor individuo.**

**¿Qué es la heurística?** La **heurística de error promedio penalizado** de un
cromosoma: se decodifica a reglas fuzzy, se ejecuta el sistema sobre los 392
autos y se compara el MPG real del dataset con el MPG obtenido por Mamdani.

```
error_i = |mpg_real − mpg_estimado|        si alguna regla produce salida
error_i = rango_mpg = max(mpg) − min(mpg)  si ninguna regla produce salida (37.6)
heurística = promedio de los 392 error_i          (menor = mejor)
```

Para cada auto **con salida**: error = |MPG real − MPG obtenido|.
Para cada auto **sin salida**: error = **37.6**, que es el rango de MPG del
dataset: max(MPG) − min(MPG) = 46.6 − 9.0 = 37.6.
Después: heurística = promedio de los errores de los **392 autos**.

Matemáticamente es un Error Absoluto Medio (MAE) con penalización para los autos
en los que ninguna regla produce salida. La penalización sale del propio dataset
y siempre es mayor que cualquier error posible de un auto con salida.

**¿Por qué puede haber autos sin salida?** El cromosoma puede contener genes 0, y
un gen 0 significa que ese antecedente **no tiene regla**. Si un automóvil solo
activa antecedentes sin regla, ninguna regla produce salida para él. Esos casos
**no se ignoran**: reciben el error de penalización 37.6. Esto evita que un
individuo parezca artificialmente bueno simplemente por no producir salida para
los casos difíciles.

- **¿Cómo se convierte en aptitud?** aptitud_i = 1 / (1 + heurística_i).
- **¿Cómo se calcula la probabilidad de selección?**
  probabilidad_i = aptitud_i / Σ aptitudes.
- Menor heurística → mayor aptitud → mayor probabilidad de ser elegido como padre.

La **misma heurística** se usa para calcular las probabilidades, comparar con el
mejor histórico, registrar la evolución y reportar la solución final.

| | Qué evalúa | Cuándo se usa |
|---|---|---|
| **Heurística** | El **cromosoma completo** (toda la base de reglas dentro del sistema Mamdani). | **Guía** al algoritmo genético durante la búsqueda. |
| **Support / confidence / lift** | **Cada regla por separado**. | **Después** de que el algoritmo terminó, solo como análisis (sección 12). |

### 10.3 El ciclo del algoritmo

1. **Crear la población inicial**: 40 cromosomas aleatorios de 27 genes en
   {0, 1, 2, 3}.
2. **Evaluar la heurística** de cada individuo.
3. **Convertir la heurística en aptitud**.
4. **Calcular la probabilidad de selección**.
5. **Seleccionar dos padres por ruleta** (dos individuos distintos, según sus
   probabilidades).
6. **Elegir un punto de cruce** k aleatorio entre 1 y 26.
7. **Generar dos hijos complementarios**:
   Hijo 1 = genes antes de k del Padre 1 + genes desde k del Padre 2;
   Hijo 2 = genes antes de k del Padre 2 + genes desde k del Padre 1.
8. **Mutación**: cada hijo tiene **10 % de probabilidad** de mutar (parámetro
   configurado); si muta, se cambia **un único gen** elegido al azar por otro
   valor distinto de {0, 1, 2, 3} (por ejemplo, gen 14: 1 → 3).
9. **Añadir los dos hijos**: 40 → 42 individuos.
10. **Eliminar aleatoriamente dos individuos entre todos los existentes**:
    42 → 40. No se protege a nadie (sin elitismo): pueden eliminarse individuos
    antiguos, los padres, el mejor de la población o los hijos (también un hijo
    que acaba de mutar). **Los hijos recién generados también pueden ser
    eliminados inmediatamente porque el descarte es completamente aleatorio.**
    No existe elitismo dentro de la población; lo único que se conserva aparte es
    la copia del mejor individuo histórico.
11. **Reevaluar la población** (heurística, aptitud y probabilidades).
12. **Actualizar el mejor individuo histórico** si aparece uno con menor
    heurística. **El mejor individuo histórico se guarda fuera de la población
    para no perder la mejor solución encontrada** (cromosoma, heurística y
    generación en que apareció); no participa en el reemplazo.
13. **Actualizar el contador de generaciones sin mejora**: vuelve a 0 si hubo
    mejora; si no, aumenta en 1.
14. **Repetir.** El algoritmo termina después de **100 generaciones consecutivas
    sin encontrar una solución mejor** o al alcanzar **500 generaciones** como
    límite de seguridad.
15. **Usar el mejor individuo histórico** (no necesariamente el mejor de la
    población final) como base de reglas.

**Una generación** = seleccionar dos padres → dos hijos → posible mutación →
añadirlos → eliminar dos → reevaluar. Cada generación crea exactamente 2 hijos.

### 10.4 Ejemplo de una generación (generación 1)

Tablas: [ejemplo_seleccion_genetica.csv](resultados/tablas/ejemplo_seleccion_genetica.csv)
(`individuo | heurística | aptitud | probabilidad | seleccionado_como_padre`, los
40 individuos) y [ejemplo_cruzamiento.csv](resultados/tablas/ejemplo_cruzamiento.csv).

| Individuo | Heurística | Aptitud | Probabilidad | Papel |
|---|---|---|---|---|
| 23 | 9.2726 | 0.0973 | 0.0306 | **Padre 2** |
| 39 | 11.1426 | 0.0824 | 0.0259 | **Padre 1** |
| 21 | 12.1689 | 0.0759 | 0.0239 | eliminado en la fase 10 |
| 34 | 10.4365 | 0.0874 | 0.0275 | eliminado en la fase 10 |

Punto de cruce **k = 6**:

| Cromosoma | Genes |
|---|---|
| Padre 1 | `3 0 3 1 3 1 │ 1 3 2 0 2 3 1 3 2 1 0 2 0 3 2 1 0 2 0 1 3` |
| Padre 2 | `1 0 1 2 0 1 │ 1 2 1 2 1 1 1 0 2 0 1 3 3 2 3 3 1 3 1 1 2` |
| Hijo 1 | `3 0 3 1 3 1 │ 1 2 1 2 1 1 1 0 2 0 1 3 3 2 3 3 1 3 1 1 2` |
| Hijo 2 | `1 0 1 2 0 1 │ 1 3 2 0 2 3 1 3 2 1 0 2 0 3 2 1 0 2 0 1 3` |

Ningún hijo mutó en esta generación.

### 10.5 Resultado de la ejecución

Figura: [evolucion_heuristica.png](resultados/figuras/evolucion_heuristica.png).
Tabla: [evolucion_genetico.csv](resultados/tablas/evolucion_genetico.csv)
(`generacion | mejor_heuristica_poblacion | mejor_heuristica_historica |
heuristica_promedio | generaciones_sin_mejora | numero_reglas_mejor_historico`).

| | Valor |
|---|---|
| Generaciones ejecutadas | 178 |
| Motivo de parada | 100 generaciones sin mejora |
| Mejor heurística de la población inicial | 5.085 |
| Mejor heurística histórica final | **4.380** |
| Generación en que apareció la mejor solución | 78 |
| Mejoras del mejor histórico | generación 76 (4.551) y generación 78 (4.380) |
| Autos en los que ninguna regla produce salida (mejor solución) | 5 de 392 |
| Heurística promedio de la población | 12.24 (gen. 0) → 6.46 (gen. 178) |

**Evolución de esta ejecución (hitos):**

```
generación 0    mejor heurística de la población inicial     5.085
generación 76   nueva mejor solución encontrada               4.551
generación 78   nueva mejor solución encontrada               4.380
generaciones 79–178   100 generaciones consecutivas sin mejora
generación 178  parada
```

El algoritmo comienza con reglas aleatorias y, generación tras generación,
intenta reducir la heurística. En esta ejecución **sí consiguió mejorar la
solución inicial** (5.085 → 4.380), y el promedio de la población bajó de 12.24 a
6.46. Como el descarte es aleatorio, la mejor solución no siempre permanece en
la población (al terminar, el mejor de la población tiene 5.070), pero la copia
externa del mejor histórico la conserva.

**Mutación: parámetro frente a lo observado.**
- Parámetro configurado: **10 %** de probabilidad de mutar por hijo.
- Observado en esta ejecución: 31 de los 356 hijos mutaron (8.7 %).
- Observado en la verificación automática (20 000 hijos simulados): 10.2 %.

Las diferencias respecto de 10 % se deben al azar; la configuración es
exactamente 10 %.

**Descarte en esta ejecución.** Hubo **178 operaciones de descarte** (una por
generación) y en cada una se eliminaron **2 individuos**: 178 × 2 = **356
eliminaciones individuales**. De ellas, 22 fueron hijos eliminados en la misma
generación en que se crearon (2 de esos hijos habían mutado).

**Detalle técnico:** el código fija el valor inicial del generador de números
aleatorios solo para que la ejecución sea reproducible; no es parte del método.

**Optimización computacional interna.** Evaluar un cromosoma con
`ctrl.ControlSystemSimulation` sobre 392 autos tarda ≈ 3–4 s. Para que el
algoritmo genético no tarde demasiado, la heurística se calcula con
`EvaluadorNumpy` ([src/sistema_mamdani.py](src/sistema_mamdani.py)), que
reproduce el mismo proceso Mamdani. `main.py` comprueba en cada ejecución que da
el mismo resultado que skfuzzy (mismas salidas y misma heurística). Con la caché
de skfuzzy activada se observaron resultados dependientes del orden al crear
muchos sistemas, por eso se usa `ControlSystemSimulation(..., cache=False)`.

## 11. Reglas encontradas

**Mejor base de reglas encontrada durante la ejecución** (el mejor individuo
histórico). El algoritmo genético es estocástico y no demuestra optimalidad
global: es la mejor solución encontrada, no la mejor posible.

Cromosoma: `[3, 2, 3, 3, 1, 0, 1, 0, 0, 0, 3, 1, 2, 1, 2, 1, 1, 2, 2, 0, 3, 1, 2, 3, 1, 2, 1]`

**22 reglas de 27.** Sin regla (gen = 0): G5, G7, G8, G9, G19.
Tabla: [reglas_genetico.csv](resultados/tablas/reglas_genetico.csv); texto:
[reglas_finales.txt](resultados/reglas_finales.txt).

| Regla | Gen | horsepower | weight | acceleration | Consecuente |
|---|---|---|---|---|---|
| R1 | G0 | baja | ligero | baja | eficiencia_alta |
| R2 | G1 | baja | ligero | media | eficiencia_media |
| R3 | G2 | baja | ligero | alta | eficiencia_alta |
| R4 | G3 | baja | medio | baja | eficiencia_alta |
| R5 | G4 | baja | medio | media | eficiencia_baja |
| R6 | G6 | baja | pesado | baja | eficiencia_baja |
| R7 | G10 | media | ligero | media | eficiencia_alta |
| R8 | G11 | media | ligero | alta | eficiencia_baja |
| R9 | G12 | media | medio | baja | eficiencia_media |
| R10 | G13 | media | medio | media | eficiencia_baja |
| R11 | G14 | media | medio | alta | eficiencia_media |
| R12 | G15 | media | pesado | baja | eficiencia_baja |
| R13 | G16 | media | pesado | media | eficiencia_baja |
| R14 | G17 | media | pesado | alta | eficiencia_media |
| R15 | G18 | alta | ligero | baja | eficiencia_media |
| R16 | G20 | alta | ligero | alta | eficiencia_alta |
| R17 | G21 | alta | medio | baja | eficiencia_baja |
| R18 | G22 | alta | medio | media | eficiencia_media |
| R19 | G23 | alta | medio | alta | eficiencia_alta |
| R20 | G24 | alta | pesado | baja | eficiencia_baja |
| R21 | G25 | alta | pesado | media | eficiencia_media |
| R22 | G26 | alta | pesado | alta | eficiencia_baja |

En `main.py` (**PASO 8**) cada regla se construye así:
`ctrl.Rule(potencia[p] & peso[w] & aceleracion[a], eficiencia[consecuente])`;
por ejemplo, R3 = `ctrl.Rule(potencia['baja'] & peso['ligero'] & aceleracion['alta'], eficiencia['eficiencia_alta'])`.

La heurística no penaliza el número de reglas, así que el algoritmo no busca la
base más pequeña.

**Autos sin salida.** Esta base tiene 5 genes con valor 0 (G5, G7, G8, G9, G19):
esos antecedentes no tienen regla. En la mejor solución encontrada, **5 de 392
autos** solo activan antecedentes sin regla, así que ninguna regla produce
salida para ellos. No es un error de ejecución: es una **limitación de la base de
reglas encontrada**. Esos 5 autos no se ignoran; cada uno cuenta con el error de
penalización 37.6 dentro de la heurística.

## 12. Support, confidence y lift

Se calculan **después** de terminar el algoritmo genético, sobre las reglas del
mejor individuo histórico y los **392 autos**:

```
algoritmo genético → mejor cromosoma histórico → reglas finales → support / confidence / lift
```

**No se usan** para crear reglas, filtrarlas antes del algoritmo, cambiar el
cromosoma, eliminar reglas ni modificar consecuentes. Responden a: **¿qué
respaldo tiene individualmente cada regla encontrada por el genético?**

Código: [src/metricas_reglas.py](src/metricas_reglas.py). Tabla:
[metricas_reglas.csv](resultados/tablas/metricas_reglas.csv).

### 12.1 Definiciones

Para una regla A → C y cada automóvil *i*:
- **alpha_i** = min(μ_horsepower, μ_weight, μ_acceleration): cuánto cumple el
  auto el antecedente;
- **beta_i** = μ_consecuente(mpg_real_i): cuánto pertenece su mpg real al
  consecuente.

| Métrica | Fórmula | Qué mide |
|---|---|---|
| **Support(A)** | (1/N) Σ alpha_i | Qué tan frecuente es el antecedente en los datos. |
| **Support(A→C)** | (1/N) Σ min(alpha_i, beta_i) | Qué tan frecuente es que antecedente y consecuente se cumplan juntos (**support** de la regla). |
| **Confidence(A→C)** | Support(A→C) / Support(A) | Entre los autos que cumplen A, cuánto cumplen también C. |
| Support(C) | (1/N) Σ beta_i | Frecuencia general del consecuente. |
| **Lift(A→C)** | Confidence / Support(C) | Cuánto más aparece C cuando se cumple A que en general. |

Las divisiones entre cero se devuelven como "sin valor" (NaN).

**Interpretación del lift:**
- **Lift > 1** → asociación positiva entre antecedente y consecuente. Un lift
  mayor que 1 indica asociación positiva, pero debe interpretarse junto con
  support, confidence y cantidad de autos que activan la regla.
- **Lift ≈ 1** → el antecedente aporta poca información adicional sobre el
  consecuente.
- **Lift < 1** → asociación menor a la esperada.

Una regla con lift alto pero **soporte muy bajo** tiene poca evidencia y no debe
calificarse automáticamente como útil: hay que mirar también confidence y número
de autos que la activan. Como cada regla tiene tres condiciones, cada
antecedente cubre una parte pequeña de los datos: el Support(A) más alto de las
reglas encontradas es 0.217.

### 12.2 Resultados

| Regla | Antecedente (potencia, peso, aceleración) | Consecuente | Support(A) | Support(regla) | Confidence | Lift | Autos activados |
|---|---|---|---|---|---|---|---|
| R1 | baja, ligero, baja | eficiencia_alta | 0.0576 | 0.0544 | 0.943 | 2.54 | 53 |
| R2 | baja, ligero, media | eficiencia_media | 0.0925 | 0.0236 | 0.256 | 1.06 | 99 |
| R3 | baja, ligero, alta | eficiencia_alta | 0.1663 | 0.1403 | 0.844 | 2.27 | 100 |
| R4 | baja, medio, baja | eficiencia_alta | 0.0086 | 0.0070 | 0.816 | 2.20 | 20 |
| R5 | baja, medio, media | eficiencia_baja | 0.0243 | 0.0064 | 0.263 | **0.68** | 48 |
| R6 | baja, pesado, baja | eficiencia_baja | 0.0008 | 0.0008 | 1.000 | 2.59 | 1 |
| R7 | media, ligero, media | eficiencia_alta | 0.0565 | 0.0396 | 0.700 | 1.88 | 72 |
| R8 | media, ligero, alta | eficiencia_baja | 0.0331 | 0.0062 | 0.188 | **0.49** | 39 |
| R9 | media, medio, baja | eficiencia_media | 0.0545 | 0.0385 | 0.707 | 2.93 | 60 |
| R10 | media, medio, media | eficiencia_baja | 0.0920 | 0.0384 | 0.417 | 1.08 | 101 |
| R11 | media, medio, alta | eficiencia_media | 0.0784 | 0.0574 | 0.733 | 3.03 | 84 |
| R12 | media, pesado, baja | eficiencia_baja | 0.0094 | 0.0074 | 0.788 | 2.04 | 20 |
| R13 | media, pesado, media | eficiencia_baja | 0.0355 | 0.0304 | 0.858 | 2.22 | 52 |
| R14 | media, pesado, alta | eficiencia_media | 0.0646 | 0.0352 | 0.545 | 2.26 | 65 |
| R15 | alta, ligero, baja | eficiencia_media | 0.0108 | 0.0089 | 0.828 | 3.43 | 26 |
| R16 | alta, ligero, alta | eficiencia_alta | 0.0007 | 0.0006 | 0.792 | 2.13 | 5 |
| R17 | alta, medio, baja | eficiencia_baja | 0.0438 | 0.0222 | 0.506 | 1.31 | 58 |
| R18 | alta, medio, media | eficiencia_media | 0.0297 | 0.0244 | 0.823 | 3.41 | 61 |
| R19 | alta, medio, alta | eficiencia_alta | 0.0127 | 0.0016 | 0.125 | **0.34** | 30 |
| R20 | alta, pesado, baja | eficiencia_baja | 0.2172 | 0.2111 | 0.972 | 2.51 | 112 |
| R21 | alta, pesado, media | eficiencia_media | 0.0523 | 0.0177 | 0.338 | 1.40 | 65 |
| R22 | alta, pesado, alta | eficiencia_baja | 0.0328 | 0.0282 | 0.859 | 2.22 | 42 |

### 12.3 Análisis de utilidad de las reglas

- **Asociación positiva (lift > 1): 19 de 22 reglas**; 3 tienen lift < 1 (R5 0.68,
  R8 0.49, R19 0.34). Esto no significa que esas 19 reglas sean "buenas": el lift
  se interpreta junto con support, confidence y autos activados.
- Las reglas con **más evidencia** son R20 (potencia alta, pesado, aceleración
  baja → eficiencia baja: support 0.211, confidence 0.97, 112 autos), R3
  (potencia baja, ligero, aceleración alta → eficiencia alta: 0.140, 0.84,
  100 autos) y R11 (media, medio, alta → eficiencia media: 0.057, 0.73,
  84 autos).
- **Confianza alta con poca evidencia:** R6 tiene confidence 1.0 y lift 2.59,
  pero solo **un automóvil** la activa (support de la regla 0.0008); R16 la
  activan 5 autos (support 0.0006). Una confianza elevada no implica
  necesariamente mucha evidencia: una regla activada por muy pocos registros
  puede presentar confidence alta. No se usan umbrales para declarar una regla
  buena o mala; se muestran las cuatro medidas juntas.
- **Asociación contraria a los datos:** R19 (potencia alta, peso medio,
  aceleración alta → eficiencia **alta**) tiene confidence 0.13 y lift 0.34.

**¿Por qué el lift no reemplaza la heurística del algoritmo genético?** El
algoritmo **no optimiza el lift**: busca la base de reglas que minimiza la
heurística **global** (error del sistema Mamdani completo, donde las reglas se
combinan en la agregación y el centroide). Support, confidence y lift analizan
**cada regla por separado**. Una regla con lift individual bajo puede formar
parte de la mejor solución porque su efecto se combina con el de otras.
**Ninguna regla se modificó ni eliminó después de mirar su lift.**

### 12.4 Comparación de los consecuentes posibles

Tabla de análisis: [comparacion_consecuentes.csv](resultados/tablas/comparacion_consecuentes.csv)
(`antecedente | lift_baja | lift_media | lift_alta | consecuente_elegido_por_AG`,
para los 27 antecedentes). Responde a "¿el consecuente elegido por el algoritmo
tiene respaldo individual en los datos?". **No se usa para cambiar las reglas.**

- El consecuente elegido coincide con el de mayor lift en **10 de las 22
  reglas** (G0, G2, G6, G12, G14, G17, G18, G22, G24, G26). Es un dato de
  análisis, no un error: que una regla no tenga el mayor lift posible no implica
  que el algoritmo genético la haya elegido incorrectamente, porque la heurística
  evalúa el comportamiento conjunto de toda la base de reglas dentro del sistema
  Mamdani, mientras que el lift evalúa cada regla individualmente.

## 13. Ejemplo completo del sistema

Automóvil real: **pontiac j2000 se hatchback** (el mismo de la sección 8.3).
Este ejemplo es una **demostración**: no representa un conjunto de prueba, ni una
validación, ni la precisión general del sistema. Se eligió porque tiene grados de
pertenencia parciales en las tres entradas, lo que permite mostrar claramente
fuzzificación → activación de varias reglas → inferencia → agregación →
centroide → MPG final.

Código: `main.py` **PASO 8 a PASO 11** y [src/proceso_mamdani.py](src/proceso_mamdani.py).
Figura: [ejemplo_mamdani.png](resultados/figuras/ejemplo_mamdani.png).
Tablas: [ejemplo_mamdani_reglas.csv](resultados/tablas/ejemplo_mamdani_reglas.csv) y
[ejemplo_mamdani_resumen.csv](resultados/tablas/ejemplo_mamdani_resumen.csv).

1. **Valores numéricos de entrada:** 85 hp, 2575 lb, 16.2 s.
2. **Fuzzificación** y 3. **grados de pertenencia:** ver la tabla de la
   sección 8.3 (potencia baja 0.459 / media 0.541; peso ligero 0.395 / medio
   0.605; aceleración media 0.541 / alta 0.459).
4. **Reglas activadas:** 7 de 22 (las demás tienen algún antecedente con μ = 0).
5. **Alpha con mínimo:**

   | Regla | SI potencia, peso, aceleración | alpha | ENTONCES |
   |---|---|---|---|
   | R2 | baja, ligero, media | min(0.459, 0.395, 0.541) = **0.395** | eficiencia_media |
   | R3 | baja, ligero, alta | min(0.459, 0.395, 0.459) = **0.395** | eficiencia_alta |
   | R5 | baja, medio, media | min(0.459, 0.605, 0.541) = **0.459** | eficiencia_baja |
   | R7 | media, ligero, media | min(0.541, 0.395, 0.541) = **0.395** | eficiencia_alta |
   | R8 | media, ligero, alta | min(0.541, 0.395, 0.459) = **0.395** | eficiencia_baja |
   | R10 | media, medio, media | min(0.541, 0.605, 0.541) = **0.541** | eficiencia_baja |
   | R11 | media, medio, alta | min(0.541, 0.605, 0.459) = **0.459** | eficiencia_media |

6. **Recorte del consecuente (función de masa):** cada consecuente se recorta al
   mayor alpha de las reglas que lo proponen: eficiencia_baja → **0.541** (R10),
   eficiencia_media → **0.459** (R11), eficiencia_alta → **0.395** (R3, R7).
7. **Agregación mediante máximo:** el conjunto agregado tiene una meseta de 0.541
   en la zona baja, luego 0.459 en la zona media y 0.395 en la zona alta, hasta
   46.6 mpg.
8. **Defuzzificación mediante centroide:**
   ∫ y·μ(y) dy = 445.410; ∫ μ(y) dy = 16.852; y* = 445.410 / 16.852 = **26.431**.
9. **MPG obtenido: 26.43** (el cálculo manual y `ctrl.ControlSystemSimulation`
   coinciden). El MPG real de este auto es 31.

## 14. Limitaciones

1. **Resultado de una sola ejecución.** La mejor base de reglas depende de la
   población inicial aleatoria y del azar del proceso; no se repitió con otros
   valores del generador aleatorio. Es el mejor conjunto encontrado en esta
   ejecución, no necesariamente el óptimo.
2. **Búsqueda lenta por diseño.** Cada generación crea solo 2 hijos y el
   descarte es completamente aleatorio (sin elitismo), así que la mejor solución
   puede salir de la población; por eso se guarda aparte el mejor histórico.
   En esta ejecución la mejora apareció en las generaciones 76 y 78.
3. **Cobertura.** La mejor base de reglas encontrada tiene 5 antecedentes sin
   regla (genes 0), y 5 de 392 autos no activan ninguna regla disponible. Es una
   limitación de esa base de reglas, no un error de ejecución: la heurística los
   penaliza con 37.6 cada uno pero no los prohíbe.
4. **Rango de salida limitado por el centroide:** entre 14.56 y 36.16 mpg.
5. **La heurística solo mide el error global:** no penaliza el número de reglas
   ni favorece reglas con más soporte o lift; por eso algunas reglas tienen lift
   < 1 (sección 12.3).
6. **Variables no incluidas:** el año del modelo (r = 0.58 con mpg) podría
   aportar información adicional que las tres entradas no capturan.

## 15. Estructura del código y mapa para la defensa

```
proyecto_ia/
├── data/                          auto-mpg.data, auto-mpg.names (dataset UCI)
├── src/
│   ├── carga_datos.py             lectura y limpieza (398 -> 392)
│   ├── exploracion.py             estadística del datasheet y gráfica de distribuciones
│   ├── variables_difusas.py       PASO 1-3: estadísticos, funciones de pertenencia, antecedentes/consecuente
│   ├── sistema_mamdani.py         cromosoma -> ctrl.Rule; ControlSystem; centroide; optimización interna
│   ├── algoritmo_genetico.py      FASE 1-15 del algoritmo genético
│   ├── metricas_reglas.py         support, confidence y lift
│   ├── proceso_mamdani.py         fuzzificación y proceso Mamdani paso a paso (PASO 1-5)
│   └── reglas_finales.py          reglas encontradas en CSV y texto
├── resultados/
│   ├── figuras/
│   │   ├── distribuciones.png            análisis del datasheet
│   │   ├── funciones_pertenencia.png     funciones con mín/Q1/mediana/Q3/máx marcados
│   │   ├── ejemplo_fuzzificacion.png     un auto dentro de las funciones + tabla de grados
│   │   ├── evolucion_heuristica.png      evolución del algoritmo genético
│   │   └── ejemplo_mamdani.png           ejemplo completo Mamdani
│   ├── tablas/
│   │   ├── estadisticas_rangos.csv       ¿de dónde salió cada valor?
│   │   ├── estadisticas_descriptivas.csv media, desviación, asimetría, atípicos
│   │   ├── correlaciones.csv             justificación de las variables
│   │   ├── ejemplo_fuzzificacion.csv     grados y fórmulas del auto de ejemplo
│   │   ├── ejemplo_seleccion_genetica.csv heurística, aptitud, probabilidad, padres
│   │   ├── ejemplo_cruzamiento.csv       padres, punto de corte, hijos
│   │   ├── evolucion_genetico.csv        heurística por generación
│   │   ├── reglas_genetico.csv           reglas encontradas
│   │   ├── metricas_reglas.csv           support, confidence, lift
│   │   ├── comparacion_consecuentes.csv  lift con los 3 consecuentes (análisis)
│   │   ├── ejemplo_mamdani_reglas.csv    reglas activadas y alpha
│   │   └── ejemplo_mamdani_resumen.csv   recorte, centroide y MPG
│   └── reglas_finales.txt         cromosoma y reglas (lingüísticas y ctrl.Rule)
├── main.py                        guion: PASO 1 ... PASO 12
├── requirements.txt
└── README.md
```

**Ejecución** (≈ 50 s): `pip install -r requirements.txt` y `python main.py`.
La consola muestra 8 líneas; el detalle queda en tablas y figuras.

**Mapa para la defensa** (cada concepto tiene un encabezado visible `# ====`):

| Pregunta del docente | Dónde está |
|---|---|
| ¿De dónde salen mín/Q1/mediana/Q3/máx? | `variables_difusas.py` → **PASO 1**; `estadisticas_rangos.csv` |
| ¿Cómo se crean las funciones de pertenencia? | `variables_difusas.py` → **PASO 2** y **PASO 3** |
| ¿Cómo se calcula un grado de pertenencia? | `proceso_mamdani.py` → **CÁLCULO DE GRADO DE PERTENENCIA** |
| Fuzzificación | `proceso_mamdani.py` → **PASO 1 - FUZZIFICACIÓN** |
| Regla `ctrl.Rule` | `sistema_mamdani.py` → **CONVERSIÓN DEL CROMOSOMA … A ctrl.Rule**; `main.py` → **PASO 8** |
| Inferencia (AND = mínimo) | `proceso_mamdani.py` → **PASO 2 - INFERENCIA MAMDANI** |
| Recorte / función de masa | `proceso_mamdani.py` → **PASO 3 - IMPLICACIÓN / RECORTE** |
| Agregación | `proceso_mamdani.py` → **PASO 4 - AGREGACIÓN** |
| Centroide | `proceso_mamdani.py` → **PASO 5 - DEFUZZIFICACIÓN POR CENTROIDE**; `sistema_mamdani.py` → `centroide_lineal` |
| Algoritmo genético | `algoritmo_genetico.py` → **FASE 1** … **FASE 15** |
| Descarte aleatorio (también de hijos) | `algoritmo_genetico.py` → **FASE 10** y `eliminar_dos_al_azar` |
| Mejor individuo histórico | `algoritmo_genetico.py` → **FASE 12** y **FASE 15** |
| Heurística | `algoritmo_genetico.py` → **HEURÍSTICA DE ERROR PROMEDIO PENALIZADO** |
| Support / confidence / lift | `metricas_reglas.py` → **SUPPORT, CONFIDENCE Y LIFT** |
| ControlSystem y simulación | `main.py` → **PASO 9** |

Orden de `main.py`: PASO 1 cargar y limpiar · PASO 2 estadísticas del datasheet ·
PASO 3 funciones de pertenencia · PASO 4 fuzzificación de ejemplo · PASO 5
algoritmo genético · PASO 6 reglas encontradas · PASO 7 support/confidence/lift ·
PASO 8 reglas `ctrl.Rule` · PASO 9 `ControlSystem` · PASO 10 ejemplo Mamdani ·
PASO 11 centroide / MPG · PASO 12 guardar tablas y figuras.

## 16. Referencias

- Quinlan, R. (1993). *Combining Instance-Based and Model-Based Learning.*
  Proceedings of the Tenth International Conference on Machine Learning,
  236–243. Morgan Kaufmann.
- Dua, D., & Graff, C. UCI Machine Learning Repository — *Auto MPG Data Set*.
- Zadeh, L. A. (1965). *Fuzzy sets.* Information and Control, 8(3), 338–353.
- Mamdani, E. H., & Assilian, S. (1975). *An experiment in linguistic synthesis
  with a fuzzy logic controller.* International Journal of Man-Machine
  Studies, 7(1), 1–13.
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization and
  Machine Learning.* Addison-Wesley.
- Agrawal, R., Imieliński, T., & Swami, A. (1993). *Mining association rules
  between sets of items in large databases.* Proceedings of ACM SIGMOD, 207–216.
- Documentación de scikit-fuzzy: https://pythonhosted.org/scikit-fuzzy/
