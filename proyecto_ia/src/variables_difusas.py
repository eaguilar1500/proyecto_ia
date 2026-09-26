"""
Variables lingüísticas y funciones de pertenencia.

Los puntos de TODAS las funciones salen de cinco estadísticos del datasheet
(392 autos): mínimo, Q1, mediana, Q3 y máximo. Ningún número se elige a mano.

    BAJO  : trapmf([mínimo, mínimo, Q1, mediana])
    MEDIO : trimf([Q1, mediana, Q3])
    ALTO  : trapmf([mediana, Q3, máximo, máximo])
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # genera imágenes sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.carga_datos import ENTRADAS, SALIDA


RUTA_FIGURAS = Path(__file__).resolve().parent.parent / "resultados" / "figuras"

# Etiquetas lingüísticas de cada variable, en orden: bajo, medio, alto.
ETIQUETAS = {
    "horsepower": ["baja", "media", "alta"],
    "weight": ["ligero", "medio", "pesado"],
    "acceleration": ["baja", "media", "alta"],
    "mpg": ["eficiencia_baja", "eficiencia_media", "eficiencia_alta"],
}

# Paso del universo = resolución con la que el dataset registra cada variable
# (horsepower y weight son enteros; acceleration y mpg tienen un decimal).
PASOS = {"horsepower": 1.0, "weight": 1.0, "acceleration": 0.1, "mpg": 0.1}

UNIDADES = {"horsepower": "hp", "weight": "lb",
            "acceleration": "s (0-60 mph)", "mpg": "millas/galón"}

COLORES = ["#2a78d6", "#eb6834", "#1baf7a"]  # bajo, medio, alto
COLOR_TEXTO = "#52514e"


# ============================================================
# PASO 1 - OBTENER MÍNIMO, Q1, MEDIANA, Q3 Y MÁXIMO
# ============================================================
def puntos_referencia(serie):
    """Los 5 estadísticos del datasheet que definen los conjuntos difusos.
    Q1 y Q3 son los percentiles 25 y 75 (interpolación lineal de pandas)."""
    return {
        "min": float(serie.min()),
        "Q1": float(serie.quantile(0.25)),
        "mediana": float(serie.median()),
        "Q3": float(serie.quantile(0.75)),
        "max": float(serie.max()),
    }


# ============================================================
# PASO 2 - CREAR FUNCIONES DE PERTENENCIA (puntos de cada conjunto)
# ============================================================
def definir_conjuntos(datos):
    """Para cada variable: tipo de función y puntos de sus 3 conjuntos.

    - Extremos -> trapecio: conceptos que se saturan (230 hp y 150 hp son
      ambos "potencia alta"); cubren con μ = 1 las colas y los atípicos.
    - Centro   -> triángulo: "medio" tiene un valor más representativo,
      la mediana, y la pertenencia baja al alejarse de ella.
    """
    definicion = {}
    for var in ENTRADAS + [SALIDA]:
        p = puntos_referencia(datos[var])
        bajo, medio, alto = ETIQUETAS[var]
        definicion[var] = {
            "referencia": p,
            "conjuntos": {
                bajo: ("trapmf", [p["min"], p["min"], p["Q1"], p["mediana"]]),
                medio: ("trimf", [p["Q1"], p["mediana"], p["Q3"]]),
                alto: ("trapmf", [p["mediana"], p["Q3"], p["max"], p["max"]]),
            },
        }
    return definicion


def tabla_rangos(datos):
    """variable | mínimo | Q1 | mediana | Q3 | máximo | conjunto | función | puntos

    Responde a "¿de dónde salió este valor?": cada punto es uno de los
    estadísticos de la misma fila.
    """
    formulas = {"trapmf_bajo": "[mínimo, mínimo, Q1, mediana]",
                "trimf": "[Q1, mediana, Q3]",
                "trapmf_alto": "[mediana, Q3, máximo, máximo]"}
    filas = []
    for var, info in definir_conjuntos(datos).items():
        p = info["referencia"]
        for n, (etiqueta, (tipo, puntos)) in enumerate(info["conjuntos"].items()):
            clave = tipo if tipo == "trimf" else f"trapmf_{'bajo' if n == 0 else 'alto'}"
            filas.append({
                "variable": var, "minimo": p["min"], "Q1": p["Q1"],
                "mediana": p["mediana"], "Q3": p["Q3"], "maximo": p["max"],
                "conjunto": etiqueta, "funcion": f"{tipo}{formulas[clave]}",
                "puntos": "[" + ", ".join(f"{x:g}" for x in np.round(puntos, 3)) + "]",
            })
    return pd.DataFrame(filas)


# ============================================================
# PASO 3 - CREAR ANTECEDENTES Y CONSECUENTE (skfuzzy.control)
# ============================================================
def _universo(var, p):
    """Universo de discurso = [mínimo, máximo] con el paso de PASOS.
    Se añaden Q1, mediana y Q3 por si no caen en la malla, para que cada
    pico llegue exactamente a μ = 1."""
    paso = PASOS[var]
    malla = np.round(np.arange(p["min"], p["max"] + paso / 2, paso), 4)
    return np.union1d(malla, [p["Q1"], p["mediana"], p["Q3"]])


def crear_variables(datos):
    """Antecedentes (horsepower, weight, acceleration) y consecuente (mpg).

    Devuelve {nombre_variable: ctrl.Antecedent / ctrl.Consequent}.
    """
    variables = {}
    for var, info in definir_conjuntos(datos).items():
        universo = _universo(var, info["referencia"])

        if var == SALIDA:
            # La salida se DEFUZZIFICA por centroide.
            variable = ctrl.Consequent(universo, var, defuzzify_method="centroid")
        else:
            # Las entradas se FUZZIFICAN.
            variable = ctrl.Antecedent(universo, var)

        for etiqueta, (tipo, puntos) in info["conjuntos"].items():
            if tipo == "trapmf":
                variable[etiqueta] = fuzz.trapmf(universo, puntos)
            else:
                variable[etiqueta] = fuzz.trimf(universo, puntos)
        variables[var] = variable
    return variables


def verificar_particion(variables):
    """Comprueba que en cada punto del universo los tres grados sumen 1
    (ningún valor queda sin cubrir). Devuelve {variable: (mín, máx) de la suma}."""
    return {var: (float(s.min()), float(s.max()))
            for var, v in variables.items()
            for s in [sum(t.mf for t in v.terms.values())]}


# ============================================================
# GRÁFICA DE LAS FUNCIONES DE PERTENENCIA
# ============================================================
def graficar_funciones(variables, datos, ruta=RUTA_FIGURAS):
    """Los 3 conjuntos de cada variable, con mín/Q1/mediana/Q3/máx marcados
    y el histograma de los datos al fondo."""
    ruta.mkdir(parents=True, exist_ok=True)
    fig, ejes = plt.subplots(2, 2, figsize=(12, 8.5))

    for ax, (var, variable) in zip(ejes.flat, variables.items()):
        ax_hist = ax.twinx()
        conteos, _, _ = ax_hist.hist(datos[var], bins=20, color="#e4e3df", edgecolor="white")
        ax_hist.set_ylim(0, conteos.max() * 1.25)
        ax_hist.set_yticks([])
        ax_hist.spines[["top", "right", "left"]].set_visible(False)
        ax.set_zorder(ax_hist.get_zorder() + 1)
        ax.patch.set_visible(False)

        for color, (etiqueta, term) in zip(COLORES, variable.terms.items()):
            ax.plot(variable.universe, term.mf, color=color, linewidth=2, label=etiqueta)

        for nombre, valor in puntos_referencia(datos[var]).items():
            ax.axvline(valor, color=COLOR_TEXTO, linewidth=0.8, linestyle=":")
            ax.text(valor, 1.04, f"{nombre}\n{valor:g}", ha="center",
                    va="bottom", fontsize=8, color=COLOR_TEXTO)

        ax.set_ylim(0, 1.18)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylabel("grado de pertenencia μ", color=COLOR_TEXTO)
        ax.set_xlabel(f"{var} ({UNIDADES[var]})", color=COLOR_TEXTO)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=COLOR_TEXTO)
        ax.legend(frameon=False, loc="center right", labelcolor=COLOR_TEXTO, fontsize=9)

    fig.suptitle(f"Funciones de pertenencia — datasheet de {len(datos)} autos\n"
                 "líneas punteadas = mín, Q1, mediana, Q3, máx; fondo = histograma")
    fig.tight_layout()
    archivo = ruta / "funciones_pertenencia.png"
    fig.savefig(archivo, dpi=120)
    plt.close(fig)
    return archivo
