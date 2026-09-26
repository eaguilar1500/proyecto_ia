"""
Análisis estadístico del datasheet (392 autos).

Sirve para conocer la forma de cada variable y justificar el diseño de las
funciones de pertenencia (uso de cuartiles, trapecios en los extremos,
triángulo en el centro). La correlación es solo descriptiva.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUTA_FIGURAS = Path(__file__).resolve().parent.parent / "resultados" / "figuras"
UNIDADES = {"horsepower": "hp", "weight": "lb",
            "acceleration": "s (0-60 mph)", "mpg": "millas/galón"}
COLOR_DATOS = "#2a78d6"
COLOR_TEXTO = "#52514e"
COLOR_REFERENCIA = "#0b0b0b"


# ============================================================
# ESTADÍSTICA DESCRIPTIVA
# ============================================================
def estadisticas(datos):
    """mínimo, Q1, mediana, media, Q3, máximo, desviación, IQR, asimetría y
    atípicos (regla 1.5·IQR) de cada variable."""
    tabla = datos.agg(["min", "max", "mean", "median", "std"]).T
    tabla["Q1"] = datos.quantile(0.25)
    tabla["Q3"] = datos.quantile(0.75)
    tabla["IQR"] = tabla["Q3"] - tabla["Q1"]
    inf, sup = tabla["Q1"] - 1.5 * tabla["IQR"], tabla["Q3"] + 1.5 * tabla["IQR"]
    tabla["atipicos"] = [((datos[c] < inf[c]) | (datos[c] > sup[c])).sum() for c in datos.columns]
    tabla["asimetria"] = datos.skew()
    tabla = tabla.rename(columns={"min": "minimo", "max": "maximo", "mean": "media",
                                  "median": "mediana", "std": "desv_std"})
    return tabla[["minimo", "Q1", "mediana", "media", "Q3", "maximo",
                  "desv_std", "IQR", "asimetria", "atipicos"]]


def correlaciones(df):
    """Correlación de Pearson de todas las variables numéricas con mpg,
    horsepower y weight (justifica qué variables se usan y cuáles se descartan)."""
    return df.select_dtypes("number").corr()[["mpg", "horsepower", "weight"]]


# ============================================================
# GRÁFICA DE DISTRIBUCIONES
# ============================================================
def graficar_distribuciones(datos, ruta=RUTA_FIGURAS):
    """Histograma de cada variable con media, mediana, Q1 y Q3 marcados
    (Q1, mediana y Q3 son después los puntos de las funciones de pertenencia)."""
    ruta.mkdir(parents=True, exist_ok=True)
    fig, ejes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(ejes.flat, datos.columns):
        ax.hist(datos[col], bins=20, color=COLOR_DATOS, edgecolor="white", linewidth=1)
        ax.axvline(datos[col].mean(), color=COLOR_REFERENCIA, linewidth=1.5,
                   label=f"media = {datos[col].mean():.1f}")
        ax.axvline(datos[col].median(), color=COLOR_REFERENCIA, linewidth=1.5,
                   linestyle="--", label=f"mediana = {datos[col].median():g}")
        q1, q3 = datos[col].quantile([0.25, 0.75])
        ax.axvline(q1, color=COLOR_TEXTO, linewidth=1, linestyle=":",
                   label=f"Q1 = {q1:g}, Q3 = {q3:g}")
        ax.axvline(q3, color=COLOR_TEXTO, linewidth=1, linestyle=":")
        ax.set_title(f"{col} ({UNIDADES[col]}) — asimetría {datos[col].skew():.2f}",
                     color=COLOR_REFERENCIA)
        ax.set_ylabel("frecuencia", color=COLOR_TEXTO)
        ax.legend(frameon=False, labelcolor=COLOR_TEXTO, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=COLOR_TEXTO)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(f"Distribución de las variables — datasheet de {len(datos)} autos")
    fig.tight_layout()
    archivo = ruta / "distribuciones.png"
    fig.savefig(archivo, dpi=120)
    plt.close(fig)
    return archivo
