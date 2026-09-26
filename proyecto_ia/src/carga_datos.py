"""
Carga y limpieza del dataset Auto MPG (UCI).

    398 registros -> se eliminan los 6 sin horsepower ('?') -> 392 registros

Los 392 automóviles completos son el datasheet de todo el proyecto: de ellos
salen los rangos fuzzy, con ellos se evalúa el algoritmo genético y sobre
ellos se calculan support, confidence y lift.

No se normaliza ni se transforma ningún valor: cada variable conserva sus
unidades originales.
"""

from pathlib import Path

import pandas as pd


RUTA_DATOS = Path(__file__).resolve().parent.parent / "data" / "auto-mpg.data"

# Nombres de columna en el orden de la sección 7 de auto-mpg.names.
COLUMNAS = [
    "mpg",           # consumo en millas por galón  -> SALIDA
    "cylinders",
    "displacement",
    "horsepower",    # potencia (hp)                -> ENTRADA (6 valores '?')
    "weight",        # peso (lb)                    -> ENTRADA
    "acceleration",  # segundos de 0 a 60 mph       -> ENTRADA
    "model_year",
    "origin",
    "car_name",
]

ENTRADAS = ["horsepower", "weight", "acceleration"]
SALIDA = "mpg"
MARCA_FALTANTE = "?"


# ============================================================
# LECTURA DEL ARCHIVO
# ============================================================
def cargar_datos(ruta=RUTA_DATOS):
    """Lee auto-mpg.data (sin encabezado, separado por espacios; el nombre del
    auto va entre comillas). No se usa na_values='?' para poder detectar y
    contar los faltantes de forma explícita."""
    return pd.read_csv(ruta, sep=r"\s+", header=None, names=COLUMNAS, quotechar='"')


# ============================================================
# LIMPIEZA: ELIMINAR SOLO LOS REGISTROS SIN HORSEPOWER
# ============================================================
def detectar_faltantes(df):
    """True en las filas cuyo horsepower es '?'."""
    return df["horsepower"].astype(str).str.strip() == MARCA_FALTANTE


def limpiar_datos(df):
    """Elimina únicamente las filas con '?' en horsepower (6 de 398) y
    convierte esa columna a número (cambia el tipo, no los valores)."""
    df_limpio = df.loc[~detectar_faltantes(df)].copy()
    df_limpio["horsepower"] = df_limpio["horsepower"].astype(float)
    return df_limpio.reset_index(drop=True)


def seleccionar_variables(df):
    """Las 3 entradas y la salida del sistema difuso."""
    return df[ENTRADAS + [SALIDA]]
