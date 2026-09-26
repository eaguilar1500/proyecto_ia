"""
Support, confidence y lift DIFUSOS de las reglas, sobre los 392 autos.

Sirven para ANALIZAR las reglas que encontró el algoritmo genético.
El algoritmo NO las optimiza y no se usan para modificar ni eliminar reglas.
"""

import numpy as np
import pandas as pd
import skfuzzy as fuzz

from src.carga_datos import SALIDA
from src.sistema_mamdani import COMBINACIONES, CONSECUENTES, N_GENES, EvaluadorNumpy


def _dividir(a, b):
    """a / b, o NaN si b = 0 (evita divisiones entre cero)."""
    return a / b if b > 0 else np.nan


def _antecedente(gen):
    p, w, a = COMBINACIONES[gen]
    return f"potencia {p} Y peso {w} Y aceleración {a}"


# ============================================================
# ALPHA Y BETA DE CADA AUTOMÓVIL
# alpha_i = min(μ_horsepower, μ_weight, μ_acceleration)   (antecedente A)
# beta_i  = μ_consecuente(mpg_real_i)                      (consecuente C)
# ============================================================
def _alpha_beta(variables, datos):
    alpha = EvaluadorNumpy(variables, datos).activacion        # N x 27
    mpg = datos[SALIDA].to_numpy(dtype=float)
    universo = variables[SALIDA].universe
    beta = np.column_stack([                                    # N x 3
        fuzz.interp_membership(universo, variables[SALIDA][CONSECUENTES[k]].mf, mpg)
        for k in (1, 2, 3)])
    return alpha, beta


# ============================================================
# SUPPORT, CONFIDENCE Y LIFT
# Support(A)    = (1/N) Σ alpha_i
# Support(A->C) = (1/N) Σ min(alpha_i, beta_i)
# Confidence    = Support(A->C) / Support(A)
# Support(C)    = (1/N) Σ beta_i
# Lift          = Confidence / Support(C)
# ============================================================
def _metricas(alpha_g, beta_k):
    support_a = alpha_g.mean()
    support_regla = np.minimum(alpha_g, beta_k).mean()
    confidence = _dividir(support_regla, support_a)
    lift = _dividir(confidence, beta_k.mean())
    return support_a, support_regla, confidence, lift


def tabla_metricas(cromosoma, variables, datos):
    """regla | antecedente | consecuente | support_antecedente | support_regla |
    confidence | lift | autos_activados   (una fila por regla encontrada)."""
    alpha, beta = _alpha_beta(variables, datos)
    filas = []
    for gen in range(N_GENES):
        valor = int(cromosoma[gen])
        if valor == 0:
            continue
        sa, sr, conf, lift = _metricas(alpha[:, gen], beta[:, valor - 1])
        filas.append({"regla": f"R{len(filas) + 1}", "gen": f"G{gen}",
                      "antecedente": _antecedente(gen), "consecuente": CONSECUENTES[valor],
                      "support_antecedente": sa, "support_regla": sr,
                      "confidence": conf, "lift": lift,
                      "autos_activados": int((alpha[:, gen] > 0).sum())})
    return pd.DataFrame(filas).round(4)


def tabla_comparacion(cromosoma, variables, datos):
    """Para los 27 antecedentes: lift con cada consecuente posible y el
    consecuente elegido por el AG. Tabla SOLO de análisis."""
    alpha, beta = _alpha_beta(variables, datos)
    nombres = {1: "baja", 2: "media", 3: "alta"}
    filas = []
    for gen in range(N_GENES):
        fila = {"gen": f"G{gen}", "antecedente": _antecedente(gen),
                "support_antecedente": alpha[:, gen].mean(),
                "autos_activados": int((alpha[:, gen] > 0).sum())}
        lifts = {}
        for k, nombre in nombres.items():
            _, _, conf, lift = _metricas(alpha[:, gen], beta[:, k - 1])
            fila[f"confidence_{nombre}"] = conf
            fila[f"lift_{nombre}"] = lift
            lifts[nombre] = lift
        validos = {k: v for k, v in lifts.items() if not np.isnan(v)}
        fila["mayor_lift"] = max(validos, key=validos.get) if validos else ""
        elegido = int(cromosoma[gen])
        fila["consecuente_elegido_por_AG"] = nombres.get(elegido, "sin regla")
        fila["coincide"] = "" if elegido == 0 else ("sí" if nombres[elegido] == fila["mayor_lift"] else "no")
        filas.append(fila)
    return pd.DataFrame(filas).round(4)
