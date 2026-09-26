"""
Puente entre el cromosoma del algoritmo genético y el sistema Mamdani.

  - Representación: qué antecedente corresponde a cada uno de los 27 genes.
  - Conversión del cromosoma a objetos ctrl.Rule.
  - Sistema skfuzzy (ctrl.ControlSystem + ctrl.ControlSystemSimulation).
  - Optimización computacional interna (NumPy) para evaluar rápido la
    heurística del algoritmo genético, comprobada contra skfuzzy.
"""

import itertools

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.carga_datos import ENTRADAS, SALIDA
from src.variables_difusas import ETIQUETAS


# ============================================================
# REPRESENTACIÓN DEL CROMOSOMA
# 3 niveles de potencia x 3 de peso x 3 de aceleración = 27 genes
# ============================================================
# Gen i  <->  COMBINACIONES[i] = (potencia, peso, aceleración)
#   G0 = (baja, ligero, baja), G1 = (baja, ligero, media), ..., G26 = (alta, pesado, alta)
COMBINACIONES = list(itertools.product(
    ETIQUETAS["horsepower"], ETIQUETAS["weight"], ETIQUETAS["acceleration"]))
N_GENES = len(COMBINACIONES)  # 27

# Valor del gen -> consecuente de la regla
CONSECUENTES = {
    0: None,                  # 0 = no existe regla
    1: ETIQUETAS[SALIDA][0],  # 1 = eficiencia_baja
    2: ETIQUETAS[SALIDA][1],  # 2 = eficiencia_media
    3: ETIQUETAS[SALIDA][2],  # 3 = eficiencia_alta
}
VALORES_GEN = list(CONSECUENTES)  # [0, 1, 2, 3]


# ============================================================
# CONVERSIÓN DEL CROMOSOMA DEL ALGORITMO GENÉTICO A ctrl.Rule
# ============================================================
def decodificar_reglas(cromosoma, variables):
    """Cada gen distinto de 0 se convierte en UN objeto ctrl.Rule:
    SI potencia es p Y peso es w Y aceleración es a ENTONCES eficiencia es c.
    El operador & de skfuzzy es el AND difuso (mínimo)."""
    potencia = variables["horsepower"]
    peso = variables["weight"]
    aceleracion = variables["acceleration"]
    eficiencia = variables[SALIDA]

    reglas = []
    for gen, valor in enumerate(cromosoma):
        if valor == 0:
            continue                               # gen 0 = no existe regla
        p, w, a = COMBINACIONES[gen]               # antecedente del gen
        consecuente = CONSECUENTES[int(valor)]     # 1, 2 o 3 -> eficiencia

        # ---------------- REGLA ctrl.Rule ----------------
        regla = ctrl.Rule(
            potencia[p]
            & peso[w]
            & aceleracion[a],
            eficiencia[consecuente],
            label=f"R{len(reglas) + 1}",
        )
        reglas.append(regla)
    return reglas


def texto_linguistico(gen, valor):
    """SI potencia es ... Y peso es ... Y aceleración es ... ENTONCES eficiencia es ..."""
    p, w, a = COMBINACIONES[gen]
    eficiencia = CONSECUENTES[int(valor)].replace("eficiencia_", "")
    return (f"SI potencia es {p} Y peso es {w} Y aceleración es {a} "
            f"ENTONCES eficiencia es {eficiencia}")


def texto_regla(gen, valor):
    """La misma regla escrita como código skfuzzy."""
    p, w, a = COMBINACIONES[gen]
    return (f"ctrl.Rule(potencia['{p}'] & peso['{w}'] & aceleracion['{a}'], "
            f"eficiencia['{CONSECUENTES[int(valor)]}'])")


# ============================================================
# SISTEMA MAMDANI DE skfuzzy: ControlSystem + ControlSystemSimulation
# ============================================================
def crear_simulacion(cromosoma, variables):
    """ctrl.ControlSystem con las reglas del cromosoma y su simulación.

    cache=False: con la caché activada se observaron resultados que
    dependían del orden al crear muchos sistemas sobre las mismas
    variables; con cache=False los resultados son reproducibles.
    """
    sistema = ctrl.ControlSystem(decodificar_reglas(cromosoma, variables))
    return ctrl.ControlSystemSimulation(sistema, cache=False)


def predecir_skfuzzy(cromosoma, variables, datos):
    """mpg de cada auto con skfuzzy; NaN si ninguna regla produce salida."""
    simulacion = crear_simulacion(cromosoma, variables)
    salida = []
    for _, auto in datos.iterrows():
        for var in ENTRADAS:
            simulacion.input[var] = auto[var]
        try:
            simulacion.compute()
            salida.append(simulacion.output[SALIDA])
        except (KeyError, ValueError):
            salida.append(np.nan)
    return np.array(salida, dtype=float)


# ============================================================
# DEFUZZIFICACIÓN POR CENTROIDE (cálculo exacto)
# y* = ∫ y·μ(y) dy / ∫ μ(y) dy
# ============================================================
def centroide_lineal(x, mu):
    """Centroide de una función lineal a tramos (igual que skfuzzy).
    En cada tramo [x1, x2] la función es una recta:
        área    = (x2 - x1)(y1 + y2) / 2
        momento = (x2 - x1)(x1(2y1 + y2) + x2(y1 + 2y2)) / 6
    y* = Σ momentos / Σ áreas
    """
    x1, x2, y1, y2 = x[:-1], x[1:], mu[:-1], mu[1:]
    dx = x2 - x1
    area = dx * (y1 + y2) / 2
    momento = dx * (x1 * (2 * y1 + y2) + x2 * (y1 + 2 * y2)) / 6
    return momento.sum() / area.sum()


def puntos_de_corte(x, mf, h):
    """Valores de x donde la función mf cruza la altura de recorte h.
    skfuzzy los añade al universo para que el recorte sea exacto."""
    cambia = np.diff(mf > h) if h == 0 else np.diff(mf >= h)
    i = np.where(cambia)[0]
    return x[i] + (h - mf[i]) * (x[i + 1] - x[i]) / (mf[i + 1] - mf[i])


# ============================================================
# OPTIMIZACIÓN COMPUTACIONAL INTERNA (NumPy)
# ============================================================
class EvaluadorNumpy:
    """Mismo proceso Mamdani que skfuzzy, pero más rápido.

    Solo sirve para que el algoritmo genético no tarde demasiado: con
    ctrl.ControlSystemSimulation, evaluar un cromosoma en 392 autos tarda
    ≈ 3-4 s. Como las funciones de pertenencia son fijas, la fuzzificación y
    la activación (AND = mínimo) de las 27 reglas posibles se calculan una
    sola vez. main.py comprueba que da el mismo resultado que skfuzzy.
    """

    def __init__(self, variables, datos):
        grados = {}
        for var in ENTRADAS:
            universo = variables[var].universe
            x = np.clip(datos[var].to_numpy(dtype=float), universo.min(), universo.max())
            grados[var] = {etq: fuzz.interp_membership(universo, term.mf, x)
                           for etq, term in variables[var].terms.items()}
        # activacion[auto, gen] = min(μ_potencia, μ_peso, μ_aceleración)
        self.activacion = np.column_stack([
            np.fmin(np.fmin(grados["horsepower"][p], grados["weight"][w]),
                    grados["acceleration"][a])
            for p, w, a in COMBINACIONES])
        self.universo = variables[SALIDA].universe
        self.mf_salida = [variables[SALIDA][CONSECUENTES[v]].mf for v in (1, 2, 3)]

    def alturas(self, cromosoma):
        """Altura de recorte de cada consecuente (máximo α de sus reglas)."""
        cromosoma = np.asarray(cromosoma)
        alturas = np.zeros((len(self.activacion), 3))
        usados = np.zeros(3, dtype=bool)
        for k in (1, 2, 3):
            genes = np.where(cromosoma == k)[0]
            if len(genes):
                alturas[:, k - 1] = self.activacion[:, genes].max(axis=1)
                usados[k - 1] = True
        return alturas, usados

    def predecir(self, cromosoma):
        """mpg de cada auto; NaN si ninguna regla produce salida."""
        alturas, usados = self.alturas(cromosoma)
        x = self.universo
        salida = np.full(len(alturas), np.nan)
        for i, h in enumerate(alturas):
            if h.max() == 0:
                continue
            cortes = [puntos_de_corte(x, self.mf_salida[k], h[k]) for k in range(3) if usados[k]]
            x_fino = np.union1d(x, np.concatenate(cortes))
            agregado = np.zeros_like(x_fino)
            for k in range(3):
                if usados[k]:   # recorte (mínimo) y agregación (máximo)
                    agregado = np.maximum(agregado, np.minimum(h[k], np.interp(x_fino, x, self.mf_salida[k])))
            salida[i] = centroide_lineal(x_fino, agregado)
        return salida


def comprobar_contra_skfuzzy(variables, datos, cromosomas):
    """Diferencia máxima entre NumPy y skfuzzy, y si coinciden los autos sin
    salida, para varios cromosomas."""
    evaluador = EvaluadorNumpy(variables, datos)
    dif_max, coinciden = 0.0, True
    for cromosoma in cromosomas:
        a, b = evaluador.predecir(cromosoma), predecir_skfuzzy(cromosoma, variables, datos)
        coinciden &= bool((np.isnan(a) == np.isnan(b)).all())
        ambos = ~np.isnan(a) & ~np.isnan(b)
        if ambos.any():
            dif_max = max(dif_max, float(np.abs(a[ambos] - b[ambos]).max()))
    return dif_max, coinciden
