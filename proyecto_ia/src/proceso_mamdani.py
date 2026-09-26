"""
Ejemplo completo del sistema Mamdani para UN automóvil real del dataset.

Las ENTRADAS (horsepower, weight, acceleration) se FUZZIFICAN.
La SALIDA (mpg) se DEFUZZIFICA por centroide.

    PASO 1 - Fuzzificación
    PASO 2 - Inferencia Mamdani (AND = mínimo)
    PASO 3 - Implicación / recorte del consecuente
    PASO 4 - Agregación (máximo)
    PASO 5 - Defuzzificación (centroide)

El resultado se compara con ctrl.ControlSystemSimulation (skfuzzy).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz

from src.carga_datos import ENTRADAS, SALIDA
from src.sistema_mamdani import (
    COMBINACIONES,
    CONSECUENTES,
    N_GENES,
    centroide_lineal,
    puntos_de_corte,
    texto_linguistico,
)
from src.variables_difusas import ETIQUETAS

RUTA_FIGURAS = Path(__file__).resolve().parent.parent / "resultados" / "figuras"
COLORES = ["#2a78d6", "#eb6834", "#1baf7a"]  # bajo, medio, alto
COLOR_TEXTO = "#52514e"
COLOR_REFERENCIA = "#0b0b0b"
COLOR_AGREGADO = "#4a3aa7"


# ============================================================
# CÁLCULO DE GRADO DE PERTENENCIA μ(x) (fórmulas manuales)
# ============================================================
def grado_pertenencia(x, tipo, puntos):
    """Devuelve (μ, fórmula usada) para un valor x.

    trapmf [a, b, c, d]:
        x entre b y c      -> μ = 1                      (meseta)
        x entre a y b      -> μ = (x - a) / (b - a)      (sube)
        x entre c y d      -> μ = (d - x) / (d - c)      (baja)
        fuera de [a, d]    -> μ = 0
    trimf [a, b, c]:
        x = b              -> μ = 1                      (pico = mediana)
        x entre a y b      -> μ = (x - a) / (b - a)
        x entre b y c      -> μ = (c - x) / (c - b)
        fuera de [a, c]    -> μ = 0

    Ejemplo: si x está entre mediana y Q3,
        μ_medio(x) = (Q3 - x) / (Q3 - mediana)
        μ_alto(x)  = (x - mediana) / (Q3 - mediana)
    """
    if tipo == "trapmf":
        a, b, c, d = puntos
        if b <= x <= c:
            return 1.0, "meseta -> 1"
        if a < x < b:
            return (x - a) / (b - a), f"({x:g} - {a:g}) / ({b:g} - {a:g})"
        if c < x < d:
            return (d - x) / (d - c), f"({d:g} - {x:g}) / ({d:g} - {c:g})"
        return 0.0, "fuera -> 0"
    a, b, c = puntos
    if x == b:
        return 1.0, "pico -> 1"
    if a < x < b:
        return (x - a) / (b - a), f"({x:g} - {a:g}) / ({b:g} - {a:g})"
    if b < x < c:
        return (c - x) / (c - b), f"({c:g} - {x:g}) / ({c:g} - {b:g})"
    return 0.0, "fuera -> 0"


# ============================================================
# PASO 1 - FUZZIFICACIÓN
# Entrada numérica -> grados de pertenencia
# ============================================================
def fuzzificar(definicion, auto):
    """Grado de cada entrada en sus tres conjuntos: {variable: {etiqueta: μ}}."""
    grados = {}
    for var in ENTRADAS:
        p = definicion[var]["referencia"]
        x = min(max(float(auto[var]), p["min"]), p["max"])  # dentro del universo
        grados[var] = {etq: grado_pertenencia(x, tipo, puntos)[0]
                       for etq, (tipo, puntos) in definicion[var]["conjuntos"].items()}
    return grados


def elegir_auto_ejemplo(definicion, datos):
    """Automóvil real para los ejemplos de fuzzificación y Mamdani.

    Criterio (no depende de las reglas): el auto cuyos tres valores caen
    más "entre dos conjuntos", es decir, con la mayor suma, en las tres
    entradas, del segundo grado de pertenencia más alto. Así el ejemplo
    muestra grados parciales y sus fórmulas en las tres variables.
    """
    def puntaje(auto):
        grados = fuzzificar(definicion, auto)
        return sum(sorted(grados[var].values())[-2] for var in ENTRADAS)
    return datos.apply(puntaje, axis=1).idxmax()


def tabla_fuzzificacion(definicion, variables, auto):
    """variable | valor | bajo | medio | alto | cálculo de cada grado.
    La columna 'comprobacion_skfuzzy' confirma que las fórmulas manuales dan
    lo mismo que fuzz.interp_membership."""
    filas = []
    for var in ENTRADAS:
        x = float(auto[var])
        fila = {"variable": var, "valor": x}
        iguales = True
        for nivel, (etq, (tipo, puntos)) in zip(("bajo", "medio", "alto"),
                                                definicion[var]["conjuntos"].items()):
            mu, formula = grado_pertenencia(x, tipo, puntos)
            fila[nivel] = round(mu, 3)
            fila[f"calculo_{nivel}"] = f"μ_{etq} = {formula}"
            mu_sk = fuzz.interp_membership(variables[var].universe, variables[var][etq].mf, x)
            iguales &= abs(mu - mu_sk) < 1e-9
        fila["comprobacion_skfuzzy"] = "igual" if iguales else "DISTINTO"
        filas.append(fila)
    return pd.DataFrame(filas)


def proceso_completo(definicion, variables, cromosoma, simulacion, auto):
    """Calcula los 5 pasos para un auto con las reglas encontradas y lo
    compara con la ctrl.ControlSystemSimulation del sistema final."""

    grados = fuzzificar(definicion, auto)                        # PASO 1
    universo = variables[SALIDA].universe

    # ============================================================
    # PASO 2 - INFERENCIA MAMDANI
    # AND = mínimo:  alpha = min(μ_potencia, μ_peso, μ_aceleración)
    # ============================================================
    reglas = []
    for gen in range(N_GENES):
        if cromosoma[gen] == 0:
            continue
        p, w, a = COMBINACIONES[gen]
        valores = [grados["horsepower"][p], grados["weight"][w], grados["acceleration"][a]]
        alpha = min(valores)
        consecuente = CONSECUENTES[int(cromosoma[gen])]

        # ============================================================
        # PASO 3 - IMPLICACIÓN / RECORTE DEL CONSECUENTE
        # μ'_C(y) = min(alpha, μ_C(y))
        # Siguiendo la terminología utilizada en clase, este resultado
        # recortado se relaciona con la denominada función de masa.
        # ============================================================
        recortado = np.fmin(alpha, variables[SALIDA][consecuente].mf)

        reglas.append({"regla": f"R{len(reglas) + 1}", "gen": f"G{gen}",
                       "texto": texto_linguistico(gen, cromosoma[gen]),
                       "mu_horsepower": valores[0], "mu_weight": valores[1],
                       "mu_acceleration": valores[2], "alpha": alpha,
                       "consecuente": consecuente, "recortado": recortado})
    activas = [r for r in reglas if r["alpha"] > 0]

    # Si varias reglas proponen el mismo consecuente, cuenta la de mayor alpha.
    alturas = {}
    for r in activas:
        alturas[r["consecuente"]] = max(alturas.get(r["consecuente"], 0), r["alpha"])

    # ============================================================
    # PASO 4 - AGREGACIÓN
    # máximo entre los consecuentes recortados de todas las reglas
    # (como skfuzzy, se añaden los puntos exactos de corte al universo)
    # ============================================================
    cortes = [puntos_de_corte(universo, variables[SALIDA][etq].mf, h) for etq, h in alturas.items()]
    y = np.union1d(universo, np.concatenate(cortes))
    agregado = np.zeros_like(y)
    for etq, h in alturas.items():
        agregado = np.maximum(agregado, np.minimum(h, np.interp(y, universo, variables[SALIDA][etq].mf)))

    # ============================================================
    # PASO 5 - DEFUZZIFICACIÓN POR CENTROIDE
    # centroide -> valor numérico de MPG
    # y* = ∫ y·μ(y) dy / ∫ μ(y) dy
    # ============================================================
    dy = np.diff(y)
    area = float(np.sum(dy * (agregado[:-1] + agregado[1:]) / 2))                    # ∫ μ(y) dy
    momento = float(np.sum(dy * (y[:-1] * (2 * agregado[:-1] + agregado[1:]) +
                                 y[1:] * (agregado[:-1] + 2 * agregado[1:])) / 6))  # ∫ y·μ(y) dy
    mpg_manual = centroide_lineal(y, agregado)

    # Mismo auto en el sistema final de skfuzzy.
    for var in ENTRADAS:
        simulacion.input[var] = auto[var]
    simulacion.compute()

    return {"auto": auto, "grados": grados, "reglas": reglas, "activas": activas,
            "alturas": alturas, "universo": universo,
            "agregado": (np.fmax.reduce([r["recortado"] for r in activas])
                         if activas else np.zeros_like(universo)),
            "area": area, "momento": momento, "mpg_manual": mpg_manual,
            "mpg_skfuzzy": float(simulacion.output[SALIDA])}


def tabla_reglas_activadas(resultado):
    """Reglas que se activan: grados de cada antecedente, alpha y consecuente."""
    columnas = ["regla", "gen", "texto", "mu_horsepower", "mu_weight",
                "mu_acceleration", "alpha", "consecuente"]
    return pd.DataFrame([{c: r[c] for c in columnas} for r in resultado["activas"]]).round(3)


def tabla_resumen(resultado):
    """Recorte, agregación, centroide y MPG obtenido."""
    filas = [{"paso": "3 - recorte", "concepto": f"altura de {etq} (máximo alpha)", "valor": round(h, 3)}
             for etq, h in resultado["alturas"].items()]
    filas += [
        {"paso": "5 - centroide", "concepto": "∫ y·μ(y) dy", "valor": round(resultado["momento"], 3)},
        {"paso": "5 - centroide", "concepto": "∫ μ(y) dy", "valor": round(resultado["area"], 3)},
        {"paso": "5 - centroide", "concepto": "MPG obtenido (cálculo manual)", "valor": round(resultado["mpg_manual"], 3)},
        {"paso": "5 - centroide", "concepto": "MPG obtenido (ctrl.ControlSystemSimulation)", "valor": round(resultado["mpg_skfuzzy"], 3)},
        {"paso": "comparación", "concepto": "MPG real del auto", "valor": resultado["auto"][SALIDA]},
    ]
    return pd.DataFrame(filas)


# ============================================================
# FIGURAS
# ============================================================
def _estilo(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=COLOR_TEXTO)
    ax.set_ylim(0, 1.08)


def _panel_entrada(ax, variables, var, auto, grados):
    variable = variables[var]
    for color, (etq, term) in zip(COLORES, variable.terms.items()):
        ax.plot(variable.universe, term.mf, color=color, linewidth=2, label=etq)
        mu = grados[var][etq]
        if mu > 0:
            ax.plot([auto[var]], [mu], "o", color=color, markersize=8, markeredgecolor="white")
            ax.annotate(f"μ = {mu:.3f}", (auto[var], mu), xytext=(6, 4),
                        textcoords="offset points", color=COLOR_REFERENCIA, fontsize=9)
    ax.axvline(auto[var], color=COLOR_REFERENCIA, linewidth=1, linestyle="--")
    ax.set_title(f"{var} = {auto[var]:g}", color=COLOR_REFERENCIA)
    ax.legend(frameon=False, fontsize=8, labelcolor=COLOR_TEXTO)
    _estilo(ax)


def graficar_fuzzificacion(definicion, variables, auto, nombre, ruta=RUTA_FIGURAS):
    """Posición del auto en las funciones de pertenencia + tabla de grados."""
    ruta.mkdir(parents=True, exist_ok=True)
    grados = fuzzificar(definicion, auto)
    fig = plt.figure(figsize=(15, 6.2))
    rejilla = fig.add_gridspec(2, 3, height_ratios=[3, 1.1])
    for i, var in enumerate(ENTRADAS):
        ax = fig.add_subplot(rejilla[0, i])
        _panel_entrada(ax, variables, var, auto, grados)
        if i == 0:
            ax.set_ylabel("grado de pertenencia μ", color=COLOR_TEXTO)
    ax_t = fig.add_subplot(rejilla[1, :])
    ax_t.axis("off")
    celdas = [[f"{var} = {auto[var]:g}"] +
              [f"{grados[var][etq]:.3f}" for etq in ETIQUETAS[var]] for var in ENTRADAS]
    t = ax_t.table(cellText=celdas, colLabels=["Variable", "Bajo", "Medio", "Alto"],
                   loc="center", cellLoc="center")
    t.scale(1, 1.4)
    fig.suptitle(f"Ejemplo de fuzzificación — {nombre}")
    fig.tight_layout()
    archivo = ruta / "ejemplo_fuzzificacion.png"
    fig.savefig(archivo, dpi=120)
    plt.close(fig)
    return archivo


def graficar_mamdani(variables, resultado, nombre, ruta=RUTA_FIGURAS):
    """Fila 1: fuzzificación. Fila 2: recortes por regla activa, y
    agregación + centroide."""
    ruta.mkdir(parents=True, exist_ok=True)
    auto = resultado["auto"]
    fig, ejes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, var in zip(ejes[0], ENTRADAS):
        _panel_entrada(ax, variables, var, auto, resultado["grados"])
        ax.set_title(f"[1] Fuzzificación: {var} = {auto[var]:g}", color=COLOR_REFERENCIA)
    ejes[0][0].set_ylabel("μ", color=COLOR_TEXTO)

    universo = resultado["universo"]
    etiquetas = list(variables[SALIDA].terms)
    activas = resultado["activas"]
    mitad = len(activas) // 2 + len(activas) % 2
    for ax, grupo in zip(ejes[1][:2], (activas[:mitad], activas[mitad:])):
        for etq, color in zip(etiquetas, COLORES):
            ax.plot(universo, variables[SALIDA][etq].mf, color=color, linewidth=1, linestyle=":")
        puestas = []
        for r in grupo:
            color = COLORES[etiquetas.index(r["consecuente"])]
            ax.fill_between(universo, r["recortado"], color=color, alpha=0.35, linewidth=0)
            ax.plot(universo, r["recortado"], color=color, linewidth=2)
            x_pico = universo[np.argmax(variables[SALIDA][r["consecuente"]].mf)]
            rep = sum(1 for x, a in puestas if abs(x - x_pico) < 10 and abs(a - r["alpha"]) < 0.08)
            puestas.append((x_pico, r["alpha"]))
            ax.annotate(f"{r['regla']}: α = {r['alpha']:.3f}", (x_pico, r["alpha"]),
                        xytext=(0, 6 + 12 * rep), textcoords="offset points",
                        ha="center", color=COLOR_REFERENCIA, fontsize=9)
        ax.set_title(f"[2-3] Inferencia y recorte: {', '.join(r['regla'] for r in grupo) or '—'}",
                     color=COLOR_REFERENCIA)
        ax.set_xlabel("mpg", color=COLOR_TEXTO)
        _estilo(ax)
    ejes[1][0].set_ylabel("μ", color=COLOR_TEXTO)

    ax = ejes[1][2]
    ax.fill_between(universo, resultado["agregado"], color=COLOR_AGREGADO, alpha=0.3, linewidth=0)
    ax.plot(universo, resultado["agregado"], color=COLOR_AGREGADO, linewidth=2, label="μ agregado (max)")
    ax.axvline(resultado["mpg_skfuzzy"], color=COLOR_REFERENCIA, linewidth=2,
               label=f"centroide = {resultado['mpg_skfuzzy']:.2f}")
    ax.axvline(auto[SALIDA], color=COLOR_TEXTO, linewidth=1.5, linestyle="--",
               label=f"mpg real = {auto[SALIDA]:g}")
    ax.set_title("[4] Agregación  +  [5] Centroide", color=COLOR_REFERENCIA)
    ax.set_xlabel("mpg", color=COLOR_TEXTO)
    ax.legend(frameon=False, fontsize=8, labelcolor=COLOR_TEXTO, loc="upper right")
    _estilo(ax)

    fig.suptitle(f"Ejemplo completo Mamdani con las reglas encontradas — {nombre}")
    fig.tight_layout()
    archivo = ruta / "ejemplo_mamdani.png"
    fig.savefig(archivo, dpi=120)
    plt.close(fig)
    return archivo
