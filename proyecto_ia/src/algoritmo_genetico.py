"""
Algoritmo genético sencillo (procedimiento visto en clase) que DEFINE la base
de reglas fuzzy.

Individuo = una posible base de reglas = cromosoma de 27 genes
            (uno por antecedente posible: 3 potencia x 3 peso x 3 aceleración).
Valor del gen: 0 = no existe regla, 1 = eficiencia baja,
               2 = eficiencia media, 3 = eficiencia alta.

Objetivo: MINIMIZAR la heurística de error promedio penalizado del sistema
Mamdani completo sobre los 392 autos.

Una generación = seleccionar dos padres -> dos hijos -> posible mutación
-> añadirlos (N + 2) -> eliminar dos al azar (N) -> reevaluar.

El algoritmo termina después de 100 generaciones consecutivas sin encontrar
una solución mejor o al alcanzar 500 generaciones como límite de seguridad.

Support, confidence y lift NO participan en la búsqueda.
Las funciones de pertenencia y el sistema Mamdani NO se modifican aquí.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.sistema_mamdani import N_GENES, VALORES_GEN


TAM_POBLACION = 40
PROB_MUTACION_HIJO = 0.10     # cada hijo: 10 % de probabilidad de mutar UN gen
MAX_SIN_MEJORA = 100          # generaciones consecutivas sin mejora -> parar
MAX_GENERACIONES = 500        # límite de seguridad

# Detalle técnico (no es parte del método): valor inicial fijo del generador
# de números aleatorios, solo para que la ejecución sea reproducible.
SEMILLA = 42


# ============================================================
# HEURÍSTICA DE ERROR PROMEDIO PENALIZADO (menor = mejor)
# ============================================================
class Heuristica:
    """Evalúa un cromosoma (una base de reglas) sobre los 392 autos.

    error_i = |mpg_real - mpg_estimado|           si alguna regla produce salida
    error_i = rango_mpg = max(mpg) - min(mpg)     si ninguna regla produce salida
    heurística = promedio de los 392 error_i

    (Internamente es un MAE penalizado.) Guarda los valores ya calculados
    para no repetir el cálculo de un mismo cromosoma.
    """

    def __init__(self, evaluador, mpg_real):
        self.evaluador = evaluador            # optimización interna (NumPy)
        self.mpg_real = np.asarray(mpg_real, dtype=float)
        self.rango_mpg = float(self.mpg_real.max() - self.mpg_real.min())
        self._memoria = {}

    def detalle(self, cromosoma):
        """(heurística, error promedio de los autos con salida, autos sin salida)"""
        clave = tuple(int(g) for g in cromosoma)
        if clave not in self._memoria:
            estimado = self.evaluador.predecir(cromosoma)
            sin_salida = np.isnan(estimado)
            errores = np.where(sin_salida, self.rango_mpg, np.abs(self.mpg_real - estimado))
            error_cubiertos = errores[~sin_salida].mean() if (~sin_salida).any() else np.nan
            self._memoria[clave] = (float(errores.mean()), error_cubiertos, int(sin_salida.sum()))
        return self._memoria[clave]

    def __call__(self, cromosoma):
        return self.detalle(cromosoma)[0]


def aptitudes_y_probabilidades(heuristicas):
    """aptitud_i = 1 / (1 + heurística_i)          menor error -> mayor aptitud
    probabilidad_i = aptitud_i / suma(aptitudes)"""
    aptitudes = 1 / (1 + np.asarray(heuristicas, dtype=float))
    return aptitudes, aptitudes / aptitudes.sum()


def mutar(hijo, rng):
    """10 % de probabilidad de cambiar UN gen al azar por otro valor distinto.
    Devuelve (hijo, (gen, antes, después)) o (hijo, None) si no muta."""
    hijo = hijo.copy()
    if rng.random() >= PROB_MUTACION_HIJO:
        return hijo, None
    gen = int(rng.integers(0, N_GENES))
    antes = int(hijo[gen])
    hijo[gen] = int(rng.choice([v for v in VALORES_GEN if v != antes]))
    return hijo, (gen, antes, int(hijo[gen]))


def eliminar_dos_al_azar(ampliada, rng):
    """Elige 2 posiciones completamente al azar entre TODOS los individuos
    (N antiguos + 2 hijos). No se protege a nadie: pueden eliminarse padres,
    hijos recién creados o mutados, o el mejor de la población."""
    eliminados = sorted(int(x) for x in rng.choice(len(ampliada), size=2, replace=False))
    return [ind for n, ind in enumerate(ampliada) if n not in eliminados], eliminados


def ejecutar(heuristica, semilla=SEMILLA):
    """Ejecuta el algoritmo genético.

    Devuelve (mejor_historico, historial, ejemplo_generacion_1, motivo_de_parada),
    donde mejor_historico = {"cromosoma", "heuristica", "generacion"}.
    """
    rng = np.random.default_rng(semilla)

    # ============================================================
    # FASE 1 - POBLACIÓN INICIAL
    # 40 cromosomas aleatorios de 27 genes con valores en {0, 1, 2, 3}
    # ============================================================
    poblacion = [rng.integers(0, len(VALORES_GEN), N_GENES) for _ in range(TAM_POBLACION)]

    # ============================================================
    # FASE 2 - EVALUACIÓN DE LA HEURÍSTICA
    # cada cromosoma es una base de reglas; se mide su error promedio
    # penalizado sobre los 392 autos
    # ============================================================
    heuristicas = [heuristica(ind) for ind in poblacion]

    # ============================================================
    # FASE 3 - CONVERSIÓN DE HEURÍSTICA A APTITUD
    # FASE 4 - CÁLCULO DE PROBABILIDADES
    # ============================================================
    aptitudes, probabilidades = aptitudes_y_probabilidades(heuristicas)

    # Mejor individuo histórico: copia EXTERNA a la población (no participa
    # en el reemplazo, así no se pierde aunque se elimine de la población).
    i = int(np.argmin(heuristicas))
    mejor_historico = {"cromosoma": poblacion[i].copy(), "heuristica": heuristicas[i],
                       "generacion": 0}
    generaciones_sin_mejora = 0
    historial = [_fila(0, heuristicas, mejor_historico, generaciones_sin_mejora)]
    ejemplo = None
    motivo = f"límite de {MAX_GENERACIONES} generaciones"

    for generacion in range(1, MAX_GENERACIONES + 1):

        # ============================================================
        # FASE 5 - SELECCIÓN DE PADRES (ruleta)
        # menor heurística -> mayor probabilidad de ser elegido
        # ============================================================
        p1, p2 = (int(x) for x in rng.choice(len(poblacion), size=2,
                                             replace=False, p=probabilidades))
        padre1, padre2 = poblacion[p1], poblacion[p2]

        # ============================================================
        # FASE 6 - PUNTO DE CRUCE k (entre 1 y 26)
        # ============================================================
        k = int(rng.integers(1, N_GENES))

        # ============================================================
        # FASE 7 - CREACIÓN DE DOS HIJOS
        # Hijo 1: genes antes de k del Padre 1 + genes desde k del Padre 2
        # Hijo 2: genes antes de k del Padre 2 + genes desde k del Padre 1
        # ============================================================
        hijo1 = np.concatenate([padre1[:k], padre2[k:]])
        hijo2 = np.concatenate([padre2[:k], padre1[k:]])

        # ============================================================
        # FASE 8 - MUTACIÓN DEL 10 %
        # cada hijo: 10 % de probabilidad; si muta, UN gen al azar cambia
        # por otro valor distinto de {0, 1, 2, 3}
        # ============================================================
        hijo1_final, mut1 = mutar(hijo1, rng)
        hijo2_final, mut2 = mutar(hijo2, rng)

        # ============================================================
        # FASE 9 - INCORPORACIÓN DE LOS HIJOS (40 -> 42)
        # ============================================================
        heuristicas_antes, aptitudes_antes, probs_antes = heuristicas, aptitudes, probabilidades
        ampliada = poblacion + [hijo1_final, hijo2_final]

        # ============================================================
        # FASE 10 - ELIMINACIÓN ALEATORIA DE DOS INDIVIDUOS (42 -> 40)
        # Los hijos recién creados también pueden ser eliminados
        # (sin elitismo: nadie está protegido)
        # ============================================================
        poblacion, eliminados = eliminar_dos_al_azar(ampliada, rng)

        # ============================================================
        # FASE 11 - REEVALUACIÓN DE LA POBLACIÓN
        # heurística, aptitud y probabilidades de la nueva población
        # ============================================================
        heuristicas = [heuristica(ind) for ind in poblacion]
        aptitudes, probabilidades = aptitudes_y_probabilidades(heuristicas)

        # ============================================================
        # FASE 12 - ACTUALIZAR MEJOR INDIVIDUO HISTÓRICO
        # FASE 13 - ACTUALIZAR GENERACIONES SIN MEJORA
        # ============================================================
        i = int(np.argmin(heuristicas))
        if heuristicas[i] < mejor_historico["heuristica"]:
            mejor_historico = {"cromosoma": poblacion[i].copy(),
                               "heuristica": heuristicas[i], "generacion": generacion}
            generaciones_sin_mejora = 0
        else:
            generaciones_sin_mejora += 1

        historial.append(_fila(generacion, heuristicas, mejor_historico, generaciones_sin_mejora))
        if ejemplo is None:   # detalle de la primera generación para la defensa
            ejemplo = {"heuristicas": heuristicas_antes, "aptitudes": aptitudes_antes,
                       "probabilidades": probs_antes, "padres": (p1, p2), "k": k,
                       "padre1": padre1, "padre2": padre2, "hijo1": hijo1, "hijo2": hijo2,
                       "hijos_finales": [hijo1_final, hijo2_final], "mutaciones": [mut1, mut2],
                       "eliminados": eliminados, "n": len(heuristicas_antes)}

        # ============================================================
        # FASE 14 - CONDICIONES DE PARADA
        # 100 sin mejora o máximo 500
        # ============================================================
        if generaciones_sin_mejora >= MAX_SIN_MEJORA:
            motivo = f"{MAX_SIN_MEJORA} generaciones sin mejora"
            break

    # ============================================================
    # FASE 15 - DEVOLVER MEJOR INDIVIDUO HISTÓRICO
    # (no necesariamente el mejor de la población final)
    # ============================================================
    return mejor_historico, historial, ejemplo, motivo


def _fila(generacion, heuristicas, mejor_historico, generaciones_sin_mejora):
    return {"generacion": generacion,
            "mejor_heuristica_poblacion": float(min(heuristicas)),
            "mejor_heuristica_historica": float(mejor_historico["heuristica"]),
            "heuristica_promedio": float(np.mean(heuristicas)),
            "generaciones_sin_mejora": generaciones_sin_mejora,
            "numero_reglas_mejor_historico": int(np.count_nonzero(mejor_historico["cromosoma"]))}


# ============================================================
# TABLAS Y GRÁFICA PARA LA DEFENSA
# ============================================================
def tabla_seleccion(ejemplo):
    """individuo | heurística | aptitud | probabilidad | seleccionado_como_padre
    (población de la primera generación)."""
    p1, p2 = ejemplo["padres"]
    filas = []
    for n, (h, a, p) in enumerate(zip(ejemplo["heuristicas"], ejemplo["aptitudes"],
                                      ejemplo["probabilidades"])):
        filas.append({"individuo": n, "heuristica": round(h, 4), "aptitud": round(a, 4),
                      "probabilidad": round(p, 4),
                      "seleccionado_como_padre": "Padre 1" if n == p1 else ("Padre 2" if n == p2 else ""),
                      "eliminado_en_fase_10": "sí" if n in ejemplo["eliminados"] else ""})
    for n, nombre in ((ejemplo["n"], "Hijo 1"), (ejemplo["n"] + 1, "Hijo 2")):
        if n in ejemplo["eliminados"]:
            filas.append({"individuo": nombre, "eliminado_en_fase_10": "sí"})
    return pd.DataFrame(filas)


def tabla_cruzamiento(ejemplo):
    """Padre 1, Padre 2, punto de corte, Hijo 1, Hijo 2 (gen por gen)."""
    def texto(m):
        return f"gen {m[0]}: {m[1]} -> {m[2]}" if m else "no muta"
    filas = [("Padre 1", ejemplo["padre1"], ""), ("Padre 2", ejemplo["padre2"], ""),
             ("Hijo 1 (cruce)", ejemplo["hijo1"], ""), ("Hijo 2 (cruce)", ejemplo["hijo2"], ""),
             ("Hijo 1 (tras mutación)", ejemplo["hijos_finales"][0], texto(ejemplo["mutaciones"][0])),
             ("Hijo 2 (tras mutación)", ejemplo["hijos_finales"][1], texto(ejemplo["mutaciones"][1]))]
    tabla = pd.DataFrame([[int(g) for g in c] for _, c, _ in filas],
                         columns=[f"G{g}" for g in range(N_GENES)])
    tabla.insert(0, "cromosoma", [f for f, _, _ in filas])
    tabla.insert(1, "punto_corte_k", ejemplo["k"])
    tabla["mutacion"] = [m for _, _, m in filas]
    return tabla


def graficar_evolucion(historial, ruta):
    """Generación vs mejor heurística histórica (menor es mejor)."""
    ruta = Path(ruta)
    ruta.mkdir(parents=True, exist_ok=True)
    g = [h["generacion"] for h in historial]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(g, [h["heuristica_promedio"] for h in historial], color="#eb6834",
            linewidth=1, alpha=0.8, label="promedio de la población")
    ax.plot(g, [h["mejor_heuristica_poblacion"] for h in historial], color="#1baf7a",
            linewidth=1, label="mejor de la población actual")
    ax.plot(g, [h["mejor_heuristica_historica"] for h in historial], color="#2a78d6",
            linewidth=2.5, label="mejor heurística histórica")
    ax.set_xlabel("generación", color="#52514e")
    ax.set_ylabel("heurística de error promedio penalizado (mpg)", color="#52514e")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#52514e")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.legend(frameon=False, labelcolor="#52514e")
    ax.set_title("Evolución del algoritmo genético (menor es mejor)")
    fig.tight_layout()
    archivo = ruta / "evolucion_heuristica.png"
    fig.savefig(archivo, dpi=120)
    plt.close(fig)
    return archivo
