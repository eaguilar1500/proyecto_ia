"""
Reglas encontradas por el algoritmo genético: el mejor conjunto de reglas
encontrado durante la ejecución, en tabla (CSV) y en texto.
"""

from pathlib import Path

import pandas as pd

from src.sistema_mamdani import (
    COMBINACIONES,
    CONSECUENTES,
    N_GENES,
    texto_linguistico,
    texto_regla,
)


def tabla_reglas(cromosoma):
    """regla | gen | horsepower | weight | acceleration | consecuente | ctrl_rule"""
    filas = []
    for gen in range(N_GENES):
        if cromosoma[gen] == 0:
            continue
        p, w, a = COMBINACIONES[gen]
        filas.append({"regla": f"R{len(filas) + 1}", "gen": f"G{gen}", "horsepower": p,
                      "weight": w, "acceleration": a,
                      "consecuente": CONSECUENTES[int(cromosoma[gen])],
                      "ctrl_rule": texto_regla(gen, cromosoma[gen])})
    return pd.DataFrame(filas)


def guardar_reglas_txt(cromosoma, archivo):
    """Mejor cromosoma y sus reglas (lingüísticas y ctrl.Rule) en texto."""
    genes = [g for g in range(N_GENES) if cromosoma[g] != 0]
    lineas = ["Mejor conjunto de reglas encontrado durante la ejecución del algoritmo genético",
              f"Cromosoma: {[int(g) for g in cromosoma]}",
              f"Reglas: {len(genes)} de {N_GENES}", "", "Reglas lingüísticas:"]
    lineas += [f"  R{n} (G{g}): {texto_linguistico(g, cromosoma[g])}"
               for n, g in enumerate(genes, start=1)]
    lineas += ["", "Reglas en skfuzzy:"]
    lineas += [f"regla{n} = {texto_regla(g, cromosoma[g])}" for n, g in enumerate(genes, start=1)]
    Path(archivo).write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return archivo

