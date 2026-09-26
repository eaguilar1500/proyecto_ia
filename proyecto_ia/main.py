"""
Sistema difuso Mamdani para la eficiencia de combustible (Auto MPG) con una
base de reglas encontrada por un algoritmo genético.

Este archivo sigue el mismo orden de la explicación (PASO 1 ... PASO 12).
La consola solo muestra una línea por etapa; el detalle queda en:
    resultados/tablas/   resultados/figuras/   resultados/reglas_finales.txt

Ejecutar desde la raíz del proyecto con:  python main.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skfuzzy import control as ctrl

from src import algoritmo_genetico as ag
from src.carga_datos import SALIDA, cargar_datos, limpiar_datos, seleccionar_variables
from src.exploracion import correlaciones, estadisticas, graficar_distribuciones
from src.metricas_reglas import tabla_comparacion, tabla_metricas
from src.proceso_mamdani import (
    elegir_auto_ejemplo,
    graficar_fuzzificacion,
    graficar_mamdani,
    proceso_completo,
    tabla_fuzzificacion,
    tabla_reglas_activadas,
    tabla_resumen,
)
from src.reglas_finales import guardar_reglas_txt, tabla_reglas
from src.sistema_mamdani import (
    COMBINACIONES,
    CONSECUENTES,
    N_GENES,
    EvaluadorNumpy,
    comprobar_contra_skfuzzy,
    predecir_skfuzzy,
)
from src.variables_difusas import (
    crear_variables,
    definir_conjuntos,
    graficar_funciones,
    tabla_rangos,
    verificar_particion,
)

RESULTADOS = Path(__file__).resolve().parent / "resultados"
TABLAS = RESULTADOS / "tablas"
FIGURAS = RESULTADOS / "figuras"


def guardar(tabla, nombre, index=False):
    TABLAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(TABLAS / nombre, index=index, encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")   # la consola de Windows no admite μ, α

    # ============================================================
    # PASO 1 - CARGAR Y LIMPIAR EL DATASET
    # 398 registros -> eliminar los 6 sin horsepower -> 392 registros
    # ============================================================
    df_original = cargar_datos()
    df = limpiar_datos(df_original)
    datos = seleccionar_variables(df)           # horsepower, weight, acceleration, mpg
    print(f"[1] Dataset: {len(df_original)} -> {len(datos)} registros válidos")

    # ============================================================
    # PASO 2 - OBTENER ESTADÍSTICAS DEL DATASHEET
    # mínimo, Q1, mediana, Q3, máximo (y forma de la distribución)
    # ============================================================
    definicion = definir_conjuntos(datos)
    guardar(tabla_rangos(datos), "estadisticas_rangos.csv")
    guardar(estadisticas(datos).round(3), "estadisticas_descriptivas.csv", index=True)
    guardar(correlaciones(df.drop(columns="car_name")).round(3), "correlaciones.csv", index=True)
    graficar_distribuciones(datos, FIGURAS)
    print("[2] Rangos fuzzy desde mín/Q1/mediana/Q3/máx -> tablas/estadisticas_rangos.csv")

    # ============================================================
    # PASO 3 - CREAR FUNCIONES DE PERTENENCIA
    # BAJO trapmf[mín, mín, Q1, mediana] | MEDIO trimf[Q1, mediana, Q3]
    # ALTO trapmf[mediana, Q3, máx, máx]
    # ============================================================
    variables = crear_variables(datos)
    graficar_funciones(variables, datos, FIGURAS)
    suma_1 = all(abs(a - 1) < 1e-9 and abs(b - 1) < 1e-9
                 for a, b in verificar_particion(variables).values())
    print(f"[3] Funciones de pertenencia creadas (suma de grados = 1: "
          f"{'sí' if suma_1 else 'NO'}) -> figuras/funciones_pertenencia.png")

    # ============================================================
    # PASO 4 - MOSTRAR FUZZIFICACIÓN DE EJEMPLO
    # entrada numérica -> grado bajo / medio / alto
    # ============================================================
    i = elegir_auto_ejemplo(definicion, datos)
    auto, nombre = datos.loc[i], df.loc[i, "car_name"]
    guardar(tabla_fuzzificacion(definicion, variables, auto), "ejemplo_fuzzificacion.csv")
    graficar_fuzzificacion(definicion, variables, auto, nombre, FIGURAS)

    # ============================================================
    # PASO 5 - EJECUTAR EL ALGORITMO GENÉTICO
    # (las 14 fases están en src/algoritmo_genetico.py)
    # ============================================================
    evaluador = EvaluadorNumpy(variables, datos)          # optimización interna
    heuristica = ag.Heuristica(evaluador, datos[SALIDA])
    mejor_historico, historial, ejemplo, motivo = ag.ejecutar(heuristica)
    mejor = mejor_historico["cromosoma"]       # las reglas finales salen de aquí
    mejor_h, error_cubiertos, sin_salida = heuristica.detalle(mejor)
    guardar(pd.DataFrame(historial), "evolucion_genetico.csv")
    guardar(ag.tabla_seleccion(ejemplo), "ejemplo_seleccion_genetica.csv")
    guardar(ag.tabla_cruzamiento(ejemplo), "ejemplo_cruzamiento.csv")
    ag.graficar_evolucion(historial, FIGURAS)
    # heurística: mejor de la población inicial -> mejor histórica (menor = mejor)
    print(f"[4] Algoritmo genético: {len(historial) - 1} generaciones | heurística "
          f"(menor = mejor): {historial[0]['mejor_heuristica_poblacion']:.3f} -> "
          f"{mejor_h:.3f} | mejor en gen. {mejor_historico['generacion']} | "
          f"sin salida: {sin_salida}/{len(datos)} | parada: {motivo}")

    # ============================================================
    # PASO 6 - MOSTRAR LAS REGLAS ENCONTRADAS
    # (mejor conjunto de reglas encontrado durante la ejecución)
    # ============================================================
    guardar(tabla_reglas(mejor), "reglas_genetico.csv")
    guardar_reglas_txt(mejor, RESULTADOS / "reglas_finales.txt")
    print(f"[5] Mejor base de reglas encontrada: {int(np.count_nonzero(mejor))} de {N_GENES} "
          f"-> tablas/reglas_genetico.csv, reglas_finales.txt")

    # ============================================================
    # PASO 7 - CALCULAR SUPPORT / CONFIDENCE / LIFT (solo análisis)
    # ============================================================
    metricas = tabla_metricas(mejor, variables, datos)
    comparacion = tabla_comparacion(mejor, variables, datos)
    guardar(metricas, "metricas_reglas.csv")
    guardar(comparacion, "comparacion_consecuentes.csv")
    print(f"[6] Support/confidence/lift -> tablas/metricas_reglas.csv | asociación "
          f"positiva (lift > 1): {(metricas['lift'] > 1).sum()}/{len(metricas)}")

    # ============================================================
    # PASO 8 - CONSTRUIR LAS REGLAS ctrl.Rule
    # cada gen distinto de 0 del mejor cromosoma -> una regla
    # ============================================================
    potencia = variables["horsepower"]
    peso = variables["weight"]
    aceleracion = variables["acceleration"]
    eficiencia = variables["mpg"]

    reglas = []
    for gen in range(N_GENES):
        if mejor[gen] == 0:
            continue                                  # 0 = no existe regla
        p, w, a = COMBINACIONES[gen]
        regla = ctrl.Rule(
            potencia[p]
            & peso[w]
            & aceleracion[a],
            eficiencia[CONSECUENTES[int(mejor[gen])]],
            label=f"R{len(reglas) + 1}",
        )
        reglas.append(regla)

    # ============================================================
    # PASO 9 - CREAR ControlSystem Y ControlSystemSimulation
    # ============================================================
    sistema = ctrl.ControlSystem(reglas)
    simulacion = ctrl.ControlSystemSimulation(sistema, cache=False)

    # Comprobación interna: la optimización NumPy reproduce el mismo Mamdani
    # que skfuzzy, y la heurística del AG es la misma con ambos cálculos.
    rng = np.random.default_rng(7)
    dif, mismos = comprobar_contra_skfuzzy(
        variables, datos, [mejor] + [rng.integers(0, 4, N_GENES) for _ in range(2)])
    estimado = predecir_skfuzzy(mejor, variables, datos)
    heur_skfuzzy = np.where(np.isnan(estimado), heuristica.rango_mpg,
                            np.abs(datos[SALIDA].to_numpy() - estimado)).mean()
    print(f"[7] Sistema Mamdani: {len(reglas)} reglas ctrl.Rule | "
          f"comprobación NumPy = skfuzzy: "
          f"{'sí' if mismos and dif < 1e-9 and abs(heur_skfuzzy - mejor_h) < 1e-9 else 'NO'}")

    # ============================================================
    # PASO 10 - EJECUTAR EL EJEMPLO MAMDANI (mismo auto del PASO 4)
    # fuzzificación -> alpha (mínimo) -> recorte -> agregación (máximo)
    # ============================================================
    resultado = proceso_completo(definicion, variables, mejor, simulacion, auto)

    # ============================================================
    # PASO 11 - CENTROIDE: MPG FINAL
    # ============================================================
    print(f"[8] Ejemplo Mamdani ({nombre}): MPG real {auto[SALIDA]:g} -> centroide "
          f"{resultado['mpg_skfuzzy']:.2f} | {len(resultado['activas'])} reglas activadas")

    # ============================================================
    # PASO 12 - GUARDAR TABLAS Y FIGURAS DEL EJEMPLO
    # ============================================================
    guardar(tabla_reglas_activadas(resultado), "ejemplo_mamdani_reglas.csv")
    guardar(tabla_resumen(resultado), "ejemplo_mamdani_resumen.csv")
    graficar_mamdani(variables, resultado, nombre, FIGURAS)
    print(f"\nTablas en {TABLAS}\nFiguras en {FIGURAS}")


if __name__ == "__main__":
    main()
