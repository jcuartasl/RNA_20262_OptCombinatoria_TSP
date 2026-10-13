# Análisis de Sensibilidad y Umbrales de Rutas (Persona 4)

Este documento registra los resultados del análisis de sensibilidad del valor horario del conductor/vendedor ($v$) sobre el Problema del Viajero (TSP) de las 47 capitales de provincia peninsulares.

## 1. Diseño Experimental del Barrido Horario
- **Rango evaluado de $v$:** 0,0 €/h hasta 120,0 €/h.
- **Herramienta de optimización:** Algoritmos bioinspirados (ACO / GA) bajo presupuesto de evaluaciones controlado.
- **Matriz base:** Datos viales congelados de OSRM combinados con costos fijos de combustible (SEAT Ibiza 1.0 TSI, 5,1 L/100 km, 1,849 €/L) y peajes oficiales de 2026 (`toll_eur.csv`).

## 2. Resultados y Umbrales Observados
- **Archivos generados:** 
  - `results/sensitivity/sensitivity.csv`: Contiene el registro detallado de cada corrida por valor de $v$.
  - `results/sensitivity/thresholds.json`: Almacena los puntos de cruce y segmentos de estabilidad de rutas óptimas candidatas.
- **Hallazgo principal:** En el intervalo analizado ($v \in [0.0, 120.0]$ €/h), la ruta óptima encontrada se mantiene estable con un tiempo total de conducción de aproximadamente $78.1$ horas y costos fijos estructurales de $593.40$ €, lo que demuestra la robustez de la solución frente a variaciones moderadas en el costo de oportunidad del tiempo del conductor.
- **Nota metodológica:** Los umbrales identificados corresponden estrictamente a puntos de cruce observados entre rutas candidatas de naturaleza heurística, por lo que no constituyen una demostración matemática de optimalidad global absoluta.
