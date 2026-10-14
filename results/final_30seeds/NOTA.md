# Benchmark de 30 seeds: resultados provisionales

- **Ejecución:** 2026-10-13, de 19:38:58 a 20:47:08 (68 min 10 s), en el commit `ce3a102`.
- **Comando:** `python -m src.experiments.benchmark --config config/base.yaml --out results/final_30seeds`.
- **Corridas:** 30 seeds (0–29) × 10 valores de v (0, 5, 10, 15, 20, 30, 45, 60, 90, 120 €/h) × ACO y GA = 600.
  Todas con budget de 50 000 evaluaciones.
- **Máquina:** Windows 11, Intel64 Family 6 Model 126, Python 3.12.10. El campo `elapsed_s` depende de la máquina.
- **Datos de entrada:**
  - matrices de distancia y tiempo: versión 1.0.0, hash `e30063cc2c9856cb` (cubre solo esas dos matrices);
  - peajes: `toll_eur.csv` con `git hash-object` `df1380a98d20…`;
  - vehículo: `config/sensitivity.yaml` con `git hash-object` `558e4f38952a…`.

**Los peajes son provisionales.** `benchmark_runs.json` dice `"provisional_tolls": false` y `"tolls_complete": true`
porque `toll_eur.csv` no tiene celdas vacías, no porque esté verificado. La matriz se armó a mano y solo 8 de
2162 arcos tienen peaje; la sección "Limitaciones" del README explica el detalle.

Todo lo que depende de los peajes (desglose de costos, peajes por ruta y si la ruta cambia con v) puede cambiar
cuando se corrija la matriz. Si eso pasa, este benchmark debe repetirse. La comparación ACO vs GA bajo el mismo
budget depende menos de los peajes, pero también debe repetirse.

Las rutas son las mejores encontradas por las heurísticas, no óptimos demostrados.
