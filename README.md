# TSP por las 47 capitales peninsulares de España: ACO y algoritmo genético

Trabajo 1, Parte 2 (Optimización combinatoria) del curso *Redes Neuronales y Algoritmos Bioinspirados*,
Universidad Nacional de Colombia, 2026-02.

Un vendedor recorre las 47 capitales de provincia de la España peninsular y vuelve a la ciudad de salida.
El costo de ir de la capital *i* a la *j* (arco dirigido, matrices asimétricas) es

```
C_ij(v) = v · T_ij + P_ij + F_ij
```

donde `v` es el valor de la hora del vendedor (€/h), `T` el tiempo por carretera (h), `P` los peajes (€) y
`F` el combustible (€) del vehículo elegido. El problema se resuelve con colonias de hormigas (ACO) y con un
algoritmo genético (GA) bajo el mismo presupuesto de evaluaciones de ruta. Las rutas que se reportan son las
**mejores encontradas** por heurísticas, no óptimos demostrados.

## Instalación

Requiere **Python 3.12** (probado con 3.12.10 en Windows 11). Las versiones de las dependencias están fijas en
`requirements.txt`.

```bash
git clone https://github.com/jcuartasl/RNA_20262_OptCombinatoria_TSP.git
cd RNA_20262_OptCombinatoria_TSP
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt      # Windows
.venv/bin/python -m pip install -r requirements.txt          # Linux / macOS
```

En los comandos de abajo, `python` es el del entorno: `.venv\Scripts\python` en Windows y `.venv/bin/python`
en Linux / macOS. Todos se corren desde la raíz del repositorio.

**Máquina sin internet:** instalar las dependencias requiere red una sola vez. Después, todo corre sin red:
las matrices, coordenadas y parámetros están guardados en el repositorio.

## Demo (un solo comando, menos de 2 minutos)

```bash
python -m src.experiments.demo
```

Con `config/demo.yaml` (v = 15 €/h, seeds 0 y 1, budget 50 000 evaluaciones) hace, en orden:

1. Benchmark corto: ACO y GA con las dos seeds (4 corridas).
2. Una corrida de ACO (seed 0) que guarda el historial completo.
3. El mapa de esa ruta, el GIF de su evolución y las figuras del benchmark.

Escribe todo en `results/demo/` y al final imprime el tiempo total. Si pasa de 120 s avisa, pero no falla.
Tiempo medido: **31 s** (Windows 11, Intel64 Family 6 Model 126, Python 3.12.10), con la red bloqueada.
La primera vez en un entorno nuevo puede tardar unos segundos más mientras matplotlib arma su caché de fuentes.

### Resultados esperados de la demo

Con los mismos datos y las mismas versiones de `requirements.txt`, la demo debe dar exactamente estos costos:

| Algoritmo | seed 0 | seed 1 |
|---|---|---|
| ACO | 1830.61 € | 1819.36 € |
| GA | 1837.77 € | 1793.01 € |

Estos valores **solo aplican** a esta versión de los datos de entrada. Si cambia alguno, cambian los costos:

| Entrada | Identificador |
|---|---|
| Matrices de distancia y tiempo (OSRM) | `dataset_version` 1.0.0, `dataset_hash_sha256_short` `e30063cc2c9856cb` en `data/processed/metadata.json` |
| Peajes `data/processed/toll_eur.csv` | `git hash-object` → `df1380a98d20…` |
| Vehículo y combustible `config/sensitivity.yaml` | `git hash-object` → `558e4f38952a…` |
| Parámetros `config/aco.yaml` / `config/genetic.yaml` | `git hash-object` → `220f042a8361…` / `89dc7628e164…` |

El hash de `metadata.json` cubre **solo** las matrices de distancia y tiempo. Por eso los peajes y el
vehículo se identifican aparte. Para comprobarlos: `git hash-object data/processed/toll_eur.csv`.

## Comandos

Cada experimento se lanza con un solo comando. Los parámetros se cambian por línea de comandos o en los YAML,
sin editar código.

| Qué | Comando | Salida |
|---|---|---|
| Una corrida | `python -m src.experiments.run_tsp --optimizer aco --config config/demo.yaml` | `results/raw/aco_v15_seed0.json` |
| Benchmark completo (30 seeds × 10 valores de v × 2 algoritmos, ~40–50 min) | `python -m src.experiments.benchmark --config config/base.yaml` | `results/raw/benchmark_runs.json`, `results/summary/benchmark_summary.csv` |
| Benchmark rápido | `python -m src.experiments.benchmark --seeds 0 1 --hourly_values 15 60` | ídem |
| Mapa de una ruta | `python -m src.visualization.map_routes --results results/raw/aco_v15_seed0.json` | `results/figures/route_aco_v15_seed0.png` |
| Mapa de una corrida del benchmark | `python -m src.visualization.map_routes --results results/raw/benchmark_runs.json --optimizer genetic --hourly_value 15` | ídem (sin `--seed`, la de menor costo) |
| GIF de la evolución | `python -m src.visualization.animation --results results/raw/aco_v15_seed0.json` | `results/gifs/aco_v15_seed0.gif` |
| Figuras del benchmark | `python -m src.visualization.plots --benchmark results/raw/benchmark_runs.json --hourly_value 15` | `results/figures/convergence_v15.png`, `final_cost_boxplot.png`, `cost_vs_v.png`, `cost_breakdown_vs_v.png`, `cost_diff_vs_v.png` |
| Sensibilidad y umbrales (Persona 4) | `python -m src.costs.sensitivity --config config/sensitivity.yaml` | `results/sensitivity/sensitivity.csv`, `thresholds.json` |
| Pruebas | `python -m pytest` | — |

El GIF sale del `history` guardado en el JSON de `run_tsp`, sin volver a optimizar. `benchmark_runs.json` no
guarda ese historial (solo una curva compacta por corrida), así que el GIF se hace desde una corrida de
`run_tsp`.

### Cambiar parámetros (también en vivo)

```bash
python -m src.experiments.run_tsp --optimizer genetic --config config/demo.yaml --hourly_value 60 --seed 3 --budget 20000
python -m src.experiments.run_tsp --optimizer aco --algo_config mi_aco.yaml     # copia de config/aco.yaml con otro rho, alpha, beta...
python -m src.visualization.animation --results results/raw/aco_v15_seed0.json --fps 6 --max_frames 30 --hold_s 4
```

## Configuración

- `config/aco.yaml` y `config/genetic.yaml`: parámetros de cada algoritmo (ACO: `n_ants`, `alpha`, `beta`,
  `rho`, `q`, `eps`; GA: `population_size`, `crossover_rate`, `mutation_rate`, `tournament_size`, `elite_count`).
- `config/base.yaml`: experimento completo (v = 0, 5, 10, 15, 20, 30, 45, 60, 90, 120 €/h; seeds 0–29; budget 50 000).
- `config/demo.yaml`: demo (v = 15 €/h, seeds 0 y 1, budget 50 000).
- `config/sensitivity.yaml`: vehículo y precio del combustible con sus fuentes. `base.yaml` y `demo.yaml` lo
  leen a través de `vehicle_config`.

El presupuesto se mide en **evaluaciones de ruta completas** y no en iteraciones, para que ACO y GA se comparen en
igualdad: una iteración de ACO y una generación de GA evalúan cantidades distintas de rutas. Con 50 000
evaluaciones los dos algoritmos se estabilizan; con 10 000, GA todavía no converge.

## Datos

`data/processed/` contiene la lista de 47 capitales con coordenadas (`cities.csv`), las matrices 47 × 47 de
distancia (km), tiempo (h) y peajes (€), y `metadata.json`. Las distancias y los tiempos salen de OSRM (perfil
`driving` sobre OpenStreetMap) y se consultaron una sola vez. Las fuentes y fechas de cada dato están en
[docs/sources.md](docs/sources.md) y las decisiones del equipo en [docs/decisions.md](docs/decisions.md).

## Salidas

| Carpeta | Contenido |
|---|---|
| `results/raw/` | JSON de cada corrida (`run_tsp`) y del benchmark |
| `results/summary/` | Resumen estadístico del benchmark por algoritmo y valor de v |
| `results/figures/` | Mapas y figuras |
| `results/gifs/` | Animaciones |
| `results/demo/` | Todo lo que produce la demo |
| `results/sensitivity/` | Barrido de sensibilidad y umbrales (Persona 4) |

Cada JSON guarda la seed, el budget, el valor de v, la configuración usada, la política de peajes y la versión
del dataset. El del benchmark guarda además el sistema operativo, el procesador y la versión de Python, porque el
tiempo de cómputo (`elapsed_s`) depende de la máquina. Los costos y las rutas no dependen de ella.

## Limitaciones

- **Rutas heurísticas.** ACO y GA no garantizan el óptimo. Los umbrales de cambio de ruta en función de v son
  observados entre las rutas candidatas encontradas, no demostrados.
- **Mapa esquemático.** Las figuras unen las capitales con segmentos rectos sobre longitud y latitud. No es la
  geometría de las carreteras, y cada figura lo dice.
- **Matriz de peajes provisional, pendiente de revisión.** Los resultados que dependen de los peajes no deben
  tomarse como definitivos. En `data/processed/toll_eur.csv`:
  - Todos los arcos se fijaron en 0 € por un "criterio de red libre", y luego se agregaron a mano los peajes de
    algunos arcos directos entre capitales por la AP-6, AP-51, AP-61 y AP-66
    ([docs/sources.md](docs/sources.md) §4.4). Solo 8 de los 2162 arcos dirigidos tienen peaje mayor que 0.
  - No salió de la ruta que recorre cada arco: las matrices de OSRM no incluyen el trazado, así que no se sabe qué
    autopistas atraviesa cada uno.
  - No tiene fuente concreta ni fecha de consulta (solo "peajes oficiales actualizados al 1 de enero de 2026"), y
    `data/raw/sources_manifest.csv` no tiene ninguna fila de peajes.
  - Es internamente inconsistente: Madrid → León y León → Oviedo cobran 16,20 € cada uno, pero Madrid → Oviedo
    (445 km, que por distancia pasa por ese corredor) vale 0 €. Del mismo modo, Madrid → Valladolid, Salamanca,
    Zamora y A Coruña valen 0 € aunque la propia matriz cobra la AP-6 en Madrid → Segovia, Ávila y León.
  - Arcos como A Coruña → Pontevedra o Bilbao → Zaragoza valen 0 €, y está pendiente verificar si su ruta usa
    autopistas de peaje.

  Por eso la observación de que la ruta preferida casi no cambia con v está **condicionada a esta matriz**: puede
  deberse a que faltan peajes y no a la red real.
- **Tiempos estáticos.** OSRM usa límites de velocidad y jerarquía vial, sin tráfico real.

## Estructura

```
config/            parámetros (base, demo, aco, genetic, sensitivity)
data/processed/    matrices y coordenadas congeladas (uso offline)
data/fixtures/     instancia de 5 ciudades para pruebas
src/data/          carga y validación de datos (Persona 1)
src/optimizers/    contrato común, ACO (Persona 2) y GA (Persona 3)
src/costs/         matriz de costos, vehículo y sensibilidad (Persona 4)
src/experiments/   runner, benchmark, exportación y demo (Persona 5)
src/visualization/ mapa, GIF y figuras (Persona 5)
tests/             pruebas (pytest)
docs/              fuentes, decisiones, reporte de sensibilidad
```
