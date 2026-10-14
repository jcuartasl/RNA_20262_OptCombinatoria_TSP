# Registro de Decisiones de Fase 0 (Parte 2: TSP Peninsular de España)

Este documento recopila las decisiones metodológicas y técnicas acordadas por **Omar (Datos Viales)** durante la fase de diseño para ser socializadas e integradas con el equipo.

---

## Decisión D-01: Lista Canónica e Indexación de las 47 Ciudades
- **Propuesta aprobada:** Orden alfabético por el nombre de la capital (desde `0: A Coruña` hasta `46: Zaragoza`).
- **Justificación:** Elimina toda ambigüedad entre integrantes al indexar matrices. Todas las filas y columnas de las matrices de distancias, tiempos, peajes, combustible y costo total tendrán idéntico orden.
- **Exclusiones verificadas:** Palma, Las Palmas de Gran Canaria, Santa Cruz de Tenerife, Ceuta y Melilla están estrictamente excluidas de los datos.

---

## Decisión D-02: Fuente de Rutas por Carretera
- **Propuesta aprobada:** Open Source Routing Machine (OSRM) público con perfil `driving` sobre la red vial de OpenStreetMap.
- **Justificación:**
  - Evita credenciales o tokens privados incrustados en el código.
  - Genera matrices viales asimétricas basadas en autovías y carreteras reales, no distancias geodésicas (Haversine).
  - Los $2{,}162$ pares dirigidos fueron extraídos y congelados en local para garantizar ejecución 100% offline.

---

## Decisión D-03: Contrato de Serialización de Matrices
- **Propuesta aprobada:** Archivos CSV (`distance_km.csv`, `time_h.csv`) con encabezados de columnas e índices de fila con los nombres de las ciudades.
- **Diagonal:** Fijada exactamente en `0.0` (sin valores NaN ni nulos) para máxima compatibilidad con NumPy y Pandas.
- **Unidades:** Kilómetros (km) para distancias y horas (h) para tiempos de viaje.

---

## Decisión D-04: Fixture de 5 Ciudades para Desarrollo Temprano (Fase 0)
- **Propuesta aprobada:** Subconjunto canónico de 5 capitales principales: Madrid, Barcelona, Valencia, Sevilla y Zaragoza.
- **Archivos generados:** `data/fixtures/toy_5cities.csv`, `toy_5cities_distance_km.csv`, `toy_5cities_time_h.csv`.
- **Propósito:** Permitir que Persona 2 (ACO), Persona 3 (GA), Persona 4 (Costos) y Persona 5 (Runner) desarrollen y ejecuten pruebas unitarias locales inmediatas sin esperar la integración completa de las 47 ciudades.

---

## Decisión D-05: Entorno Aislado de Preparación
- **Propuesta aprobada:** Todo el desarrollo se ubica en `staging_tsp_data/`, manteniendo el repositorio clonado `RNA_20262_OptCombinatoria_TSP` limpio hasta que el equipo defina la rama base y la estructura inicial.
