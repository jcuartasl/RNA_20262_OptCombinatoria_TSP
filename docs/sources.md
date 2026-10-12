# Procedencia de Datos, Metodología Vial y Fuentes (Parte 2: TSP Peninsular de España)

**Autor:** Omar (Datos: Ciudades, Rutas Viales, Tiempos, Distancias y Fuentes)  
**Curso:** Redes Neuronales y Algoritmos Bioinspirados  
**Semestre:** 2026-02 — Universidad Nacional de Colombia  

---

## 1. Definición del Universo de Ciudades

De acuerdo con el enunciado oficial del trabajo (pág. 2) y la Guía Operativa de la Parte 2 (§1), el problema combinatorio del vendedor viajero (TSP) se formula sobre **las 47 capitales de provincia de la España peninsular**:
- **Provincias peninsulares (47):** A Coruña, Albacete, Alicante, Almería, Ávila, Badajoz, Barcelona, Bilbao, Burgos, Cáceres, Cádiz, Castellón de la Plana, Ciudad Real, Córdoba, Cuenca, Girona, Granada, Guadalajara, Huelva, Huesca, Jaén, León, Lleida, Logroño, Lugo, Madrid, Málaga, Murcia, Ourense, Oviedo, Palencia, Pamplona, Pontevedra, Salamanca, San Sebastián, Santander, Segovia, Sevilla, Soria, Tarragona, Teruel, Toledo, Valencia, Valladolid, Vitoria-Gasteiz, Zamora y Zaragoza.
- **Exclusiones mandatorias por el profesor:**
  - **Palma (Mallorca, Illes Balears):** Territorio insular no accesible por carretera peninsular.
  - **Las Palmas de Gran Canaria y Santa Cruz de Tenerife (Canarias):** Territorios insulares no accesibles por carretera.
  - **Ceuta y Melilla:** Ciudades autónomas situadas en el norte de África, no categorizadas como provincias ni conectadas por red vial continua con la península.

Las coordenadas geográficas (latitud, longitud en WGS84) se centraron en los núcleos urbanos y enlaces con la red vial principal, verificadas a partir de la infraestructura de datos del Instituto Geográfico Nacional (IGN) y OpenStreetMap.

---

## 2. Motor de Enrutamiento y Obtención de Matrices

Para modelar la red vial de forma fidedigna y evitar la alucinación de calcular distancias euclidianas o geodésicas ortodrómicas (Haversine), se empleó el motor **Open Source Routing Machine (OSRM)** sobre la base de datos vial de **OpenStreetMap (OSM)**:
- **Servicio:** OSRM Table Service (`/table/v1/driving`).
- **Perfil de enrutamiento:** `driving` (vehículo turismo por carreteras asfaltadas, autovías y autopistas).
- **Parámetros solicitados:** `annotations=distance,duration`.
- **Pares dirigidos consultados:** $47 \times 46 = 2{,}162$ trayectos dirigidos ($i \to j, \, i \ne j$).
- **Conversión de unidades:**
  - Distancias en metros convertidas a **kilómetros (km)** dividiendo por $1{,}000$.
  - Tiempos en segundos convertidos a **horas (h)** dividiendo por $3{,}600$.

### 2.1 Propiedades de las Matrices Obtenidas
- **Asimetría vial real:** A diferencia de las distancias euclidianas, las matrices son asimétricas ($D_{ij} \ne D_{ji}$ y $T_{ij} \ne T_{ji}$). Las autovías de acceso, enlaces de circunvalación, vías de un solo sentido y pendientes orográficas generan diferencias reales entre el trayecto de ida y vuelta.
- **Diagonal definida:** Se fijó estrictamente $D_{ii} = 0.0$ y $T_{ii} = 0.0$. La diagonal no representa un traslado válido entre ciudades distintas.
- **Velocidades medias implícitas:** La velocidad media calculada ($D_{ij} / T_{ij}$) oscila entre $57.8 \text{ km/h}$ y $97.1 \text{ km/h}$, con una media global de $87.5 \text{ km/h}$, lo cual es coherente con los límites y trazados de la red de carreteras del Estado en España.

---

## 3. Almacenamiento Offline y Reproducibilidad

El enunciado estipula como requisito indispensable:
> *"Los datos del TSP (coordenadas, matriz de costos y sus componentes) guardados en el repositorio, para que la ejecución no dependa de conexión a internet ni de APIs externas."*

Por consiguiente:
1. Las matrices brutas obtenidas de OSRM se congelaron en los archivos versionados:
   - `data/processed/cities.csv`
   - `data/processed/distance_km.csv`
   - `data/processed/time_h.csv`
2. Los cargadores de Python (`load_cities()`, `load_distance_matrix()`, `load_time_matrix()`) en `src/data/road_matrix.py` leen **exclusivamente de disco local** sin emitir llamadas de red, permitiendo que las pruebas y los experimentos de optimización (ACO, GA) corran completamente offline y en menos de dos minutos.
3. Se generó un *fixture* reducido de 5 ciudades principales (`data/fixtures/toy_5cities.csv`) para permitir que los demás integrantes (Persona 2: ACO, Persona 3: GA, Persona 4: Costos y Persona 5: Runner) ejecuten pruebas de integración sin requerir la matriz de 47 nodos.

---

## 4. Riesgos de Rutas Alternativas y Limitaciones

1. **Autopistas de peaje vs. autovías libres:**  
   En España, tras los rescates y levantamientos de peajes entre 2018 y 2021 (como la AP-7 en Cataluña/Levante, AP-1 en Burgos-Armiñón y tramos de la AP-2), varias autopistas pasaron a ser gratuitas. OSRM con perfil `driving` selecciona la ruta más rápida considerando límites de velocidad, lo que en algunos corredores puede priorizar vías de peaje remanentes (ej. túneles o radiales de acceso) y en otros autovías libres. La Persona 4 (Costos y Peajes) debe cruzar estos arcos con las tarifas vigentes del Ministerio de Transportes y Movilidad Sostenible.
2. **Tráfico dinámico:**  
   El grafo de OSRM utiliza tiempos de conducción basados en límites de velocidad y jerarquía vial teórica, sin congestión en tiempo real por accidentes o congestión estacional. Esta abstracción estática es la estándar para modelos de optimización combinatoria tipo TSP.

---

## 5. Referencias Bibliográficas (Formato APA 7.ª Edición)

- Instituto Geográfico Nacional. (2024). *Información geográfica de referencia: Municipios y poblaciones de España*. Ministerio de Transportes y Movilidad Sostenible. https://www.ign.es
- Luxen, D., & Vetter, C. (2011). Real-time routing with OpenStreetMap data. En *Proceedings of the 19th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems* (pp. 513–516). ACM. https://doi.org/10.1145/2093973.2094062
- OpenStreetMap Contributors. (2024). *Planet dump road network*. OpenStreetMap Foundation. https://www.openstreetmap.org
