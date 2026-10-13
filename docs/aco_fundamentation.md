# Optimización por Colonia de Hormigas (ACO)

## 1. Ecuaciones de actualización

Se minimiza el costo del ciclo $L(\pi)=\sum_{k=1}^{n}C_{\pi_k\pi_{k+1}}$, con $\pi_{n+1}=\pi_1$, sobre las permutaciones $\pi$ de las $n$ ciudades. La matriz de costos $C\in\mathbb{R}^{n\times n}_{\ge 0}$ es finita y no necesariamente simétrica. Se usa una colonia de $m$ hormigas (*Ant System*). La matriz de feromona $\boldsymbol\tau$ ($n\times n$) es compartida; cada hormiga $k$ construye una ruta $\pi^k$ y el algoritmo conserva la mejor ruta global $\pi^*$ (*best-so-far*).

**Inicialización:**

$$\eta_{ij}=\frac{1}{\max(C_{ij},\varepsilon)}\ (i\neq j),\qquad \eta_{ii}=0,\qquad \bar\eta_{ij}=\left(\frac{\eta_{ij}}{\max_{a\neq b}\eta_{ab}}\right)^{\beta}$$

$$\tau_{ij}^{0}=\frac{m}{L_{nn}}\ \ \forall\, i,j,\qquad \pi^*=\varnothing,\qquad L^*=+\infty$$

donde $L_{nn}$ es el costo de la ruta del vecino más cercano partiendo de la ciudad 0. Sólo fija la escala inicial de la feromona.

**Ciudad inicial** (para cada hormiga $k$):

$$s_k\sim U\{1,\dots,n\}$$

**Probabilidad de transición** (hormiga $k$ en la ciudad $i$; $J_k(i)$ = ciudades aún no visitadas por $k$):

$$p^{k}_{ij}=\frac{\tilde\tau_{ij}^{\,\alpha}\;\bar\eta_{ij}}{\sum_{l\in J_k(i)}\tilde\tau_{il}^{\,\alpha}\;\bar\eta_{il}}\quad (j\in J_k(i)),\qquad \tilde\tau_{ij}=\frac{\tau_{ij}}{\max_{a,b}\tau_{ab}}$$

Las normalizaciones de $\boldsymbol\tau$ y $\bar\eta$ son factores constantes que se cancelan en el cociente: no cambian las probabilidades, sólo evitan desbordamientos con $\alpha$ y $\beta$ altos. Las ciudades visitadas tienen probabilidad exactamente 0. La ciudad siguiente se elige por ruleta.

**Costo de la ruta:**

$$L_k=L(\pi^k)=\sum_{t=1}^{n}C_{\pi^k_t\,\pi^k_{t+1}},\qquad \pi^k_{n+1}=\pi^k_1$$

**Evaporación** (cada arco $(i,j)$):

$$\tau_{ij}\leftarrow(1-\rho)\,\tau_{ij}$$

**Depósito** (cada hormiga deposita sobre los arcos dirigidos de su ruta, incluido el de cierre $\pi^k_n\to\pi^k_1$):

$$\Delta\tau^{k}_{ij}=\begin{cases}\dfrac{Q}{\max(L_k,\varepsilon)} & \text{si } (i\to j)\in\pi^k\\[6pt] 0 & \text{en otro caso}\end{cases}$$

**Actualización de feromona** (una vez por iteración, después de evaluar las $m$ rutas):

$$\tau_{ij}\leftarrow(1-\rho)\,\tau_{ij}+\sum_{k=1}^{m}\Delta\tau^{k}_{ij}$$

**Mejor ruta global** (se revisa en cada evaluación):

$$\pi^*\leftarrow\pi^k,\quad L^*\leftarrow L_k,\quad\text{sólo si } L_k<L^*$$

Con $\alpha,\beta\ge 0$, $\rho\in(0,1]$, $Q>0$, $\varepsilon>0$. No hay elitismo, límites $\tau_{min}/\tau_{max}$ ni búsqueda local: todas las hormigas depositan y la feromona no se trunca.

---

## 2. Parámetros

| Parámetro | Valor usado | Significado |
|---|---|---|
| $m$ (`n_ants`) | 30 | Número de hormigas. Cada iteración cuesta $m$ evaluaciones de $L$. Entero $\ge 1$. |
| $\alpha$ (`alpha`) | 1.0 | Peso de la feromona en la probabilidad de transición. $\alpha=0$ ignora la feromona (búsqueda voraz estocástica). |
| $\beta$ (`beta`) | 2.0 | Peso de la visibilidad $\eta=1/C$. $\beta=0$ ignora el costo local y deja guiar sólo a la feromona. |
| $\rho$ (`rho`) | 0.5 | Tasa de evaporación en $(0,1]$: fracción de feromona que se pierde por iteración. $\rho=1$ borra la memoria previa. |
| $Q$ (`q`) | 1.0 | Constante de depósito: cada hormiga aporta $Q/L_k$ a sus arcos (más corta la ruta, más feromona). $Q>0$. |
| $\varepsilon$ (`eps`) | 1.0 | Piso de costo en $\eta$ y en el depósito: evita dividir por cero. Está en las unidades de $C$: todo costo menor que $\varepsilon$ se trata como $\varepsilon$. $\varepsilon>0$. |
| $\tau^{0}$ | $m/L_{nn}$ | Feromona inicial uniforme, escalada con la ruta del vecino más cercano. Se calcula, no se configura. |
| $C$ | matriz $n\times n$ | Costos del ciclo (puede ser asimétrica). Debe ser cuadrada, $n\ge 2$, finita y sin valores negativos. |
| $B$ (`budget`) | definido por el runner | Máximo de evaluaciones (entero $\ge 1$). Una iteración completa necesita $m$ evaluaciones. |
| `seed` | definida por el runner | Entero $\ge 0$. Semilla única del generador (ciudad inicial y ruleta de todas las hormigas). |

Todos los parámetros de ACO son obligatorios y se leen de `config/aco.yaml`: el código rechaza claves desconocidas y valida los rangos indicados.

---

## 3. Ciclo de vida de una evaluación

Una "evaluación" es una llamada a `RouteEvaluator.evaluate` con una ruta completa. Ese método valida que la ruta sea una permutación, calcula el costo del ciclo (incluye el arco de retorno) y suma 1 al contador $N_f$. ACO no usa gradientes, por lo que $E_{eq}=N_f$. El ciclo de vida de las evaluaciones en ACO es:

| Etapa | Qué ocurre | Dónde |
|---|---|---|
| 1. Validación | Se valida la matriz $C$ (cuadrada, finita, no negativa; copia de sólo lectura) y los parámetros de ACO (todos presentes, ninguno desconocido, rangos correctos). | `common.py` (`validate_cost_matrix`), `aco.py` (`_get_parameters`) |
| 2. Reinicio | `run()` llama a `reset()`: se recrea el generador desde `seed` y un evaluador con contador en 0. | `common.py` (`run`, `reset`) |
| 3. Estado inicial | Se calculan $\bar\eta$ y $\tau^0=m/L_{nn}$. El vecino más cercano usa `route_cost`, que no pasa por el contador: no se cobra. | `aco.py` (`_reset_state`) |
| 4. Verificación de presupuesto | Mientras `remaining > 0`, se fija `ants = min(m, remaining)`. | `aco.py` (`_run`) |
| 5. Construcción | Las `ants` rutas se construyen en paralelo con la ruleta sobre $\tilde\tau^{\alpha}\bar\eta$. Es sólo aritmética: no se cobra. | `aco.py` (`_construct_routes`) |
| 6. Evaluación | Se evalúa cada ruta con `evaluate`: hasta $m$ evaluaciones. Si una ruta mejora estrictamente a $L^*$, se actualiza $\pi^*$. Se corta si `remaining` llega a 0. | `aco.py` (`_evaluate_routes`), `common.py` (`evaluate`) |
| 7. Iteración incompleta | Si se evaluaron menos de $m$ rutas, el ciclo termina sin actualizar $\boldsymbol\tau$ y sin contar la iteración. | `aco.py` (`_run`) |
| 8. Actualización de feromona | Sólo con $m$ rutas evaluadas: primero evaporación, luego depósito de las $m$ hormigas. No se cobra. | `aco.py` (`_update_pheromone`) |
| 9. Registro | Se añade el paso a `history` (`best_cost`, `evaluations`, `best_route`) y se incrementa `iterations`. | `common.py` (`end_iteration`) |
| 10. Cierre | Si hubo evaluaciones después del último registro (iteración incompleta), se añade una última entrada a `history` sin sumar iteración. Se devuelve el resultado con `stop_reason`, la semilla y una copia de la configuración. | `common.py` (`run`) |

**Conteo total:** no hay evaluaciones iniciales; cada iteración completa cuesta $m$. Con $T=\lfloor B/m\rfloor$ iteraciones completas y $r=B \bmod m$:

$$N_f=m\,T+r$$


**Punto clave:** la actualización es síncrona. Todas las hormigas de una iteración construyen su ruta con la misma $\boldsymbol\tau$, y ésta sólo cambia después de evaluar las $m$ rutas. Las $r$ hormigas de la iteración incompleta sí se evalúan y pueden mejorar $\pi^*$, pero no depositan feromona, porque el presupuesto ya no permite otra iteración. Las rutas inválidas nunca se cuentan: `evaluate` valida antes de incrementar $N_f$.
