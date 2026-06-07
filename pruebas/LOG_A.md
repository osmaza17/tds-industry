# Log instancia A (FOCUS: threshold-cv)

Foco: elegir el umbral por K-fold / leave-one-fault-out (LOO) en vez de un único split.
Anchor (campeón robusto `best`, medias del prompt): M1≈0.53, M2≈0.85, M3≈0.77, M4≈0.85,
M5≈0.82, M6≈0.93, Macro≈0.79.

## Mecánica añadida al harness (`exp_A.py`)
- Flag `cv_thresh=True`: si un motor tiene ≥2 secuencias de fallo reales, el umbral + closing
  se eligen por **leave-one-fault-out**. Cada fold deja fuera UNA secuencia de fallo + su
  reparto round-robin de secuencias normales. El umbral se elige maximizando la **media de F1
  por fold** (NO el F1 del pool de OOF concatenado: eso es degenerado por la inversión
  temp↔fallo — colapsa a "predecir todo positivo", p.ej. M4 thr=0.05 → 8522 fallos).
- Flag `cv_deploy`: `'all'` (reentrena el modelo final con TODAS las secuencias) o `'folds'`
  (despliega el ensemble de los modelos de fold; misma escala de probabilidad que el umbral).

## Exp `cv_loo` (cv_thresh + cv_deploy='all') — DESCARTADO

LOO threshold + modelo final reentrenado en todas las secuencias. 4 semillas, excluyendo M3.

| Motor | s1 | s2 | s3 | s4 | media | `best` | veredicto |
|---|---|---|---|---|---|---|---|
| M1 | 0.661 | 0.130 | 0.096 | 0.043 | **0.233** | 0.53 | mucho peor (s1 fue suerte) |
| M2 | 0.846 | 0.907 | 0.862 | 0.764 | 0.845 | 0.85 | = |
| M4 | 0.247 | 0.247 | 0.252 | 0.258 | **0.251** | 0.85 | colapso robusto |
| M5 | 0.929 | 0.948 | 0.943 | 0.819 | 0.910 | 0.82 | algo mejor |
| M6 | 0.741 | 0.878 | 0.934 | 0.827 | 0.845 | 0.93 | peor |

**Veredicto: DESCARTO.** Macro de esos 5 motores: 0.617 vs 0.796 del campeón.
**Causa raíz (hallazgo clave):** desajuste train/deploy. El umbral se calibra sobre modelos de
fold entrenados con UNA sola secuencia de fallo, pero el modelo final ve LAS DOS → es más nítido
y mejor calibrado, así que el umbral del fold queda demasiado conservador (M4: ~84 fallos
predichos → recall se hunde). La variabilidad de M1 NO se arregla (la inversión la hace
intrínsecamente inestable). Esto explica POR QUÉ el single-split del campeón funciona: su modelo
de validación ES su modelo de despliegue (escala de probabilidad emparejada); es de hecho un
único fold de LOO bien emparejado.

→ Hipótesis siguiente (`cv_ens2`): mantener el umbral LOO pero **desplegar el ensemble de los
  modelos de fold** (escala emparejada) en vez de reentrenar en todo. Debería arreglar el
  desajuste de M4.

## Dato clave descubierto
Con los datos adicionales (`additional_data`), los motores tienen MUCHAS más secuencias de
fallo de lo que asumía: M1 tiene **7** secuencias de fallo, M6 tiene **10** (no 2). Por tanto
LOO son 7–10 folds. Aun así, **el fallo de M2/M4 está concentrado en 1 secuencia**
(`20240325_155003`), lo que rompe el promediado de umbral por fold (ver abajo).

## Exp `cv_ens2` (cv_thresh + cv_deploy='folds') — DESCARTADO
Umbral LOO (media F1 por fold) + despliegue del ensemble de modelos de fold (escala emparejada).
Seed 1, excl. M3: M1=0.163, M2=0.913, M4=**0.261**, M5=0.943, M6=0.929.
**Veredicto: DESCARTO** (1 semilla basta: M4 sigue colapsado en 0.26, M1 0.16). El ensemble de
folds SUAVIZA las probas → con thr=0.5 predice AÚN MENOS fallos (M4: 61) → recall hundido.
El problema es el VALOR del umbral, no el despliegue.

## Exp `cv_wgt` (media F1 por fold ponderada por nº de fallos) — DESCARTADO
Idea: que la secuencia con todo el fallo (M2/M4) mande en el umbral en vez de diluirse entre
folds casi sin fallo. Seed 1, excl. M3: M1=0.712, M2=**0.079**, M4=**0.157**, M5=0.910, M6=0.925.
**Veredicto: DESCARTO.** Sobre-corrige: el umbral colapsa a 0.05 → predice 40–60% positivos
(M2: 5965, M4: 8522 fallos) → F1 se hunde. M1=0.712 otra vez alto pero es ruido de modelo.

## CONCLUSIÓN del foco threshold-cv (negativo, robusto)

**Ningún esquema de umbral por CV bate el single-split emparejado del campeón** para los motores
difíciles. El umbral óptimo vive en una banda estrecha y los métodos de CV se pasan por los dos
lados:

| Método | Umbral típico M4 | Fallos pred. M4 | F1 M4 |
|---|---|---|---|
| Pool OOF concatenado | 0.05 | 8522 | colapso (predice todo) |
| Media F1 por fold (equal) | 0.5 | 84 | 0.25 (muy conservador) |
| Media F1 por fold (pond. fallos) | 0.05 | 8522 | 0.16 (colapso) |
| **Campeón single-split** | (sweet spot) | ~cientos | **0.85–0.90** |

**Causas raíz (dos, combinadas):**
1. **Fallo concentrado** (M2/M4 en 1 secuencia): equal-weight diluye la señal (umbral muy alto);
   fault-weight la sobre-pondera (umbral colapsa). El single-split, al validar sobre un trozo que
   incluye la secuencia de fallo grande, cae en el punto medio bueno.
2. **Desajuste train/deploy:** el umbral calibrado en modelos de fold (menos datos) no transfiere
   al modelo final (más datos, probas más nítidas). El single-split del campeón NO sufre esto: su
   modelo de validación ES su modelo de despliegue.

**M1 (el cuello) NO es un problema de umbral:** su F1 oscila 0.04↔0.71 por semilla en TODOS los
métodos. La varianza viene del MODELO (inyección sintética + inversión temp↔fallo), no del umbral.
Ningún criterio de selección de umbral lo estabiliza.

**Mejor config robusta en mi foco = el campeón `best` sin cambios** (single-split emparejado).
No hay mejora de umbral-por-CV que aceptar. El foco threshold-cv queda agotado como palanca:
el cuello (M1) y los motores concentrados (M2/M4) son inmunes a la elección de umbral.

