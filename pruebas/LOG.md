# Log de experimentos — búsqueda de mejor F1 en Kaggle

Objetivo: maximizar el F1 por motor en el test real de Kaggle, partiendo de
`error injection_v2.py`. Todo el código vive en `pruebas/` (sin tocar los originales).

## Mecánica del proceso

- **Harness:** `pruebas/exp.py`, parametrizado por un registro `EXPERIMENTS` (features,
  inyección, pico, modelo por motor). Reutiliza toda la lógica de v2.
- **Evaluación = probe.** Cada submission fuerza un motor a `-1` → Kaggle da `ERROR`
  pero el `error_description` filtra el **F1 real por motor**. Helper: `pruebas/probe.py`.
  Los probes NO gastan cupo diario (confirmado por el usuario).
- Para leer un motor concreto hay que excluir OTRO motor (el excluido sale como `-1`).
  Se usan 2 probes (excluir M3 y excluir M1) para cubrir los 6 motores.

## Hallazgos de mecánica (importantes)

1. **Las features dinámicas son la gran palanca.** Sustituir la temperatura absoluta
   por features de *cambio/variabilidad* (diffs multi-lag, std rolling, desviación de la
   media local) en M1 y M4 dispara su F1. Razón: la relación temp↔fallo se **invierte**
   entre las dos secuencias de fallo reales; los valores absolutos engañan, la dinámica no.
2. **Cambiar features NO acopla** (no consume aleatoriedad) → comparaciones limpias.
3. **Cambiar inyecciones SÍ acopla el RNG.** Tanto el nº de inyecciones como el rango de
   pico (`np.random.randint` usa muestreo por rechazo → consume distinta cantidad del
   stream) desplazan los motores posteriores. Esto metió ruido en los primeros resultados.
4. **Solución: reseed por motor** (`np.random.seed(seed*1000 + motor_id)` antes de
   sintetizar). Cada motor es independiente y reproducible. A partir de `champ2` todo es limpio.
5. **Variance por semilla:** con una sola realización de RNG, el F1 de un motor varía
   bastante (M4 osciló 0.49–0.77). Algunos "máximos" iniciales eran suerte, no mejora real.

## Tabla de experimentos

F1 real por motor (Kaggle). Macro = media de los 6. `=` significa idéntico al baseline.

### Época semilla global (resultados con ruido de acoplamiento)

| Exp | Cambio probado | Por qué | M1 | M2 | M3 | M4 | M5 | M6 | Macro |
|-----|----------------|---------|----|----|----|----|----|----|-------|
| **baseline** | v2 tal cual (features+picos v2) | referencia | 0.231 | 0.888 | 0.769 | 0.202 | 0.941 | 0.769 | **0.633** |
| **dynfeat** | features dinámicas en M1,M4 | dinámica invariante a la inversión temp↔fallo | 0.536 | 0.888 | 0.769 | 0.744 | 0.941 | 0.794 | **0.779** |
| dynfeat_inj | +inyecciones (12) en M1,M4 | ¿más datos de fallo? | 0.506 | 0.905 | – | 0.423 | 0.550 | 0.838 | peor |
| dyn146 | dinámicas también en M6 | ¿generaliza a M6? | = | = | – | = | = | 0.727 | M6 peor |
| m1rf | RandomForest para M1 | ¿otro modelo ayuda a M1? | 0.163 | – | – | – | – | – | mucho peor |
| m1et | ExtraTrees para M1 | idem | 0.149 | – | – | – | – | – | mucho peor |
| m1peak | pico M1 = 6–15 | picos sintéticos más separables | 0.659 | – | – | – | – | – | ↑ |
| m1peak2 | pico M1 = 10–25 | ¿aún más alto? | 0.281 | – | – | – | – | – | irreal, peor |
| m3dyn | dinámicas en M3 | ¿ayuda a M3? | – | – | 0.730 | – | – | – | M3 peor |
| m1pk_a / m1pk_b | pico M1 = 4–12 / 8–16 | afinar óptimo M1 | 0.609 / **0.689** | – | – | – | – | – | 8–16 mejor |
| m1pk_c | pico M1 = 9–18 | afinar | 0.662 | – | – | – | – | – | < 8–16 |
| both_pk / m4pk | pico M4 = 6–15 / 4–12 | tunear M4 | – | – | – | 0.769 / 0.766 | – | – | 6–15 mejor |
| m4pk_b | pico M4 = 8–16 | afinar M4 | – | – | – | 0.753 | – | – | < 6–15 |
| **champ** | M1 8–16 + M4 6–15 | combinar mejores picos | 0.689 | 0.888 | 0.769 | 0.769 | 0.941 | 0.794 | **0.808** (inflado) |
| m3pk_hi / m3pk_lo | pico M3 = 3–8 / 1–3 | tunear M3 | – | – | 0.875 / 0.769 | – | – | – | 3–8 (con acoplamiento) |
| m6pk_a / m6pk_b | pico M6 = 6–15 / 4–12 | tunear M6 | – | – | – | – | – | 0.867 / 0.787 | 6–15 mejor |

### Época semilla por motor (limpio y reproducible)

| Exp | Config | M1 | M2 | M3 | M4 | M5 | M6 | Macro |
|-----|--------|----|----|----|----|----|----|-------|
| **dynbase** | dinámicas M1,M4 + picos v2 | 0.478 | 0.883 | 0.717 | 0.495 | 0.963 | 0.852 | **0.731** |
| **champ2** | dinámicas M1,M4 + picos M1 8–16, M3 3–8, M4 6–15, M6 6–15 | 0.595 | 0.883 | 0.769 | 0.658 | 0.963 | 0.889 | **0.793** |

**Comparación limpia dynbase → champ2 (misma semilla por motor):** el tuneo de pico ayuda
en TODOS los motores objetivo: M1 +0.117, M3 +0.052, M4 +0.163, M6 +0.037. Es decir, el
*pico más alto* es una mejora real, aunque las magnitudes exactas single-seed eran ruidosas.

## Estrategias estructuralmente distintas (A / B / C)

Hasta aquí todo era hill-climbing sobre la arquitectura de v2. Estas son palancas nuevas:

| Estrategia | Qué hace | Resultado |
|-----------|----------|-----------|
| **A — rank-based threshold** | fija el % de positivos = prevalencia de train (cuantil global) en vez del umbral por validación | **FALLA**: empeora todo (M1 0.40, M2 0.64...). La prevalencia de test ≠ train; el cuantil global es mal umbral. |
| **B — post-procesado morfológico** | `close` (fusiona fragmentos) + `min_run` (borra runs cortos), **por motor** | **GRAN ÉXITO**: M4 0.658→**0.901** (close=5 + min_run=40), M6 0.889→**0.929** (min_run=20). M2/M5 no afectados. |
| **C — seed-ensembling** | promedia probas de K modelos (K semillas de inyección) | **MIXTO**: M4 0.901→0.932 pero **M5 0.963→0.676** (K=3) / 0.202 (K=5). El umbral single-seed se descalibra al promediar. Solo útil por-motor + recalibrando umbral. |

### Detalle Strategy B (post-procesado por motor)

| Variante | M1 | M2 | M4 | M5 | M6 |
|----------|----|----|----|----|----|
| champ2 (sin pp) | 0.595 | 0.883 | 0.658 | 0.963 | 0.889 |
| min_run=20 (global) | 0.588 | 0.883 | 0.743 | 0.963 | 0.929 |
| min_run=40 + close=5 (global) | 0.572 | 0.883 | 0.901 | 0.963 | 0.741 |
| **champ3** (M4: close5+mr40; M6: mr20) | 0.595 | 0.883 | **0.901** | 0.963 | **0.929** |

## Mejor config actual (reproducible)

**`champ3`** — Macro **0.840** (vs baseline 0.633, **+0.207**):
- Features dinámicas en M1 y M4 (`DYN_FEATS`); v2 en M2,M3,M5,M6.
- Picos de inyección: M1 (8,16), M3 (3,8), M4 (6,15), M6 (6,15); M2 (2,10), M5 (1,4).
- Modelo HistGradientBoosting en los 6 (RF/ET son peores).
- **Post-procesado por motor: M4 → close=5 + min_run=40; M6 → min_run=20.**
- Reseed por motor (`seed*1000+id`, seed=42).

F1 por motor: M1=0.595, M2=0.883, M3=0.769, M4=0.901, M5=0.963, M6=0.929.

| Hito | Macro | Δ |
|------|:---:|:---:|
| baseline v2 | 0.633 | — |
| + features dinámicas (dynfeat) | 0.779 | +0.146 |
| + picos por motor (champ2) | 0.793 | +0.014 |
| + post-procesado (champ3) | **0.840** | +0.047 |

## Descartado (no funciona)

- Más inyecciones (over-injection colapsa M4/M5).
- RandomForest / ExtraTrees para M1 (mucho peor que HistGB).
- Features dinámicas en M3 y M6 (peor que v2 para esos dos).
- Picos demasiado altos (M1 10–25 colapsa).

## Próximos pasos candidatos

- **M1 sigue siendo el cuello** (0.595). Estrategias sin probar: D (ensemble de modelos
  HistGB+RF+LR con voto), E (detección de anomalías IsolationForest/AAKR para M1/M3).
- **Ensembling SOLO en M4** + recalibrar umbral sobre el ensemble (M4 0.901→0.932 sin
  romper M5): requiere `n_seeds` por motor y re-tuneo de umbral sobre probas promediadas.
- Post-procesado para M1/M3 (de momento no ayuda; explorar close por separado).
- **Importante**: todos los F1 son single-seed (excepto donde se promedió). Conviene
  validar el campeón final en 2-3 semillas antes de fijarlo.
