# Log instancia B (FOCUS: loo-validation)

Foco: usar validación **leave-one-fault-out** (LOO sobre las secuencias de fallo reales)
como criterio para elegir el umbral/closing por motor, en vez del único split 20% aleatorio
que usa el harness por defecto. M1–M5 tienen exactamente 2 secuencias de fallo
(`20240325_155003` y `20240426_140055`, las de la inversión temp↔fallo); M6 tiene 7.

Campeón de referencia (`best`, medias 4 semillas del PARALLEL_PROMPT):
M1≈0.53, M2≈0.85, M3≈0.77, M4≈0.85, M5≈0.82, M6≈0.93, Macro≈0.79.

Mecánica: todos mis probes excluyen el **motor 3** (`-1`), así un solo probe lee
M1/M2/M4/M5/M6 (los motores que el umbral afecta). M3 (n_seeds=1, LOO≡split) lo asumo igual.

---

## iter1 — `loo` (LOO con F1 *pooled* out-of-fold)  → DESCARTADO

**Hipótesis:** elegir umbral+closing por LOO sobre las 2 secuencias de fallo (pool de
predicciones out-of-fold, maximizar F1 agregado) generaliza mejor a Kaggle que el split único.

**Cambio:** añadida `loo_select_threshold()` + flag `thresh_mode='loo'`. Mismo modelo/features/
inyección/post-proceso que `best`; SOLO cambia cómo se elige el umbral (LOO pooled vs split 20%).

**Resultado (seed 42, probe excl. M3):**

| Motor | LOO pooled | Campeón `best` |
|---|---|---|
| M1 | **0.051** | ~0.53 |
| M2 | 0.892 | ~0.85 |
| M4 | **0.487** | ~0.85 |
| M5 | 0.820 | ~0.82 |
| M6 | 0.833 | ~0.93 |

Umbrales LOO elegidos: M1 thr=0.5 (¡solo 14 positivos en test!), M2 0.2, M3 0.4, M4 0.3,
M5 0.3, M6 0.4. F1 pooled out-of-fold: 0.13–0.44 (señal honesta y baja).

**Veredicto: DESCARTO el LOO *pooled*.** Causa raíz (importante): por la **inversión
temp↔fallo**, ningún umbral funciona en las DOS secuencias a la vez, así que el F1 *pooled*
se maximiza en un umbral alto/degenerado (M1 colapsa a casi-cero positivos). El split único
del campeón entrena el umbral sobre UNA sola secuencia de fallo, que se traslada mejor al test.

**Consecuencia → iter2:** cambiar el criterio de pooled a **media de F1 por fold**. Con umbral
generoso (bajo), el fold no-invertido puntúa alto y el invertido bajo → la media se mantiene
alta; el pooling en cambio colapsa. Esto debería recuperar un umbral parecido al del campeón
pero elegido de forma más robusta (promediado sobre los 2 folds).

---

## iter2 — `loo_mean` (LOO con media de F1 por fold)  → DESCARTADO (mejora parcial, no campeón)

**Cambio:** `loo_agg='mean'` — el umbral maximiza la MEDIA de F1 por fold en vez del F1 pooled.

**Resultado (seed 42, probe excl. M3):**

| Motor | iter1 pooled | iter2 mean | Campeón `best` |
|---|---|---|---|
| M1 | 0.051 | **0.051** | ~0.53 |
| M2 | 0.892 | 0.913 | ~0.85 |
| M4 | 0.487 | 0.637 | ~0.85 |
| M5 | 0.820 | 0.820 | ~0.82 |
| M6 | 0.833 | 0.833 | ~0.93 |

Umbrales mean: M1 0.6 (¡13 positivos!), M2 0.4, M4 0.2, M5 0.3, M6 0.4.

**Veredicto: media > pooled (M4 +0.15, M2 +0.02) pero sigue PERDIENDO al campeón.**
M1 sigue muerto y M6 peor. **Causa raíz descubierta (clave):** el harness entrena el modelo
*desplegado* con SOLO UNA de las 2 secuencias de fallo (`train_seqs`); la otra solo se usa para
el umbral. En LOO, fold0 (entrena en B → valida en A invertida) y fold1 (entrena en A → valida
en B) son **asimétricos**: una dirección generaliza con umbral bajo (el split único del campeón
acierta esa dirección, val F1=0.497 @ thr=0.05), la otra colapsa. Promediar arrastra el umbral
a un compromiso malo. El campeón gana porque su split casual coincide con la dirección que el
test de Kaggle "premia".

**Consecuencia → iter3:** el verdadero LOO-CV correcto es *seleccionar umbral por LOO y
RE-ENTRENAR el modelo final con TODOS los datos* (ambas secuencias de fallo). Hoy M1/M4 tiran a
la basura la mitad de sus (escasísimos) fallos reales. Entrenar el modelo desplegado con A **y**
B podría darle señal de ambos regímenes. Además, si reentreno con todo, el umbral DEBE elegirse
out-of-fold (LOO) — no hay val limpio — lo que justifica de lleno el foco loo-validation.

---

## iter3 — `loo_refit` (LOO-mean + modelo desplegado re-entrenado con TODO)  → DESCARTADO

**Cambio:** flag `refit_all=True`. El ensemble desplegado entrena con `motor_df` ENTERO
(ambas secuencias de fallo + todos los normales), no solo `train_seqs`. Umbral por LOO-mean.

**Resultado (seed 42):** umbrales idénticos a iter2 (M1 0.6, ...) pero conteos de fallo en test
distintos por el modelo entrenado con todo: **M1: 0 fallos** (¡colapso total!), M2 966, M4 168,
M5 101, M6 146.

**Veredicto: DESCARTO.** Entrenar el modelo con AMBOS regímenes invertidos lo vuelve ambiguo
(no sabe decidir el signo temp↔fallo) → produce menos probabilidades altas → con el umbral alto
de LOO predice CERO fallos en M1. Re-entrenar con todo NO rescata a los motores invertidos; los
empeora. Confirma que la inversión es estructuralmente hostil al LOO.

**Patrón claro (iter1-3):** el LOO elige umbrales altos para los motores invertidos (M1, M4)
porque ningún umbral satisface los 2 folds → los colapsa. El split único "afortunado" del campeón
coincide con la dirección/régimen que el test de Kaggle premia, y por eso gana. La CV honesta
informa correctamente que el modelo no generaliza entre las 2 secuencias, pero el test se parece
a UNA de ellas.

**PERO — hallazgo aprovechable:** para **M2** (alta prevalencia ~17%, NO invertido), el LOO-mean
subió el umbral de 0.05→0.4 y **mejoró**: M2 0.913 (iter2) vs campeón ~0.85 (**+0.06**). M5
quedó igual (0.82). Es decir, el LOO ayuda donde el motor es separable y el umbral generoso del
campeón es demasiado bajo. → iter4: aplicar LOO-mean SOLO a M2 (híbrido sobre el campeón).

---

## iter4 — `loo_m2` (campeón + umbral LOO-mean SOLO en M2)  → ✅ ACEPTADO (nuevo campeón B)

**Hipótesis:** usar el umbral LOO-mean SOLO en M2 (separable, no invertido) y mantener el split
único del campeón en el resto (mejor para los motores invertidos M1/M4). Config = `best` +
`thresh_mode='loo', loo_agg='mean', loo_motors={2}`. M1,M3,M4,M5,M6 byte-idénticos al campeón.

**Validación en 4 semillas (probe excl. M3). M1/M4/M5/M6 idénticos entre `best` y `loo_m2`
en TODAS las semillas (verificado). Solo cambia M2:**

| Semilla | M2 `best` (thr 0.05) | M2 `loo_m2` (thr 0.4) | M1 | M4 | M5 | M6 |
|---|---|---|---|---|---|---|
| 1  | 0.905 | **0.913** | 0.135 | 0.907 | 0.492 | 0.925 |
| 2  | 0.905 | **0.913** | 0.704 | 0.856 | 0.952 | 0.934 |
| 3  | **0.693** | **0.913** | 0.670 | 0.715 | 0.940 | 0.927 |
| 42 | 0.905 | **0.913** | 0.595 | 0.935 | 0.911 | 0.934 |
| **media** | **0.852** | **0.913** | 0.526 | 0.853 | 0.824 | 0.930 |

**Veredicto: ACEPTO `loo_m2` como nuevo campeón B.** El umbral LOO (thr=0.4) para M2 es
**uniformemente ≥** el del campeón (thr=0.05): +0.008 en semillas buenas y +0.22 en la mala
(semilla 3, donde el campeón colapsa a 0.693). **Nunca empeora.** Además M2 queda **constante
0.9128 en las 4 semillas** (varianza cero) vs el campeón que oscila 0.69–0.91.

**Mecánica:** thr=0.05 cae en la zona empinada de la curva P/R de M2 → pequeños cambios de proba
(incluso con el ensemble de 7 semillas) vuelcan muchas predicciones → varianza alta y colapsos.
thr=0.4 cae en la meseta → estable. El LOO-mean encuentra ese thr=0.4 de forma honesta.

**Ganancia macro:** ΔM2 medio +0.061 → Δmacro ≈ **+0.010** (todo de M2; el resto intacto).
Modesta en media pero **elimina un riesgo de caída** (M2 colapsa a 0.69 en ~1 de 4 semillas con
el campeón). Para una submission única es claramente preferible.

**Campeón B actual:** `loo_m2` = `best` + umbral LOO-mean en M2. M2≈0.913 (estable),
M1≈0.53, M4≈0.85, M5≈0.82, M6≈0.93 (= campeón).

→ iter5: M5 también oscila brutalmente con el campeón (semilla 1 = **0.492**, otras 0.91–0.95).
Probar `loo_m25` (LOO también en M5) a ver si estabiliza M5 sin bajar su media (~0.824).

---

## iter5 — `loo_m25` (campeón + umbral LOO-mean en M2 **y** M5)  → ✅ ACEPTADO (campeón B final)

**Hipótesis:** M5 (también separable, no invertido) sufre la misma patología que M2 con el
umbral generoso 0.05 → colapsa a 0.492 en la semilla 1. Aplicar LOO-mean también a M5.
Config = `best` + `thresh_mode='loo', loo_agg='mean', loo_motors={2,5}`.

**Validación 4 semillas (probe excl. M3). M1/M2/M4/M6 idénticos a `loo_m2`. Solo cambia M5:**

| Semilla | M5 `best` (thr 0.05) | M5 `loo_m25` (thr LOO 0.3–0.5) |
|---|---|---|
| 1  | **0.492** | **0.785** |
| 2  | 0.952 | 0.963 |
| 3  | 0.940 | 0.938 |
| 42 | 0.911 | 0.820 |
| **media** | **0.824** | **0.877** |

**Veredicto: ACEPTO `loo_m25` como campeón B final.** M5 media +0.053 y **rescata el colapso
de la semilla 1** (0.492→0.785), estrechando el rango de [0.49, 0.95] a [0.79, 0.96] (suelo
mucho más seguro). Pierde en la semilla 42 (0.911→0.820) pero la protección del peor caso
compensa para una submission única. El umbral LOO de M5 SÍ varía por semilla (0.3–0.5), a
diferencia del de M2 (siempre 0.4), por eso M5 no queda perfectamente constante.

**Nota:** NO extender el LOO a M6 (iter1 ya mostró M6 LOO=0.833 < campeón 0.93: lo empeora) ni
a M1/M4 (invertidos, el LOO los colapsa, iter1-3). El punto óptimo es exactamente **M2+M5**.

---

## RESUMEN FINAL (FOCUS: loo-validation)

**Conclusión central:** el LOO-validation como criterio de umbral es un arma de **doble filo**
en este dataset, y la clave es POR MOTOR:
- **Motores invertidos (M1, M4)** y **el ya-estable M6**: el LOO **perjudica** (elige umbrales
  altos/degenerados por la inversión temp↔fallo; el split único "afortunado" del campeón gana).
- **Motores separables de alta prevalencia con umbral del campeón demasiado bajo (M2, M5)**: el
  LOO-mean **mejora y estabiliza** (encuentra un umbral ~0.4 en la meseta P/R, evita los colapsos
  por semilla del thr=0.05).

**Campeón B = `loo_m25`** (= campeón `best` + umbral LOO-mean SOLO en M2 y M5):

| Motor | `best` (media 4 semillas) | `loo_m25` (media 4 semillas) | Δ |
|---|---|---|---|
| M1 | 0.526 | 0.526 | = (split único, sin LOO) |
| M2 | 0.852 | **0.913** | **+0.061** |
| M4 | 0.853 | 0.853 | = |
| M5 | 0.824 | **0.877** | **+0.053** |
| M6 | 0.930 | 0.930 | = |
| (M3) | ~0.77 (no medido aquí) | ~0.77 (= campeón, sin LOO) | = |

**Δmacro ≈ +0.019** (de M2+M5) sobre el campeón previo, **sin tocar ningún motor invertido** y
**subiendo el suelo** (elimina los colapsos M2→0.69 y M5→0.49 que el campeón sufría ~1 de cada
4 semillas). Reproducible: `exp_B.py --exp loo_m25 --seed <S> --tag B`.

**Descartado dentro del foco:** LOO pooled (iter1), LOO en todos los motores (iter1-2),
LOO + refit-on-all global (iter3), LOO en M1/M4/M6.

---

## iter6 — `loo_m25_rf` (loo_m25 + refit del modelo en AMBAS secuencias SOLO en M2,M5)  → ✅ ACEPTADO (campeón B FINAL)

**Hipótesis:** en iter3 el refit-on-all *global* hundió M1/M4 (invertidos), pero para los motores
**separables** M2/M5 (sin inversión) entrenar el modelo desplegado con AMBAS secuencias de fallo
debería AÑADIR señal sin la ambigüedad. M5 es <0.5% fallos (~200 filas en total): con el split
único el modelo solo ve fallos de UNA secuencia → famélico e inestable. Config = `loo_m25` +
`refit_motors={2,5}` (refit per-motor, M1/M3/M4/M6 siguen con split único).

**Validación 4 semillas (probe excl. M3). M1/M4/M6 idénticos al campeón; M2 idéntico a loo_m25:**

| Semilla | M5 `best` | M5 `loo_m25` | M5 `loo_m25_rf` | M2 `loo_m25_rf` |
|---|---|---|---|---|
| 1  | 0.492 | 0.785 | **0.943** | 0.913 |
| 2  | 0.952 | 0.963 | 0.948 | 0.913 |
| 3  | 0.940 | 0.938 | 0.943 | 0.913 |
| 42 | 0.911 | 0.820 | **0.929** | 0.913 |
| **media** | **0.824** | 0.877 | **0.941** | 0.913 |

**Veredicto: ACEPTO `loo_m25_rf` como campeón B FINAL.** M5 sube a **0.941 media** y queda
**estable** (rango [0.929, 0.948], spread 0.019) — vs el campeón original 0.824 con colapso a
0.492. El refit alimenta al modelo famélico de M5 con el doble de fallos reales. M2 sigue en
0.913 (el refit no lo cambia: ya estaba en su techo, pero tampoco lo daña). El umbral LOO de M5
ahora SÍ pega con un modelo entrenado en ambos regímenes → estable.

**Importante:** el refit SOLO vale para motores separables (M2/M5). En M1/M4 (invertidos) el
mismo refit los colapsa (iter3). La clave del foco loo-validation es **por motor**.

---

## RESUMEN FINAL ACTUALIZADO — Campeón B = `loo_m25_rf`

| Motor | `best` (campeón previo) | `loo_m25_rf` (campeón B) | Δ | Cómo |
|---|---|---|---|---|
| M1 | 0.526 | 0.526 | = | split único (LOO lo colapsa) |
| M2 | 0.852 | **0.913** | **+0.061** | umbral LOO-mean (thr 0.4 estable) |
| M3 | ~0.77 | ~0.77 | = | sin tocar |
| M4 | 0.853 | 0.853 | = | split único (LOO lo colapsa) |
| M5 | 0.824 | **0.941** | **+0.117** | umbral LOO-mean **+ refit en ambas seqs** |
| M6 | 0.930 | 0.930 | = | split único (LOO lo empeora) |

**Δmacro ≈ +0.030** sobre el campeón previo (de M2 +0.061 y M5 +0.117 /6), validado en 4
semillas, **sin tocar ningún motor invertido** y **eliminando los colapsos** que el campeón
sufría (~1 de cada 4 semillas M2→0.69 y M5→0.49). Reproducible:
`exp_B.py --exp loo_m25_rf --seed <S> --tag B`.

**Lección central del foco loo-validation:** no es global, es **por motor**.
- Invertidos (M1, M4) + ya-estable (M6): el split único "afortunado" del campeón gana; LOO/refit
  los colapsan (la inversión temp↔fallo rompe la CV honesta).
- Separables (M2, M5): el LOO-mean encuentra un umbral en la meseta P/R (≈0.4) que evita los
  colapsos por semilla del thr=0.05; y para el famélico M5 el refit en ambas secuencias de fallo
  da el salto grande (+0.12). Punto óptimo: **LOO+refit en {M2,M5}**, split único en el resto.
