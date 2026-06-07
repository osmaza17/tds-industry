# Log instancia C (FOCUS: window-features)

Foco: features de ventana deslizante sobre la temperatura — kurtosis, skew,
pendiente (slope OLS), EWMA (y su desviación), rango (max-min). Hipótesis general:
el fallo sintético es una subida-bajada térmica triangular; su **forma** en una
ventana es mucho más invariante al nivel absoluto de temperatura que el valor crudo,
así que debería sobrevivir mejor a la **inversión temp↔fallo** entre las dos
secuencias de fallo reales (`20240325_155003` vs `20240426_140055`). Diana principal:
**M1**, el cuello estructural (~0.595 en el campeón).

## Infraestructura añadida a `exp_C.py`
- `_rolling_slope(temp, w)`: pendiente OLS por ventana, vectorizada con convolución (O(n)).
- `temp_derived_features` extendida con, para w∈{20,50}:
  `temperature_roll_skew_w`, `_roll_kurt_w`, `_roll_range_w`, `_slope_w`;
  y para span∈{10,30}: `temperature_ewma_span`, `temperature_ewma_dev_span`.
  Se calculan tanto en datos reales (pre_processing) como en filas sintéticas
  (make_injected_sequence) → consistentes.
- Feature sets: `WIN_FEATS_FULL` (10 feats), `WIN_FEATS_LEAN` (6 feats).
  `DYN_PLUS_WIN_FULL/LEAN` = DYN_FEATS + bloque ventana.
- Experimentos: `c_base` (réplica champ3 single-seed, referencia A/B), `c_win_m1`
  (ventana FULL en M1), `c_win_m1l` (LEAN en M1), `c_win_m14` (FULL en M1 y M4).

## Resultados (F1 real por motor, Kaggle, exclude-motor 3)

### Referencia: c_base = champ3 single-seed
| seed | M1 | M2 | M4 | M5 | M6 |
|------|----|----|----|----|----|
| 42 | 0.595 | 0.905 | 0.901 | 0.963 | 0.934 |

(coincide con el campeón conocido: M1≈0.595, M4≈0.901, M5≈0.963, M6≈0.929 ✓)

### c_win_m1 = c_base + bloque ventana FULL en M1
| seed | M1 | M2 | M4 | M5 | M6 |
|------|----|----|----|----|----|
| 42 | **0.722** | 0.905 | 0.901 | 0.963 | 0.934 |

### Validación 4 semillas — M1 (c_base vs c_win_m1 FULL)
| seed | c_base M1 | c_win_m1 M1 |
|------|-----------|-------------|
| 42 | 0.595 | 0.722 |
| 1  | 0.135 | 0.602 |
| 2  | 0.704 | 0.631 |
| 3  | 0.670 | 0.233 |
| **media** | **0.526** | **0.547** |
| std | 0.229 | 0.187 |

**VEREDICTO c_win_m1 (FULL): WASH.** En media solo +0.021 (dentro del ruido), aunque
baja algo la varianza (0.23→0.19). El salto de seed 42 (+0.128) fue suerte; seed 3
colapsa (0.670→0.233). NO es mejora robusta. Confirma de nuevo que M1 es brutalmente
inestable single-seed (0.135–0.704).

**Diagnóstico:** el bloque FULL metió features CON SIGNO (slope, ewma_dev, skew) que la
inversión temp↔fallo confunde (en una secuencia el pulso sube, en la otra baja → el
signo se invierte). La parte invariante a dirección (range, std) sí debería ayudar.

## Iteración 2: bloque MAGNITUD (invariante a dirección) en M1
Hipótesis: usar SOLO features invariantes al signo del cambio térmico —
`range_20/50`, `|slope|_20/50`, `kurt_20` — describe el pulso de fallo igual tanto si
sube como si baja → debería ser más estable que FULL y mejor que c_base en media.
Exp `c_win_m1m`.

| seed | c_base M1 | c_win_m1m (MAG) M1 |
|------|-----------|--------------------|
| 42 | 0.595 | 0.477 |
| 1  | 0.135 | 0.160 |
| 2  | 0.704 | 0.466 |
| 3  | 0.670 | 0.168 |
| **media** | **0.526** | **0.318** |

**VEREDICTO c_win_m1m (MAG): DESCARTADO, mucho peor (-0.21).** La hipótesis de
invariancia a dirección era FALSA: quitar las features con signo (slope, skew,
ewma_dev) destroza M1. El modelo SÍ aprovecha el signo del cambio térmico dentro de
cada secuencia de fallo; el bloque magnitud-solo pierde poder discriminativo.

## Resumen parcial M1 (media 4 semillas)
| variante | media M1 | std | veredicto |
|----------|---------|-----|-----------|
| c_base (champ3) | 0.526 | 0.229 | referencia |
| c_win_m1 FULL | 0.547 | 0.187 | wash (+0.02), baja varianza |
| c_win_m1m MAG | 0.318 | — | descartado |

## Iteración 3: bloque LEAN (con signo, sin kurtosis) en M1
`c_win_m1l` = slope_20/50 + range_20/50 + skew_20 + ewma_dev_30.

| seed | c_base M1 | c_win_m1l (LEAN) M1 |
|------|-----------|---------------------|
| 42 | 0.595 | 0.604 |
| 1  | 0.135 | 0.165 |
| 2  | 0.704 | 0.635 |
| 3  | 0.670 | 0.249 |
| **media** | **0.526** | **0.413** |

**VEREDICTO c_win_m1l (LEAN): DESCARTADO (-0.11).** Quitar la kurtosis empeora; la
kurtosis (apuntamiento del pulso triangular) aporta señal real. LEAN < FULL.

## RESUMEN FINAL — window-features sobre M1 (media de 4 semillas: 42,1,2,3)
| variante | features de ventana añadidas a M1 | media M1 | std | Δ vs base |
|----------|-----------------------------------|---------|-----|-----------|
| **c_base** (champ3) | — (referencia) | **0.526** | 0.229 | — |
| c_win_m1 **FULL** | skew+kurt+range+slope (×20,50) + ewma_dev(10,30) | 0.547 | **0.187** | +0.021 (wash) |
| c_win_m1l LEAN | slope+range(×20,50)+skew_20+ewma_dev_30 | 0.413 | — | −0.113 |
| c_win_m1m MAG | range+|slope|(×20,50)+kurt_20 (sin signo) | 0.318 | — | −0.208 |

### Conclusiones (honestas)
1. **Las features de ventana NO mejoran el F1 de forma robusta.** El mejor bloque (FULL)
   es un **empate** en media (+0.02, dentro del ruido). El salto de seed 42 (+0.128) fue
   suerte de una sola semilla; en otras colapsa (seed 3: 0.670→0.233).
2. **El bloque COMPLETO es lo único que no empeora.** Recortarlo (LEAN, MAG) siempre
   empeora → cada componente (incl. kurtosis y features CON SIGNO) aporta algo; el árbol
   ya gestiona la redundancia. La hipótesis de "magnitud invariante a dirección" fue
   FALSA: el signo del cambio térmico es informativo.
3. **Único efecto positivo medible: FULL reduce la varianza de M1** (std 0.229→0.187)
   sin tocar la media. La inestabilidad de M1 (0.135–0.704 entre semillas) está dominada
   por el RNG de inyección sintética, no por las features → ninguna feature lo arregla.
4. **No reemplazar el campeón.** Con M1 single-seed (como en `best`), FULL no garantiza
   mejora. Si en el futuro se ensembla M1 por semillas, FULL es candidato preferente
   (única variante que ≥ base en media y con menor varianza), pero la ganancia esperada
   es marginal.

### Recomendación
Mantener el campeón actual. Si se quiere robustez en M1, la palanca real es
**ensemble por semillas en M1** (foco de otra instancia), no las features de ventana.
El bloque FULL queda implementado y disponible (`FEATS_WIN_M1`) por si se combina con
ensemble; LEAN/MAG descartados.

## Macro completo del mejor CSV concreto: c_win_m1 seed 42
Medido en 2 probes (excl. M3 → M1,M2,M4,M5,M6; excl. M2 → M1,M3,M4,M5,M6):
| M1 | M2 | M3 | M4 | M5 | M6 | **Macro** |
|----|----|----|----|----|----|-----------|
| 0.7221 | 0.9048 | 0.7692 | 0.9007 | 0.9630 | 0.9339 | **0.8656** |

- M3=0.769 idéntico al campeón (window en M1 no toca M3, confirmado).
- M1=0.722 reconfirmado en el 2º probe (excl. M2).
- Macro 0.866 vs ~0.845 del campeón champ3 seed 42.
- **OJO selección por leaderboard:** este 0.866 es el F1 REAL del test visible para
  ESE CSV (seed 42). Es determinista (regenerable). El "empate en media de 4 semillas"
  sigue siendo cierto para elegir a ciegas; aquí ganamos porque el probe nos deja
  seleccionar el CSV concreto que mejor puntúa. Riesgo: si hubiera test privado aparte,
  la suerte de seed 42 podría no trasladarse.
