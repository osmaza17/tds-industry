# CLAUDE.md — TDS Industry: Detección de Fallos en Motores Servo-Robóticos

## Descripción general del proyecto

Curso de **Machine Learning Industrial** (Universidad, Junio 2026). El trabajo principal es un **Data Challenge en Kaggle** para detectar fallos en 6 motores servo-robóticos monitorizados con sensores. El proyecto se divide en Travaux Dirigés (TD) sucesivos, cada uno enfocado en una parte del pipeline ML. Los informes se redactan en LaTeX (`output/*.tex`) y los experimentos en Jupyter (`td*.ipynb`).

**Alumno:** Óscar Martínez Zamora

---

## El dataset

| Parámetro | Valor |
|---|---|
| Filas × columnas | 39 309 × 26 |
| Frecuencia | 10 Hz (1 muestra cada 0.1 s) |
| Secuencias de entrenamiento | 15 (con etiquetas) |
| Secuencias de test Kaggle | 8 (sin etiquetas) |
| Secuencias totales en el CSV | 23 |

**Columnas por motor** (×6 motores):
- `data_motor_i_position` — posición del encoder [0, 1000]
- `data_motor_i_temperature` — temperatura en °C [0, 100]
- `data_motor_i_voltage` — voltaje en mV [6000, 9000]
- `data_motor_i_label` — etiqueta binaria 0=normal, 1=fallo

**Columna de agrupación:** `test_condition` — identificador de secuencia (timestamp).

### Tasa de fallos por motor

| Motor | Fallos aprox. | Tasa |
|---|---|---|
| M1 | ~1 600 | ~4.1% |
| M2 | ~6 700 | ~17.0% |
| M3 | <200 | <0.5% |
| M4 | ~6 700 | ~17.0% |
| M5 | <200 | <0.5% |
| M6 | ~2 000 | ~5.1% |

**M3 y M5** tienen tan pocos fallos que la clasificación supervisada estándar no es viable — se recomienda detección de anomalías (one-class SVM, Isolation Forest, Autoencoder).

La mayoría de los fallos de M2/M4 se concentran en la sesión `20240325_155003`.

---

## Archivos clave

```
kaggle_data_challenge/
├── utility.py                        # Funciones: carga CSV, CV por secuencia, sliding window
└── kaggle_data_challenge/
    ├── training_data/                # CSVs de entrenamiento (con etiquetas)
    ├── testing_data/                 # CSVs de test (sin etiquetas)
    └── sample_submission.csv

output/
├── td4.tex / td4.pdf                 # Informe: EDA, limpieza, feature engineering
├── td5.tex / td5.pdf                 # Informe: SVM, Random Forest, desbalanceo
├── td6.tex / td6.pdf                 # Informe: Benchmarking modelos (AAKR, RF, GB, LR)
└── td7.tex / td7.pdf                 # Informe: AAKR desde cero

td4.ipynb                             # EDA + preprocesamiento
td5.ipynb                             # SVM + RF en datasets sintéticos
td6.ipynb                             # Benchmarking real dataset Kaggle
td7.ipynb                             # AAKR implementación propia
td8.ipynb                             # (más reciente)

predict_overheating_v4.ipynb          # Versiones de la submission
motor_submission*.csv                 # Submissions generadas
```

---

## Estructura de los TDs y qué se ha hecho

### TD4 — Exploración, Limpieza y Selección de Características

**Task 1 — EDA:**
- Estadísticas descriptivas: 0 NaN, tipos numéricos, longitudes de secuencia 700–3500 muestras
- Series temporales: posición oscilatoria entre 2 set-points, temperatura creciente lenta, voltaje ruidoso
- Histogramas: posición **bimodal** (IQR no válido para outliers), temperatura con colas a 255°C (saturación ADC), voltaje ~normal
- PCA: sin normalizar → voltaje domina 70% varianza; normalizado → 8 PCs para 85% varianza (datos genuinamente multidimensionales)

**Task 2 — Preprocesamiento:**
- Normalización: `StandardScaler` ajustado SOLO en train, aplicado en test (sin re-fit)
- Outliers — 2 etapas:
  1. Recorte por rango físico (e.g. temperatura > 100°C → NaN + forward-fill causal)
  2. Z-score |z| > 3 calculado **por secuencia** (no globalmente)
  - Total eliminados: **6 840 valores** (0.97% del total)
- Suavizado: media móvil causal, ventana 5 muestras (0.5 s), `center=False`, `min_periods=1`, **por secuencia**

**Task 3 — Feature selection:**
- Violin plots + d de Cohen para poder discriminativo
- Correlación de Pearson: voltajes M1–M4 muy correlacionados (r > 0.92) → redundantes
- Características recomendadas (10 de 18):
  - Todas las temperaturas (independientes por motor)
  - Voltaje M2 y M4 (d ~1.8–1.9)
  - Posición M4 (d ~1.8) y M1 (d ~1.0)

### TD5 — SVM, Random Forest y Datos Desbalanceados

(Dataset sintético: `make_moons`, `make_classification`)

**Ejercicio 1 — SVM:**
- Kernel lineal: accuracy ~0.85 (limitación estructural)
- Kernel RBF default (C=1): ~0.94
- Kernel RBF tuneado (C=10, GridSearchCV 5-fold): **~0.95**
- SVM vs Regresión Logística: +10 pp en datos no lineales

**Ejercicio 2 — Random Forest:**
- `n_estimators` no causa overfitting; estabiliza el modelo (satura ~100 árboles)
- `max_depth` controla bias-varianza: depth=2 underfitting, depth=10 óptimo, depth=None overfitting
- Mejor configuración: `max_depth=10, n_estimators=100` → test acc ~0.95
- RF vs árbol único: +7 pp por reducción de varianza vía ensemble

**Ejercicio 3 — Datos desbalanceados (ratio 9:1):**
- Modelo base (LR sin corrección): accuracy ~97% pero **recall fallo = 0.71** (1 de cada 3 fallos perdido)
- `class_weight='balanced'`: recall fallo mejora a ~0.90
- SMOTE y undersampling aleatorio: mejoras comparables con distintos trade-offs
- **Métrica prioritaria: recall de la clase fallo, NO accuracy**

### TD6 — Benchmarking de Modelos por Motor

(Dataset real Kaggle, 6 motores)

**Modelos comparados:** AAKR, Random Forest, Gradient Boosting (HistGB), Regresión Logística

**Protocolo:** Validación cruzada 5-fold **por secuencia** (leave-one-sequence-out), búsqueda anidada de hiperparámetros, métricas: precisión, recall, F1-score.

**Resultado:** Random Forest con pesos balanceados (`class_weight='balanced'`) domina. AAKR es robusto para motores con muy pocos fallos (M3, M5).

**Workflow Kaggle:**
1. Entrenar mejor modelo por motor sobre el conjunto completo de entrenamiento
2. Predecir sobre las 8 secuencias de test
3. Generar `motor_submission.csv`
4. Subir con Kaggle CLI

### TD7 — AAKR desde cero

**Algoritmo AAKR:**
- No paramétrico, basado en instancias: almacena observaciones normales como "memoria"
- Ante un punto de consulta, estima el comportamiento esperado mediante media ponderada con **kernel gaussiano**
- El residuo `r = ||x_q - x̂_q||` es la señal de fallo
- Parámetro clave: ancho de banda `h` (controla localidad de la estimación)

**Ejercicio 1 — AAKR 1D:** Implementación, efecto de h en bias-varianza
**Ejercicio 2 — AAKR multivariado:** Extensión a 2 sensores correlacionados, detección por residuos

**Ventaja clave de AAKR:** Solo necesita datos normales para entrenar → ideal para M3/M5 donde los fallos son rarísimos.

---

## Reglas y decisiones críticas del proyecto

### Validación: SIEMPRE por secuencia completa
El split train/test y cualquier validación cruzada debe hacerse **a nivel de `test_condition`** (secuencia entera), nunca por fila. Razón: muestras adyacentes de la misma secuencia están temporalmente correlacionadas (temperatura cambia muy despacio). Dividir por fila produce **data leakage temporal** que infla dramáticamente el F1 local pero no se traslada a Kaggle.

```python
# CORRECTO
sequences = df['test_condition'].unique()
train_seqs, test_seqs = train_test_split(sequences, test_size=0.2)

# MAL (leakage)
X_train, X_test = train_test_split(df, test_size=0.2)  # NO
```

### La validación local engaña
Con split por fila: F1 local ~0.94 → Kaggle real ~0.17. Con split por secuencia honesto: F1 ~0.15–0.42. El mejor score real obtenido en Kaggle es ~**0.42**. No prometerse F1 > 0.5 con validación honesta por secuencia.

### Techo estructural del F1
Con validación por secuencia, F1 > 0.7 es estructuralmente inalcanzable con features de sensor por fila. La causa: la relación temperatura↔fallo **se invierte entre las dos secuencias con fallos reales** (`20240325_155003` y `20240426_140055`). Un modelo que aprende "temp alta→fallo" en una acierta al revés en la otra.

### Métricas: usar F1 (macro), no accuracy
Accuracy es engañosa con clases tan desbalanceadas. Usar `f1_score(..., average='macro')` o métricas por clase. Para modelos, `class_weight='balanced'` o SMOTE.

### Preprocesamiento por secuencia
Todas las operaciones que dependen de estadísticos locales (z-score, normalización, forward-fill, suavizado) deben calcularse **dentro de cada secuencia** con `groupby('test_condition')`.

### StandardScaler: fit solo en train
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train[features])
X_test  = scaler.transform(df_test[features])   # NO fit_transform
```

### Orden del pipeline
1. Recorte por rango físico → forward-fill causal
2. Z-score por secuencia (umbral |z| > 3)
3. Suavizado rolling mean causal (ventana 5, center=False)
4. Normalización StandardScaler (fit solo en train)
5. Feature selection (10 características recomendadas)

---

## Lo que queremos conseguir

1. **Maximizar F1 macro en Kaggle** para las 8 secuencias de test (objetivo inmediato)
2. **Submission honesta:** predicciones que generalicen a secuencias no vistas, no que memoricen el train
3. **Informes TD completos:** cada TD tiene un informe LaTeX con metodología, código, figuras y justificaciones del "por qué" de cada decisión

### Bottleneck actual
El mayor problema es la **calibración de prevalencia**: los modelos entrenados con balanceo 50/50 sobre-predicen en test (donde los fallos son raros). Estrategia pendiente: **umbral basado en rango** (rank-based threshold) — para cada motor, predecir como fallo el top-p% de muestras ordenadas por probabilidad, donde p = prevalencia observada en entrenamiento.

---

## Ideas / experimentos que podríamos probar

### Features
- **Ventana deslizante** (5–60 s a 10 Hz): media, std, rango, kurtosis → captura cambios en variabilidad, no solo valores instantáneos
- **Diferencias (diffs)**: `position_diff`, `temp_diff` con lags múltiples (1, 5, 10 muestras)
- **Residuos AAKR** como feature para clasificador supervisado posterior
- **Normalización por secuencia** antes de extraer features (pero cuidado: transductivo, puede inestabilizar secuencias cortas sin fallos)
- Reducir a las ~10–15 features con mayor poder discriminativo por motor

### Modelos
- **AAKR puro** (solo datos normales): especialmente para M3 y M5
- **XGBoost con inyección sintética de fallos**: inyectar copias de muestras de fallo con ruido gaussiano para aumentar datos de fallos raros
- **Isolation Forest / One-class SVM**: para M3/M5 (< 0.5% fallos)
- **Autoencoder de reconstrucción**: umbral en error de reconstrucción como score de anomalía
- **LSTM/GRU**: modelos recurrentes para capturar dependencias temporales que los clasificadores tabulares ignoran
- **Ensemble de modelos**: combinar AAKR + RF + GB por votación ponderada

### Validación y calibración
- **Umbral rank-based**: para cada motor, predecir como fallo el top-k% con mayor probabilidad (k = prevalencia de entrenamiento de ese motor)
- **Platt scaling / isotonic regression**: calibrar probabilidades de RF para que reflejen mejor la prevalencia real
- **Leave-one-fault-out CV**: dejar fuera una de las 2 secuencias con fallos como test, entrenar en la otra — la CV más honesta posible
- Explorar si features de dinámica (`diff`, residuos) tienen AUC > 0.5 en la secuencia de test al revés

---

## Kaggle

**Competición:** `robot-predictive-maintenance-season-2026`

**CLI:**
```bash
# Subir submission
.venv/Scripts/python.exe -m kaggle competitions submit \
  -c robot-predictive-maintenance-season-2026 \
  -f motor_submission.csv \
  -m "descripción"

# Ver scores pasados
.venv/Scripts/python.exe -m kaggle competitions submissions \
  -c robot-predictive-maintenance-season-2026
```

**Límite:** ~5 submissions/día (resetea medianoche UTC)
**Token:** `C:\Users\oscar\.kaggle\access_token`

**Mejor score obtenido:** F1 ≈ 0.42 (submission "honest")

---

## Entorno

- Python venv en `.venv/` (Python 3.x)
- Librerías: numpy, pandas, scikit-learn, scipy, imbalanced-learn, xgboost, matplotlib, seaborn
- LaTeX: pdflatex en `output/`; tras compilar, **limpiar `.out`, `.log`, `.aux`, `.toc`**
- VSCode: si el kernel se cuelga, reiniciar y asegurarse de seleccionar el venv correcto
