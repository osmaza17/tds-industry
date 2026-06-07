# `pruebas/` — Banco de experimentos para el F1 de Kaggle

Carpeta de **experimentación** del Data Challenge (detección de fallos en 6 motores).
Aquí se itera para subir el F1 macro real en Kaggle **sin tocar los scripts originales**
del proyecto. Todo lo de aquí es trabajo de laboratorio: harnesses, sus salidas (CSVs),
logs de ejecución y diarios de experimentos.

## Origen y jerarquía (de dónde sale esto)

```
kaggle_data_challenge/utility.py        infraestructura base del curso
        ▼
predict_overheating.py                  primer script de submission        [LEGACY → legacy/]
        ▼
error injection_v1.py                   refactor con inyección sintética   [LEGACY → legacy/]
        ▼
error injection_v2.py                   padre directo del harness (probing por motor)
        ▼
pruebas/exp.py                          harness parametrizado (registro EXPERIMENTS)
        ▼
pruebas/exp_A.py  exp_B.py  exp_C.py  exp_Z.py    instancias paralelas (1 foco c/u)
```

`exp.py` reescribe `error injection_v2.py` parametrizado por un dict `EXPERIMENTS`
(features / inyección / pico / modelo por motor) para probar ideas sin editar los
originales. `exp_A/B/C/Z.py` son **copias de `exp.py` corridas en paralelo**, cada una
con un FOCO de estrategia distinto y un `--tag` propio (A/B/C/Z) para no pisarse en la
misma cuenta de Kaggle. Ver `notes/PARALLEL_PROMPT.md` para el esquema de paralelización.

> **Nota:** los `.py` deben quedarse en la **raíz de `pruebas/`** — calculan la raíz del
> proyecto como `dirname(dirname(__file__))` y escriben sus CSVs en su propio directorio.
> Moverlos a una subcarpeta rompe la carga de datos. Por eso solo se han reorganizado las
> *salidas* (CSVs, logs), no el código.

## Mecánica común: probing por motor

Kaggle solo muestra un F1 macro agregado. Cada submission fuerza **un motor a `-1`**
(`--exclude-motor N`): el scoring falla con `ERROR` pero el `error_description` filtra el
**F1 real por motor** del test. Con 2 probes complementarios (p. ej. excluir M3 y excluir M1)
se cubren los 6 motores. Los probes **no gastan cupo diario**. Helpers: `probe.py`,
`probe_tagged.py`. (Detalle completo en `../CLAUDE.md`.)

## Estructura de la carpeta

```
pruebas/
├── README.md                  este archivo
│
├── exp.py                     harness base (sin tag → CSVs sub_<exp>.csv en outputs/base/)
├── exp_A.py exp_B.py          instancias paralelas (tag A/B/C/Z)
├── exp_C.py exp_Z.py
├── sweep_Z.py                 barrido de semillas + ensamblado best-seed-por-motor (instancia Z)
├── stitch_C.py                pega columnas de motor de varios CSVs en uno (CLI: out.csv N=src.csv …)
├── probe.py                   sube 1 CSV y devuelve el F1 por motor del error_description
├── probe_tagged.py            igual que probe.py pero etiquetando la submission
│
├── notes/                     diarios de experimentos (markdown) — el "por qué" de cada decisión
│   ├── LOG.md                 mecánica general + hallazgos del harness base (exp.py)
│   ├── LOG_A.md               instancia A — FOCUS: threshold-cv (umbral por K-fold/LOO)
│   ├── LOG_B.md               instancia B — FOCUS: loo-validation (leave-one-fault-out)
│   ├── LOG_C.md               instancia C — FOCUS: window-features (forma de la ventana térmica)
│   ├── LOG_Z.md               instancia Z — FOCUS: real-augment + cross-motor
│   └── PARALLEL_PROMPT.md     prompt/esquema para correr las instancias en paralelo
│
├── runlogs/                   stdout crudo de las corridas (run_*.log) — referencia, no fuente
│
└── outputs/                   TODAS las submission CSVs (salidas de los harnesses)
    ├── base/                  generadas por exp.py (sin tag): baseline, champ*, dyn*, m1*…
    ├── A/                     generadas por exp_A.py (tag A)
    ├── B/                     generadas por exp_B.py (tag B)
    ├── C/                     generadas por exp_C.py (tag C)  ← incluye el campeón (ver abajo)
    ├── Z/                     generadas por exp_Z.py / sweep_Z.py (tag Z)
    └── _intermediate/         artefactos transitorios del sweep/probes
                               (pred_s*_full.csv, _probe_x*.csv, sub_swp_maxcfg*.csv, …)
```

### Convención de nombres de los CSV

`sub_<tag>_<exp>[_s<seed>][_excl<N>].csv`

- sin `<tag>` → corrida de `exp.py` (carpeta `outputs/base/`).
- `<exp>` = clave del dict `EXPERIMENTS` del harness correspondiente.
- `_s<seed>` = semilla (sin sufijo ⇒ seed 42 por defecto).
- `_FULL` / `_full` = CSV **completo y submittable** (todos los motores con 0/1, ninguno a `-1`),
  frente al CSV de probe que lleva un motor a `-1`.

### Campeón actual

`outputs/C/sub_C_c_win_m1_FULL_seed42.csv` — `exp_C.py --exp c_win_m1 --seed 42 --exclude-motor 0`.
Es `champ3` (HistGB + inyección sintética + post-proceso) **+ bloque de ventana FULL en M1**.
Probado seed 42: M1≈0.722, M2≈0.905, M4≈0.901, M5≈0.963, M6≈0.934. Ver `notes/LOG_C.md`
para el veredicto multi-semilla (la mejora de M1 fue parcialmente suerte de la seed).

## Cómo correr

```bash
# desde la raíz del proyecto
.venv/Scripts/python.exe pruebas/exp_C.py --exp c_win_m1 --seed 42 --exclude-motor 0 --tag C
# (--exclude-motor 0 = CSV completo; usa N∈1..6 para generar un probe de ese motor)
```

Los harnesses escriben siempre en la **raíz de `pruebas/`**; mueve manualmente el CSV a la
subcarpeta `outputs/<instancia>/` que corresponda para mantener el orden.
