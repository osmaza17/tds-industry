# `legacy/` — código obsoleto archivado

Scripts y notebooks **superados** por el pipeline actual. Se conservan solo como
historial; **no forman parte del flujo de trabajo vigente** y no deben usarse para
generar submissions nuevas.

| Fichero | Qué era | Reemplazado por |
|---|---|---|
| `predict_overheating.py` | Primer script de submission (RF/ExtraTrees/XGBoost, `inject_failure` básico). | `error injection_v2.py` → `pruebas/exp_*.py` |
| `predict_overheating_v2.ipynb` | Versión notebook antigua de la submission. | idem |
| `predict_overheating_v3.ipynb` | Versión notebook antigua de la submission. | idem |
| `error injection_v1.py` | Refactor con inyección sintética "MATLAB-style"; primera versión del probing por motor. | `error injection_v2.py` (config de inyección por motor) |

## Qué SÍ sigue vigente (no está aquí, sigue en la raíz)

- **`error injection_v2.py`** — herramienta de probing por motor documentada en `CLAUDE.md`
  y padre directo del harness actual `pruebas/exp_*.py`.
- **`error_injection_code.md`** — especificación de la competición + fuente MATLAB original
  de la inyección de fallos (referencia, no código obsoleto).
- **`pruebas/exp_*.py`** — harness de experimentos actual (ver `pruebas/README.md`).

> Nota: `error injection_v1.py` calculaba la raíz del proyecto como `dirname(__file__)`,
> así que al estar dentro de `legacy/` ya no resuelve bien la ruta a `kaggle_data_challenge/`.
> No se ha corregido a propósito: es código archivado, no pensado para ejecutarse.
