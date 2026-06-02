"""
TD9 - data loading and cleaning for the fault-detection data challenge.

Sources (all under ``../TD6/kaggle_data_challenge``):
- ``training_data``  : labelled, used as the *trusted* CV/validation pool.
- ``testing_data``   : unlabelled, aligned with ``sample_submission.csv``.
- ``additional_data``: labelled sequences produced by previous student groups.
  These are noisier, so they go through ``quality_filter_additional`` before
  being trusted, and they are only ever used to *train* (never to validate).

Cleaning mirrors the course/TD4 approach:
- physical-range clip + forward-fill on each raw motor CSV,
- IQR-based soft clipping with bounds fit on the *training* data only (so test
  and additional data cannot leak information into the clip limits).

Why the quality filter matters: the failure mode is *defined* by an abnormal
temperature rise (see the MATLAB ``inject_failure`` model). A labelled fault
segment whose temperature does not actually rise above the sequence baseline is
almost certainly mislabelled, so we void those labels rather than feed the model
contradictory targets.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_ROOT = (HERE / ".." / "TD6" / "kaggle_data_challenge").resolve()
TD6_PKG = DATA_ROOT  # utility.py lives here

if str(TD6_PKG) not in sys.path:
    sys.path.insert(0, str(TD6_PKG))

from utility import (  # noqa: E402
    read_all_csvs_one_test,
    read_all_test_data_from_path,
)

TRAIN_DIR = str(DATA_ROOT / "training_data") + "/"
TEST_DIR = str(DATA_ROOT / "testing_data") + "/"
ADDITIONAL_DIR = DATA_ROOT / "additional_data"
SAMPLE_SUBMISSION = DATA_ROOT / "sample_submission.csv"

# Quality-filter thresholds.
MIN_SEQ_LEN = 50           # drop additional sequences shorter than this
TEMP_RISE_MARGIN = 1.0     # required rise (deg) of fault temp above baseline


def remove_outliers_physical(df: pd.DataFrame) -> None:
    """In-place physical-range clean of one motor's raw columns (ffill gaps)."""
    df["temperature"] = df["temperature"].where(df["temperature"] <= 100, np.nan)
    df["temperature"] = df["temperature"].where(df["temperature"] >= 0, np.nan)
    df["temperature"] = df["temperature"].ffill()
    df["voltage"] = df["voltage"].where(df["voltage"] >= 6000, np.nan)
    df["voltage"] = df["voltage"].where(df["voltage"] <= 9000, np.nan)
    df["voltage"] = df["voltage"].ffill()
    df["position"] = df["position"].where(df["position"] >= 0, np.nan)
    df["position"] = df["position"].where(df["position"] <= 1000, np.nan)
    df["position"] = df["position"].ffill()


def sensor_measure_columns() -> list[str]:
    return [
        f"data_motor_{m}_{s}"
        for m in range(1, 7)
        for s in ("position", "temperature", "voltage")
    ]


def fit_iqr_clip_bounds(
    df: pd.DataFrame, *, iqr_factor: float = 1.5, min_count: int = 50
) -> tuple[pd.Series, pd.Series]:
    """Per-sensor IQR clip bounds fit on training data (clip, never drop rows)."""
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for c in sensor_measure_columns():
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(x) < min_count:
            continue
        q1, q3 = float(x.quantile(0.25)), float(x.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower[c] = q1 - iqr_factor * iqr
        upper[c] = q3 + iqr_factor * iqr
    return pd.Series(lower), pd.Series(upper)


def clip_sensor_columns(df: pd.DataFrame, lower: pd.Series, upper: pd.Series) -> pd.DataFrame:
    out = df.copy()
    for c in lower.index:
        if c in out.columns:
            out[c] = out[c].clip(lower[c], upper[c])
    return out


def load_training() -> pd.DataFrame:
    """Trusted, labelled training pool."""
    df = read_all_test_data_from_path(TRAIN_DIR, remove_outliers_physical, is_plot=False)
    df["source"] = "train"
    return df.reset_index(drop=True)


def load_testing() -> pd.DataFrame:
    """Unlabelled test pool, aligned with ``sample_submission.csv`` row order."""
    df = read_all_test_data_from_path(TEST_DIR, remove_outliers_physical, is_plot=False)
    df["source"] = "test"
    return df.reset_index(drop=True)


def _list_sequence_dirs(group_dir: Path) -> list[str]:
    return sorted(
        d for d in os.listdir(group_dir) if os.path.isdir(group_dir / d)
    )


def load_additional_raw() -> pd.DataFrame:
    """Load every additional_data sequence (labels included), physically cleaned.

    ``test_condition`` is prefixed with the group folder so sequence ids stay
    unique across groups (important for GroupKFold grouping).
    """
    frames: list[pd.DataFrame] = []
    if not ADDITIONAL_DIR.exists():
        return pd.DataFrame()
    for group in sorted(os.listdir(ADDITIONAL_DIR)):
        gdir = ADDITIONAL_DIR / group
        if not gdir.is_dir():
            continue
        for seq in _list_sequence_dirs(gdir):
            seq_path = str(gdir / seq)
            uniq_id = f"{group}__{seq}"
            try:
                df = read_all_csvs_one_test(seq_path, uniq_id, remove_outliers_physical)
            except Exception:
                continue
            if df is None or len(df) == 0:
                continue
            df["source"] = "additional"
            frames.append(df.reset_index(drop=True))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out


def quality_filter_additional(
    df_add: pd.DataFrame, *, min_len: int = MIN_SEQ_LEN, margin: float = TEMP_RISE_MARGIN
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trust-filter additional data at the (sequence, motor) granularity.

    A motor's labels within a sequence are voided (set to NaN) when the labelled
    fault rows do not show a temperature rise above the sequence's normal
    baseline. Voided labels are dropped per-motor by the trainer, so a single
    bad motor never discards an otherwise-good sequence.

    Returns ``(filtered_df, audit_table)``.
    """
    if df_add.empty:
        return df_add, pd.DataFrame()

    out = df_add.copy()
    audit_rows: list[dict] = []
    drop_short: list[str] = []

    for seq, g in out.groupby("test_condition", sort=False):
        if len(g) < min_len:
            drop_short.append(seq)
            continue
        idx = g.index
        for m in range(1, 7):
            lab_col = f"data_motor_{m}_label"
            temp_col = f"data_motor_{m}_temperature"
            if lab_col not in out.columns or temp_col not in out.columns:
                continue
            lab = out.loc[idx, lab_col]
            n_pos = int((lab == 1).sum())
            if n_pos == 0:
                continue
            fault_temp = out.loc[idx, temp_col][lab == 1]
            base_temp = out.loc[idx, temp_col][lab == 0]
            baseline = float(base_temp.median()) if len(base_temp) else float(fault_temp.min())
            rise = float(fault_temp.max()) - baseline
            trusted = rise >= margin
            if not trusted:
                out.loc[idx, lab_col] = np.nan
            audit_rows.append(
                dict(sequence=seq, motor=m, n_pos=n_pos, temp_rise=round(rise, 2), trusted=trusted)
            )

    if drop_short:
        out = out[~out["test_condition"].isin(drop_short)].reset_index(drop=True)

    audit = pd.DataFrame(audit_rows)
    return out, audit


def load_all(*, use_iqr_clip: bool = True) -> dict:
    """One-shot loader.

    Returns a dict with:
    - ``train``      : trusted training pool (labels 0/1)
    - ``test``       : test pool (aligned with submission)
    - ``additional`` : quality-filtered additional pool (labels 0/1/NaN)
    - ``audit``      : additional-data trust audit table
    - ``clip_lo``/``clip_hi`` : the IQR bounds (fit on train)
    """
    df_train = load_training()
    df_test = load_testing()
    df_add_raw = load_additional_raw()
    df_add, audit = quality_filter_additional(df_add_raw)

    if use_iqr_clip:
        lo, hi = fit_iqr_clip_bounds(df_train)
        df_train = clip_sensor_columns(df_train, lo, hi)
        df_test = clip_sensor_columns(df_test, lo, hi)
        if not df_add.empty:
            df_add = clip_sensor_columns(df_add, lo, hi)
    else:
        lo, hi = pd.Series(dtype=float), pd.Series(dtype=float)

    return dict(
        train=df_train,
        test=df_test,
        additional=df_add,
        audit=audit,
        clip_lo=lo,
        clip_hi=hi,
    )


if __name__ == "__main__":
    data = load_all()
    print("train rows     :", len(data["train"]))
    print("test rows      :", len(data["test"]))
    print("additional rows:", len(data["additional"]))
    print("\nadditional trust audit:")
    if not data["audit"].empty:
        print(data["audit"].to_string(index=False))
