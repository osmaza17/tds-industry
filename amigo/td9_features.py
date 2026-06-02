"""
TD9 - engineered + temporal features for fault detection.

Adapted from the TD6 feature set. All rolling/difference features are computed
*per sequence* (``test_condition``) so signals never leak across sequences.

Design rationale (ties to the professor's notes):
- Temporal context (point 3): rolling stats, drift vs a 200-step baseline, and
  rate/acceleration of temperature give the model recent history instead of a
  single instantaneous sample.
- Cross-motor coupling (point 4): per-motor voltage deviation from peers and
  "what are the OTHER motors doing right now" features let the model learn that
  a still motor's voltage spike is innocent when a neighbour starts moving.
- Failure physics: temperature is the defining signal, so explicit
  rise-above-recent-min / range / acceleration features are provided.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import td9_shape as shp  # noqa: E402

ROLL_STD_WIN = 50
ROLL_BASELINE_WIN = 200
MOVE_EPS = 0.5
MOVE_SMOOTH_WIN = 20

# Robust (long, causal) temperature-baseline settings. The defining fault signal
# is "temperature elevated above the motor's NORMAL level", but a fault can last
# thousands of samples (motors 2/4), which contaminates a short rolling *mean*
# baseline. A long, low-quantile / expanding-minimum baseline cannot be pulled up
# quickly by a sustained high fault, so the elevation stays visible mid-fault.
LONG_BASELINE_WIN = 600  # 60 s trailing window
BASELINE_Q = 0.10  # low quantile -> tracks the "cool"/normal floor
EARLY_WIN = 200  # samples used to estimate each motor's normal temperature spread

# Motors that receive the shape-aware feature block, chosen on held-out evidence
# (td9_shape_eval.py): a decisive lift on motor 3 and held-out gains on 1/2/5,
# while it hurts the strong motor 4 and the held-out F1 of motor 6.
SHAPE_MOTORS = frozenset({1, 2, 3, 5})


def enrich_basic(df: pd.DataFrame) -> pd.DataFrame:
    """Mean voltage across motors + per-motor temperature delta from seq start."""
    out = df.copy()
    vcols = [f"data_motor_{i}_voltage" for i in range(1, 7)]
    if all(c in out.columns for c in vcols):
        out["mean_motor_voltage"] = out[vcols].mean(axis=1)
    if "test_condition" not in out.columns:
        return out
    for i in range(1, 7):
        tcol = f"data_motor_{i}_temperature"
        if tcol not in out.columns:
            continue
        first = out.groupby("test_condition", sort=False)[tcol].transform("first")
        out[f"{tcol}_delta0"] = out[tcol] - first
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with all engineered columns appended."""
    out = enrich_basic(df)
    gb_key = "test_condition"

    def grp_diff(col: str) -> pd.Series:
        return out.groupby(gb_key, sort=False)[col].diff().fillna(0.0)

    def grp_roll(col: str, window: int, op: str) -> pd.Series:
        s = out.groupby(gb_key, sort=False)[col].transform(
            lambda x: getattr(x.rolling(window, min_periods=1), op)()
        )
        return s.fillna(0.0)

    for i in range(1, 7):
        pos = f"data_motor_{i}_position"
        temp = f"data_motor_{i}_temperature"
        volt = f"data_motor_{i}_voltage"

        out[f"m{i}_velocity"] = grp_diff(pos)
        out[f"m{i}_accel"] = grp_diff(f"m{i}_velocity")
        out[f"m{i}_temp_rate"] = grp_diff(temp)
        out[f"m{i}_volt_rate"] = grp_diff(volt)

        out[f"m{i}_pos_roll_std"] = grp_roll(pos, ROLL_STD_WIN, "std")
        out[f"m{i}_temp_roll_std"] = grp_roll(temp, ROLL_STD_WIN, "std")
        out[f"m{i}_volt_roll_std"] = grp_roll(volt, ROLL_STD_WIN, "std")

        temp_baseline = out.groupby(gb_key, sort=False)[temp].transform(
            lambda x: x.rolling(ROLL_BASELINE_WIN, min_periods=1).mean()
        )
        out[f"m{i}_temp_drift"] = out[temp] - temp_baseline

        is_move_raw = (out[f"m{i}_velocity"].abs() > MOVE_EPS).astype(float)
        out[f"m{i}_is_moving_raw"] = is_move_raw
        out[f"m{i}_is_moving"] = (
            out.groupby(gb_key, sort=False)[f"m{i}_is_moving_raw"]
            .transform(lambda x: x.rolling(MOVE_SMOOTH_WIN, min_periods=1).mean())
            .fillna(0.0)
        )

    volt_cols = [f"data_motor_{i}_voltage" for i in range(1, 7)]
    volt_sum = out[volt_cols].sum(axis=1)
    for i in range(1, 7):
        out[f"m{i}_volt_dev_peers"] = (6.0 * out[f"data_motor_{i}_voltage"] - volt_sum) / 5.0

    vel_cols = [f"m{i}_velocity" for i in range(1, 7)]
    out["any_motor_moving"] = (out[vel_cols].abs() > MOVE_EPS).any(axis=1).astype(float)
    out["n_motors_moving"] = (out[vel_cols].abs() > MOVE_EPS).sum(axis=1).astype(float)

    abs_vels = {i: out[f"m{i}_velocity"].abs() for i in range(1, 7)}
    abs_volt_rates = {i: out[f"m{i}_volt_rate"].abs() for i in range(1, 7)}
    moving_flags = {i: (abs_vels[i] > MOVE_EPS).astype(float) for i in range(1, 7)}

    for i in range(1, 7):
        others = [j for j in range(1, 7) if j != i]

        out[f"m{i}_other_vel_sum"] = sum(abs_vels[j] for j in others)
        out[f"m{i}_other_vel_recent"] = (
            out.groupby(gb_key, sort=False)[f"m{i}_other_vel_sum"]
            .transform(lambda s: s.rolling(MOVE_SMOOTH_WIN, min_periods=1).mean())
            .fillna(0.0)
        )

        out[f"m{i}_other_volt_rate_sum"] = sum(abs_volt_rates[j] for j in others)
        out[f"m{i}_other_volt_rate_recent"] = (
            out.groupby(gb_key, sort=False)[f"m{i}_other_volt_rate_sum"]
            .transform(lambda s: s.rolling(MOVE_SMOOTH_WIN, min_periods=1).mean())
            .fillna(0.0)
        )

        out[f"m{i}_other_moving_count"] = sum(moving_flags[j] for j in others)
        out[f"m{i}_other_moving_recent"] = (
            out.groupby(gb_key, sort=False)[f"m{i}_other_moving_count"]
            .transform(lambda s: s.rolling(MOVE_SMOOTH_WIN, min_periods=1).mean())
            .fillna(0.0)
        )

    for i in range(1, 7):
        temp = f"data_motor_{i}_temperature"
        roll_min = out.groupby(gb_key, sort=False)[temp].transform(
            lambda x: x.rolling(ROLL_BASELINE_WIN, min_periods=1).min()
        )
        roll_max = out.groupby(gb_key, sort=False)[temp].transform(
            lambda x: x.rolling(ROLL_BASELINE_WIN, min_periods=1).max()
        )
        out[f"m{i}_temp_above_min200"] = out[temp] - roll_min
        out[f"m{i}_temp_range200"] = roll_max - roll_min
        out[f"m{i}_temp_accel"] = (
            out.groupby(gb_key, sort=False)[f"m{i}_temp_rate"].diff().fillna(0.0)
        )

    # --- Robust long/causal temperature-baseline elevation (fault-resistant) ---
    for i in range(1, 7):
        temp = f"data_motor_{i}_temperature"
        g = out.groupby(gb_key, sort=False)[temp]

        # 1) Trailing low-quantile baseline over a long window (60 s). A sustained
        #    fault occupies the high quantiles, so the 10th percentile stays near
        #    the normal floor and the elevation signal survives long faults.
        qbase = g.transform(
            lambda x: x.rolling(LONG_BASELINE_WIN, min_periods=20).quantile(BASELINE_Q)
        )
        # 2) Causal expanding low-quantile, anchored to the start of the sequence.
        expq = g.transform(lambda x: x.expanding(min_periods=20).quantile(BASELINE_Q))
        # 3) Running minimum since sequence start (hardest-to-contaminate floor).
        cmin = g.transform("cummin")
        # Fall back to the running min early on (before min_periods is reached).
        qbase = qbase.fillna(cmin)
        expq = expq.fillna(cmin)

        out[f"m{i}_temp_above_qbase"] = out[temp] - qbase
        out[f"m{i}_temp_above_expq"] = out[temp] - expq
        out[f"m{i}_temp_above_cummin"] = out[temp] - cmin

        # 4) Normalise the elevation by the motor's own early-sequence temperature
        #    spread so the tiny motor-3 rise (~+1.5 C) is still visible relative to
        #    a quiet motor, while noisy motors are scaled down. (+1 C floor.)
        early_scale = g.transform(lambda x: x.head(EARLY_WIN).std())
        out[f"m{i}_temp_elev_norm"] = out[f"m{i}_temp_above_expq"] / (
            early_scale.fillna(0.0) + 1.0
        )

    # --- Shape-aware temperature features (matched filter / CUSUM / run-length) ---
    # Encode the known fault trajectory (linear rise over n/3, slower decay over
    # 2 n/3) directly. Computed per sequence on the robust elevation signal so they
    # are baseline-relative and never leak across sequences.
    shape_names = ["shape_match", "shape_scale", "cusum_up", "elev_runlen", "slope_asym"]
    group_pos = out.groupby(gb_key, sort=False).indices  # key -> positional idx array
    for i in range(1, 7):
        elev_vals = out[f"m{i}_temp_above_expq"].to_numpy(dtype=float)
        buffers = {nm: np.zeros(len(out)) for nm in shape_names}
        for idx in group_pos.values():
            feats = shp.shape_features(elev_vals[idx])
            for nm in shape_names:
                buffers[nm][idx] = feats[nm]
        for nm in shape_names:
            out[f"m{i}_{nm}"] = buffers[nm]

    # De-fragment: the many column inserts above leave the frame fragmented.
    return out.copy()


def feature_list_for_motor(motor_idx: int) -> list[str]:
    """Feature columns for one motor's classifier (sees the whole robot state)."""
    feats: list[str] = []
    for j in range(1, 7):
        feats.append(f"data_motor_{j}_position")
        feats.append(f"data_motor_{j}_temperature")
        feats.append(f"data_motor_{j}_voltage")

    for j in range(1, 7):
        feats.extend(
            [
                f"m{j}_velocity",
                f"m{j}_accel",
                f"m{j}_temp_rate",
                f"m{j}_temp_accel",
                f"m{j}_volt_rate",
                f"m{j}_pos_roll_std",
                f"m{j}_temp_roll_std",
                f"m{j}_volt_roll_std",
                f"m{j}_temp_drift",
                f"m{j}_temp_above_min200",
                f"m{j}_temp_range200",
                f"m{j}_temp_above_qbase",
                f"m{j}_temp_above_expq",
                f"m{j}_temp_above_cummin",
                f"m{j}_temp_elev_norm",
                f"m{j}_is_moving",
                f"m{j}_volt_dev_peers",
            ]
        )

    feats.extend(
        [
            f"m{motor_idx}_other_vel_recent",
            f"m{motor_idx}_other_volt_rate_recent",
            f"m{motor_idx}_other_moving_recent",
        ]
    )

    # Shape-aware features describing the target motor's own temperature trajectory.
    # Adopted per-motor on held-out evidence (see td9_shape_eval.py): they give a
    # decisive lift on motor 3 and help motors 1/2/5 on held-out additional_data,
    # but hurt the strong motor 4 (OOF AUC/F1) and motor 6 (held-out F1), so those
    # two keep the original feature set. The columns still exist for all motors;
    # only the target motor's are selected here.
    if motor_idx in SHAPE_MOTORS:
        feats.extend(
            [
                f"m{motor_idx}_shape_match",
                f"m{motor_idx}_shape_scale",
                f"m{motor_idx}_cusum_up",
                f"m{motor_idx}_elev_runlen",
                f"m{motor_idx}_slope_asym",
            ]
        )

    feats.append("mean_motor_voltage")
    for j in range(1, 7):
        feats.append(f"data_motor_{j}_temperature_delta0")

    feats.extend(["any_motor_moving", "n_motors_moving"])
    return feats
