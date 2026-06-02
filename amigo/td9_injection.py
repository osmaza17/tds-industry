"""
TD9 - synthetic fault injection.

The challenge organisers simulate failures with a known MATLAB routine
(``inject_failure``): inside a chosen fault window the temperature rises to a
random peak ``+ randi([2, 10])`` above the boundary temperature and then decays,
with the *rise being twice as fast as the descent* (n_rise = floor(n/3)).

We reproduce that exact behaviour in Python and use it to manufacture extra,
physically-consistent fault examples on *normal* sequences. This is used only to
augment the training side (never validation), and is aimed primarily at the
motors with almost no real faults (motors 3 and 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MOTOR_SENSORS = ("position", "temperature", "voltage")


def inject_failure(
    temperature: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
    *,
    peak_low: int = 2,
    peak_high: int = 10,
) -> np.ndarray:
    """Python port of the organisers' MATLAB ``inject_failure``.

    Overwrites the temperature inside the ``label == 1`` region with a
    rise-then-decay profile (rise 2x faster than the descent). Returns a new
    array; the input is not modified.

    The peak temperature is drawn from ``randi([peak_low, peak_high])`` above the
    boundary temperature. The organisers' default is ``[2, 10]``; for motors
    whose *real* faults show only a tiny temperature rise (motors 3 and 5) we
    pass a smaller, more varied range so the synthetic faults match the physics.
    """
    temperature = np.asarray(temperature, dtype=float).copy()
    label = np.asarray(label)
    mask = label == 1
    tmp = temperature[mask].astype(float)
    n_seq = len(tmp)
    if n_seq == 0:
        return temperature

    n_rise = n_seq // 3
    if n_rise < 1:
        return temperature  # window too short to host a meaningful profile

    temp_start = tmp[0]
    temp_end = tmp[-1]
    temp_high = max(temp_start, temp_end) + int(rng.integers(peak_low, peak_high + 1))

    rise_span = int(round(temp_high - temp_start))
    # --- rising phase (0-based port of the MATLAB inclusive loops) ---
    if rise_span >= 1:
        step_rise = (n_rise // (rise_span + 1)) or 1
        i = 0
        for i in range(1, rise_span + 1):
            lo = (i - 1) * step_rise
            hi = min(i * step_rise, n_rise)
            if lo >= n_rise:
                break
            tmp[lo:hi] = temp_start + (i - 1)
        filled = i * step_rise
        if filled < n_rise:
            tmp[filled:n_rise] = temp_high
    else:
        tmp[:n_rise] = temp_high

    # --- descending phase (twice as slow as the rise) ---
    down_span = int(round(temp_high - temp_end))
    if down_span >= 1:
        step_down = (2 * n_rise // down_span) or 1
        i = 0
        for i in range(1, down_span):
            lo = n_rise + (i - 1) * step_down
            hi = min(n_rise + i * step_down, n_seq)
            if lo >= n_seq:
                break
            tmp[lo:hi] = temp_high - i
        filled = n_rise + i * step_down
        if filled < n_seq:
            tmp[filled:] = temp_end
    else:
        tmp[n_rise:] = temp_end

    temperature[mask] = tmp
    return temperature


def make_injected_sequence(
    seq_df: pd.DataFrame,
    motor: int,
    start: int,
    length: int,
    rng: np.random.Generator,
    new_id: str,
    *,
    peak_low: int = 2,
    peak_high: int = 10,
) -> pd.DataFrame:
    """Copy one normal sequence and inject a synthetic fault on ``motor``.

    Only ``motor``'s label/temperature change; every other column (including the
    other motors) is preserved so cross-motor features stay realistic.
    """
    out = seq_df.copy().reset_index(drop=True)
    n = len(out)
    start = max(0, min(start, n - 1))
    end = min(start + length, n)
    if end - start < 3:
        return out.iloc[0:0]

    lab_col = f"data_motor_{motor}_label"
    temp_col = f"data_motor_{motor}_temperature"

    label = np.zeros(n, dtype=int)
    label[start:end] = 1
    new_temp = inject_failure(
        out[temp_col].to_numpy(), label, rng, peak_low=peak_low, peak_high=peak_high
    )

    out[temp_col] = new_temp
    out[lab_col] = label
    # Void every OTHER motor's label on this synthetic sequence so it trains only
    # the motor it was made for. Otherwise a base sequence chosen to host a
    # motor-3 fault would also re-feed its (real) motor-6 labels, duplicating
    # those faults many times over and biasing motor 6's model.
    for j in range(1, 7):
        if j != motor:
            out[f"data_motor_{j}_label"] = np.nan
    out["test_condition"] = new_id
    out["source"] = "injected"
    return out


def _normal_sequences_for_motor(df_train: pd.DataFrame, motor: int) -> list[str]:
    """Training sequences with no real fault for ``motor`` (safe to overwrite)."""
    lab_col = f"data_motor_{motor}_label"
    keep: list[str] = []
    for seq, g in df_train.groupby("test_condition", sort=False):
        if int((g[lab_col] == 1).sum()) == 0:
            keep.append(seq)
    return keep


def synthesize_for_motor(
    df_train: pd.DataFrame,
    motor: int,
    *,
    n_sequences: int,
    rng: np.random.Generator,
    min_len: int = 120,
    max_len: int = 400,
    peak_low: int = 2,
    peak_high: int = 10,
) -> pd.DataFrame:
    """Manufacture ``n_sequences`` synthetic-fault sequences for one motor."""
    base_ids = _normal_sequences_for_motor(df_train, motor)
    if not base_ids:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    made = 0
    attempts = 0
    while made < n_sequences and attempts < n_sequences * 5:
        attempts += 1
        seq_id = base_ids[int(rng.integers(0, len(base_ids)))]
        seq_df = df_train[df_train["test_condition"] == seq_id]
        n = len(seq_df)
        if n < min_len + 10:
            continue
        length = int(rng.integers(min_len, min(max_len, n - 5) + 1))
        start = int(rng.integers(5, n - length - 1)) if n - length - 1 > 5 else 5
        new_id = f"inject_m{motor}_{made:03d}"
        inj = make_injected_sequence(
            seq_df, motor, start, length, rng, new_id,
            peak_low=peak_low, peak_high=peak_high,
        )
        if len(inj) == 0:
            continue
        frames.append(inj)
        made += 1

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Per-motor injection profiles. Real motor-3/5 faults show only a tiny, varied
# temperature rise (~+1..+3 C) over a wide range of durations, so we inject many
# subtle, short-to-long faults for them rather than the organisers' default
# +2..+10 C peaks. The other motors keep the default profile and a few examples.
PER_MOTOR_INJECT: dict[int, dict] = {
    1: dict(n=4, peak=(2, 10), dur=(120, 400)),
    2: dict(n=4, peak=(2, 10), dur=(120, 400)),
    3: dict(n=16, peak=(1, 4), dur=(40, 500)),
    4: dict(n=4, peak=(2, 10), dur=(120, 400)),
    5: dict(n=16, peak=(1, 4), dur=(40, 500)),
    6: dict(n=4, peak=(2, 10), dur=(120, 400)),
}


def synthesize_all(
    df_train: pd.DataFrame,
    *,
    per_motor: dict[int, dict] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the full injected pool.

    ``per_motor`` maps motor -> ``{"n", "peak": (low, high), "dur": (min, max)}``.
    Defaults (``PER_MOTOR_INJECT``) focus on the fault-starved motors (3 and 5)
    with subtle, physically-matched temperature rises while adding a few default
    examples for the others.
    """
    if per_motor is None:
        per_motor = PER_MOTOR_INJECT
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for motor, cfg in per_motor.items():
        n = int(cfg.get("n", 0))
        if n <= 0:
            continue
        peak_low, peak_high = cfg.get("peak", (2, 10))
        min_len, max_len = cfg.get("dur", (120, 400))
        df_m = synthesize_for_motor(
            df_train, motor, n_sequences=n, rng=rng,
            min_len=min_len, max_len=max_len,
            peak_low=peak_low, peak_high=peak_high,
        )
        if not df_m.empty:
            frames.append(df_m)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import td9_data as d

    data = d.load_all()
    inj = synthesize_all(data["train"])
    print("injected rows:", len(inj))
    if not inj.empty:
        for m in range(1, 7):
            lc = f"data_motor_{m}_label"
            seqs = inj[inj[lc] == 1]["test_condition"].nunique()
            pos = int((inj[lc] == 1).sum())
            print(f"  motor {m}: {pos} synthetic fault rows across {seqs} sequences")
