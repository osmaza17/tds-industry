"""
TD9 - shape-aware temperature features.

The organisers' ``inject_failure`` makes every fault a deterministic temperature
trajectory: a linear rise to a random peak over ``n/3`` samples, then a decay
over ``2 n/3`` (the rise is twice as fast as the decay). A per-sample classifier
cannot "see" that shape from instantaneous values, so here we encode it directly:

- a multi-scale **matched filter** against the canonical rise-then-decay template
  (best for the short, subtle faults of motor 3 that elevation alone misses);
- a one-sided **CUSUM** that accumulates sustained elevation (best for the long
  faults of motors 2/4);
- the **run-length** of the current elevation episode (how long we have stayed up);
- a **slope-asymmetry** signal that fires on the "rose, now decaying" pattern.

Everything is a pure function of the temperature *elevation above baseline* and is
applied per ``test_condition`` (no cross-sequence leakage) by ``td9_features``.
"""

from __future__ import annotations

import numpy as np

# Short matched-filter scales (in samples). We deliberately keep these short: they
# target the brief, subtle faults (motor 3 ~26 samples) that the elevation/CUSUM
# features handle poorly, while long faults are covered by CUSUM + run-length.
DEFAULT_SCALES = (15, 30, 60, 120)
CUSUM_SLACK = 0.5      # deg C of elevation ignored as noise before accumulating
ELEV_MARGIN = 1.0      # deg C above baseline counted as "elevated"
ASYM_LAG = 15          # samples between the rise window and the fall window


def rise_decay_template(length: int) -> np.ndarray:
    """Canonical fault shape: linear rise over L/3, linear decay over 2L/3.

    Returned zero-mean and unit-norm so a sliding dot product is a correlation.
    """
    length = max(int(length), 3)
    n_rise = max(length // 3, 1)
    rise = np.linspace(0.0, 1.0, n_rise, endpoint=False)
    decay = np.linspace(1.0, 0.0, length - n_rise)
    tmpl = np.concatenate([rise, decay])
    tmpl = tmpl - tmpl.mean()
    norm = np.linalg.norm(tmpl)
    return tmpl / norm if norm > 0 else tmpl


def _sliding_corr(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Normalised correlation of a window centred at each sample with ``template``."""
    L = len(template)
    n = len(x)
    pad_left = L // 2
    pad_right = L - 1 - pad_left
    xp = np.pad(x, (pad_left, pad_right), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(xp, L)  # (n, L)
    win = win - win.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(win, axis=1)
    norm[norm == 0] = 1.0
    return (win @ template) / norm


def matched_filter(elev: np.ndarray, scales=DEFAULT_SCALES) -> tuple[np.ndarray, np.ndarray]:
    """Best correlation with the rise-decay template over several scales.

    Returns ``(best_corr, best_scale)`` per sample: how strongly the local shape
    matches a fault, and the template length that matched best (a fault-duration
    hint). ``best_corr`` is clipped at 0 (anti-correlation is not informative).
    """
    elev = np.asarray(elev, dtype=float)
    n = len(elev)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    best = np.full(n, -2.0)
    best_scale = np.zeros(n)
    for L in scales:
        if L > n:
            continue
        corr = _sliding_corr(elev, rise_decay_template(L))
        upd = corr > best
        best = np.where(upd, corr, best)
        best_scale = np.where(upd, float(L), best_scale)
    best[best < 0] = 0.0
    return best, best_scale


def cusum_up(elev: np.ndarray, slack: float = CUSUM_SLACK) -> np.ndarray:
    """One-sided CUSUM: accumulates elevation above ``slack``, resets to 0 below.

    Vectorised via the running-minimum identity
    ``S_i = P_i - min(0, min_{j<=i} P_j)`` where ``P`` is the cumulative sum.
    """
    elev = np.asarray(elev, dtype=float)
    if len(elev) == 0:
        return np.zeros(0)
    p = np.cumsum(elev - slack)
    running_min = np.minimum.accumulate(p)
    return p - np.minimum(running_min, 0.0)


def elevation_runlen(elev: np.ndarray, margin: float = ELEV_MARGIN) -> np.ndarray:
    """Consecutive samples that elevation has stayed above ``margin`` (episode age).

    Vectorised: for each elevated sample, the run length is its index minus the
    index of the most recent non-elevated sample.
    """
    elev = np.asarray(elev, dtype=float)
    n = len(elev)
    if n == 0:
        return np.zeros(0)
    above = elev > margin
    idx = np.arange(n)
    last_below = np.maximum.accumulate(np.where(~above, idx, -1))
    return np.where(above, idx - last_below, 0).astype(float)


def slope_asymmetry(elev: np.ndarray, lag: int = ASYM_LAG) -> np.ndarray:
    """Fires on the "rose, then now decaying" pattern (the fault's decay side).

    ``rose`` = elevation gain from ``2*lag`` to ``lag`` ago; ``fell`` = elevation
    drop from ``lag`` ago to now. Their product (both positive) peaks on the slow
    decay that follows a fault peak - the signature the template misses at the tail.
    """
    elev = np.asarray(elev, dtype=float)
    n = len(elev)
    if n == 0:
        return np.zeros(0)

    def shift(a, k):
        out = np.empty_like(a)
        if k <= 0:
            return a.copy()
        out[:k] = a[0]
        out[k:] = a[:-k]
        return out

    e_lag = shift(elev, lag)
    e_2lag = shift(elev, 2 * lag)
    rose = np.clip(e_lag - e_2lag, 0.0, None)
    fell = np.clip(e_lag - elev, 0.0, None)
    return rose * fell


def shape_features(elev: np.ndarray, scales=DEFAULT_SCALES) -> dict[str, np.ndarray]:
    """All shape features for one sequence's elevation array."""
    match, scale = matched_filter(elev, scales)
    return {
        "shape_match": match,
        "shape_scale": scale,
        "cusum_up": cusum_up(elev),
        "elev_runlen": elevation_runlen(elev),
        "slope_asym": slope_asymmetry(elev),
    }
