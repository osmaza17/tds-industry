"""
TD9 - episode-level decision layer (Viterbi decoding).

The per-sample classifier decides each 0.1 s instant independently, so its output
flickers (fault / normal / fault) even though a real fault is one *contiguous*
temperature episode. This module commits the output to whole episodes: it runs the
exported per-sample fault probabilities through a 2-state ("normal", "fault") hidden
Markov model and returns the single most likely contiguous state path (Viterbi).

- The transition "stickiness" (probability of staying in the fault state) is set
  from the *real* mean fault-episode length per motor, not tuned on the leaderboard.
- A scalar entry-bias is bisection-solved so the decoded positive rate matches the
  same trusted per-motor flag budget we already use, so this only changes the
  *shape* of the predictions (scattered -> contiguous), never the count.

It reuses the saved probabilities, so no retraining is needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RATE_FLOOR = 1e-4


def _group_slices(groups: np.ndarray) -> list[tuple[int, int]]:
    if len(groups) == 0:
        return []
    codes = pd.factorize(groups, sort=False)[0]
    cuts = np.where(np.diff(codes) != 0)[0] + 1
    bounds = np.concatenate(([0], cuts, [len(codes)]))
    return [(int(s), int(e)) for s, e in zip(bounds[:-1], bounds[1:])]


def mean_episode_length(label: np.ndarray, groups: np.ndarray) -> float:
    """Average length (in samples) of contiguous fault runs across sequences."""
    lengths: list[int] = []
    for s, e in _group_slices(groups):
        seg = np.asarray(label[s:e]) == 1
        if not seg.any():
            continue
        # run-length encode the True runs
        idx = np.where(np.diff(seg.astype(int)) != 0)[0] + 1
        bounds = np.concatenate(([0], idx, [len(seg)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if seg[a]:
                lengths.append(b - a)
    return float(np.mean(lengths)) if lengths else 50.0


def viterbi_path(prob: np.ndarray, a11: float, a00: float, bias: float) -> np.ndarray:
    """Most likely normal/fault path for one sequence's fault probabilities.

    ``a11`` = P(stay fault), ``a00`` = P(stay normal); ``bias`` is subtracted from
    the fault log-emission to control how readily the path enters/holds the fault
    state (higher bias -> fewer fault samples). Returns a 0/1 array.
    """
    n = len(prob)
    if n == 0:
        return np.zeros(0, dtype=int)
    eps = 1e-6
    p = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    le1 = np.log(p) - bias
    le0 = np.log(1 - p)
    la11, la10 = np.log(a11), np.log(1 - a11)
    la00, la01 = np.log(a00), np.log(1 - a00)

    d0 = np.empty(n)
    d1 = np.empty(n)
    b0 = np.zeros(n, dtype=np.int8)
    b1 = np.zeros(n, dtype=np.int8)
    d0[0] = le0[0]
    d1[0] = le1[0]
    for t in range(1, n):
        c00 = d0[t - 1] + la00
        c10 = d1[t - 1] + la10
        if c00 >= c10:
            d0[t], b0[t] = c00 + le0[t], 0
        else:
            d0[t], b0[t] = c10 + le0[t], 1
        c11 = d1[t - 1] + la11
        c01 = d0[t - 1] + la01
        if c11 >= c01:
            d1[t], b1[t] = c11 + le1[t], 1
        else:
            d1[t], b1[t] = c01 + le1[t], 0

    path = np.zeros(n, dtype=int)
    path[n - 1] = 1 if d1[n - 1] >= d0[n - 1] else 0
    for t in range(n - 1, 0, -1):
        path[t - 1] = b1[t] if path[t] == 1 else b0[t]
    return path


def decode_all(prob: np.ndarray, groups: np.ndarray, a11: float, a00: float, bias: float) -> np.ndarray:
    out = np.zeros(len(prob), dtype=int)
    for s, e in _group_slices(groups):
        out[s:e] = viterbi_path(prob[s:e], a11, a00, bias)
    return out


def decode_to_rate(
    prob: np.ndarray,
    groups: np.ndarray,
    target_rate: float,
    a11: float,
    *,
    a00: float = 0.995,
    iters: int = 26,
) -> np.ndarray:
    """Viterbi-decode so the contiguous output matches ``target_rate`` of samples."""
    target = float(np.clip(target_rate, RATE_FLOOR, 0.5))

    def rate_for(bias: float) -> tuple[float, np.ndarray]:
        path = decode_all(prob, groups, a11, a00, bias)
        return float(path.mean()), path

    # Higher bias -> fewer fault samples, so rate decreases with bias: bisection.
    lo, hi = -25.0, 25.0
    path = decode_all(prob, groups, a11, a00, 0.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        rate, path = rate_for(mid)
        if rate > target:
            lo = mid  # too many positives -> raise bias
        else:
            hi = mid
    return path
