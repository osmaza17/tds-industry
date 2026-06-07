"""
final_test/combine.py — build a submission taking each motor's column from a chosen (source, threshold).

Per-motor independence of macro-F1 lets us mix: e.g. M2 from the cross_pair model (M4 features help it),
the rest from the plain champion model, each at its own private-optimal threshold.

Spec: dict {motor: (probs_csv, meta_json, threshold_or_None)}  (None -> that source's val threshold)
"""
import os, sys
import pandas as pd
import make_sub as M

OUT = M.OUT


def combine(name, spec):
    cols = {}
    for motor, (probs, meta, thr) in spec.items():
        M.set_source(probs, meta)
        kw = {'thr': {motor: thr}} if thr is not None else {}
        _, _ = M.build(f'_cmb_{motor}', postproc=M.CHAMP_PP, **kw)
        df = pd.read_csv(os.path.join(OUT, f'_cmb_{motor}.csv'))
        cols[motor] = df[f'data_motor_{motor}_label'].values
    out = pd.read_csv(os.path.join(OUT, '_cmb_%d.csv' % list(spec)[0]))  # template
    for motor in range(1, 7):
        if motor in cols:
            out[f'data_motor_{motor}_label'] = cols[motor]
    path = os.path.join(OUT, f'{name}.csv')
    out.to_csv(path, index=False)
    counts = {f'M{m}': int((out[f'data_motor_{m}_label'] == 1).sum()) for m in range(1, 7)}
    return path, counts


PLAIN = ('test_probs.csv', 'train_prev.json')
CROSS = ('test_probs_cross_pair.csv', 'train_prev_cross_pair.json')

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'privmax'
    if which == 'privmax':
        spec = {1: (*PLAIN, 0.5), 2: (*CROSS, 0.7), 3: (*PLAIN, 0.3),
                4: (*PLAIN, None), 5: (*PLAIN, 0.2), 6: (*PLAIN, None)}
    else:  # balanced
        spec = {1: (*PLAIN, 0.1), 2: (*CROSS, 0.7), 3: (*PLAIN, None),
                4: (*PLAIN, None), 5: (*PLAIN, 0.3), 6: (*PLAIN, None)}
    path, counts = combine(f'FINAL2_{which}', spec)
    print(f'FINAL2_{which}', counts, path)
