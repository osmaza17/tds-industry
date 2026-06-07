"""
final_test/sweep.py — per-motor threshold sweep against the PRIVATE score.

macro-F1 = mean of independent per-motor F1, so we optimise one motor at a time:
build = champion for all motors EXCEPT the swept motor (which gets a trial threshold/rank),
submit, read private. The swept motor's private F1 = 6*(priv - PRIV_BASE) + F1_BASE_motor.

Usage:
    python final_test/sweep.py M2 thr 0.1 0.3 0.5 0.7
    python final_test/sweep.py M4 rank 0.02 0.05 0.09
    python final_test/sweep.py M3 thr 0.2 0.4 0.6     (lower than champ 0.7 to recover faults)
Results appended to final_test/results.csv
"""
import sys, os, csv, time
import make_sub as M
from run import submit_read

OUT = M.OUT
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')

# champion private per-motor F1 (from zero-tests, final_test/LOG.md). None = unknown.
PRIV_BASE = 0.30437
CHAMP_PRIV_F1 = {1: 0.0, 2: 0.264, 3: 0.0, 4: 1.0, 5: 0.023, 6: 0.539}


def log_result(row):
    new = not os.path.exists(RESULTS)
    with open(RESULTS, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['name', 'motor', 'mode', 'value', 'public', 'private',
                        'priv_delta_vs_champ', 'derived_motor_priv_f1', 'counts'])
        w.writerow(row)


def main():
    motor = int(sys.argv[1].lstrip('Mm'))
    mode  = sys.argv[2]                       # 'thr' or 'rank'
    values = [float(x) for x in sys.argv[3:]]
    for v in values:
        name = f'S_M{motor}_{mode}_{v}'
        kw = {('thr' if mode == 'thr' else 'rank'): {motor: v}}
        path, counts = M.build(name, postproc=M.CHAMP_PP, **kw)
        desc = f'sweep M{motor} {mode}={v} (others=champ)'
        pub, priv, st = submit_read(path, desc)
        if priv is None:
            print(f'{name}: {st}'); continue
        dpriv = float(priv) - PRIV_BASE
        derived = round(6 * dpriv + (CHAMP_PRIV_F1[motor] if CHAMP_PRIV_F1[motor] is not None else 0), 3)
        log_result([name, motor, mode, v, pub, priv, round(dpriv, 4), derived, counts[f'M{motor}']])
        base_note = '' if CHAMP_PRIV_F1[motor] is not None else ' (champ base unknown -> derived is delta only)'
        pcount = counts[f'M{motor}']
        print(f'{name}: pub={pub} priv={priv}  dpriv={dpriv:+.4f}  '
              f'M{motor}_priv_F1~={derived}{base_note}  (preds={pcount})', flush=True)
        time.sleep(3)


if __name__ == '__main__':
    main()
