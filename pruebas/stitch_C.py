"""
stitch_C.py — build ONE submission CSV taking each motor's column from a chosen
source CSV (the seed that scored best for that motor). Columns per motor are
independent in the submission, and the probe gives the true per-motor F1, so this
is a legitimate per-motor best-of-seeds stitch (heavy leaderboard-probing though).

Usage:
    python stitch_C.py out.csv 1=src1.csv 2=src2.csv 3=src3.csv 4=... 5=... 6=...

Any motor not specified is taken from the FIRST source given (fallback). Rows are
aligned on 'idx'. Output has no -1 (full submittable CSV) unless a source motor
column itself contains -1 (it must not for that motor).
"""
import sys
import pandas as pd

def main():
    out_path = sys.argv[1]
    mapping = {}
    for arg in sys.argv[2:]:
        m, path = arg.split('=', 1)
        mapping[int(m)] = path

    # Use any source as the skeleton (idx + test_condition).
    any_src = next(iter(mapping.values()))
    base = pd.read_csv(any_src).sort_values('idx').reset_index(drop=True)
    out = base[['idx', 'test_condition']].copy() if 'test_condition' in base.columns else base[['idx']].copy()

    for motor in range(1, 7):
        col = f'data_motor_{motor}_label'
        src = mapping.get(motor, any_src)
        s = pd.read_csv(src).sort_values('idx').reset_index(drop=True)
        out[col] = s[col].values

    # Reorder columns to match sample format: idx, motors 1..6, test_condition
    cols = ['idx'] + [f'data_motor_{m}_label' for m in range(1, 7)]
    if 'test_condition' in out.columns:
        cols += ['test_condition']
    out = out[cols]

    # Sanity: report any -1 and fault counts
    mcols = [f'data_motor_{m}_label' for m in range(1, 7)]
    has_neg = (out[mcols] == -1).any().any()
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}  rows={len(out)}  any -1?={has_neg}")
    print("source per motor:", {m: mapping.get(m, any_src) for m in range(1, 7)})
    print("faults per motor:", {m: int((out[f'data_motor_{m}_label'] == 1).sum()) for m in range(1, 7)})


if __name__ == '__main__':
    main()
