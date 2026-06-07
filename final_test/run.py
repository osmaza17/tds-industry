"""
final_test/run.py — submit a CSV, wait for scoring, print public+private.
Usage: python final_test/run.py <csv_path> "<description>"
"""
import sys, time
from kaggle.api.kaggle_api_extended import KaggleApi

COMP = 'robot-predictive-maintenance-season-2026'


def submit_read(csv_path, desc, timeout=400):
    api = KaggleApi(); api.authenticate()
    for attempt in range(6):
        try:
            api.competition_submit(csv_path, desc, COMP); break
        except Exception as e:
            print(f'  retry {attempt+1}: {e.__class__.__name__}', flush=True); time.sleep(15)
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = [s for s in api.competition_submissions(COMP) if s.description == desc]
        if m:
            st = str(m[0].status)
            if 'PENDING' not in st and 'RUNNING' not in st:
                s = m[0]
                return s.public_score, s.private_score, str(s.status).split('.')[-1]
        time.sleep(6)
    return None, None, 'TIMEOUT'


if __name__ == '__main__':
    csv, desc = sys.argv[1], sys.argv[2]
    pub, priv, st = submit_read(csv, desc)
    print(f'RESULT | {desc}\n  status={st}  public={pub}  private={priv}')
