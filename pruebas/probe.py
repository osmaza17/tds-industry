"""
Submit a probe CSV (one motor = -1) to Kaggle and read back the per-motor F1
that leaks through the scoring error message. Probes do NOT consume the daily
submission quota.

Usage:
    python probe.py ../pruebas/sub_dynfeat.csv "dynfeat: dynamic features M1/M4"
"""
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

COMP = 'robot-predictive-maintenance-season-2026'


def main():
    csv_path = sys.argv[1]
    message  = sys.argv[2] if len(sys.argv) > 2 else 'probe'

    api = KaggleApi()
    api.authenticate()

    print(f"Submitting {csv_path} ...")
    api.competition_submit(csv_path, message, COMP)

    # Poll until the latest submission resolves
    for _ in range(30):
        time.sleep(6)
        subs = api.competition_submissions(COMP)
        s = subs[0]
        status = str(s.status)
        if 'PENDING' not in status and 'RUNNING' not in status:
            print(f"\nStatus: {status}")
            if s.public_score:
                print(f"Public score: {s.public_score}")
            if s.error_description:
                print("\n--- Per-motor F1 (error_description) ---")
                print(s.error_description)
            return
    print("Timed out waiting for scoring.")


if __name__ == '__main__':
    main()
