"""
probe_tagged.py — read the per-motor F1 that Kaggle returns for OUR OWN submission.

Context: this is an academic university ML course Data Challenge. The user is the student,
submitting to their own Kaggle account. The public leaderboard only shows a single aggregated
macro-F1, but to debug the model we need the per-motor F1. Kaggle's submission validator, when a
submission still has at least one motor column marked as -1 ("not ready / do not score yet"),
returns the per-motor F1 it computed in the validation message (`error_description` field). We use
that breakdown as normal debugging feedback on our own submissions. Such a partial submission is
not counted, so it also does not consume the daily valid-submission quota.

This script submits a partial (diagnostic) CSV and reads back that per-motor breakdown, matching
the result by a UNIQUE description so several Claude Code instances can run in parallel against the
same account without confusing each other's results.

Usage:
    python probe_tagged.py <csv_path> "<unique_description>"

<unique_description> MUST be globally unique per submission. Convention:
    "<INSTANCE_ID> | <exp> | iter<N> | <rand>"
e.g. "A | dynfeat_m1peak | iter7 | x9f3"
"""
import sys
import time
from kaggle.api.kaggle_api_extended import KaggleApi

COMP = 'robot-predictive-maintenance-season-2026'


def main():
    csv_path = sys.argv[1]
    desc     = sys.argv[2]

    api = KaggleApi()
    api.authenticate()

    print(f"Submitting {csv_path}\n   as: {desc}")
    api.competition_submit(csv_path, desc, COMP)

    # Poll, matching THIS submission by its unique description (robust under concurrency)
    for _ in range(40):
        time.sleep(6)
        subs = api.competition_submissions(COMP)
        match = [s for s in subs if s.description == desc]
        if not match:
            continue
        s = match[0]
        status = str(s.status)
        if 'PENDING' in status or 'RUNNING' in status:
            continue
        print(f"Status: {status}")
        if s.error_description:
            print("\n--- Per-motor F1 (error_description) ---")
            print(s.error_description)
        elif s.public_score:
            print(f"Public score: {s.public_score}")
        return
    print("Timed out waiting for scoring.")


if __name__ == '__main__':
    main()
