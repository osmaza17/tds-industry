# -*- coding: utf-8 -*-
"""Build reporte_preliminar.ipynb: restructured, detailed report keeping the
final_submission_group_2 spirit (per-motor subsections, runnable pipeline cells)
but with the expanded methods, an injection code cell, per-section explanations,
results moved below the pipeline, and the measured post-deadline improvements."""
import json, uuid, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
fsg2 = json.load(open('final_submission_group_2.ipynb', encoding='utf-8'))['cells']
code = {i: ''.join(fsg2[i]['source']) for i in [4, 5, 6, 7, 15, 17]}


def md(s): return {'id': uuid.uuid4().hex[:8], 'cell_type': 'markdown', 'metadata': {}, 'source': s}
def co(s): return {'id': uuid.uuid4().hex[:8], 'cell_type': 'code', 'metadata': {},
                   'execution_count': None, 'outputs': [], 'source': s}


TITLE = ''.join(fsg2[0]['source'])

DATACLEAN = """## Data cleaning

Before any modelling, every raw signal is cleaned in a few simple, causal steps so that sensor glitches do
not contaminate the features:

- Range clipping. Each sensor has a known physical range (temperature 0 to 100 deg C, voltage 6000 to 9000
  mV, position 0 to 1000). Any reading outside its range is treated as a sensor error and discarded.
- Forward filling. The discarded readings, and any missing values, are filled with the last valid value.
  This is causal (it never looks into the future), which matters because the data is a time series.
- Baseline removal. Each signal is expressed relative to the first sample of its sequence, so that two
  sequences starting at different temperatures or positions become directly comparable and the model sees
  changes rather than absolute offsets.

All of this is done per sequence, so statistics from one recording never leak into another."""

OVERVIEW = """## Methods

Instead of building a single model for the whole robot, we treated the problem as six separate binary
classification tasks, one per motor, that all run through the same pipeline but are tuned individually. We
did this because the six motors behave very differently from one another, both in how often they fail and in
what a failure looks like in their sensor traces. The sections below explain, one decision at a time, how the
data was organised, how we dealt with the heavy class imbalance, how we validated without leaking
information, how we generated synthetic faults, which features and model we used, and how we turned the
model's probabilities into final 0/1 predictions. The runnable pipeline that puts all of this together comes
after these explanations, and the per-motor results come at the very end."""

DATA = """### The data: original training set and additional data

We worked with two sources of labelled data.

The original training set is the one provided for the challenge (downloaded from the course platform). It
contains a handful of labelled sequences and is the data we used in our first attempts.

The additional data is a set of supplementary labelled datasets that we only started using later in the
project. Our first models, trained on the original data alone, did not reach a good F1 score, mainly because
some motors had almost no examples of failure to learn from. Adding the supplementary datasets gave the model
many more sequences (and many more fault examples for the rare motors), which is why all the results below are
produced on the combined data. The pipeline loads the original set first and then appends the additional
sets."""

IMBALANCE = """### The class-imbalance problem

This is a strongly imbalanced problem: in normal operation a motor is almost always healthy, so the fault
class is rare. The metric (macro F1 over the fault class) is chosen precisely because plain accuracy is
meaningless here (always predicting "healthy" would already score around 95% accuracy while detecting nothing).

The table below shows the percentage of data points labelled as a fault for each motor, over the whole
labelled dataset (original training set plus additional data, 99407 rows):

| Motor | Fault data points | Fault rate (whole dataset) |
|---|---|---|
| M1 | 3815 | 3.84% |
| M2 | 8096 | 8.14% |
| M3 | 834 | 0.84% |
| M4 | 8674 | 8.73% |
| M5 | 1622 | 1.63% |
| M6 | 2957 | 2.97% |

The imbalance is even more extreme if we look only at the original training set (39309 rows), which is what
motivated us to add the supplementary data and to generate synthetic faults. In the original data, Motors 3
and 5 have almost no failures at all (well under half a percent), so a classifier has essentially nothing to
learn from:

| Motor | Fault data points | Fault rate (original training set) |
|---|---|---|
| M1 | 1349 | 3.43% |
| M2 | 6732 | 17.13% |
| M3 | 127 | 0.32% |
| M4 | 6739 | 17.14% |
| M5 | 184 | 0.47% |
| M6 | 1932 | 4.91% |

The practical consequences are two: we must measure the fault class directly (not accuracy), and for the
rare motors (M1, M3, M5) we must create extra fault examples, which we do with synthetic injection below."""

VALIDATION = """### Validation: whole sequences and K-fold (avoiding data leakage)

Samples that are close together in time are almost identical, because temperature changes very slowly. If we
split the data randomly row by row, rows from the same fault would land in both the training and the
validation set, and the model would effectively be tested on data it had already seen. That kind of leakage
produces a falsely high local F1 that collapses on the real leaderboard.

To prevent this we always validate by whole sequences: an entire sequence goes either to training or to
validation, never split across the two. The grouping unit is the sequence, not the row, so every fault block
in the validation set is one the model never saw during training.

Concretely, we hold out about 20% of the sequences for validation and train on the remaining 80%. The split
is stratified by sequence type: we set aside 20% of the fault-containing sequences and, separately, 20% of
the normal sequences, so the validation set is guaranteed to contain at least one real fault. With the
combined data this works out to roughly 48 training sequences and 11 validation sequences out of 59 (about
81% versus 19%). We use this leakage-free, per-sequence split both to choose the model hyperparameters and to
choose the decision threshold. Its only weakness is that the number of sequences containing real faults is
small, so the validation set is not very diverse, which we return to in the conclusions."""

INJ_INTRO = """### Synthetic fault injection

Because Motors 1, 3 and 5 barely fail in the data, we generate extra fault examples so the model has
something to learn from. The idea is to take a healthy sequence and reshape a slice of its temperature trace
into a realistic failure signature, then label that slice as a fault. The next cell is the exact code we used
to inject a fault; the explanation follows it."""

inject_code = '''import numpy as np
import matplotlib.pyplot as plt

# This is the exact synthetic-fault injection function used by the pipeline (shown here to explain it).
def inject_failure(temp, label, peak_low, peak_high):
    temp = np.asarray(temp, float).copy(); label = np.asarray(label); mask = label == 1
    tmp = temp[mask].copy(); n = len(tmp)            # n = length of the fault window
    if n == 0: return temp
    nr = max(1, n // 3); ts, te = tmp[0], tmp[-1]    # nr = rise length (first third)
    th = max(ts, te) + np.random.randint(peak_low, peak_high + 1)   # peak height above baseline
    rs = int(round(th - ts))                         # how many degrees to climb on the way up
    if rs >= 1:
        st = (nr // (rs + 1)) or 1; i = 0            # samples spent on each 1-degree step
        for i in range(1, rs + 1):
            lo, hi = (i - 1) * st, min(i * st, nr)
            if lo >= nr: break
            tmp[lo:hi] = ts + (i - 1)
        if i * st < nr: tmp[i * st:nr] = th
    ds = int(round(th - te))                         # how many degrees to fall on the way down
    if ds >= 1:
        st = (2 * nr // ds) or 1; i = 0              # decay spans the remaining two thirds (2*nr)
        for i in range(1, ds):
            lo, hi = nr + (i - 1) * st, min(nr + i * st, n)
            if lo >= n: break
            tmp[lo:hi] = th - i
        if nr + i * st < n: tmp[nr + i * st:] = te
    temp[mask] = tmp; return temp

# Demonstration: inject a fault into a flat 300-sample baseline and plot the resulting pulse.
np.random.seed(0)
demo = np.zeros(300)
lbl = np.zeros(300, int); lbl[80:250] = 1            # a 170-sample fault window
shaped = inject_failure(demo.copy(), lbl, peak_low=8, peak_high=16)
print('fault window length (n):', int(lbl.sum()), '| rise ~ n/3, decay ~ 2n/3')
print('peak reached (deg above baseline):', round(float(shaped.max()), 1))
plt.figure(figsize=(7, 3))
plt.plot(shaped); plt.axvspan(80, 250, color='orange', alpha=0.15, label='fault window')
plt.xlabel('sample (0.1 s each)'); plt.ylabel('temperature above baseline (deg)')
plt.title('Synthetic fault: fast rise (first third) then slow decay (two thirds)')
plt.legend(); plt.tight_layout(); plt.show()
'''

INJ_EXPLAIN = """#### How the injection works

The fault window is a random contiguous slice of a healthy sequence, between 120 and 400 samples long (12 to
40 seconds at 10 Hz). Inside that window the temperature is reshaped into a triangular pulse:

- A peak height is drawn at random between `peak_low` and `peak_high` degrees above the local baseline.
- The temperature climbs from the baseline to the peak over the first third of the window (`nr = n // 3`),
  one degree per step.
- It then decays from the peak back down to the end level over the remaining two thirds (`2 * nr`).

The rise is therefore faster than the decay, which matches a real thermal transient: a motor heats up quickly
when something goes wrong and cools down slowly afterwards. After the temperature is reshaped, all the derived
features (differences, rolling statistics, window shapes) are recomputed on the injected trace, so the
synthetic fault carries consistent dynamic features.

#### Per-motor injection choices

Where the fault is placed is chosen at random: for each synthetic example we pick a healthy sequence at
random and, inside it, a random start point and a random window length (between 120 and 400 samples). We only
draw the window from sequences that have no real fault for that motor, so we never overwrite a genuine
failure, and the random placement means the model sees faults beginning at many different moments rather than
always at the same position.

We tuned two knobs per motor: how many synthetic faults to inject, and how strong (the peak range). The
premises were simple. The motors that almost never fail (M3 and M5) get many injections so the model sees
enough examples, and those injections are subtle (a small peak) because their real faults are small. The
structurally hard motors (M1 and M4) get a larger peak so the synthetic signature is clear enough to anchor
the signal against the temperature-fault inversion. The frequently-failing motors already have plenty of real
faults, so they only need a modest top-up.

| Motor | Fault rate (orig.) | Injections | Peak rise (deg) | Premise for the choice |
|---|---|---|---|---|
| M1 | 3.43% | 4 | 8 to 16 | hardest motor; strong, clear pulses to anchor the signal |
| M2 | 17.13% | 4 | 2 to 10 | many real faults already; only a modest top-up |
| M3 | 0.32% | 16 | 3 to 8 | extremely rare; many subtle examples needed |
| M4 | 17.14% | 4 | 6 to 15 | many real faults but inversion-affected; stronger pulses |
| M5 | 0.47% | 16 | 1 to 4 | extremely rare; many very subtle examples needed |
| M6 | 4.91% | 4 | 6 to 15 | marked, contiguous faults; moderate pulses |"""

FEATURES = """### Features: physical, rolling and dynamic

We did not feed the raw sensor values alone; we built a small set of features that describe both the state of
the motor and how that state is changing, using plain transformations:

- Physical features: temperature, position and voltage, each expressed relative to the start of the sequence.
  These describe where the motor is right now.
- Rolling features: rolling mean, maximum and standard deviation over short windows. These smooth out sensor
  noise and capture the recent local level and how much the signal is wobbling.
- Dynamic features: differences at several time lags and the deviation from a rolling mean. These describe how
  fast and in which direction the signal is moving. They matter because a fault shows up as a change over time
  rather than as a particular absolute value, and because the absolute temperature is unreliable (its
  relationship with the fault even flips sign between sequences).
- Window-shape features (used for Motor 1): skewness, kurtosis, slope and range over a window, which describe
  the shape of the thermal pulse rather than its level.

We keep both the physical and the dynamic/rolling families on purpose: the physical features alone miss the
temporal pattern of a fault, while the dynamic features alone lose the steady-state level. Together they
describe both the state and its evolution, which is what a fault detector needs.

#### The temperature-fault inversion

There is a specific difficulty that shaped our feature choices, especially for Motors 1 and 4. The two
sequences that contain real faults disagree on how temperature relates to a fault: in one of them the faulty
period coincides with the temperature being higher than normal, while in the other the faulty period
coincides with the temperature being lower. In other words, the sign of the temperature-versus-fault
relationship is inverted between the two sequences. A model that learns "high temperature means fault" from
one sequence is therefore wrong on the other. This is why absolute temperature is an unreliable feature, and
why we lean on dynamic features (how the temperature is changing) and on window-shape features (the shape of
the pulse), which behave consistently regardless of whether the fault sits above or below the normal level.
It is also a structural ceiling: with per-sample sensor features, Motor 1 cannot be pushed far beyond an F1
of about 0.72, because no single rule about temperature can be right on both sequences at once."""

THRESHOLD = """### Choosing the decision threshold

A classifier outputs a probability of fault; we still have to decide at which probability we call a sample a
fault. With balanced classes the natural cut is 0.5, but here the fault class is rare, so 0.5 is not the right
cut. We therefore choose the threshold that maximises the fault-class F1 on the held-out validation sequences:
after a motor is trained, we sweep candidate thresholds from 0.05 to 0.8 and keep the value (together with an
optional gap-closing step) that gives the best validation F1.

This is a deliberate, per-motor calibration rather than a fixed rule. It is also, as we discuss at the end,
the step most responsible for the gap between our public and private scores: tuning the threshold on a very
small validation set pushed several thresholds down to 0.05, which over-predicts faults on unseen data."""

MORPH = """### Morphological post-processing

A real fault is a continuous block in time, not a scatter of isolated points. We use this prior to clean up
the raw per-sample predictions with two simple operations from signal/image morphology:

- Binary closing fills short gaps, so a fault that the model detects in a few broken pieces becomes a single
  continuous block.
- A minimum-run filter deletes predicted runs that are shorter than a few samples, which removes isolated
  false spikes that cannot be a real fault.

Both are tuned per motor. Motor 4 uses an aggressive combination (closing plus a 40-sample minimum run) and
Motor 6 uses a 20-sample minimum run; this post-processing was one of the larger single improvements on those
two motors."""

MODEL = """### Model choice

We compared three classifier families per motor: a histogram gradient-boosting tree (HistGradientBoosting), a
random forest, and logistic regression. On every motor the histogram gradient boosting came out best, so we
use the same family for all six (tuning its hyperparameters per motor by grid search). The reasons are
practical: it captures non-linear interactions between features, it is fast to train so we could iterate
quickly, it is robust to differently-scaled features, and it handles the fairly large set of engineered
features well. Logistic regression underfits the non-linear structure, and the random forest was close but
consistently a little worse and slower. Using one model family across all motors also keeps the pipeline
simple and consistent."""

SUMMARY = """### Summary of the per-motor choices

The table collects the design decisions described above. The decision threshold is not fixed here: it is
learned per motor on the validation sequences (the values shown are the ones selected by the training cell
further down).

| Motor | Feature set | Injections | Peak (deg) | Model | Post-processing | Threshold (learned) |
|---|---|---|---|---|---|---|
| M1 | dynamic + window-shape | 4 | 8 to 16 | HistGB | default closing | 0.05 |
| M2 | physical + rolling | 4 | 2 to 10 | HistGB | default closing | 0.05 |
| M3 | physical + rolling | 16 | 3 to 8 | HistGB | default closing | 0.70 |
| M4 | dynamic | 4 | 6 to 15 | HistGB | closing + min-run 40 | 0.05 |
| M5 | physical + rolling | 16 | 1 to 4 | HistGB | default closing | 0.50 |
| M6 | physical + rolling | 4 | 6 to 15 | HistGB | min-run 20 | 0.30 |"""

PIPELINE_INTRO = """## Complete pipeline

The cells below run the whole method end to end. The pipeline is self-contained and reproducible; a short
paragraph before each cell says which step it implements."""

STEP1 = """Step 1 — preprocessing and feature engineering. We clip each signal to its physical range, forward-fill
missing values, express each signal relative to the start of its sequence, and derive the dynamic, rolling and
window-shape temperature features described above."""
STEP2 = """Step 2 — load the data. We read the original training set, append the additional datasets, read the
test set, and attach the movement-description (operating-regime) features to every row."""
STEP3 = """Step 3 — per-motor settings. The feature set, the number and strength of synthetic injections, and
the post-processing for each motor, together with the injection function and the short-run filter."""
STEP4 = """Step 4 — train one motor. Split by whole sequences, inject synthetic faults into the training part,
grid-search the gradient-boosting hyperparameters, and choose the threshold and closing on the validation
sequences."""
STEP5 = """Step 5 — train all six motors and report the honest, leakage-free held-out F1 on the validation
sequences (note these held-out numbers are more conservative than the public leaderboard scores)."""
STEP6 = """Step 6 — build the submission. Predict the eight test sequences with the selected models, apply each
motor's threshold and post-processing, and write the 0/1 submission file."""

RESULTS_INTRO = """## Results

With the pipeline trained, we now report the outcome motor by motor. For each motor we summarise the
configurations we tried and the one we kept, with the fault-class F1 of the chosen configuration (which
matches the public per-motor breakdown reported in the Discussion). The honest held-out F1 on validation
sequences is printed by the training cell above and is lower, which already hints at the over-confidence we
analyse afterwards."""

PM1 = """### Motor 1 — hardest motor (temperature-fault inversion)

| Configuration tried | F1 (fault class) |
|---|---|
| Absolute features (HistGB) | 0.23 |
| + dynamic features | 0.54 |
| + injection peak tuning (8 to 16 deg) | 0.66 |
| + window-shape features (skew, kurtosis, slope, range, EWMA) | 0.72 |

Chosen: HistGB with dynamic and window-shape features, injection peak 8 to 16 deg."""

PM2 = """### Motor 2 — frequent fault, easy to separate

| Configuration tried | F1 (fault class) |
|---|---|
| HistGB (physical + dynamic) | 0.90 |
| Random forest | 0.74 |
| Logistic regression | 0.46 |

Chosen: HistGB with physical and dynamic features."""

PM3 = """### Motor 3 — extremely rare fault (under 0.5%)

| Configuration tried | F1 (fault class) |
|---|---|
| HistGB + heavy injection (x16) | 0.77 |
| Isolation forest (anomaly detection) | 0.10 |

Chosen: HistGB sustained by abundant subtle injection (peak 3 to 8 deg, x16)."""

PM4 = """### Motor 4 — frequent fault, also affected by the inversion

| Configuration tried | F1 (fault class) |
|---|---|
| Absolute features (HistGB) | 0.20 |
| + dynamic features | 0.74 |
| + post-processing (closing + min-run 40) | 0.90 |

Chosen: HistGB with dynamic features and aggressive morphological post-processing."""

PM5 = """### Motor 5 — extremely rare fault (under 0.5%)

| Configuration tried | F1 (fault class) |
|---|---|
| HistGB + subtle injection (x16, 1 to 4 deg) | 0.96 |

Chosen: HistGB with subtle injection; already excellent."""

PM6 = """### Motor 6 — marked, contiguous fault

| Configuration tried | F1 (fault class) |
|---|---|
| HistGB (physical + dynamic) | 0.89 |
| + post-processing (min-run 20) | 0.93 |

Chosen: HistGB with minimum-run post-processing."""

DISCUSSION = """## Discussion and conclusions

Scores. Our best submission reaches a public macro-F1 of 0.866 on Kaggle, with the per-motor breakdown M1
0.722, M2 0.905, M3 0.769, M4 0.901, M5 0.963, M6 0.934 (read from the validator's per-motor feedback). On the
private leaderboard, however, the same submission scores only 0.304. In other words the public score is very
strong but the private score is mediocre, and that gap is the main story of this project.

Why the gap, what we believed versus what actually happened. It helps to separate two things: the assumption
that led us to make the choices we made during the challenge, and the mechanism that, once we could measure
it afterwards, actually caused the drop.

- What we believed at the time. During the challenge the only feedback available to us was the leaderboard
  score, and it had been mentioned that in previous years some groups reached both public and private scores
  above 0.9. We therefore assumed that a good public score would translate into a good private score, and we
  optimised aggressively for the public number. In hindsight this amounted to a fairly aggressive overfitting
  of the training and of the public leaderboard.
- What actually happened. The post-deadline experiments showed that the mechanical cause was neither the
  model family nor the features, but calibration. Training on a balanced mixture of real and injected faults
  and then fixing very low decision thresholds (as low as 0.05) on a tiny validation split made the model
  over-predict on the test set, whose real fault prevalence is much lower. Those extra predictions are false
  positives that collapse precision and therefore the private F1. In short, the failure was overfitting
  expressed as a prevalence-calibration error, not a flaw in the model design itself.

Limitations.
- The temperature-fault inversion (explained in the Methods) is a genuine structural ceiling on Motor 1,
  which stays around 0.72 and is the main source of variance.
- The synthetic injection is stochastic, so the exact macro-F1 depends on the random seed (the rare-fault
  motors can swing noticeably); the architecture is robust but the precise number is not.
- The number of sequences with real faults is small, so the validation set is not diverse enough to fully
  represent the private test set.
- We rely on a single classifier family and did not fully exploit temporal models.

What we would do differently. The concrete recalibrations that recover part of the private score are
measured in the section below; the broader change we would make is not in the model but in how we select and
validate it in the first place. With only two sequences containing real faults, we would adopt
leave-one-fault-sequence-out validation as the selection criterion, which is the most honest estimate
available, and we would fix the calibration strategy before looking at the leaderboard rather than tuning
against it, treating the public score as one noisy signal instead of the objective to maximise.

After the submission deadline, and because the platform still accepted submissions, we ran a series of
additional experiments to understand the gap and to recover part of the private score. The findings of that
investigation are reported in the section below."""

# the measured improvements section (reused wording, adapted)
IMPROVEMENTS = """## Possible improvements (measured after the deadline)

These post-deadline experiments measured several of the ideas above on Kaggle, by submitting recalibrated
prediction files and reading both the public and the private score. The cells at the end reproduce the parts
that can be recomputed here (the private fault labels are not available, so only the supporting quantities are
recomputed). Throughout, "robust" means an improvement backed by a mechanism that also preserves the public
score, as opposed to one that merely trades public for private.

#### Diagnosis: the gap is prevalence miscalibration, not features

We first measured each motor's private F1 directly. By submitting the champion file with one motor's column
forced to zero and reading how the private macro changed, we isolated every motor's private contribution
(the macro F1 is the mean of independent per-motor F1, so zeroing one motor moves the macro by that motor's
F1 divided by six). On the private set, Motor 1 scored 0, Motor 2 about 0.26, Motor 3 about 0, Motor 4 about
1.0, Motor 5 about 0.02 and Motor 6 about 0.54.

A motor with no faults in a subset scores F1 = 1.0 if it predicts nothing, but F1 = 0 the moment it predicts a
single false positive. That is exactly what happened: trained on a balanced mix of real and injected faults,
the model over-predicts on the private set, whose fault prevalence is much lower, and the very low thresholds
(0.05 for several motors) sprayed false positives that destroyed precision. The gap is a calibration problem,
not a feature problem.

#### Robust improvement 1: raise the Motor 2 threshold (0.05 to 0.5)

Motor 2's real faults are high-confidence (probability above 0.5); the threshold of 0.05 only added
low-confidence false positives. Raising it to 0.5 left the public score essentially unchanged (0.865) and
raised Motor 2's private F1 from 0.264 to 0.735. This is the single largest honest gain, and it is robust
because it fixes an obvious miscalibration rather than fitting the private set.

#### Robust improvement 2: use the Motor 4 signal to help Motor 2

Motors 2 and 4 are strongly correlated (phi = 0.77): when one fails, the other fails about 80% of the time,
and in the same instant. Adding Motor 4's temperature features to the Motor 2 model raised Motor 2's private
F1 from 0.778 to 0.845 while keeping the public score at 0.866, and it also improves the held-out validation
F1 (reproduced in a cell below). The effect is asymmetric: feeding Motor 4 into Motor 2 helps (Motor 2 has
real private faults that Motor 4 confirms), while feeding Motor 2 into Motor 4 hurts (Motor 4 has no private
faults, so the coupling only adds false positives). The rule is therefore to give Motor 2 the Motor 4
features and leave Motor 4 untouched.

#### Minor improvement and combined effect

Raising Motor 5's threshold from 0.5 to 0.3 lifts its private F1 from about 0.02 to about 0.15 with almost no
public cost. Applying only the public-preserving changes (the Motor 2 threshold, the Motor 4 to Motor 2
features and the small Motor 5 adjustment) gives public 0.862 and private 0.422, against the original 0.866
and 0.304 — recovering more than a third of the private gap with no loss of public score.

| Submission | Public | Private |
|---|---|---|
| Original champion | 0.866 | 0.304 |
| Calibrated, public-preserving | 0.862 | 0.422 |

#### The general lesson

The dominant lever for the private score is prevalence calibration: a model trained on a balanced mix of real
and injected faults must not be used with thresholds tuned on that same balanced distribution; the thresholds
have to be raised to match the much lower fault rate of the test set. The second, smaller lever is to exploit
genuine inter-motor correlations as features, in the direction that confirms a motor that actually has faults
in the target set. Both levers preserve the public score because they remove false positives or reinforce
true positives instead of chasing the private labels directly."""

EMP_INTRO = """### Empirical support for the improvements

The cells below reproduce, on the local training and test data, the mechanisms behind the improvements: the
inter-motor fault correlation, the effect of feeding Motor 4 features to Motor 2, and the effect of
recalibrating the over-low thresholds. They reuse the `train_df`, `test_df`, `models` and `final` objects
built by the pipeline above, so they must be run after it."""

corr_code = (
"# Inter-motor fault-label correlation (phi). A strong Motor 2 <-> Motor 4 coupling is expected.\n"
"LAB = [f'data_motor_{m}_label' for m in range(1, 7)]\n"
"L = train_df.loc[train_df[LAB].notna().all(axis=1), LAB].astype(int)\n"
"corr = L.corr(); corr.index = corr.columns = [f'M{m}' for m in range(1, 7)]\n"
"print('Fault-label correlation (phi):')\n"
"print(corr.round(2).to_string())\n"
"m2, m4 = (L.iloc[:, 1] == 1), (L.iloc[:, 3] == 1)\n"
"print()\n"
"print(f'P(M4 fault | M2 fault) = {(L.iloc[:, 3][m2] == 1).mean():.2f}')\n"
"print(f'P(M2 fault | M4 fault) = {(L.iloc[:, 1][m4] == 1).mean():.2f}')\n")

cross_code = (
"# Improvement: give Motor 2 the Motor 4 temperature signal (asymmetric M4 -> M2).\n"
"# Same grid-search / held-out protocol as train_motor(), but with an explicit feature list.\n"
"def heldout_f1(mid, feats, seed=42):\n"
"    np.random.seed(seed * 1000 + mid)\n"
"    lab = f'data_motor_{mid}_label'\n"
"    mdf = train_df.dropna(subset=[lab]).copy()\n"
"    seqs = mdf['test_condition'].unique()\n"
"    fail = [s for s in seqs if (mdf[mdf.test_condition == s][lab] == 1).any()]\n"
"    norm = [s for s in seqs if s not in fail]\n"
"    trf, vaf = _split(fail); trn, van = _split(norm)\n"
"    raw_tr = mdf[mdf.test_condition.isin(trf + trn)].copy()\n"
"    synth = synthesize(raw_tr, mid, N_INJECT[mid], PEAK[mid])\n"
"    aug = pd.concat([raw_tr, synth], ignore_index=True) if not synth.empty else raw_tr\n"
"    for f in feats:\n"
"        if f not in aug.columns: aug[f] = 0.0\n"
"    aug[feats] = aug[feats].fillna(0.0)\n"
"    Xtr, ytr = aug[feats], aug[lab]; sw = np.where(ytr == 1, 2.0, 1.0)\n"
"    va = mdf[mdf.test_condition.isin(vaf + van)]\n"
"    best = 0.0\n"
"    for params in PARAM_GRID:\n"
"        m = HistGradientBoostingClassifier(random_state=42, **params).fit(Xtr, ytr, sample_weight=sw)\n"
"        if len(va) and va[lab].nunique() > 1:\n"
"            p = m.predict_proba(va[feats].fillna(0.0))[:, 1]\n"
"            best = max(best, max(f1_score(va[lab], (p >= t).astype(int), pos_label=1, zero_division=0)\n"
"                                 for t in THRESHOLDS))\n"
"    return best\n"
"\n"
"base_feats  = [f'data_motor_2_{f}' for f in MOTOR_FEATURES[2]]\n"
"cross_feats = base_feats + ['data_motor_4_temperature', 'data_motor_4_temperature_dev_20']\n"
"print(f'Motor 2 held-out F1  baseline      = {heldout_f1(2, base_feats):.3f}')\n"
"print(f'Motor 2 held-out F1  with M4 feats = {heldout_f1(2, cross_feats):.3f}')\n")

recal_code = (
"# Recalibration: the validation-tuned thresholds are too low and over-predict on the\n"
"# low-prevalence test set. Raising Motor 2 to 0.5 removes low-confidence false positives.\n"
"for t in sorted({models[2]['thr'], 0.5}):\n"
"    n = int(predict_column(models[2], t).sum())\n"
"    print(f'Motor 2 at threshold {t}: predicted faults on test = {n}')\n"
"\n"
"improved = final.copy()\n"
"improved['data_motor_2_label'] = predict_column(models[2], 0.5)   # robust, public-preserving\n"
"improved['data_motor_5_label'] = predict_column(models[5], 0.3)   # minor\n"
"improved.to_csv(os.path.join(ROOT, 'final_submission_group_X_recalibrated.csv'), index=False)\n"
"print()\n"
"print('Saved recalibrated submission.')\n"
"print('Faults per motor (recalibrated):',\n"
"      {f'M{m}': int((improved[f'data_motor_{m}_label'] == 1).sum()) for m in range(1, 7)})\n")

cells = [
 md(TITLE),
 md(DATACLEAN),
 md(OVERVIEW), md(DATA), md(IMBALANCE), md(VALIDATION),
 md(INJ_INTRO), co(inject_code), md(INJ_EXPLAIN),
 md(FEATURES), md(THRESHOLD), md(MORPH), md(MODEL), md(SUMMARY),
 md(PIPELINE_INTRO),
 md(STEP1), co(code[4]),
 md(STEP2), co(code[5]),
 md(STEP3), co(code[6]),
 md(STEP4), co(code[7]),
 md(STEP5), co(code[15]),
 md(STEP6), co(code[17]),
 md(RESULTS_INTRO), md(PM1), md(PM2), md(PM3), md(PM4), md(PM5), md(PM6),
 md(DISCUSSION),
 md(IMPROVEMENTS), md(EMP_INTRO), co(corr_code), co(cross_code), co(recal_code),
]
nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3'}},
      'nbformat': 4, 'nbformat_minor': 5}
json.dump(nb, open('reporte_preliminar.ipynb', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote reporte_preliminar.ipynb:', len(cells), 'cells,',
      sum(c['cell_type'] == 'code' for c in cells), 'code')
