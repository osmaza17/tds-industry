"""Build reporte_preliminar.ipynb: report narrative + working pipeline cells + demo cells."""
import json, re, uuid, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

text = open('final_test/REPORT.md', encoding='utf-8').read()
parts = re.split(r'(?=^## )', text, flags=re.M)
parts = [re.sub(r'\n-{3,}\n', '\n', p).strip('\n') for p in parts if p.strip()]
intro = parts[0] + '\n\n' + parts[1]

src = json.load(open('final_submission_group_2.ipynb', encoding='utf-8'))
code = {i: ''.join(src['cells'][i]['source']) for i in [4, 5, 6, 7, 15, 17]}


def md(s): return {'id': uuid.uuid4().hex[:8], 'cell_type': 'markdown', 'metadata': {}, 'source': s}
def co(s): return {'id': uuid.uuid4().hex[:8], 'cell_type': 'code', 'metadata': {},
                   'execution_count': None, 'outputs': [], 'source': s}


PIPE_INTRO = ("### Shared pipeline (used by all motors)\n\n"
 "The code below is self-contained and reproduces the winning pipeline end-to-end: preprocessing, "
 "feature engineering, synthetic injection, the per-motor model factory and the morphological "
 "post-processing. Each subsequent code cell builds on the previous one.")
TRAIN_INTRO = ("### Train all six motors and report held-out F1\n\n"
 "We fit every motor with its per-motor configuration and report the fault-class F1 on the held-out "
 "validation sequences (an honest, leakage-free estimate on the training data).")
SUB_INTRO = ("### Prepare final submission\n\n"
 "Using exactly the per-motor models selected above, we predict the eight test sequences, apply each "
 "motor decision threshold and morphological post-processing, and write the 0/1 submission file.")
EMP_INTRO = ("### Empirical support for the improvements\n\n"
 "The cells below reproduce, on the local training/test data, the mechanisms behind the improvements "
 "described above: the inter-motor fault correlation, the effect of feeding Motor 4 features to Motor 2, "
 "and the effect of recalibrating the over-low decision thresholds. The public/private Kaggle scores "
 "quoted in the text were measured by submitting the corresponding files; the private fault labels are "
 "not available locally, so only the supporting quantities are recomputed here.")

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
 md(intro), md(parts[2]), md(parts[3]), md(parts[4]),
 md(PIPE_INTRO), co(code[4]), co(code[5]), co(code[6]), co(code[7]),
 md(parts[5]), md(TRAIN_INTRO), co(code[15]),
 md(parts[6]), md(SUB_INTRO), co(code[17]),
 md(parts[7]), md(parts[8]),
 md(EMP_INTRO), co(corr_code), co(cross_code), co(recal_code),
 md(parts[9]),
]
nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3'}},
      'nbformat': 4, 'nbformat_minor': 5}
json.dump(nb, open('reporte_preliminar.ipynb', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('wrote reporte_preliminar.ipynb:', len(cells), 'cells,',
      sum(c['cell_type'] == 'code' for c in cells), 'code')
