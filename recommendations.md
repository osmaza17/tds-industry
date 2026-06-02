# Recommendations & Advice — Industry & Maintenance 4.0 Project

Extracted from the group WhatsApp chat. Sources: Wassim (W), Yiling (Y), and external friend of Yiling (F).


## Kaggle Submission Trick (unlimited test feedback)

*(Y, 30/05 & 02/06)*

To bypass the 5-submission daily limit and still see per-motor results:

1. Set **one motor's prediction column to `-1`** in the submission file.
2. The submission will fail (error message shown), so it **does not count** against the limit.
3. Kaggle still shows the scores for the other motors in the error output.

This gives near-unlimited partial feedback on individual motors.

---

## Recommended Pipeline

*(Y, 30/05 — from a friend who scored ~0.7)*

1. **Clean the data.**
2. **Feature analysis** for each motor individually.
3. **Inject failure data** (using the Kaggle utility code) **only on the training set** — keep the validation set with original (uninjected) failure data to avoid data leakage.
4. **Train-validation split ratio: 4:1 (80% train, 20% validation).**
5. Use the **validation set to tune hyperparameters** (max depth, leaf count, etc. for gradient boosting / XGBoost).
6. Train **XGBoost on all 6 motors** — this combination yielded ~0.7 F1 on Kaggle for the reference friend.

Key principle: inject failures to balance classes in training, but evaluate on real failure distribution in validation.

---

## Model Recommendations

- **Primary model: XGBoost** — best results reported (~0.7 F1). *(Y, 29/05 & 30/05)*
- Models explored and benchmarked: Logistic Regression, Random Forest, Gradient Boosting, AAKR, KNN.
- **AAKR is NOT a classification model** — it cannot produce class labels for a Kaggle submission directly. Use it for anomaly scoring only. *(Oscar, 24/05)*
- **KNN** was mentioned in class videos as an option worth trying. *(W, 24/05)*
- Diversity of models across motors is valuable for the write-up ("we tried three different setups"). *(W, 24/05)*

---

## Feature Engineering

- **Feature engineering is critical.** Using all raw features makes models slower AND worse (curse of dimensionality). *(Oscar, 25/05; W, 24/05)*
- Do per-motor feature selection/analysis — some motors have zero overheating while others don't; motors differ. *(Y, 19/05)*
- Feature engineering should be a focus for the final sessions. *(W, 24/05)*

---

## Model Tuning (Gradient Boosting / XGBoost)

*(Y, 02/06)*

Key hyperparameters to adjust per motor:
- **max depth**
- **min samples per leaf**
- Other model-specific parameters

Use the 20% validation set to tune these — do not tune on the full training set.

---



## Metric

The evaluation metric is **F1 score** (not accuracy). Do not optimize or report accuracy. *(Oscar & W, 24/05)*

---
