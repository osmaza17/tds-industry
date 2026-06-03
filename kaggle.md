# Robot Predictive Maintenance — Season 2026

## Overview
The aim of this competition is to develop machine learning models for fault detection in predictive maintenance. The robot comprises **6 motors**. A condition-monitoring program collects in real-time the **position, temperature and voltage** of each motor at **10 Hz** (1 data point every 0.1 s). Using these features, you must detect whether a given motor is working normally at a given time. Performance is evaluated by the **F1 score** of detecting failures in the test dataset.

## Failure definition
The failure mode is an **abnormal motor temperature rise** (e.g. a blocked motor). The failure is *simulated* to avoid damaging the motors:
1. Select the beginning and end of the failure period.
2. Pick a peak temperature randomly.
3. Generate the temperature trend assuming temperature **rises 2× faster than it descends**.

MATLAB source used to simulate the failure behaviour:

```matlab
function temperature = inject_failure(temperature, label)
    % Get the sequence where failure is injected.
    tmp_temp = temperature(label==1);
    % Length of the sequence.
    n_seq = length(tmp_temp);
    % Decide the sequence where temperature rises: temperature
    % rise is assumed 2 times faster than temperature drop.
    n_rise = floor(n_seq/3);
    if ~isempty(tmp_temp)
        % Get the starting and ending temperature
        temp_start = tmp_temp(1);
        temp_end = tmp_temp(end);

        % Generate a random highest temperature
        temp_high = max(temp_start, temp_end) + randi([2, 10]);
        % Generate the temperature rise
        step_size_rise = floor(n_rise/(temp_high-temp_start+1));
        for i = 1:temp_high-temp_start
            tmp_temp((i-1)*step_size_rise+1:i*step_size_rise) = temp_start+i-1;
        end
        tmp_temp(i*step_size_rise+1:n_rise) = temp_high;

        % Generate temperature decrease
        step_size_down = floor(2*n_rise/(temp_high-temp_end));
        for i = 1:temp_high-temp_end-1
            tmp_temp(n_rise+1+(i-1)*step_size_down:n_rise+i*step_size_down) = temp_high-i;
        end
        tmp_temp(n_rise+i*step_size_down+1:end) = temp_end;
    end
    temperature(label==1) = tmp_temp;
end
```

## Objectives
Predict the health state of each of the 6 motors:
- `0` — working normally
- `1` — abnormal temperature rise

## Evaluation
Predict the labels for all six motors. The score is the **average F1 score over the six motors**. Follow the format of `sample_submission.csv`.

```python
"""
Average F1 Score for the six motors.
"""
import pandas as pd
import pandas.api.types
from sklearn.metrics import f1_score


class ParticipantVisibleError(Exception):
    pass


def cal_classification_perf(y_true, y_pred):
    # Set precision to 1 when both y_true and y_pred contain no positives.
    if sum(y_true) == 0 and sum(y_pred) == 0:
        f1 = f1_score(y_true, y_pred, zero_division=1)
    else:
        f1 = f1_score(y_true, y_pred, zero_division=0)
    return f1


def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str = 'idx') -> float:
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    results = []
    for i in range(6):
        col_name = f'data_motor_{i+1}_label'
        y_true = solution[col_name].values
        y_pred = submission[col_name].values
        results.append(cal_classification_perf(y_true, y_pred))

    return sum(results) / len(results)
```

## Citation
Zhiguo Zeng. *Robot predictive maintenance - Season 2026.* https://kaggle.com/competitions/robot-predictive-maintenance-season-2026, 2026. Kaggle.
