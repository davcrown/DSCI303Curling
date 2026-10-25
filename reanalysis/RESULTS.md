# Re-analysis of DSCI303Curling shot-quality models

Data: CSAS 2026 Data Challenge (github.com/CSAS-Data-Challenge/2026), 26,066 rated shots, 344 games.
Target: `Points` 0–4 (class shares: 4 = 56%, 3 = 16%, 2 = 14%, 0 = 11%, 1 = 3%).
Run: `sh fetch_data.sh` (downloads CSVs to `../data`), then `python -I run.py` from this folder → `results.json` (about 5 min).
Requires pandas, numpy, scikit-learn.

## 1. Why the paper got ~90%

| Setup (original features, random forest) | Accuracy | Macro-F1 |
|---|---|---|
| A. Paper protocol: oversample all data, then random split | **0.925** | 0.925 |
| B. Random split, oversample training only | 0.512 | 0.383 |
| C. Split by game, no oversampling | 0.551 | 0.404 |
| Majority class ("always 4") | 0.562 | 0.144 |

In setup A, 88.7% of test rows are exact copies of training rows. The 90% comes from leakage.
With an honest split, the paper's features do no better than always guessing 4 on accuracy.

## 2. Fixed pipeline (5-fold GroupKFold by game)

Fixes: positions are differenced within each end only (the first shot is compared to the pre-placed layout);
0 = unthrown and 4095 = removed are handled explicitly instead of being treated as coordinates or median-filled;
the thrown stone is identified; power play is assigned at end level; hammer, score and house-state context are added.

| Model | Acc | Bal. acc | Macro-F1 | QWK | MAE | AUC (3–4 vs 0–2) |
|---|---|---|---|---|---|---|
| Majority class | 0.562 | 0.200 | 0.144 | 0.000 | 0.959 | 0.500 |
| Logistic regression | 0.551 | 0.507 | 0.462 | 0.637 | 0.695 | 0.877 |
| **Random forest** | **0.705** | 0.555 | **0.563** | **0.786** | **0.425** | **0.930** |
| Gradient boosting | 0.650 | **0.583** | 0.555 | 0.761 | 0.487 | 0.926 |
| MLP | 0.697 | 0.498 | 0.507 | 0.762 | 0.455 | 0.922 |
| RF + train-only oversampling | 0.677 | 0.556 | 0.552 | 0.776 | 0.454 | 0.927 |

Fold SD is ≤ 0.013 on all metrics. Oversampling did not help.

Random forest recall by class: 0 → 0.84, 1 → 0.21, 2 → 0.54, 3 → 0.33, 4 → 0.86.
The middle grades (1 and 3) are the hard part.

Held-out permutation importance (macro-F1): shot number in end, change in scoring state,
thrown stone's distance to the button, opponent stones removed, then task indicators.

Per task: takeouts and double takeouts are predicted well (QWK ≈ 0.80). Guards and clearings
barely beat their majority grade (QWK 0.23 / 0.56): their success depends on intent the layout doesn't show.
