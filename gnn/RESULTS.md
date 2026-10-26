# GNN v1: shot quality from stone-relationship graphs

Run from this folder: `sh ../reanalysis/fetch_data.sh` (if `../data` is empty), then `python -I graphs.py` (builds `graphs.npz`, ~2 min), then `python -I train.py` (~45 min on GPU) → `results.json`.
Requires torch, scikit-learn, pandas, numpy (no PyTorch Geometric: graphs have ≤13 nodes, so batches are dense).

## Setup
- **Graph per shot:** one node per stone in play before or after the shot, plus a button node.
  Node features: owner, just-thrown, pre-placed, before/after position and zone, displacement, removed.
  Fully connected edges: same team, offset/distance before and after, contact, guarding (j in i's lane toward the hog line),
  thrown-stone→moved-stone.
- **Model:** context-conditioned node encoder → 3 edge-conditioned attention layers (GATv2-style, 4 heads, 64 hidden)
  → readout [thrown-stone embedding | attention pool | context] → cumulative-link ordinal head (0–4).
  Mirror augmentation (x ↦ −x, handle flipped, power-play side swapped).
- **Ablation:** identical model with the attention layers replaced by per-node MLPs (stones never exchange information).
- **Evaluation:** the same 5 GroupKFold-by-game folds as `reanalysis/`; 10% of training games held out for early stopping on QWK;
  3 seeds per fold. The RF baseline is retrained here on the same folds (`--rf-only` refits just the RF).

## Results (pooled over folds; GNN rows are the mean of 3 seeds, SD across seeds ≤ 0.004)

| Model | Acc | Bal. acc | Macro-F1 | QWK | MAE | Success AUC |
|---|---|---|---|---|---|---|
| Majority class | 0.562 | 0.200 | 0.144 | 0.000 | 0.959 | 0.500 |
| Random forest (handcrafted features) | 0.704 | **0.553** | **0.561** | 0.786 | 0.426 | 0.930 |
| No-edge ablation | 0.710 | 0.517 | 0.526 | 0.776 | 0.430 | 0.928 |
| **GNN** | 0.719 | 0.530 | 0.541 | 0.794 | 0.406 | 0.933 |
| GNN, 3-seed ensemble | **0.729** | 0.538 | 0.551 | **0.802** | **0.391** | **0.941** |

**Paired fold-level QWK:**
- GNN − no-edge: +0.018 (better in 5/5 folds, +0.009 to +0.026).
- GNN − RF: +0.008 (better in 4/5 folds, −0.003 to +0.024).

**Recall by grade (0–4):**
- RF: 0.83 / 0.21 / 0.55 / 0.33 / 0.86
- GNN: 0.81 / 0.12 / 0.52 / 0.29 / 0.91

The GNN leans toward the majority grade 4. Its ordinal loss is unweighted, while the RF uses balanced class weights.

**QWK by task** (GNN = 3-seed ensemble):

| Task | n | GNN | No-edge | RF |
|---|---|---|---|---|
| Draw | 13,488 | **0.845** | 0.833 | 0.836 |
| Raise | 2,716 | **0.759** | 0.743 | 0.734 |
| Double takeout | 2,086 | 0.791 | 0.771 | **0.800** |
| Guard | 2,052 | **0.256** | 0.178 | 0.229 |
| Takeout | 1,992 | 0.790 | 0.763 | **0.798** |
| Promotion takeout | 1,044 | **0.786** | 0.765 | 0.745 |
| Hit and roll | 1,025 | **0.716** | 0.669 | 0.695 |
| Clearing | 936 | **0.610** | 0.565 | 0.547 |
| Wick | 453 | **0.754** | 0.716 | 0.644 |
| Front | 160 | 0.296 | 0.357 | **0.378** |
| Freeze | 113 | **0.628** | 0.471 | 0.478 |

## Reading
- **Relationships between stones help, consistently but modestly.** Edges add +0.018 QWK over the same model without them in every fold.
  The biggest gains are on relational tasks: freeze (+0.16), wick (+0.04), hit and roll (+0.05), clearing (+0.05).
- **The GNN matches or slightly beats a well-engineered RF** without hand-designed scoring features.
  The gap (+0.008 QWK, 4/5 folds) is small and within the range a tuned RF might close.
- **Guards remain unsolved** (QWK ≈ 0.26 for every model). A guard's grade depends on where the thrower *meant* it to stop,
  which the layout alone doesn't reveal.
- **Middle grades are still the weak spot** (grade 1 recall 0.12).

Caveats:
- The RF numbers above use the corrected `reanalysis/features.py`. A pre-placed stone hit on the first shot of an end
  used to show zero displacement; both pipelines now reconstruct its starting spot from the power-play setting
  (`features.preplaced_start`). The fix changed 974 rows (840 first shots now register a moved stone) but left the RF
  essentially unchanged (QWK 0.7864 → 0.7863), so it does not explain the GNN's lead. GNN inputs were verified
  byte-identical before and after, so the GNN rows were not retrained; the RF was refit with `train.py --rf-only`.
  Pre-fix numbers: `results_before_rf_fix.json`.
- No hyperparameter search was done for the GNN beyond defaults.

## Next steps
- Class-weighted or threshold-tuned ordinal loss to recover grade 1 and grade 3 recall.
- Pretrain on next-layout prediction (all shots, including unrated), then fine-tune.
- Expected-points value model and an end-level sequence model (see the design outline).
