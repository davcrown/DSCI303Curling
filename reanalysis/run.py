"""Re-run DSCI303Curling models: reproduce the leaky 90% result, then evaluate honestly.

Usage: python -I run.py   (expects ../data/*.csv)
"""
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             cohen_kappa_score, mean_absolute_error, roc_auc_score,
                             confusion_matrix)

sys.path.insert(0, ".")
import features as f

SEED = 42
rng = np.random.default_rng(SEED)


def oversample(X, y, seed=SEED):
    """Random oversampling to the majority class size (same as imblearn RandomOverSampler)."""
    r = np.random.default_rng(seed)
    y = np.asarray(y)
    n = np.bincount(y).max()
    idx = np.concatenate([r.choice(np.where(y == c)[0], n, replace=True) for c in np.unique(y)])
    return X.iloc[idx].reset_index(drop=True), pd.Series(y[idx])


def metrics(y, p):
    return {"acc": accuracy_score(y, p), "bal_acc": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro"),
            "qwk": cohen_kappa_score(y, p, weights="quadratic"), "mae": mean_absolute_error(y, p)}


def rf(**kw):
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED, **kw)


MODELS = {
    "Majority class": lambda: DummyClassifier(strategy="most_frequent"),
    "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "Random forest": lambda: rf(class_weight="balanced", min_samples_leaf=2),
    "Gradient boosting": lambda: HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced", max_iter=400, learning_rate=0.05),
    "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier((128, 64), alpha=1e-3, early_stopping=True, max_iter=300, random_state=SEED)),
}


def cv_eval(X, y, groups, make_model, over=False, n_splits=5):
    pred = np.empty(len(y), dtype=int)
    proba = np.zeros((len(y), 5))
    fold_scores = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        Xtr, ytr = X.iloc[tr], y.iloc[tr]
        if over:
            Xtr, ytr = oversample(Xtr, ytr)
        m = make_model().fit(Xtr, ytr)
        pred[te] = m.predict(X.iloc[te])
        if hasattr(m, "predict_proba"):
            proba[te] = m.predict_proba(X.iloc[te])
        fold_scores.append(metrics(y.iloc[te], pred[te]))
    out = metrics(y, pred)
    out.update({k + "_sd": float(np.std([s[k] for s in fold_scores])) for k in ("acc", "macro_f1", "qwk")})
    succ = (y >= 3).astype(int)
    out["success_auc"] = roc_auc_score(succ, proba[:, 3] + proba[:, 4]) if proba.sum() else np.nan
    return out, pred


def main():
    stones, ends, games = f.load("../data")
    report = {}

    # ---------------- Original features ----------------
    Xo, yo, keep_o = f.original_features(stones, ends)
    go = f.game_ids(stones)[keep_o]

    # A. Paper's protocol: oversample the WHOLE dataset, then random split
    Xr, yr = oversample(Xo.reset_index(drop=True), yo.reset_index(drop=True))
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=SEED, stratify=yr)
    m = rf(class_weight="balanced").fit(Xtr, ytr)
    pA = m.predict(Xte)
    tr_keys = set(map(tuple, Xtr.round(6).to_numpy()))
    dup = np.mean([tuple(r) in tr_keys for r in Xte.round(6).to_numpy()])
    report["A_paper_protocol"] = {**metrics(yte, pA), "n_train": len(ytr), "n_test": len(yte),
                                  "test_rows_also_in_train": float(dup)}

    # B. Original features, random split, oversample TRAIN only
    Xtr, Xte, ytr, yte = train_test_split(Xo, yo, test_size=0.2, random_state=SEED, stratify=yo)
    Xtr_o, ytr_o = oversample(Xtr.reset_index(drop=True), ytr.reset_index(drop=True))
    pB = rf(class_weight="balanced").fit(Xtr_o, ytr_o).predict(Xte)
    report["B_orig_random_split_train_oversample"] = metrics(yte, pB)

    # C. Original features, grouped by game
    Xo_, yo_ = Xo.reset_index(drop=True), yo.reset_index(drop=True)
    report["C_orig_group_split"] = {}
    for name in ["Majority class", "Logistic regression", "Random forest"]:
        report["C_orig_group_split"][name], _ = cv_eval(Xo_, yo_, go, MODELS[name])
    report["C_orig_group_split"]["Random forest + train oversample"], _ = cv_eval(Xo_, yo_, go, MODELS["Random forest"], over=True)

    # ---------------- Fixed features ----------------
    F, y, keep = f.fixed_features(stones, ends, games)
    g = f.game_ids(stones)[keep]
    report["D_fixed_group_split"] = {}
    preds = {}
    for name, mk in MODELS.items():
        report["D_fixed_group_split"][name], preds[name] = cv_eval(F, y, g, mk)
        print(name, {k: round(v, 3) for k, v in report["D_fixed_group_split"][name].items()}, flush=True)
    report["D_fixed_group_split"]["Random forest + train oversample"], _ = cv_eval(F, y, g, MODELS["Random forest"], over=True)

    # Diagnostics for the best model
    best = max(["Random forest", "Gradient boosting", "MLP", "Logistic regression"],
               key=lambda k: report["D_fixed_group_split"][k]["qwk"])
    p = preds[best]
    report["best_model"] = best
    report["confusion_matrix"] = confusion_matrix(y, p).tolist()
    report["per_class_recall"] = (np.diag(confusion_matrix(y, p)) / np.bincount(y)).round(3).tolist()
    task_cols = [c for c in F.columns if c.startswith("task_")]
    task = F[task_cols].idxmax(axis=1).str.replace("task_", "")
    report["per_task"] = (pd.DataFrame({"task": task, "y": y, "p": p})
                          .groupby("task").apply(lambda d: pd.Series({
                              "n": len(d), "acc": accuracy_score(d.y, d.p),
                              "majority_acc": d.y.value_counts(normalize=True).iloc[0],
                              "qwk": cohen_kappa_score(d.y, d.p, weights="quadratic") if d.y.nunique() > 1 else np.nan}))
                          .round(3).sort_values("n", ascending=False).to_dict(orient="index"))

    # Held-out permutation importance (one grouped split)
    tr, te = next(GroupKFold(5).split(F, y, g))
    m = MODELS[best]().fit(F.iloc[tr], y.iloc[tr])
    pi = permutation_importance(m, F.iloc[te], y.iloc[te], scoring="f1_macro", n_repeats=5, random_state=SEED, n_jobs=-1)
    report["perm_importance_top15"] = (pd.Series(pi.importances_mean, index=F.columns)
                                       .sort_values(ascending=False).head(15).round(4).to_dict())

    with open("results.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
