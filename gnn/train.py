"""GroupKFold-by-game comparison: GNN vs no-edge ablation vs random forest.

Usage:
  python -I graphs.py          # once, builds graphs.npz
  python -I train.py           # full run -> results.json
  python -I train.py --quick   # smoke test (1 fold, 1 seed, few epochs)

Folds are identical to reanalysis/run.py (same row order and game groups), so the
random-forest numbers here are paired with the GNN fold by fold.
"""
import argparse
import copy
import json
import os
import sys
import time
import numpy as np
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score,
                             mean_absolute_error, roc_auc_score, recall_score)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from graphs import NODE_X, EDGE_X  # noqa: E402
from model import ShotGNN, ordinal_loss, ordinal_probs  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUMERIC_CTX = ["shot_in_end", "end", "score_diff_before"]


def metrics(y, p, p_success):
    return {"acc": accuracy_score(y, p), "bal_acc": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro"),
            "qwk": cohen_kappa_score(y, p, weights="quadratic"), "mae": mean_absolute_error(y, p),
            "success_auc": roc_auc_score(y >= 3, p_success),
            "recall_by_class": recall_score(y, p, average=None, labels=range(5)).round(3).tolist()}


class Data:
    def __init__(self, g):
        self.node = torch.tensor(g["node"], device=DEV)
        self.edge = torch.tensor(g["edge"], device=DEV)
        self.mask = torch.tensor(g["mask"], device=DEV)
        self.thrown = torch.tensor(g["thrown"], device=DEV)
        self.ctx_raw = torch.tensor(g["ctx"], device=DEV)
        self.y = torch.tensor(g["y"], device=DEV)
        cols = list(g["ctx_cols"])
        self.num_idx = [cols.index(c) for c in NUMERIC_CTX]
        self.handle_idx = cols.index("handle")
        self.pp_idx = (cols.index("powerplay_1"), cols.index("powerplay_2"))

    def set_scaler(self, train_idx):
        x = self.ctx_raw[train_idx][:, self.num_idx]
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6

    def batch(self, idx, augment=False):
        node, edge, ctx = self.node[idx].clone(), self.edge[idx].clone(), self.ctx_raw[idx].clone()
        ctx[:, self.num_idx] = (ctx[:, self.num_idx] - self.mu) / self.sd
        if augment:  # mirror across the centre line: negate x, flip handle, swap power-play side
            flip = torch.rand(len(idx), device=DEV) < 0.5
            fn = flip.view(-1, 1, 1)
            node[..., NODE_X] = torch.where(fn, -node[..., NODE_X], node[..., NODE_X])
            fe = flip.view(-1, 1, 1, 1)
            edge[..., EDGE_X] = torch.where(fe, -edge[..., EDGE_X], edge[..., EDGE_X])
            ctx[flip, self.handle_idx] = 1 - ctx[flip, self.handle_idx]
            a, b = self.pp_idx
            ctx[flip, a], ctx[flip, b] = ctx[flip, b].clone(), ctx[flip, a].clone()
        return node, edge, self.mask[idx], self.thrown[idx], ctx


@torch.no_grad()
def predict(model, data, idx, bs=2048):
    model.eval()
    probs, p_succ = [], []
    for k in range(0, len(idx), bs):
        logits = model(*data.batch(idx[k:k + bs]))
        probs.append(ordinal_probs(logits))
        p_succ.append(torch.sigmoid(logits[:, 2]))
    return torch.cat(probs).cpu().numpy(), torch.cat(p_succ).cpu().numpy()


def train_gnn(data, tr, va, use_edges, seed, epochs, dims, bs=256, patience=12):
    torch.manual_seed(seed)
    model = ShotGNN(*dims, use_edges=use_edges).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr_t = torch.tensor(tr, device=DEV)
    y_va = data.y[va].cpu().numpy()
    best, best_state, wait = -1, None, 0
    gen = torch.Generator(device=DEV).manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = tr_t[torch.randperm(len(tr_t), device=DEV, generator=gen)]
        for k in range(0, len(perm), bs):
            idx = perm[k:k + bs]
            loss = ordinal_loss(model(*data.batch(idx, augment=True)), data.y[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        probs, _ = predict(model, data, torch.tensor(va, device=DEV))
        q = cohen_kappa_score(y_va, probs.argmax(1), weights="quadratic")
        if q > best:
            best, best_state, wait = q, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, ep + 1, best


def rf_fold_preds(g, folds):
    y = g["y"]
    pred, succ, fold_qwk = np.zeros(len(y), int), np.zeros(len(y)), []
    for tr_all, te in folds:
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42, class_weight="balanced",
                                    min_samples_leaf=2).fit(g["handcrafted"][tr_all], y[tr_all])
        pred[te] = rf.predict(g["handcrafted"][te])
        succ[te] = rf.predict_proba(g["handcrafted"][te])[:, 3:].sum(1)
        fold_qwk.append(cohen_kappa_score(y[te], pred[te], weights="quadratic"))
    return pred, succ, fold_qwk


def rf_only(g):
    """Refit only the RF (e.g. after a handcrafted-feature fix) and update results.json in place.

    Valid only when the GNN inputs are unchanged since the GNN run.
    """
    path = os.path.join(HERE, "results.json")
    report = json.load(open(path))
    y = g["y"]
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, g["groups"]))
    pred, succ, fq = rf_fold_preds(g, folds)
    report["Random forest"] = metrics(y, pred, succ)
    report["fold_qwk"]["Random forest"] = fq
    d = np.array(report["fold_qwk"]["GNN"]) - np.array(fq)
    report["paired_qwk_diff"]["GNN - RF"] = {"mean": float(d.mean()), "per_fold": d.round(4).tolist(),
                                             "folds_positive": int((d > 0).sum())}
    names = list(g["task_names"])
    for t, row in report["per_task_qwk"].items():
        m = g["task"] == names.index(t)
        row["Random forest"] = round(float(cohen_kappa_score(y[m], pred[m], weights="quadratic")), 3)
    report["rf_refit"] = "RF refit with --rf-only after first-shot pre-placed displacement fix"
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print("Random forest", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in report["Random forest"].items()})
    print(json.dumps({"fold_qwk_rf": fq, "GNN - RF": report["paired_qwk_diff"]["GNN - RF"],
                      "per_task": report["per_task_qwk"]}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rf-only", action="store_true", help="refit only the RF and update results.json")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    n_folds, seeds, epochs = (1, 1, 3) if args.quick else (5, args.seeds, args.epochs)

    g = dict(np.load(os.path.join(HERE, "graphs.npz"), allow_pickle=True))
    if args.rf_only:
        return rf_only(g)
    data = Data(g)
    y, groups = g["y"], g["groups"]
    dims = (g["node"].shape[2], g["edge"].shape[3], g["ctx"].shape[1])
    print(f"device={DEV} n={len(y)} dims={dims}", flush=True)

    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups))[:n_folds]
    variants = {"GNN": True, "No-edge ablation": False}
    pred = {v: np.zeros((seeds, len(y), 5)) for v in variants}
    succ = {v: np.zeros((seeds, len(y))) for v in variants}
    rf_pred, rf_succ, rf_q = rf_fold_preds(g, folds)
    fold_qwk = {v: [] for v in variants}
    fold_qwk["Random forest"] = rf_q
    log = []

    for fi, (tr_all, te) in enumerate(folds):
        inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=fi)
        a, b = next(inner.split(tr_all, y[tr_all], groups[tr_all]))
        tr, va = tr_all[a], tr_all[b]
        data.set_scaler(torch.tensor(tr, device=DEV))
        te_t = torch.tensor(te, device=DEV)


        for v, use_edges in variants.items():
            qs = []
            for s in range(seeds):
                t0 = time.time()
                model, ep, vq = train_gnn(data, tr, va, use_edges, seed=100 * fi + s, epochs=epochs, dims=dims)
                pred[v][s, te], succ[v][s, te] = predict(model, data, te_t)
                q = cohen_kappa_score(y[te], pred[v][s, te].argmax(1), weights="quadratic")
                qs.append(q)
                log.append({"fold": fi, "variant": v, "seed": s, "epochs": ep, "val_qwk": vq, "test_qwk": q,
                            "sec": round(time.time() - t0, 1)})
                print(log[-1], flush=True)
            fold_qwk[v].append(float(np.mean(qs)))

    te_all = np.concatenate([te for _, te in folds])
    yt = y[te_all]
    report = {"n_folds": n_folds, "seeds": seeds, "per_run": log, "fold_qwk": fold_qwk}
    report["Random forest"] = metrics(yt, rf_pred[te_all], rf_succ[te_all])
    for v in variants:
        per_seed = [metrics(yt, pred[v][s, te_all].argmax(1), succ[v][s, te_all]) for s in range(seeds)]
        agg = {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0] if k != "recall_by_class"}
        agg.update({k + "_sd_seeds": float(np.std([m[k] for m in per_seed])) for k in ("acc", "macro_f1", "qwk")})
        agg["recall_by_class"] = np.mean([m["recall_by_class"] for m in per_seed], 0).round(3).tolist()
        ens = pred[v][:, te_all].mean(0)
        agg["seed_ensemble"] = metrics(yt, ens.argmax(1), succ[v][:, te_all].mean(0))
        report[v] = agg

    # paired fold-level QWK differences
    def diff(a, b):
        d = np.array(fold_qwk[a]) - np.array(fold_qwk[b])
        return {"mean": float(d.mean()), "per_fold": d.round(4).tolist(), "folds_positive": int((d > 0).sum())}
    report["paired_qwk_diff"] = {"GNN - No-edge": diff("GNN", "No-edge ablation"),
                                 "GNN - RF": diff("GNN", "Random forest")}

    # per-task QWK (seed-ensembled GNN vs no-edge vs RF)
    names = g["task_names"]
    per_task = {}
    for t in np.unique(g["task"][te_all]):
        m = g["task"][te_all] == t
        if m.sum() < 50 or len(np.unique(yt[m])) < 2:
            continue
        row = {"n": int(m.sum())}
        for v in variants:
            row[v] = cohen_kappa_score(yt[m], pred[v][:, te_all].mean(0)[m].argmax(1), weights="quadratic")
        row["Random forest"] = cohen_kappa_score(yt[m], rf_pred[te_all][m], weights="quadratic")
        per_task[str(names[t])] = {k: round(float(x), 3) for k, x in row.items()}
    report["per_task_qwk"] = per_task

    out = os.path.join(HERE, "results_quick.json" if args.quick else "results.json")
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    for k in ["Random forest", "GNN", "No-edge ablation"]:
        print(k, {m: (round(x, 3) if isinstance(x, float) else x) for m, x in report[k].items() if m != "seed_ensemble"})
    print(json.dumps({"paired": report["paired_qwk_diff"], "per_task": per_task}, indent=1))


if __name__ == "__main__":
    main()
