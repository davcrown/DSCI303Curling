"""Build one dense graph per rated shot.

Nodes: up to 12 stones + 1 button node (index 12). A stone is a node if it is in play
before or after the shot (so the just-thrown stone and stones removed by this shot are
included; unthrown stones and stones removed earlier are not).
Node features carry the before/after state, so stone identity across the shot is implicit.
Edges: fully connected among present nodes, with relational features (distance, offset,
same team, contact, guarding) before and after the shot.
Context: shot-level features (task, handle, hammer, score, ...) from reanalysis/features.py.

Run `python -I graphs.py` to build and cache graphs.npz.
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "reanalysis"))
import features as f  # noqa: E402

N = 13                       # 12 stones + button
BUTTON_NODE = 12
SCALE = f.HOUSE_R            # positions are expressed in house radii
CONTACT = 2 * f.STONE_R + 25
GUARD_LANE = 1.5 * f.STONE_R

# Node feature layout (names kept so augmentation can find the x-coordinates)
STATE = ["in_play", "out", "x", "y", "dist", "in_house", "guard_zone"]
NODE_FEATS = (["is_button", "own", "is_thrown", "is_preplaced"]
              + [f"pre_{k}" for k in STATE] + [f"post_{k}" for k in STATE]
              + ["disp", "moved", "removed"])
EDGE_FEATS = ["same_team",
              "pre_valid", "pre_dx", "pre_dy", "pre_dist", "pre_contact", "pre_guards",
              "post_valid", "post_dx", "post_dy", "post_dist", "post_contact", "post_guards",
              "thrown_to_moved"]
NODE_X = [NODE_FEATS.index("pre_x"), NODE_FEATS.index("post_x")]
EDGE_X = [EDGE_FEATS.index("pre_dx"), EDGE_FEATS.index("post_dx")]


def _status(xy):
    """0 unthrown, 1 in play, 2 out (any 4095 coordinate)."""
    st = np.ones(12, dtype=int)
    st[(xy[:, 0] == 0) & (xy[:, 1] == 0)] = 0
    st[(xy[:, 0] == 4095) | (xy[:, 1] == 4095)] = 2
    return st


def _state(xy, st):
    rel = (xy - f.BUTTON) / SCALE
    inplay = st == 1
    d = np.where(inplay, np.linalg.norm(rel, axis=1), 0.0)
    in_house = inplay & (d * SCALE <= f.HOUSE_R + f.STONE_R)
    guard = inplay & (xy[:, 1] > f.BUTTON[1] + f.HOUSE_R + f.STONE_R) & (xy[:, 1] < 2900)
    rel = np.where(inplay[:, None], rel, 0.0)
    return np.column_stack([inplay, st == 2, rel[:, 0], rel[:, 1], d, in_house, guard]).astype(np.float32)


def _pair(xy, st):
    """Pairwise relational features for one layout: valid, dx, dy, dist, contact, guards."""
    inplay = st == 1
    valid = inplay[:, None] & inplay[None, :]
    delta = (xy[None, :, :] - xy[:, None, :])            # [i, j] = j - i
    dist = np.linalg.norm(delta, axis=2)
    contact = valid & (dist < CONTACT)
    # j guards i: j lies between i and the hog line (higher y) within a stone-width lane
    guards = valid & (delta[:, :, 1] > 0) & (np.abs(delta[:, :, 0]) < GUARD_LANE) & (xy[None, :, 1] < 2900)
    z = lambda a: np.where(valid, a, 0.0)
    return [valid, z(delta[:, :, 0] / SCALE), z(delta[:, :, 1] / SCALE), z(dist / SCALE), contact, guards]


def build(data_dir=os.path.join(HERE, "..", "data")):
    stones, ends, games = f.load(data_dir)
    ctx_df, y, keep = f.fixed_features(stones, ends, games)

    s = stones
    XY = np.stack([s[[f"stone_{i}_x" for i in range(1, 13)]].to_numpy(float),
                   s[[f"stone_{i}_y" for i in range(1, 13)]].to_numpy(float)], axis=2)
    end_id = s.groupby(f.KEY_END, sort=False).ngroup().to_numpy()
    first_in_end = np.r_[True, end_id[1:] != end_id[:-1]]
    pp = ends.dropna(subset=["PowerPlay"]).drop_duplicates(f.KEY_END).set_index(f.KEY_END)["PowerPlay"]
    end_pp = [int(pp.get(tuple(k), 0)) for k in s[f.KEY_END].to_numpy()]
    teams = s["TeamID"].to_numpy()

    # Which team owns stones 1-6 in each end: the team whose shot first brings one of them into play
    low_team = {}
    for i in range(len(s)):
        e = end_id[i]
        if e in low_team:
            continue
        pre = _status(XY[i - 1]) if not first_in_end[i] else np.isin(np.arange(12), [0, 6]).astype(int)
        new = np.where((pre == 0) & (_status(XY[i]) != 0))[0]
        if len(new):
            if new[0] < 6:
                low_team[e] = teams[i]
            else:
                other = teams[(end_id == e) & (teams != teams[i])]
                low_team[e] = other[0] if len(other) else -1

    n = len(s)
    X_node = np.zeros((n, N, len(NODE_FEATS)), np.float32)
    X_edge = np.zeros((n, N, N, len(EDGE_FEATS)), np.float32)
    mask = np.zeros((n, N), bool)
    thrown_idx = np.full(n, -1, np.int64)
    pre_xy = pre_st = None
    for i in range(n):
        xy, st = XY[i], _status(XY[i])
        if first_in_end[i]:
            pre_xy = f.preplaced_start(xy, st, end_pp[i])
            pre_st = np.isin(np.arange(12), [0, 6]).astype(int)
        own = (np.arange(12) < 6) if teams[i] == low_team.get(end_id[i], teams[i]) else (np.arange(12) >= 6)
        thrown = np.where((pre_st == 0) & (st != 0))[0]
        t = thrown[0] if len(thrown) else -1
        thrown_idx[i] = t

        present = (pre_st == 1) | (st == 1)
        if t >= 0:
            present[t] = True
        stayed = (pre_st == 1) & (st == 1)
        disp = np.where(stayed, np.linalg.norm(xy - pre_xy, axis=1), 0.0)
        if t >= 0:
            disp[t] = 0.0
        moved = disp > f.MOVE_EPS
        removed = (pre_st == 1) & (st == 2)

        nf = np.column_stack([np.zeros(12), own, np.arange(12) == t, np.isin(np.arange(12), [0, 6]),
                              _state(pre_xy, pre_st), _state(xy, st),
                              disp / SCALE, moved, removed]).astype(np.float32)
        X_node[i, :12] = nf * present[:, None]
        X_node[i, BUTTON_NODE, 0] = 1.0
        mask[i, :12] = present
        mask[i, BUTTON_NODE] = True

        # stones + button as a 13-point layout (button always "in play" at the button)
        xy13_pre = np.vstack([pre_xy, f.BUTTON]); st13_pre = np.r_[pre_st, 1]
        xy13 = np.vstack([xy, f.BUTTON]); st13 = np.r_[st, 1]
        own13 = np.r_[own, False]
        same = (own13[:, None] == own13[None, :]) & (np.arange(N) != BUTTON_NODE)[:, None] & (np.arange(N) != BUTTON_NODE)[None, :]
        pre_f = _pair(xy13_pre, st13_pre)
        post_f = _pair(xy13, st13)
        hit = np.zeros((N, N), bool)
        if t >= 0:
            hit[t, :12] = moved | removed
        ef = np.stack([same] + pre_f + post_f + [hit], axis=2).astype(np.float32)
        m2 = np.r_[present, True]
        X_edge[i] = ef * (m2[:, None] & m2[None, :])[:, :, None]
        pre_xy, pre_st = xy, st

    keep = np.asarray(keep)
    ctx_cols = ["shot_in_end", "end", "is_hammer", "timeout", "player2", "handle", "score_diff_before"] + \
               [c for c in ctx_df.columns if c.startswith(("task_", "powerplay_"))]
    task_cols = [c for c in ctx_df.columns if c.startswith("task_")]
    return dict(
        node=X_node[keep], edge=X_edge[keep], mask=mask[keep], thrown=thrown_idx[keep],
        ctx=ctx_df[ctx_cols].to_numpy(np.float32), ctx_cols=np.array(ctx_cols),
        y=y.to_numpy(np.int64), groups=f.game_ids(stones)[keep],
        task=ctx_df[task_cols].to_numpy().argmax(1), task_names=np.array([c[5:] for c in task_cols]),
        handcrafted=ctx_df.to_numpy(np.float32), handcrafted_cols=np.array(ctx_df.columns),
    )


if __name__ == "__main__":
    g = build()
    np.savez_compressed(os.path.join(HERE, "graphs.npz"), **g)
    print({k: v.shape for k, v in g.items()})
    print("nodes per graph (incl. button):", np.bincount(g["mask"].sum(1)))
    print("thrown identified:", (g["thrown"] >= 0).mean().round(4))
