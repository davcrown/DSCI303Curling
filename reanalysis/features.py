"""Feature construction for shot-quality (Points 0-4) prediction.

Two feature sets:
  original_features() - faithful reproduction of the DSCI303Curling pipeline, bugs included
  fixed_features()    - per-end state differencing with sentinel handling and game context

Coordinate conventions (CSAS 2026 data dictionary):
  x in (0,1500), y in (0,3000); 4095 = out of play; 0 = not yet thrown.
  Button at (750, 800). Hog line y=2900, back line y=200.
  Stones 1-6 belong to one team, 7-12 to the other; stones 1 and 7 are pre-placed.
  Coordinates in each row are the layout AFTER that shot.
"""
import numpy as np
import pandas as pd

KEY_GAME = ["CompetitionID", "SessionID", "GameID"]
KEY_END = KEY_GAME + ["EndID"]
BUTTON = np.array([750.0, 800.0])
HOUSE_R = 578.0   # 1.829 m house radius at ~316 units/m (sheet width 4.75 m = 1500 units)
STONE_R = 46.0    # ~0.145 m stone radius
MOVE_EPS = 15.0   # displacement below this is treated as image-extraction noise
VALID_POINTS = [0, 1, 2, 3, 4]
# Pre-placed positions by end-level PowerPlay value (0 none, 1 right, 2 left): (guard, house).
# Modal first-shot positions in the data; e.g. 1945/2039 non-power-play ends have a stone at (750, 1916).
PREPLACED = {0: ((750, 1916), (750, 650)),
             1: ((1089, 1916), (1150, 850)),
             2: ((411, 1916), (350, 850))}

TASKS = {0: "Draw", 1: "Front", 2: "Guard", 3: "Raise", 4: "Wick", 5: "Freeze",
         6: "Takeout", 7: "HitRoll", 8: "Clearing", 9: "DoubleTakeout",
         10: "PromotionTakeout", 11: "Through"}


def load(data_dir="data"):
    stones = pd.read_csv(f"{data_dir}/Stones.csv")
    ends = pd.read_csv(f"{data_dir}/Ends.csv")
    games = pd.read_csv(f"{data_dir}/Games.csv")
    return stones, ends, games


def game_ids(stones):
    return stones.groupby(KEY_GAME).ngroup().to_numpy()


# --------------------------------------------------------------------------
# Original pipeline (finalmodels.py lines 40-170), vectorised but equivalent
# --------------------------------------------------------------------------
def original_features(stones, ends):
    results = stones.merge(ends, on=KEY_END[:4] + ["TeamID"], how="left")
    results["PowerPlay"] = results["PowerPlay"].fillna(0)
    shot_info = results.iloc[:, 7:].copy()
    shot_info["TimeOut"] = shot_info["TimeOut"].fillna(0)
    shot_info["PowerPlay"] = shot_info["PowerPlay"].replace([0, 1, 2], ["None", "Left", "Right"])

    d = pd.DataFrame(index=shot_info.index)
    for s in range(1, 13):
        x, y = shot_info[f"stone_{s}_x"], shot_info[f"stone_{s}_y"]
        # previous ROW of the whole file, no grouping by end/game; 0 treated as a position
        dist = np.sqrt((x - x.shift(1)) ** 2 + (y - y.shift(1)) ** 2)
        dist[(x == 4095) | (y == 4095)] = np.nan
        d[f"stone_{s}_distance"] = dist
    dcols = list(d.columns)
    d["Task"] = shot_info["Task"].astype(str).map({str(k): v for k, v in TASKS.items()})
    d["Handle"] = shot_info["Handle"].astype(str)
    d["Points"] = shot_info["Points"]
    d["TimeOut"] = shot_info["TimeOut"]
    d["PowerPlay"] = shot_info["PowerPlay"]
    d["Result"] = shot_info["Result"]
    d["OverallAvgStoneDistance"] = d[dcols].mean(axis=1)

    keep = d["Points"].isin(VALID_POINTS)
    d = d[keep].copy()
    # committed code calls np.isnan on the mixed-type frame (TypeError); numeric-only is the
    # closest runnable version. NaNs here are stones at 4095 -> counts stones OUT of play.
    d["StonesInPlay"] = d.select_dtypes("number").isna().sum(axis=1)
    for c in dcols + ["OverallAvgStoneDistance"]:
        d[c] = d[c].fillna(d[c].median())   # median computed on full data before split
    enc = pd.get_dummies(d, columns=["Task", "Handle", "PowerPlay"], dummy_na=False)
    X = enc.drop(columns=["Points", "Result"]).astype(float)
    y = enc["Points"].astype(int)
    return X, y, keep.to_numpy()


# --------------------------------------------------------------------------
# Fixed pipeline
# --------------------------------------------------------------------------
def _layout(row_xy):
    """row_xy: (12, 2) array -> status per stone: 0 unthrown, 1 in play, 2 out."""
    status = np.ones(12, dtype=int)
    status[(row_xy[:, 0] == 0) & (row_xy[:, 1] == 0)] = 0
    status[(row_xy[:, 0] == 4095) | (row_xy[:, 1] == 4095)] = 2
    return status


def _state_summary(xy, status, own):
    """Summary of a layout from the thrower's perspective. own: bool mask of thrower's stones."""
    inplay = status == 1
    d = np.where(inplay, np.linalg.norm(xy - BUTTON, axis=1), np.inf)
    in_house = inplay & (d <= HOUSE_R + STONE_R)
    guard = inplay & (xy[:, 1] > BUTTON[1] + HOUSE_R + STONE_R) & (xy[:, 1] < 2900)
    # stones counting: own stones closer than the closest opponent stone in the house
    opp_best = d[~own & in_house].min() if (~own & in_house).any() else np.inf
    own_best = d[own & in_house].min() if (own & in_house).any() else np.inf
    own_count = int(((d < opp_best) & own & in_house).sum())
    opp_count = int(((d < own_best) & ~own & in_house).sum())
    return {
        "own_inplay": int((inplay & own).sum()), "opp_inplay": int((inplay & ~own).sum()),
        "own_house": int((in_house & own).sum()), "opp_house": int((in_house & ~own).sum()),
        "own_guard": int((guard & own).sum()), "opp_guard": int((guard & ~own).sum()),
        "score_state": own_count - opp_count,   # + = thrower would score this many
    }


def fixed_features(stones, ends, games):
    s = stones.copy()
    xs = s[[f"stone_{i}_x" for i in range(1, 13)]].to_numpy(float)
    ys = s[[f"stone_{i}_y" for i in range(1, 13)]].to_numpy(float)
    XY = np.stack([xs, ys], axis=2)             # (n, 12, 2)
    end_id = s.groupby(KEY_END, sort=False).ngroup().to_numpy()
    first_in_end = np.r_[True, end_id[1:] != end_id[:-1]]

    # team that owns stones 1-6: thrower of the first shot that brings a stone 2-6 into existence
    team_of_low = {}
    for e, grp in s.assign(_e=end_id).groupby("_e"):
        idx = grp.index.to_numpy()
        prev = None
        for i in idx:
            st = _layout(XY[i])
            pre = prev if prev is not None else _preplaced(st)
            new = np.where((pre == 0) & (st != 0))[0]
            if len(new):
                team = s.at[i, "TeamID"]
                low = new[0] < 6
                other = grp["TeamID"][grp["TeamID"] != team]
                other = other.iloc[0] if len(other) else -1
                team_of_low[e] = team if low else other
                break
            prev = st

    # end-level context from Ends
    pp = ends.dropna(subset=["PowerPlay"]).set_index(KEY_END)["PowerPlay"]
    pp = pp[~pp.index.duplicated()]
    hammer = s.loc[s.groupby(end_id)["ShotID"].idxmax(), KEY_END + ["TeamID"]].set_index(KEY_END)["TeamID"]
    e2 = ends.sort_values(KEY_GAME + ["EndID"]).copy()
    e2["ScoreBefore"] = e2.groupby(KEY_GAME + ["TeamID"])["Result"].cumsum() - e2["Result"]
    score_before = e2.set_index(KEY_END + ["TeamID"])["ScoreBefore"]
    end_teams = s.groupby(end_id)["TeamID"].unique().to_dict()

    rows = []
    prev_xy, prev_st = None, None
    for i in range(len(s)):
        xy, st = XY[i], _layout(XY[i])
        team = s.at[i, "TeamID"]
        low_team = team_of_low.get(end_id[i], team)
        own = np.arange(12) < 6 if team == low_team else np.arange(12) >= 6
        if first_in_end[i]:
            pre_st = _preplaced(st)
            pre_xy = preplaced_start(xy, st, int(pp.get(tuple(s.loc[i, KEY_END]), 0)))
        else:
            pre_xy, pre_st = prev_xy, prev_st

        thrown = np.where((pre_st == 0) & (st != 0))[0]
        t = thrown[0] if len(thrown) else -1
        others = np.ones(12, bool)
        if t >= 0:
            others[t] = False
        was_in = (pre_st == 1) & others
        removed = was_in & (st == 2)
        stayed = was_in & (st == 1)
        disp = np.where(stayed, np.linalg.norm(xy - pre_xy, axis=1), 0.0)

        before = _state_summary(pre_xy, pre_st, own)
        after = _state_summary(xy, st, own)
        r = {
            "shot_in_end": s.at[i, "ShotID"],
            "end": s.at[i, "EndID"],
            "is_hammer": int(hammer.get(tuple(s.loc[i, KEY_END]), -1) == team),
            "timeout": int(s.at[i, "TimeOut"] == 1),
            "player2": int(s.at[i, "PlayerID"] == 2),
            "handle": s.at[i, "Handle"],
            "task": s.at[i, "Task"],
            "powerplay": int(pp.get(tuple(s.loc[i, KEY_END]), 0)),
            "thrown_identified": int(t >= 0),
            "thrown_out": int(t >= 0 and st[t] == 2),
            "thrown_dist_button": (np.linalg.norm(xy[t] - BUTTON) if t >= 0 and st[t] == 1 else 3000.0),
            "thrown_dx": (xy[t, 0] - 750 if t >= 0 and st[t] == 1 else 0.0),
            "thrown_dy": (xy[t, 1] - 800 if t >= 0 and st[t] == 1 else 0.0),
            "own_removed": int((removed & own).sum()), "opp_removed": int((removed & ~own).sum()),
            "own_moved": int(((disp > MOVE_EPS) & own).sum()), "opp_moved": int(((disp > MOVE_EPS) & ~own).sum()),
            "own_disp": float(disp[own].sum()), "opp_disp": float(disp[~own].sum()),
            "max_disp": float(disp.max()),
        }
        r.update({f"pre_{k}": v for k, v in before.items()})
        r.update({f"post_{k}": v for k, v in after.items()})
        r["d_score_state"] = after["score_state"] - before["score_state"]
        r["d_own_house"] = after["own_house"] - before["own_house"]
        r["d_opp_house"] = after["opp_house"] - before["opp_house"]
        sb = score_before.get(tuple(s.loc[i, KEY_END]) + (team,), 0)
        opp_team = [tm for tm in end_teams[end_id[i]] if tm != team]
        so = score_before.get(tuple(s.loc[i, KEY_END]) + (opp_team[0],), 0) if opp_team else 0
        r["score_diff_before"] = sb - so
        rows.append(r)
        prev_xy, prev_st = xy, st

    F = pd.DataFrame(rows)
    keep = s["Points"].isin(VALID_POINTS).to_numpy()
    F, y = F[keep].reset_index(drop=True), s.loc[keep, "Points"].astype(int).reset_index(drop=True)
    F["task"] = F["task"].map(TASKS).fillna("Other")
    F = pd.get_dummies(F, columns=["task", "powerplay"], dtype=float)
    return F.astype(float), y, keep


def preplaced_start(xy_after, st_after, pp):
    """Positions of stones 1 and 7 before the first thrown shot of an end.

    The after-shot layout can't be used: the first shot may move or remove a pre-placed stone.
    Assigns stones 1/7 to the guard and house spots by minimising distance from the canonical
    spots to where each stone is after the first shot (removed stones cost nothing).
    """
    guard, house = (np.array(p, float) for p in PREPLACED.get(pp, PREPLACED[0]))

    def cost(i, spot):
        return 0.0 if st_after[i] != 1 else np.linalg.norm(xy_after[i] - spot)

    pre = np.zeros((12, 2))
    if cost(0, guard) + cost(6, house) <= cost(0, house) + cost(6, guard):
        pre[0], pre[6] = guard, house
    else:
        pre[0], pre[6] = house, guard
    return pre


def _preplaced(status_after_first):
    """Before the first thrown shot only the pre-placed stones (1 and 7) exist."""
    pre = np.zeros(12, dtype=int)
    for k in (0, 6):
        pre[k] = 1 if status_after_first[k] != 0 else 0
    return pre
