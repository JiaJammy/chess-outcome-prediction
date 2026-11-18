"""Step 3: player-history features, using ONLY games that had surely FINISHED before the current game started.

Run from the project folder:   python src/player_history.py
Output: data/history_features.csv.gz (one row per game in the two 2025 blitz files, key = site)

How "earlier" is decided (strict, no leakage):
  Lichess gives only the START time of a game. For every game we compute a safe LATEST POSSIBLE END:
     end_upper = start + clock time both players used + clock time the player to move still had + 10 s
  (the player to move could have used their remaining time before resigning / losing on time).
  A past game counts for a player only if its end_upper is BEFORE the current game's start.

History source: ALL rated games (any time control) in both Lichess files (2025-01-01 00:00 - 21:53 UTC).
Features for each player (white_ / black_ prefix):
  hist_games          number of earlier finished games (0 if none)
  hist_winrate10      share of wins in their last 10 earlier games      (missing if no earlier games)
  hist_score10        score in last 10 earlier games (win 1, draw 0.5)   (missing if no earlier games)
  hist_score_white10  score in last 10 earlier games played as White     (missing if none as White)
  hist_score_black10  score in last 10 earlier games played as Black     (missing if none as Black)
"""
import os
import re
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.features import parse_time_control

FILES = [os.path.join(ROOT, f"data/hf/lichess_2025_01_part{i}.parquet") for i in (0, 1)]
TARGETS = [os.path.join(ROOT, "data/games_2025_01_blitz.csv.gz"), os.path.join(ROOT, "data/games_2025_01_blitz_part1.csv.gz")]
OUT = os.path.join(ROOT, "data/history_features.csv.gz")
LAST_N = 10
SAFETY_SECONDS = 10
_CLK = re.compile(r"\[%clk (\d+):(\d+):(\d+)\]")
_MOVE = re.compile(r"(?:^|\s)\d+\.(?!\.)")       # white move numbers like '12.' (not '12...')


def latest_end_seconds(movetext, tc):
    """Upper bound on how long the game lasted (seconds)."""
    base, inc = parse_time_control(tc)
    clocks = [int(h) * 3600 + int(m) * 60 + int(s) for h, m, s in _CLK.findall(movetext or "")]
    if base != base or not clocks:          # no time control or no clocks: be very safe (1 day)
        return 86_400.0
    n = len(clocks)
    w, b = clocks[0::2], clocks[1::2]
    used_w = base + inc * len(w) - w[-1]
    used_b = (base + inc * len(b) - b[-1]) if b else 0
    remaining_to_move = (w[-1] if n % 2 == 0 else (b[-1] if b else base))
    return float(max(used_w, 0) + max(used_b, 0) + remaining_to_move + SAFETY_SECONDS)


def _read_group(args):
    path, i = args
    t = pq.ParquetFile(path).read_row_group(i, columns=["Site", "UTCDate", "UTCTime", "White", "Black",
                                                       "Result", "TimeControl", "movetext"]).to_pylist()
    return [(r["Site"], f"{r['UTCDate']} {r['UTCTime']}", r["White"], r["Black"], r["Result"],
             latest_end_seconds(r["movetext"], r["TimeControl"])) for r in t]


def load_all_games():
    jobs = [(p, i) for p in FILES for i in range(pq.ParquetFile(p).metadata.num_row_groups)]
    with Pool(max(1, os.cpu_count() // 2)) as pool:
        rows = [r for part in pool.map(_read_group, jobs) for r in part]
    g = pd.DataFrame(rows, columns=["site", "start", "white", "black", "result", "dur"])
    g = g[g["result"].isin(["1-0", "0-1", "1/2-1/2"])].reset_index(drop=True)
    g["start"] = pd.to_datetime(g["start"])
    g["end"] = g["start"] + pd.to_timedelta(g["dur"], unit="s")
    return g


def long_table(g):
    """One row per (game, player): who, colour, when, and the player's score in that game."""
    w = pd.DataFrame({"site": g["site"], "player": g["white"], "color": "white", "start": g["start"], "end": g["end"],
                      "score": g["result"].map({"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0})})
    b = pd.DataFrame({"site": g["site"], "player": g["black"], "color": "black", "start": g["start"], "end": g["end"],
                      "score": g["result"].map({"0-1": 1.0, "1/2-1/2": 0.5, "1-0": 0.0})})
    L = pd.concat([w, b], ignore_index=True)
    L["win"] = (L["score"] == 1.0).astype(float)
    return L


def history_lookup(L, Q, prefix):
    """For each query row (player, start) in Q: stats over that player's rows in L with end < start.
    Returns a DataFrame aligned with Q."""
    L = L.sort_values(["player", "end"]).reset_index(drop=True)
    L["pos"] = np.arange(len(L))
    first = L.groupby("player")["pos"].transform("min").values
    cum_score = L.groupby("player")["score"].cumsum().values
    cum_win = L.groupby("player")["win"].cumsum().values
    Qs = Q[["qid", "player", "start"]].sort_values("start")
    m = pd.merge_asof(Qs, L[["player", "end", "pos"]].sort_values("end"), left_on="start", right_on="end",
                      by="player", allow_exact_matches=False, direction="backward")   # strictly earlier end
    m = m.set_index("qid").reindex(Q["qid"])
    has = m["pos"].notna().values
    pos = m["pos"].fillna(0).astype(int).values
    k = np.where(has, pos - first[pos] + 1, 0)                     # number of earlier games
    back = pos - LAST_N
    prev_ok = has & (back >= first[pos])
    s_prev = np.where(prev_ok, cum_score[np.maximum(back, 0)], 0.0)
    w_prev = np.where(prev_ok, cum_win[np.maximum(back, 0)], 0.0)
    n_last = np.minimum(k, LAST_N)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(has, (cum_score[pos] - s_prev) / n_last, np.nan)
        win = np.where(has, (cum_win[pos] - w_prev) / n_last, np.nan)
    return pd.DataFrame({f"{prefix}games": k, f"{prefix}winrate10": win, f"{prefix}score10": score}, index=Q.index)


if __name__ == "__main__":
    t0 = time.time()
    g = load_all_games()
    print(f"history source games: {len(g)}  ({time.time() - t0:.0f} s)", flush=True)
    L = long_table(g)

    targets = pd.concat([pd.read_csv(p, usecols=["site", "white", "black", "utc_date", "utc_time"]) for p in TARGETS],
                        ignore_index=True)
    targets["start"] = pd.to_datetime(targets["utc_date"] + " " + targets["utc_time"])
    out = targets[["site"]].copy()
    for side in ["white", "black"]:
        Q = pd.DataFrame({"qid": np.arange(len(targets)), "player": targets[side], "start": targets["start"]})
        a = history_lookup(L, Q, f"{side}_hist_")
        cw = history_lookup(L[L["color"] == "white"], Q, f"{side}_hcw_")
        cb = history_lookup(L[L["color"] == "black"], Q, f"{side}_hcb_")
        out[f"{side}_hist_games"] = a[f"{side}_hist_games"].values
        out[f"{side}_hist_winrate10"] = a[f"{side}_hist_winrate10"].values
        out[f"{side}_hist_score10"] = a[f"{side}_hist_score10"].values
        out[f"{side}_hist_score_white10"] = cw[f"{side}_hcw_score10"].values
        out[f"{side}_hist_score_black10"] = cb[f"{side}_hcb_score10"].values
    out.to_csv(OUT, index=False)
    print(f"saved {OUT}: {len(out)} rows ({time.time() - t0:.0f} s)")
    print(out.describe().T.round(3))
    # keep the source tables for the leakage check
    L.to_parquet(os.path.join(ROOT, "data/history_long.parquet"), index=False)
