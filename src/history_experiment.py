"""Step 3 checks and results.

Run from the project folder:   python src/history_experiment.py
 A. Leakage proof on 2,000 random (game, player) pairs:
    1. recompute every history number the slow, simple way (filter that player's games by hand) -> must match
    2. every counted game's latest possible end is before the current game's start
    3. the current game itself is never counted
 B. Accuracy with vs without history: 5-fold CV on the same 200,000 TRAIN games as the Stockfish test.
Outputs: results/history_leakage_check.csv, results/history_cv.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.features import add_model_columns, RATING_COLS_V2, TIME_COLS_V2, BOARD_COLS_V2, CLOCK_COLS_V2
from src.player_history import LAST_N

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

RS = 42
LABELS = {"0-1": 0, "1/2-1/2": 1, "1-0": 2}
BASE = RATING_COLS_V2 + TIME_COLS_V2 + BOARD_COLS_V2 + CLOCK_COLS_V2
SF_COLS = ["sf_cp", "sf_expected"]
HIST_COLS = [f"{s}_hist_{a}" for s in ("white", "black")
             for a in ("games", "winrate10", "score10", "score_white10", "score_black10")] + ["diff_hist_score10"]


def add_hist_diff(df):
    df["diff_hist_score10"] = df["white_hist_score10"] - df["black_hist_score10"]
    return df


def leakage_check(n=2000):
    L = pd.read_parquet(os.path.join(ROOT, "data/history_long.parquet"))
    H = pd.read_csv(os.path.join(ROOT, "data/history_features.csv.gz"))
    T = pd.concat([pd.read_csv(os.path.join(ROOT, p), usecols=["site", "white", "black", "utc_date", "utc_time"])
                   for p in ["data/games_2025_01_blitz.csv.gz", "data/games_2025_01_blitz_part1.csv.gz"]], ignore_index=True)
    T = T.merge(H, on="site")
    T["start"] = pd.to_datetime(T["utc_date"] + " " + T["utc_time"])
    sample = T.sample(n=n, random_state=RS)
    byp = {p: d.sort_values("end") for p, d in L[L["player"].isin(set(sample["white"]) | set(sample["black"]))].groupby("player")}
    bad, own_counted, late = 0, 0, 0
    for _, r in sample.iterrows():
        for side in ("white", "black"):
            d = byp.get(r[side], L.iloc[0:0])
            e = d[d["end"] < r["start"]]                       # slow, simple filter
            own_counted += int((e["site"] == r["site"]).any())
            late += int((e["end"] >= r["start"]).any())
            last = e.tail(LAST_N)
            exp = {"games": len(e),
                   "winrate10": last["win"].mean() if len(e) else np.nan,
                   "score10": last["score"].mean() if len(e) else np.nan}
            ew, eb = e[e["color"] == "white"].tail(LAST_N), e[e["color"] == "black"].tail(LAST_N)
            exp["score_white10"] = ew["score"].mean() if len(ew) else np.nan
            exp["score_black10"] = eb["score"].mean() if len(eb) else np.nan
            for k, v in exp.items():
                got = r[f"{side}_hist_{k}"]
                if not ((np.isnan(v) and np.isnan(got)) or abs(v - got) < 1e-9):
                    bad += 1
    res = pd.DataFrame([{"pairs_checked": 2 * n, "numbers_checked": 2 * n * 5, "mismatches": bad,
                         "current_game_counted": own_counted, "counted_game_not_finished_before_start": late}])
    res.to_csv(os.path.join(ROOT, "results/history_leakage_check.csv"), index=False)
    print(res.to_string(index=False), flush=True)


def cv_compare():
    df = pd.read_csv(os.path.join(ROOT, "data/games_2025_01_blitz.csv.gz"))
    y_all = df["result"].map(LABELS).values
    itr, _ = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RS, stratify=y_all)
    pick, _ = train_test_split(itr, train_size=200_000, random_state=RS, stratify=y_all[itr])
    df = df.iloc[pick].reset_index(drop=True)
    sf = pd.read_csv(os.path.join(ROOT, "data/sf_eval_train200k.csv.gz"))
    assert (sf["site"].values == df["site"].values).all()
    df[SF_COLS] = sf[SF_COLS].values
    df = add_hist_diff(df.merge(pd.read_csv(os.path.join(ROOT, "data/history_features.csv.gz")), on="site", how="left"))
    df = add_model_columns(df)
    y = df["result"].map(LABELS).values
    print("share of players with no earlier game:",
          round(((df["white_hist_games"] == 0).mean() + (df["black_hist_games"] == 0).mean()) / 2, 3),
          " median earlier games:", df[["white_hist_games", "black_hist_games"]].median().tolist(), flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    models = {"Logistic Regression": lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                                           StandardScaler(), LogisticRegression(max_iter=3000)),
              "HistGradientBoosting": lambda: HistGradientBoostingClassifier(random_state=RS)}
    rows = []
    for fname, cols in [("base", BASE), ("base + history", BASE + HIST_COLS),
                        ("base + Stockfish", BASE + SF_COLS), ("base + Stockfish + history", BASE + SF_COLS + HIST_COLS)]:
        for mname, make in models.items():
            s = cross_validate(make(), df[cols].astype(float), y, cv=cv, scoring=["accuracy", "f1_macro"])
            rows.append({"features": fname, "model": mname, "n_train_games": len(y),
                         "cv_acc_mean": round(s["test_accuracy"].mean(), 4), "cv_acc_std": round(s["test_accuracy"].std(), 4),
                         "cv_macroF1_mean": round(s["test_f1_macro"].mean(), 4), "cv_always_white": round((y == 2).mean(), 4)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results/history_cv.csv"), index=False)


if __name__ == "__main__":
    t0 = time.time()
    if "--skip-leakage" not in sys.argv:
        leakage_check()
    cv_compare()
    print(f"done ({time.time() - t0:.0f} s)")
