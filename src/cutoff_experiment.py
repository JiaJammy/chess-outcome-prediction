"""Accuracy vs cutoff move (10, 15, 20, 25, 30 moves) on 2025 blitz games.

Run from the project folder:   python src/cutoff_experiment.py
Output: results/accuracy_by_cutoff.csv and results/accuracy_by_cutoff.png

Steps:
 1. pick a random 200,000 blitz games (same rules as build_dataset.py, random_state 42)
 2. for each cutoff, make the features using ONLY the moves/clocks up to that cutoff
 3. keep only games still running after that cutoff (so the result is not already on the board)
 4. 80/20 split, then score: always White, ratings-only Logistic Regression, HGB with all features
Extra: the same thing on ONE fixed group of games (those still running after move 30),
so we can see the effect of more moves without the group of games changing.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.features import (features_from_movetext, add_model_columns,
                          RATING_COLS_V2, TIME_COLS_V2, BOARD_COLS_V2, CLOCK_COLS_V2)
from src.build_dataset import keep_row, HEADER_COLS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

IN_FILE = os.path.join(ROOT, "data/hf/lichess_2025_01_part0.parquet")
OUT_DIR = os.path.join(ROOT, "results")
N_SAMPLE = 200_000
CUTOFF_MOVES = [10, 15, 20, 25, 30]
RS = 42
ALL_FEATURES = RATING_COLS_V2 + TIME_COLS_V2 + BOARD_COLS_V2 + CLOCK_COLS_V2
LABELS = {"0-1": 0, "1/2-1/2": 1, "1-0": 2}          # black_win=0, draw=1, white_win=2 (same as midterm)


def load_sample():
    """Pass 1: find eligible games using headers only. Pass 2: read move text of the sampled games."""
    pf = pq.ParquetFile(IN_FILE)
    head_cols = HEADER_COLS + ["WhiteTitle", "BlackTitle"]
    eligible, offset = [], 0
    for i in range(pf.metadata.num_row_groups):
        rows = pf.read_row_group(i, columns=head_cols).to_pylist()
        eligible += [offset + j for j, r in enumerate(rows) if keep_row(r)]
        offset += len(rows)
    chosen = set(np.random.RandomState(RS).choice(eligible, N_SAMPLE, replace=False).tolist())
    games, offset = [], 0
    for i in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(i).num_rows
        if any(offset + j in chosen for j in range(n)):
            rows = pf.read_row_group(i, columns=HEADER_COLS + ["movetext"]).to_pylist()
            games += [r for j, r in enumerate(rows) if offset + j in chosen]
        offset += n
    print(f"eligible blitz games: {len(eligible)}, sampled: {len(games)}", flush=True)
    return games


def _features(args):
    rows, plies = args
    out = []
    for r in rows:
        h = {c: (None if r[c] is None else str(r[c])) for c in HEADER_COLS}
        try:
            out.append(features_from_movetext(h, r["movetext"], early_plies=plies))
        except ValueError:
            out.append(None)
    return out


def score(df, plies, tag):
    """Split, train, and score one table of games. Returns one result row."""
    df = add_model_columns(df, early_plies=plies)
    y = df["result"].map(LABELS).values
    itr, ite = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RS, stratify=y)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    lr.fit(df.loc[itr, RATING_COLS_V2], y[itr])
    p_lr = lr.predict(df.loc[ite, RATING_COLS_V2])
    hgb = HistGradientBoostingClassifier(random_state=RS)
    hgb.fit(df.loc[itr, ALL_FEATURES].astype(float), y[itr])
    p_hgb = hgb.predict(df.loc[ite, ALL_FEATURES].astype(float))
    return {"games": tag, "cutoff_move": plies // 2, "cutoff_ply": plies,
            "n_games": len(df), "n_train": len(itr), "n_test": len(ite),
            "test_acc_always_white": round((y[ite] == 2).mean(), 4),
            "test_acc_ratings_logreg": round(accuracy_score(y[ite], p_lr), 4),
            "test_macroF1_ratings_logreg": round(f1_score(y[ite], p_lr, average="macro"), 4),
            "test_acc_hgb_all": round(accuracy_score(y[ite], p_hgb), 4),
            "test_macroF1_hgb_all": round(f1_score(y[ite], p_hgb, average="macro"), 4)}


def make_chart(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = res[res["games"] == "still running after cutoff"]
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    x = r["cutoff_move"]
    lines = [("test_acc_hgb_all", "HGB, all features", "#2a78d6", "-"),
             ("test_acc_ratings_logreg", "Ratings only (LogReg)", "#eb6834", "-"),
             ("test_acc_always_white", "Always White (baseline)", "#8a8984", "--")]
    for col, name, color, ls in lines:
        ax.plot(x, r[col], ls, color=color, lw=2, marker="o", ms=6, mec="#fcfcfb", mew=1.5, label=name,
                solid_capstyle="round")
        ax.annotate(f"{r[col].iloc[-1]:.3f}", (x.iloc[-1], r[col].iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color="#52514e")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{m}\n(n={n:,})" for m, n in zip(x, r["n_games"])], fontsize=9, color="#52514e")
    ax.set_xlabel("Cutoff (moves per side seen by the model)", color="#0b0b0b")
    ax.set_ylabel("Test accuracy", color="#0b0b0b")
    ax.set_title("Accuracy vs cutoff move - 2025 Lichess blitz (games still running after cutoff)",
                 fontsize=11, color="#0b0b0b", loc="left")
    ax.grid(axis="y", color="#e4e3df", lw=1); ax.grid(axis="x", visible=False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors="#52514e")
    ax.set_xlim(x.min() - 1, x.max() + 3)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(path, facecolor=fig.get_facecolor()); plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    games = load_sample()
    chunks = [games[i:i + 2000] for i in range(0, len(games), 2000)]
    tables = {}
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        for m in CUTOFF_MOVES:
            plies = 2 * m
            feats = [f for part in pool.map(_features, [(c, plies) for c in chunks]) for f in part]
            tables[m] = pd.DataFrame([f if f is not None else {} for f in feats])   # row i = game i
            print(f"cutoff {m}: features done ({time.time() - t0:.0f} s)", flush=True)

    # games still running after move 30 (used for the fixed-group comparison)
    long_games = tables[CUTOFF_MOVES[-1]]["game_continues"] == 1

    results = []
    for m in CUTOFF_MOVES:
        t = tables[m]
        keep = t["game_continues"] == 1
        results.append(score(t[keep].reset_index(drop=True), 2 * m, "still running after cutoff"))
        results.append(score(t[long_games].reset_index(drop=True), 2 * m, f"same games (running after move {CUTOFF_MOVES[-1]})"))
        print(results[-2], "\n", results[-1], flush=True)

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OUT_DIR, "accuracy_by_cutoff.csv"), index=False)
    make_chart(res, os.path.join(OUT_DIR, "accuracy_by_cutoff.png"))
    print(f"saved results ({time.time() - t0:.0f} s)")
