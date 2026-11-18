"""Step 2 small test: does a Stockfish evaluation at move 10 help?

Run from the project folder:   python src/stockfish_experiment.py
Uses ONLY training data (the untouched test file is not opened):
 1. same 80/20 split of the main 2025 file as the final model (random_state 42), keep the TRAIN part
 2. random 200,000 of those train games (random_state 42)
 3. Stockfish (depth 10, 8 engines in parallel) on the board AT MOVE 10 only (column fen_cutoff)
 4. 5-fold cross-validation, with vs without the 2 Stockfish columns, for Logistic Regression and HGB
Outputs: data/sf_eval_train200k.csv.gz (evaluations, reusable), results/stockfish_cv.csv
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import chess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.features import add_model_columns, RATING_COLS_V2, TIME_COLS_V2, BOARD_COLS_V2, CLOCK_COLS_V2
from src.stockfish_features import open_engine, evaluate, DEPTH

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

RS = 42
N_SAMPLE = 200_000
N_ENGINES = 8
MAIN_FILE = os.path.join(ROOT, "data/games_2025_01_blitz.csv.gz")
SF_FILE = os.path.join(ROOT, "data/sf_eval_train200k.csv.gz")
OUT = os.path.join(ROOT, "results/stockfish_cv.csv")
BASE = RATING_COLS_V2 + TIME_COLS_V2 + BOARD_COLS_V2 + CLOCK_COLS_V2
SF_COLS = ["sf_cp", "sf_expected"]
LABELS = {"0-1": 0, "1/2-1/2": 1, "1-0": 2}

_eng = None


def _init():
    global _eng
    _eng = open_engine()


def _eval(fen):
    return evaluate(_eng, chess.Board(fen), DEPTH)


if __name__ == "__main__":
    t0 = time.time()
    df = pd.read_csv(MAIN_FILE)
    y_all = df["result"].map(LABELS).values
    itr, _ = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RS, stratify=y_all)
    pick, _ = train_test_split(itr, train_size=N_SAMPLE, random_state=RS, stratify=y_all[itr])
    df = df.iloc[pick].reset_index(drop=True)
    print(f"train sample: {len(df)} games", flush=True)

    if os.path.exists(SF_FILE):
        sf = pd.read_csv(SF_FILE)
    else:
        with Pool(N_ENGINES, initializer=_init) as pool:
            res = pool.map(_eval, df["fen_cutoff"].tolist(), chunksize=200)
        sf = pd.DataFrame(res, columns=SF_COLS)
        sf.insert(0, "site", df["site"].values)
        sf.to_csv(SF_FILE, index=False)
        print(f"stockfish done ({time.time() - t0:.0f} s)", flush=True)
    assert (sf["site"].values == df["site"].values).all()          # same games, same order
    df[SF_COLS] = sf[SF_COLS].values

    df = add_model_columns(df)
    y = df["result"].map(LABELS).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    models = {"Logistic Regression": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                                           LogisticRegression(max_iter=3000)),
              "HistGradientBoosting": lambda: HistGradientBoostingClassifier(random_state=RS)}
    rows = []
    for fname, cols in [("without Stockfish", BASE), ("with Stockfish", BASE + SF_COLS),
                        ("ratings + Stockfish only", RATING_COLS_V2 + SF_COLS)]:
        for mname, make in models.items():
            s = cross_validate(make(), df[cols].astype(float), y, cv=cv, scoring=["accuracy", "f1_macro"])
            rows.append({"features": fname, "model": mname, "n_train_games": len(y),
                         "cv_acc_mean": round(s["test_accuracy"].mean(), 4), "cv_acc_std": round(s["test_accuracy"].std(), 4),
                         "cv_macroF1_mean": round(s["test_f1_macro"].mean(), 4),
                         "cv_always_white": round((y == 2).mean(), 4)})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"saved {OUT} ({time.time() - t0:.0f} s)")
