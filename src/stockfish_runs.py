"""Step 2 (faster plan): Stockfish runs needed for Step 4.

Run from the project folder:   python src/stockfish_runs.py
 Job 1: untouched test file (part1): random 200,000 games (random_state 42), Stockfish at move 10
        -> data/sf_eval_part1_200k.csv.gz   (site, sf_cp, sf_expected)
 Job 2: per-cutoff training data (moves 15, 20, 25, 30): games from the TRAIN part of the main file
        (same 80/20 split, random_state 42), still running after the cutoff, random 100,000 per cutoff,
        features + Stockfish at that cutoff only
        -> data/cutoff_{move}_train100k.csv.gz
 (Move 10 reuses data/sf_eval_train200k.csv.gz from the small test.)
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import chess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.features import features_from_movetext
from src.build_dataset import keep_row, HEADER_COLS
from src.stockfish_features import open_engine, evaluate, DEPTH

from sklearn.model_selection import train_test_split

RS = 42
N_ENGINES = 8
LABELS = {"0-1": 0, "1/2-1/2": 1, "1-0": 2}
CUTOFFS = [15, 20, 25, 30]
N_PER_CUTOFF = 100_000
N_CANDIDATES = 220_000      # enough so that >= 100k are still running after move 30 (~57% are)

_eng = None


def _init():
    global _eng
    _eng = open_engine()


def _eval(fen):
    return evaluate(_eng, chess.Board(fen), DEPTH)


def run_stockfish(fens):
    with Pool(N_ENGINES, initializer=_init) as pool:
        return pool.map(_eval, fens, chunksize=200)


def _features(args):
    rows, plies = args
    out = []
    for r in rows:
        h = {c: (None if r[c] is None else str(r[c])) for c in HEADER_COLS}
        out.append(features_from_movetext(h, r["movetext"], early_plies=plies))
    return out


if __name__ == "__main__":
    t0 = time.time()

    # ---- Job 1: untouched test file
    out1 = os.path.join(ROOT, "data/sf_eval_part1_200k.csv.gz")
    if not os.path.exists(out1):
        te = pd.read_csv(os.path.join(ROOT, "data/games_2025_01_blitz_part1.csv.gz"), usecols=["site", "fen_cutoff"])
        te = te.sample(n=200_000, random_state=RS).reset_index(drop=True)
        res = run_stockfish(te["fen_cutoff"].tolist())
        sf = pd.DataFrame(res, columns=["sf_cp", "sf_expected"])
        sf.insert(0, "site", te["site"].values)
        sf.to_csv(out1, index=False)
        print(f"job 1 done: {len(sf)} test positions ({time.time() - t0:.0f} s)", flush=True)

    # ---- Job 2: per-cutoff training data
    main = pd.read_csv(os.path.join(ROOT, "data/games_2025_01_blitz.csv.gz"), usecols=["site", "result"])
    y = main["result"].map(LABELS).values
    itr, _ = train_test_split(np.arange(len(main)), test_size=0.2, random_state=RS, stratify=y)
    train_sites = main["site"].values[itr]
    cand = set(np.random.RandomState(RS).choice(train_sites, N_CANDIDATES, replace=False).tolist())
    pf = pq.ParquetFile(os.path.join(ROOT, "data/hf/lichess_2025_01_part0.parquet"))
    games = []
    for i in range(pf.metadata.num_row_groups):
        for r in pf.read_row_group(i, columns=HEADER_COLS + ["WhiteTitle", "BlackTitle", "movetext"]).to_pylist():
            if r["Site"] in cand and keep_row(r):
                games.append(r)
    print(f"candidate train games: {len(games)} ({time.time() - t0:.0f} s)", flush=True)
    chunks = [games[i:i + 2000] for i in range(0, len(games), 2000)]
    for m in CUTOFFS:
        out = os.path.join(ROOT, f"data/cutoff_{m}_train100k.csv.gz")
        if os.path.exists(out):
            continue
        with Pool(max(1, os.cpu_count() - 1)) as pool:
            feats = [f for part in pool.map(_features, [(c, 2 * m) for c in chunks]) for f in part]
        df = pd.DataFrame(feats)
        df = df[df["game_continues"] == 1].sample(n=N_PER_CUTOFF, random_state=RS).reset_index(drop=True)
        res = run_stockfish(df["fen_cutoff"].tolist())
        df[["sf_cp", "sf_expected"]] = np.array(res)
        df.to_csv(out, index=False)
        print(f"cutoff {m}: {len(df)} games saved ({time.time() - t0:.0f} s)", flush=True)
    print("all done")
