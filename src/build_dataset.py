"""Build the final-project dataset from one Lichess month (Hugging Face copy of the official Lichess database).

Run from the project folder:   python src/build_dataset.py
Input : data/hf/lichess_2025_01_part0.parquet   (download link is in README / PROJECT_CONTEXT.md)
Output: data/games_2025_01_blitz.csv.gz

Rules (decided 2026-11-18):
  - blitz only (Lichess rule: base + 40 * increment is 180..479 seconds)
  - both ratings present, result is 1-0 / 0-1 / 1/2-1/2
  - no BOT accounts
  - game must still be going after move 10 (more than 20 plies), so the result is not already on the board
"""
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import features_from_movetext, parse_time_control, is_blitz

IN_FILE = "data/hf/lichess_2025_01_part0.parquet"
OUT_FILE = "data/games_2025_01_blitz.csv.gz"
HEADER_COLS = ["Site", "UTCDate", "UTCTime", "White", "Black", "WhiteElo", "BlackElo", "ECO", "TimeControl", "Result"]


def keep_row(r):
    if r["Result"] not in ("1-0", "0-1", "1/2-1/2"):
        return False
    if pd.isna(r["WhiteElo"]) or pd.isna(r["BlackElo"]):
        return False
    if r["WhiteTitle"] == "BOT" or r["BlackTitle"] == "BOT":
        return False
    base, inc = parse_time_control(r["TimeControl"])
    return base == base and is_blitz(base, inc)        # base == base is False for NaN


def work(batch):
    out = []
    for r in batch:
        if not keep_row(r):
            continue
        h = {c: (None if r[c] is None else str(r[c])) for c in HEADER_COLS}
        try:
            f = features_from_movetext(h, r["movetext"])
        except ValueError:          # a broken move text (very rare) -> skip the game
            continue
        if f["game_continues"] == 1:
            out.append(f)
    return out


def work_no_filter(batch):
    """Same as work() but keeps every game (used to copy the midterm setup exactly)."""
    out = []
    for r in batch:
        h = {c: (None if r[c] is None else str(r[c])) for c in HEADER_COLS}
        try:
            out.append(features_from_movetext(h, r["movetext"]))
        except ValueError:
            continue
    return out


if __name__ == "__main__":
    # optional: python src/build_dataset.py <input.parquet> <output.csv.gz> [--no-filter]
    if len(sys.argv) >= 3:
        IN_FILE, OUT_FILE = sys.argv[1], sys.argv[2]
    fn = work_no_filter if "--no-filter" in sys.argv else work
    t0 = time.time()
    pf = pq.ParquetFile(IN_FILE)
    cols = HEADER_COLS + ["WhiteTitle", "BlackTitle", "movetext"]
    print("games in file:", pf.metadata.num_rows)
    # read the file in pieces (row groups) so memory stays small
    batches = (pf.read_row_group(i, columns=cols).to_pylist() for i in range(pf.metadata.num_row_groups))
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        rows = [row for part in pool.imap(fn, batches) for row in part]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)
    print(f"kept {len(df)} games -> {OUT_FILE}  ({time.time() - t0:.0f} s)")
