"""Does the Stockfish version change the predictions?

The live app (Streamlit Cloud) has Stockfish 17.1; the models were trained with Stockfish 19.
1. Upload data/demo_games.pgn to the live app and click "Download predictions (CSV)".
2. Run:  python src/stockfish_version_check.py chess_predictions.csv
This predicts the same games here with Stockfish 19 and compares, game by game.
Saves results/stockfish_version_check.csv (one row per game) and prints a summary.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import ROOT, load_models, read_games, predict_all, stockfish_engine

live_csv = sys.argv[1]
pgn = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "data", "demo_games.pgn")

# --- Stockfish 17.1 (live app) ---
live = pd.read_csv(live_csv)
live = live.rename(columns={"Game": "game", "Stockfish": "sf17_pawns", "White win %": "sf17_white_pct",
                            "Draw %": "sf17_draw_pct", "Black win %": "sf17_black_pct",
                            "Most likely": "sf17_most_likely"})
live["sf17_pawns"] = pd.to_numeric(live["sf17_pawns"], errors="coerce")   # "–" -> NaN

# --- Stockfish 19 (this computer) ---
engine = stockfish_engine()
print("local engine:", engine.id.get("name"))
with open(pgn, encoding="utf-8", errors="replace") as f:
    local = predict_all(read_games(f.read()), load_models(), engine)
engine.quit()
names = np.array(["White win", "Draw", "Black win"])
probs = local[["p_white", "p_draw", "p_black"]].to_numpy()
local["sf19_most_likely"] = np.where(local["p_white"].notna(), names[np.nan_to_num(probs).argmax(axis=1)], "–")

out = pd.DataFrame({
    "game": local["game"], "white": local["white"], "black": local["black"], "result": local["result"],
    "cutoff_move": local["cutoff_move"],
    "sf19_pawns": local["stockfish_pawns"].round(2), "sf17_pawns": live["sf17_pawns"],
    "sf19_white_pct": (local["p_white"] * 100).round(1), "sf17_white_pct": live["sf17_white_pct"].round(1),
    "sf19_black_pct": (local["p_black"] * 100).round(1), "sf17_black_pct": live["sf17_black_pct"].round(1),
    "sf19_most_likely": local["sf19_most_likely"], "sf17_most_likely": live["sf17_most_likely"],
})
out["pawns_diff"] = (out["sf17_pawns"] - out["sf19_pawns"]).round(2)
out["white_pct_diff"] = (out["sf17_white_pct"] - out["sf19_white_pct"]).round(1)
out["most_likely_changed"] = out["sf19_most_likely"] != out["sf17_most_likely"]
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
out.to_csv(os.path.join(ROOT, "results", "stockfish_version_check.csv"), index=False)

pred = out[out["sf19_white_pct"].notna()]
d = pred["white_pct_diff"].abs()
p = pred["pawns_diff"].abs()
print(f"games predicted: {len(pred)} of {len(out)}")
print(f"most-likely result changed: {int(pred['most_likely_changed'].sum())} of {len(pred)}")
print(f"White-win % change (points): median {d.median():.1f}, mean {d.mean():.1f}, max {d.max():.1f}")
print(f"Stockfish score change (pawns): median {p.median():.2f}, max {p.max():.2f}; "
      f"same score in {int((p == 0).sum())} games")
print(out.to_string(index=False))
