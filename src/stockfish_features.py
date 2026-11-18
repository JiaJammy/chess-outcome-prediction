"""Stockfish evaluation of the position AT THE CUTOFF (nothing after it).

Used by the notebook scripts and the app. Stockfish looks only at the board position after
ply 20 (or the chosen cutoff) and says who stands better. It never sees later moves or the result.
"""
import os
import shutil

import numpy as np
import chess
import chess.engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_EXE = os.path.join(ROOT, "tools", "stockfish", "stockfish", "stockfish-windows-x86-64-universal.exe")
DEPTH = 10
CP_CLIP = 1000          # evaluations are capped at +-10 pawns (mate also counts as +-10 pawns)


def find_stockfish():
    """Local Windows copy first, otherwise a 'stockfish' on the PATH (e.g. Linux / Streamlit Cloud)."""
    if os.path.exists(LOCAL_EXE):
        return LOCAL_EXE
    for p in [shutil.which("stockfish"), "/usr/games/stockfish", "/usr/bin/stockfish"]:
        if p and os.path.exists(p):
            return p
    return None


def open_engine(path=None):
    eng = chess.engine.SimpleEngine.popen_uci(path or find_stockfish())
    eng.configure({"Threads": 1, "Hash": 16})       # 1 thread per engine; we run many engines in parallel
    return eng


def evaluate(engine, board, depth=DEPTH):
    """Return (centipawns from White's view, clipped; expected score for White 0..1)."""
    # game=object() makes Stockfish forget the previous position first (new game each time),
    # so the same position always gets the same score.
    info = engine.analyse(board, chess.engine.Limit(depth=depth), game=object())
    score = info["score"].white()
    cp = score.score(mate_score=100_000)
    cp = float(np.clip(cp, -CP_CLIP, CP_CLIP))
    expected = score.wdl(model="sf", ply=board.ply()).expectation()     # win=1, draw=0.5, loss=0
    return cp, float(expected)
