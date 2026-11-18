"""Prediction logic used by app.py (kept separate so it can be tested without the web page).

For each game:
 1. read it with python-chess (from an uploaded .pgn file or pasted moves)
 2. pick the LATEST cutoff the game has reached (10, 15, 20, 25 or 30 moves per side)
 3. make the features using only the moves up to that cutoff (same code as training: src/features.py)
 4. Stockfish evaluates the board at that cutoff (if Stockfish is available)
 5. the model for that cutoff gives the chances of White win / draw / Black win
"""
import io
import os

import numpy as np
import pandas as pd
import joblib
import chess
import chess.pgn

from src.features import extract_game_features_v2, add_model_columns, to_int, parse_time_control, is_blitz
from src.stockfish_features import find_stockfish, open_engine, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTOFFS = [10, 15, 20, 25, 30]          # moves per side
MAX_GAMES = 500


def load_models(models_dir=os.path.join(ROOT, "models")):
    """Returns {(cutoff_move, uses_stockfish): bundle}."""
    models = {}
    for m in CUTOFFS:
        models[(m, True)] = joblib.load(os.path.join(models_dir, f"model_move{m}.joblib"))
        models[(m, False)] = joblib.load(os.path.join(models_dir, f"model_move{m}_nosf.joblib"))
    return models


def read_games(text):
    """Read every game in a PGN text. Pasted moves without headers also work ('1. e4 e5 2. Nf3 ...')."""
    games, stream = [], io.StringIO(text)
    while len(games) < MAX_GAMES:
        try:
            g = chess.pgn.read_game(stream)
        except Exception:          # badly broken text
            break
        if g is None:
            break
        if not list(g.mainline_moves()):       # no moves at all (empty or not chess) -> skip
            continue
        games.append(g)
    return games


def game_info(g):
    h = g.headers
    plies = sum(1 for _ in g.mainline_moves())
    base, inc = parse_time_control(h.get("TimeControl"))
    has_clock = any(node.clock() is not None for node in g.mainline())
    return {"white": h.get("White", "?"), "black": h.get("Black", "?"),
            "white_elo": to_int(h.get("WhiteElo")), "black_elo": to_int(h.get("BlackElo")),
            "result": h.get("Result", "*"), "plies": plies, "time_control": h.get("TimeControl", "-"),
            "time_control_known": bool(base == base),
            "blitz": bool(base == base and is_blitz(base, inc)), "has_clock": has_clock,
            "illegal_move": bool(g.errors)}


def choose_cutoff(plies, result):
    """Latest cutoff the game has reached. A finished game must still be going AFTER the cutoff
    (like the training data); an unfinished game ('*') may stop exactly at the cutoff."""
    ok = [m for m in CUTOFFS if plies > 2 * m or (result == "*" and plies == 2 * m)]
    return max(ok) if ok else None


def predict_game(g, info, models, engine=None, white_elo=None, black_elo=None):
    """Returns a dict with the cutoff used, Stockfish eval and the three chances (or a reason why not)."""
    cut = choose_cutoff(info["plies"], info["result"])
    if cut is None:
        return {"cutoff_move": None, "note": "too short (≤10 moves)"}
    plies = 2 * cut
    f = extract_game_features_v2(g, early_plies=plies)
    if white_elo is not None:
        f["white_elo"] = white_elo
    if black_elo is not None:
        f["black_elo"] = black_elo
    if f["white_elo"] != f["white_elo"] or f["black_elo"] != f["black_elo"]:      # NaN check
        return {"cutoff_move": cut, "note": "type ratings above"}
    sf_cp = np.nan
    use_sf = engine is not None
    if use_sf:
        try:
            f["sf_cp"], f["sf_expected"] = evaluate(engine, chess.Board(f["fen_cutoff"]))
            sf_cp = f["sf_cp"]
        except Exception:          # engine crashed: fall back to the model without Stockfish
            use_sf = False
    bundle = models[(cut, use_sf)]
    row = add_model_columns(pd.DataFrame([f]), early_plies=plies)
    p = bundle["model"].predict_proba(row[bundle["features"]].astype(float))[0]
    probs = dict(zip(bundle["labels"], p))
    return {"cutoff_move": cut, "stockfish_pawns": sf_cp / 100 if sf_cp == sf_cp else np.nan,
            "p_white": probs["white_win"], "p_draw": probs["draw"], "p_black": probs["black_win"],
            "fen_cutoff": f["fen_cutoff"], "used_stockfish": use_sf, "note": ""}


def predict_all(games, models, engine=None, typed_ratings=None, progress=None):
    """games: list of python-chess games. typed_ratings: {index: (white_elo, black_elo)} for games without ratings."""
    rows = []
    for i, g in enumerate(games):
        info = game_info(g)
        we, be = (typed_ratings or {}).get(i, (None, None))
        r = predict_game(g, info, models, engine, we, be)
        if we is not None:                      # show the ratings that were actually used
            info["white_elo"], info["black_elo"] = we, be
        rows.append({"game": i + 1, **info, **r})
        if progress:
            progress(i + 1, len(games))
    cols = ["game", "white", "black", "white_elo", "black_elo", "result", "plies", "time_control",
            "time_control_known", "blitz", "has_clock",
            "illegal_move", "cutoff_move", "stockfish_pawns", "p_white", "p_draw", "p_black", "fen_cutoff",
            "used_stockfish", "note"]
    return pd.DataFrame(rows).reindex(columns=cols)      # same columns even if no game could be predicted


def stockfish_engine():
    """Open Stockfish if it is installed, else None (the app then uses the no-Stockfish models)."""
    path = find_stockfish()
    if path is None:
        return None
    try:
        return open_engine(path)
    except Exception:
        return None
