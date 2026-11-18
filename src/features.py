"""Feature code for the chess project.

This one file is used by BOTH the notebook and (later) the Streamlit app,
so the features are always made in exactly the same way.
"""
import io
import re

import numpy as np
import pandas as pd
import chess
import chess.pgn

EARLY_PLIES = 20   # 10 moves each = 20 plies (we only look at this part of the game)

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def to_int(x):
    """Turn text like '1650' into a number. Missing or bad values become NaN."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return np.nan


# =====================================================================
# VERSION 1 (midterm). Kept unchanged so the midterm cells give the same numbers.
# =====================================================================
def extract_game_features(game, early_plies=EARLY_PLIES):
    """Replay the first `early_plies` plies of one game and return its features as a dict."""
    h = game.headers
    board = chess.Board()
    trace, w_cap, b_cap = [], 0, 0          # trace = material balance after every ply
    w_castle = b_castle = np.nan            # ply number of first castle (NaN = did not castle)
    ply = 0
    for move in game.mainline_moves():
        if ply >= early_plies:
            break
        cap, castle, white_turn = board.is_capture(move), board.is_castling(move), board.turn
        board.push(move)
        ply += 1
        if cap:
            w_cap += int(white_turn)
            b_cap += int(not white_turn)
        if castle:
            if white_turn and np.isnan(w_castle):
                w_castle = ply
            if (not white_turn) and np.isnan(b_castle):
                b_castle = ply
        trace.append(sum(len(board.pieces(p, chess.WHITE)) * v - len(board.pieces(p, chess.BLACK)) * v
                         for p, v in PIECE_VALUES.items()))
    return {"white": h.get("White"), "black": h.get("Black"),
            "white_elo": to_int(h.get("WhiteElo")), "black_elo": to_int(h.get("BlackElo")),
            "eco": h.get("ECO"), "time_control": h.get("TimeControl"), "result": h.get("Result"),
            "white_castle_ply": w_castle, "black_castle_ply": b_castle,
            "white_captures_early": w_cap, "black_captures_early": b_cap,
            "material_balance_at_cutoff": trace[-1] if trace else np.nan,
            "material_swing_early": (max(trace) - min(trace)) if trace else np.nan,
            "plies_recorded": ply}


def game_iterator(source, url, n_games):
    """Give games one by one.
    source = "stream" -> read from the Lichess .pgn.zst link (no full download)
    source = a file path -> read from a .pgn file on the computer
    """
    if source == "stream":
        import requests, zstandard
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(r.raw), encoding="utf-8")
            for _ in range(n_games):
                g = chess.pgn.read_game(text)
                if g is None:
                    break
                yield g
    else:
        with open(source) as f:
            for _ in range(n_games):
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                yield g


# =====================================================================
# VERSION 2 (final project, added 2026-11-18)
# Only uses information known after move 10 (ply 20): the moves, the board,
# the clocks, the ratings and the time control.
# Never uses the result, Termination or RatingDiff (those give away the answer).
# =====================================================================
CENTRE = [chess.D4, chess.E4, chess.D5, chess.E5]
MINOR_START = {chess.WHITE: [(chess.KNIGHT, chess.B1), (chess.KNIGHT, chess.G1), (chess.BISHOP, chess.C1), (chess.BISHOP, chess.F1)],
               chess.BLACK: [(chess.KNIGHT, chess.B8), (chess.KNIGHT, chess.G8), (chess.BISHOP, chess.C8), (chess.BISHOP, chess.F8)]}
QUEEN_START = {chess.WHITE: chess.D1, chess.BLACK: chess.D8}


def parse_time_control(tc):
    """'180+2' -> (180, 2) seconds. Anything else -> (NaN, NaN)."""
    try:
        base, inc = str(tc).split("+")
        return int(base), int(inc)
    except ValueError:
        return np.nan, np.nan


def is_blitz(base, inc):
    """Lichess rule: estimated game time = base + 40 * increment. Blitz = 3 to 8 minutes."""
    est = base + 40 * inc
    return 180 <= est < 480


_TOKEN = re.compile(r"\{[^}]*\}|[^\s{}]+")
_CLK = re.compile(r"\[%clk (\d+):(\d+):(\d+)")
_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def parse_movetext(movetext, max_plies):
    """Read Lichess move text like '1. e4 { [%clk 0:03:00] } 1... e5 ...'.
    Returns (list of SAN moves, list of clock seconds left after each move).
    Stops after max_plies moves, so we never read further than needed."""
    sans, clocks = [], []
    for tok in _TOKEN.findall(movetext):
        if tok.startswith("{"):                       # a comment: may hold the clock
            m = _CLK.search(tok)
            if m and len(clocks) < len(sans):
                h, mi, s = map(int, m.groups())
                clocks.append(float(h * 3600 + mi * 60 + s))
            continue
        if tok in _RESULTS:
            break
        if tok.rstrip(".").isdigit() or tok.startswith("$"):   # move number like '12.' or a mark like '$1'
            continue
        tok = tok.rstrip("!?")                        # remove marks like 'Qh4?' -> 'Qh4'
        if not tok:
            continue
        if len(sans) >= max_plies:
            break
        if len(clocks) < len(sans):                   # previous move had no clock
            clocks.append(np.nan)
        sans.append(tok)
    while len(clocks) < len(sans):
        clocks.append(np.nan)
    return sans, clocks


def _material(board, color):
    return sum(len(board.pieces(p, color)) * v for p, v in PIECE_VALUES.items())


def _side_features(board, color):
    """Board features for one side, measured at the cutoff position (after ply 20)."""
    opp = not color
    developed = sum(1 for piece, sq in MINOR_START[color] if board.piece_at(sq) != chess.Piece(piece, color))
    queen_moved = int(board.piece_at(QUEEN_START[color]) != chess.Piece(chess.QUEEN, color))
    centre_pawns = sum(1 for sq in CENTRE if board.piece_at(sq) == chess.Piece(chess.PAWN, color))
    b = board.copy(stack=False)
    b.turn = color
    mobility = b.legal_moves.count()                  # how many legal moves this side has
    hanging_n, hanging_val = 0, 0                     # own pieces attacked and not defended
    for sq, pc in board.piece_map().items():
        if pc.color == color and pc.piece_type != chess.KING:
            if board.is_attacked_by(opp, sq) and not board.is_attacked_by(color, sq):
                hanging_n += 1
                hanging_val += PIECE_VALUES[pc.piece_type]
    k = board.king(color)
    shield = 0                                        # own pawns just in front of the king
    if k is not None:
        f, r = chess.square_file(k), chess.square_rank(k)
        step = 1 if color == chess.WHITE else -1
        for dfile in (-1, 0, 1):
            for drank in (1, 2):
                ff, rr = f + dfile, r + step * drank
                if 0 <= ff < 8 and 0 <= rr < 8 and board.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, color):
                    shield += 1
    return {"minors_developed": developed, "queen_moved": queen_moved, "centre_pawns": centre_pawns,
            "mobility": mobility, "hanging_count": hanging_n, "hanging_value": hanging_val,
            "king_shield": shield,
            "pawns": len(board.pieces(chess.PAWN, color)),
            "minors": len(board.pieces(chess.KNIGHT, color)) + len(board.pieces(chess.BISHOP, color)),
            "rooks": len(board.pieces(chess.ROOK, color)), "queens": len(board.pieces(chess.QUEEN, color))}


def features_v2(headers, moves, clocks=None, early_plies=EARLY_PLIES, total_plies=None):
    """Main v2 function.
    moves  = list of SAN strings (like 'Nf3') or chess.Move objects
    clocks = list of seconds left after each move (or None)"""
    board = chess.Board()
    trace = []
    cap = {chess.WHITE: 0, chess.BLACK: 0}
    checks = {chess.WHITE: 0, chess.BLACK: 0}
    castle_ply = {chess.WHITE: np.nan, chess.BLACK: np.nan}
    castle_side = {chess.WHITE: 0, chess.BLACK: 0}      # 0 = not castled, 1 = kingside, 2 = queenside
    king_walk = {chess.WHITE: 0, chess.BLACK: 0}         # 1 = king moved without castling
    ply = 0
    for mv in moves:
        if ply >= early_plies:
            break
        move = board.parse_san(mv) if isinstance(mv, str) else mv
        side = board.turn
        if board.is_capture(move):
            cap[side] += 1
        if board.is_castling(move):
            if np.isnan(castle_ply[side]):
                castle_ply[side] = ply + 1
                castle_side[side] = 1 if board.is_kingside_castling(move) else 2
        elif board.piece_type_at(move.from_square) == chess.KING:
            king_walk[side] = 1
        board.push(move)
        ply += 1
        if board.is_check():
            checks[side] += 1
        trace.append(_material(board, chess.WHITE) - _material(board, chess.BLACK))

    base, inc = parse_time_control(headers.get("TimeControl"))
    row = {"site": headers.get("Site"), "utc_date": headers.get("UTCDate"), "utc_time": headers.get("UTCTime"),
           "white": headers.get("White"), "black": headers.get("Black"),
           "white_elo": to_int(headers.get("WhiteElo")), "black_elo": to_int(headers.get("BlackElo")),
           "eco": headers.get("ECO"), "time_control": headers.get("TimeControl"),
           "base_time": base, "increment": inc, "result": headers.get("Result"),
           "plies_recorded": ply,
           "game_continues": int(total_plies > early_plies) if total_plies is not None else np.nan,
           "material_balance_at_cutoff": trace[-1] if trace else np.nan,
           "material_swing_early": (max(trace) - min(trace)) if trace else np.nan,
           "in_check_at_cutoff": int(board.is_check())}
    for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        row[f"{name}_castle_ply"] = castle_ply[color]
        row[f"{name}_castle_side"] = castle_side[color]
        row[f"{name}_king_walk"] = king_walk[color]
        row[f"{name}_captures_early"] = cap[color]
        row[f"{name}_checks"] = checks[color]
        for k, v in _side_features(board, color).items():
            row[f"{name}_{k}"] = v
    # clocks: seconds left after each player's 10th move
    w_clk = b_clk = np.nan
    if clocks is not None and ply == early_plies and len(clocks) >= early_plies:
        w_clk, b_clk = clocks[early_plies - 2], clocks[early_plies - 1]
    row["white_clock"], row["black_clock"] = w_clk, b_clk
    row["fen_cutoff"] = board.fen()        # the board position at the cutoff (used only for Stockfish)
    return row


def features_from_movetext(headers, movetext, early_plies=EARLY_PLIES):
    """For the Hugging Face / Lichess parquet data (all moves in one text string)."""
    sans, clocks = parse_movetext(movetext, early_plies + 1)   # +1 only to know if the game goes on
    return features_v2(headers, sans[:early_plies], clocks[:early_plies], early_plies, total_plies=len(sans))


def extract_game_features_v2(game, early_plies=EARLY_PLIES):
    """For a python-chess Game (for example an uploaded .pgn file in the app)."""
    moves, clocks = [], []
    for node in game.mainline():
        moves.append(node.move)
        c = node.clock()
        clocks.append(np.nan if c is None else c)
    return features_v2(game.headers, moves[:early_plies], clocks[:early_plies], early_plies, total_plies=len(moves))


# ---------- extra columns made from the raw v2 features (same code for notebook and app) ----------
def add_model_columns(df, early_plies=EARLY_PLIES):
    """Add the extra columns the v2 models use.
    early_plies must be the same cutoff used to make the features (20 for the main move-10 model)."""
    df = df.copy()
    df["elo_diff"] = df["white_elo"] - df["black_elo"]
    df["avg_elo"] = (df["white_elo"] + df["black_elo"]) / 2
    df["elo_expected"] = 1 / (1 + 10 ** (-df["elo_diff"] / 400))      # Elo formula: expected score of White
    df["white_castled"] = df["white_castle_ply"].notna().astype(int)
    df["black_castled"] = df["black_castle_ply"].notna().astype(int)
    df["white_castle_ply"] = df["white_castle_ply"].fillna(early_plies + 1)     # never castled -> one after the cutoff
    df["black_castle_ply"] = df["black_castle_ply"].fillna(early_plies + 1)
    for a in ["minors_developed", "mobility", "hanging_value", "king_shield", "checks", "captures_early",
              "centre_pawns", "pawns", "minors", "rooks", "queens"]:
        df[f"diff_{a}"] = df[f"white_{a}"] - df[f"black_{a}"]
    df["clock_diff"] = df["white_clock"] - df["black_clock"]
    df["white_clock_frac"] = df["white_clock"] / df["base_time"].replace(0, np.nan)
    df["black_clock_frac"] = df["black_clock"] / df["base_time"].replace(0, np.nan)
    eco = df["eco"].fillna("A00").astype(str)
    fam = eco.str[0]
    for f in "BCDE":
        df[f"eco_{f}"] = (fam == f).astype(int)
    df["eco_num"] = fam.map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}).fillna(0) * 100 + \
        pd.to_numeric(eco.str[1:], errors="coerce").fillna(0)          # A00..E99 -> 0..499
    return df


RATING_COLS_V2 = ["white_elo", "black_elo", "elo_diff", "avg_elo", "elo_expected"]
TIME_COLS_V2 = ["base_time", "increment"]
CLOCK_COLS_V2 = ["white_clock", "black_clock", "clock_diff", "white_clock_frac", "black_clock_frac"]
BOARD_COLS_V2 = (["material_balance_at_cutoff", "material_swing_early", "in_check_at_cutoff",
                  "eco_B", "eco_C", "eco_D", "eco_E", "eco_num"]
                 + [f"{s}_{a}" for s in ("white", "black") for a in
                    ["castle_ply", "castle_side", "castled", "king_walk", "captures_early", "checks",
                     "minors_developed", "queen_moved", "centre_pawns", "mobility", "hanging_count",
                     "hanging_value", "king_shield"]]
                 + [f"diff_{a}" for a in ["minors_developed", "mobility", "hanging_value", "king_shield", "checks",
                                          "captures_early", "centre_pawns", "pawns", "minors", "rooks", "queens"]])
