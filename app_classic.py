"""Chess result predictor - the FIRST, simple version of the app (kept as a backup; the main app is app.py).

Run from the project folder:   streamlit run app_classic.py
Then open http://localhost:8501 in a browser.
"""
import os

import numpy as np
import pandas as pd
import streamlit as st
import chess
import chess.svg

from src.predict import (load_models, read_games, game_info, predict_all, stockfish_engine, MAX_GAMES, CUTOFFS)
from src.style import style_for_games, STYLE_LABELS

ROOT = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Chess result predictor", page_icon="♟", layout="wide")


# ---------- things loaded once and shared ----------
@st.cache_resource
def get_models():
    return load_models()


@st.cache_resource
def get_engine():
    if os.environ.get("CHESS_APP_NO_STOCKFISH") == "1":     # only used to test the no-Stockfish path
        return None
    return stockfish_engine()          # None if Stockfish is not installed


@st.cache_resource
def get_style_model():
    import joblib
    p = os.path.join(ROOT, "models", "style_kmeans.joblib")
    return joblib.load(p) if os.path.exists(p) else None


@st.cache_data(show_spinner=False)
def run_styles(text):
    return style_for_games(read_games(text), get_style_model())


@st.cache_data
def confidence_table():
    p = os.path.join(ROOT, "results", "confidence_bands.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data(show_spinner=False)
def run_predictions(text, typed_items, use_stockfish):
    games = read_games(text)
    engine = get_engine() if use_stockfish else None
    return predict_all(games, get_models(), engine, typed_ratings=dict(typed_items))


models = get_models()
engine = get_engine()
final = models[(10, True)]

# ---------- sidebar: how it works ----------
with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. The app reads each game.\n"
        f"2. It looks at the game only up to the **latest cutoff it has reached** "
        f"({', '.join(map(str, CUTOFFS))} moves per side).\n"
        "3. It makes the same features used in training (ratings, time control, clocks, board, "
        "and a **Stockfish** evaluation of the board at the cutoff).\n"
        "4. The model for that cutoff gives the chances of each result.\n\n"
        "Nothing after the cutoff is used.")
    if engine is not None:
        st.success(f"Stockfish: found ({engine.id.get('name', 'Stockfish')}, depth 10)")
        if "Stockfish 19" not in engine.id.get("name", "Stockfish 19"):
            st.caption("The models were trained with Stockfish 19 evaluations; this server has a different "
                       "version, so its scores can differ a little.")
    else:
        st.warning("Stockfish: not found. Using the models without Stockfish (about 2-3 points less accurate).")
    tr = final.get("test_result", {})
    if tr:
        st.subheader("Test result (move 10)")
        st.markdown(f"- Accuracy **{tr['test_acc']:.1%}** on {tr['n_test']:,} unseen 2025 blitz games\n"
                    f"- Always guessing White: {tr['test_acc_always_white']:.1%}")
    st.caption("Trained on Lichess blitz games played on 1 January 2025. Draws are rare (about 4%), "
               "so the model almost never picks 'draw' as most likely.")

# ---------- input ----------
st.title("♟ Chess result predictor")
st.write("Upload a PGN file (one or many games) or paste moves. For each game the app shows the chances of "
         "**White win / draw / Black win**, judged from the first 10-30 moves only.")

tab_up, tab_paste = st.tabs(["Upload .pgn file", "Paste moves"])
with tab_up:
    up = st.file_uploader("PGN file from Lichess or chess.com (one or many games)", type=["pgn", "txt"])
with tab_paste:
    pasted = st.text_area("Paste a PGN, or just the moves (for example: 1. e4 e5 2. Nf3 Nc6 ...)", height=140)

text = up.getvalue().decode("utf-8", errors="replace") if up is not None else pasted
if not text or not text.strip():
    st.info("Upload a file or paste moves to start.")
    st.stop()

games = read_games(text)
if not games:
    st.error("No games could be read. Please check the file or the pasted moves.")
    st.stop()
if len(games) >= MAX_GAMES:
    st.warning(f"Only the first {MAX_GAMES} games are used.")
infos = [game_info(g) for g in games]

# ---------- ratings that are missing ----------
missing = [i for i, inf in enumerate(infos) if np.isnan(inf["white_elo"]) or np.isnan(inf["black_elo"])]
typed = {}
if missing:
    st.subheader("Type the missing ratings")
    st.caption("These games have no ratings in the file. Type each player's rating (Lichess blitz rating if you know it).")
    edit = pd.DataFrame({"Game": [i + 1 for i in missing],
                         "White": [infos[i]["white"] for i in missing],
                         "Black": [infos[i]["black"] for i in missing],
                         "White rating": [1500 if np.isnan(infos[i]["white_elo"]) else int(infos[i]["white_elo"]) for i in missing],
                         "Black rating": [1500 if np.isnan(infos[i]["black_elo"]) else int(infos[i]["black_elo"]) for i in missing]})
    edited = st.data_editor(edit, hide_index=True, disabled=["Game", "White", "Black"],
                            column_config={"White rating": st.column_config.NumberColumn(min_value=400, max_value=3500, step=1),
                                           "Black rating": st.column_config.NumberColumn(min_value=400, max_value=3500, step=1)},
                            key=f"ratings_{hash(text)}")
    typed = {int(r["Game"]) - 1: (float(r["White rating"]), float(r["Black rating"])) for _, r in edited.iterrows()}

# ---------- predict ----------
with st.spinner(f"Analysing {len(games)} game(s)..."):
    df = run_predictions(text, tuple(sorted(typed.items())), engine is not None)

ok = df["p_white"].notna()
n_ok = int(ok.sum())
c1, c2, c3 = st.columns(3)
c1.metric("Games read", len(df))
c2.metric("Games predicted", n_ok)
c3.metric("Too short (10 moves or fewer)", int(df["note"].str.startswith("too short").sum()))

notes = []
known = df["time_control_known"].astype(bool)
not_blitz = known & ~df["blitz"].astype(bool)
if not_blitz.any():
    notes.append(f"{int(not_blitz.sum())} game(s) are not blitz (the model was trained on blitz, so treat these as rough).")
if (~known).any():
    notes.append(f"{int((~known).sum())} game(s) have no time control; the model assumes a typical blitz game.")
if (~df["has_clock"]).any():
    notes.append(f"{int((~df['has_clock']).sum())} game(s) have no clock times; the model still works but is a bit less sure.")
if df["illegal_move"].any():
    notes.append("Some games contain an illegal or unreadable move; only the moves before it were used.")
for n in notes:
    st.caption("ℹ️ " + n)

# ---------- results table ----------
st.subheader("Predictions")
DASH = "–"         # shown instead of an empty value (e.g. for games that are too short)
show = pd.DataFrame({
    "Game": df["game"],
    "White": df["white"] + " (" + df["white_elo"].map(lambda v: "?" if np.isnan(v) else str(int(v))) + ")",
    "Black": df["black"] + " (" + df["black_elo"].map(lambda v: "?" if np.isnan(v) else str(int(v))) + ")",
    "Moves": np.ceil(df["plies"] / 2).astype(int),
    "Result": df["result"].replace({"*": "not finished"}),
    "Move used": df["cutoff_move"].map(lambda v: DASH if pd.isna(v) else str(int(v))),
    "Stockfish": df["stockfish_pawns"].map(lambda v: DASH if pd.isna(v) else f"{v:+.2f}"),
    "White win %": df["p_white"] * 100,
    "Draw %": df["p_draw"] * 100,
    "Black win %": df["p_black"] * 100,
    "Most likely": np.where(ok, np.select([df["p_white"] >= df[["p_draw", "p_black"]].max(axis=1),
                                           df["p_black"] >= df[["p_draw", "p_white"]].max(axis=1)],
                                          ["White win", "Black win"], "Draw"), DASH),
    "Note": df["note"].fillna("").replace("", DASH),
})
pct = lambda label: st.column_config.ProgressColumn(label, min_value=0, max_value=100, format="%.0f%%", width=80)
st.dataframe(show, hide_index=True, width="stretch",
             column_config={"Game": st.column_config.NumberColumn(width=48),
                            "White": st.column_config.TextColumn(width=135),
                            "Black": st.column_config.TextColumn(width=135),
                            "Moves": st.column_config.NumberColumn(width=50),
                            "Result": st.column_config.TextColumn(width=65),
                            "Move used": st.column_config.TextColumn(width=70, help="The cutoff the model used (moves per side)"),
                            "Stockfish": st.column_config.TextColumn(width=70, help="Stockfish score at that move, in pawns (+ means White is better)"),
                            "White win %": pct("White win"), "Draw %": pct("Draw"), "Black win %": pct("Black win"),
                            "Most likely": st.column_config.TextColumn(width=80),
                            "Note": st.column_config.TextColumn(width=150)})

conf = confidence_table()
if conf is not None:
    r70 = conf[conf["model_at_least_sure"] == ">= 70%"]
    if len(r70):
        st.caption(f"How much to trust it: on unseen test games, when the move-10 model was at least 70% sure "
                   f"it was right {r70['test_acc'].iloc[0]:.0%} of the time; over all games it was right "
                   f"{conf['test_acc'].iloc[0]:.0%} of the time (always guessing White: "
                   f"{conf['test_acc_always_white_same_games'].iloc[0]:.0%}).")

st.download_button("Download predictions (CSV)", show.to_csv(index=False).encode("utf-8"),
                   file_name="chess_predictions.csv", mime="text/csv")

# ---------- one game in detail ----------
if n_ok:
    st.subheader("Look at one game")
    choices = df.loc[ok, "game"].tolist()
    pick = st.selectbox("Game", choices, format_func=lambda i: f"Game {i}: {df.loc[i - 1, 'white']} vs {df.loc[i - 1, 'black']}")
    r = df.loc[pick - 1]
    left, right = st.columns([1, 1])
    with left:
        st.markdown(chess.svg.board(chess.Board(r["fen_cutoff"]), size=340), unsafe_allow_html=True)
        st.caption(f"Board after move {int(r['cutoff_move'])} (the position the model judged).")
    with right:
        st.metric("White win", f"{r['p_white']:.0%}")
        st.metric("Draw", f"{r['p_draw']:.0%}")
        st.metric("Black win", f"{r['p_black']:.0%}")
        if r["used_stockfish"]:
            st.write(f"Stockfish at move {int(r['cutoff_move'])}: **{r['stockfish_pawns']:+.2f}** pawns (+ means White is better).")
        st.write(f"Actual result in the file: **{r['result']}**")


# ---------- player style groups (clustering) ----------
style = get_style_model()
if style is not None:
    st.subheader("Player style groups")
    st.caption(f"For every player with **{style['min_games']} or more games** in this file (only games longer than "
               f"10 moves count). The groups were found with K-Means clustering on {style['n_players']:,} Lichess "
               "blitz players, using only how they play the first 10 moves (castling, captures, material swings, "
               "development, mobility, opening variety) - not ratings or results. The groups overlap, so treat "
               "this as a rough description of style.")
    sty = run_styles(text)
    if len(sty) == 0:
        st.info(f"No player has {style['min_games']} or more games (longer than 10 moves) in this file, "
                "so no style group can be given. Upload more games of the same player.")
    else:
        for name, r in sty.iterrows():
            st.markdown(f"- **{name}** ({int(r['n_games'])} games): **{r['group_name']}** - {r['group_meaning']}")
        st.caption("Pick a player to compare their numbers with their group and with all players.")
        who = st.selectbox("Player", sty.index.tolist())
        r = sty.loc[who]
        st.markdown(f"**{who}** is in the group **{r['group_name']}**: {r['group_meaning']}")
        centres = style["scaler"].inverse_transform(style["kmeans"].cluster_centers_)   # group averages
        feats = style["features"]
        comp = pd.DataFrame({"Style feature (first 10 moves)": [STYLE_LABELS[c] for c in feats],
                             who: [r[c] for c in feats],
                             f"{r['group_name']} group average": centres[int(r["group"])],
                             "All players average": style["scaler"].mean_})
        st.dataframe(comp, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f") for c in comp.columns[1:]})
        if not r["opening_variety_known"]:
            st.caption("ℹ️ This player's games have no opening (ECO) code, so opening variety was set to the average.")

