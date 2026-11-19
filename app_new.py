"""Chess result predictor - NEW look (demo on branch ui-redesign).

Same models and same logic as app.py; only the look is new. app.py is not changed.
Run from the project folder:   streamlit run app_new.py --server.port 8502
Then open http://localhost:8502 in a browser.

Pages (chosen with ?page=... in the address, so a page can be opened directly):
  home      - landing page with the 3D board (ui/landing.html)
  predict   - upload games and see the predictions (everything the old app shows)
  model     - the project in short cards + two charts
  different - comparison with other chess-prediction projects
"""
import os

import numpy as np
import pandas as pd
import streamlit as st
from streamlit import config as st_config
import chess
import chess.svg

from src.predict import (load_models, read_games, game_info, predict_all, stockfish_engine, MAX_GAMES, CUTOFFS)
from src.style import style_for_games, STYLE_LABELS
from src.ui_parts import (esc, logo_svg, logo_img, piece_icon, pct, section_title, note_box, chart_cutoff,
                          chart_confidence, legend)

ROOT = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Predict the Winner", page_icon="♛", layout="wide", initial_sidebar_state="collapsed")

# ---------- dark theme for THIS app only (app.py keeps its normal light theme) ----------
# Streamlit reads the theme from its settings. We set them here instead of in .streamlit/config.toml,
# because that file would also change the old app. The first visit reruns once so the theme is used.
THEME = {"theme.base": "dark", "theme.backgroundColor": "#111216", "theme.secondaryBackgroundColor": "#1a1b20",
         "theme.textColor": "#f5f5f5", "theme.primaryColor": "#f5f5f5"}
if st_config.get_option("theme.backgroundColor") != THEME["theme.backgroundColor"]:
    for k, v in THEME.items():
        st_config.set_option(k, v)
    st.rerun()

PAGES = {"home": "Home", "predict": "Predict", "model": "The model", "different": "How it's different"}
page = st.query_params.get("page", "home")
if page not in PAGES:
    page = "home"


def read_text(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
        return f.read()


# ---------- things loaded once and shared (same as app.py) ----------
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
def results_csv(name):
    p = os.path.join(ROOT, "results", name)
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data(show_spinner=False)
def run_predictions(text, typed_items, use_stockfish):
    games = read_games(text)
    engine = get_engine() if use_stockfish else None
    return predict_all(games, get_models(), engine, typed_ratings=dict(typed_items))


# ---------- look: CSS + top menu ----------
st.html(f"<style>{read_text('ui', 'style.css')}"
        f".st-key-nav_{page} button p {{ color: #f5f5f5 !important; }}"
        f".st-key-nav_{page} button {{ border-bottom: 1px solid #f5f5f5 !important; }}</style>")


def go(p):
    st.query_params["page"] = p


with st.container(key="topnav"):
    cols = st.columns([3.4, 0.75, 0.95, 1.2, 1.75], vertical_alignment="center")
    cols[0].html(f'<div class="brand">{logo_img(34)}<span>Predict the winner</span></div>')
    for col, (key, label) in zip(cols[1:], PAGES.items()):
        col.button(label.upper(), key=f"nav_{key}", type="tertiary", on_click=go, args=(key,))


def hero(title, line):
    st.html(f'<div class="hero"><div class="rule"><h1>{esc(title)}</h1></div><p>{esc(line)}</p></div>')


def site_foot():
    st.html('<div class="site-foot"><span>CSET-211 · Bennett University · Jia Jain</span>'
            '<span>Lichess blitz · 1 January 2025</span>'
            '<a href="https://github.com/JiaJammy/chess-outcome-prediction" target="_blank">Code on GitHub</a></div>')


# =====================================================================
# PAGE 1 - HOME (3D board)
# =====================================================================
def page_home():
    pieces_js = read_text("ui", "pieces.js")          # the 3D pieces, shared by both 3D pictures
    with st.container(key="landing"):
        st.iframe(read_text("ui", "landing.html").replace("{{LOGO}}", logo_svg(78, width_px=1.4))
                  .replace("{{PIECES_JS}}", pieces_js), height=720)

    # ---- 1) split section: our 3D king and queen on light grey | what the app does ----
    with st.container(key="split"):
        left, right = st.columns(2, gap=None)
        with left:
            st.iframe(read_text("ui", "duo.html").replace("{{PIECES_JS}}", pieces_js)
                      .replace("{{LOGO_DARK}}", logo_svg(160, stroke="#16171b")), height=740)
        with right:
            details = ["623,722 real Lichess blitz games", "Only the first 10-30 moves are used",
                       "Stockfish evaluation of the position", "3 player styles found by clustering",
                       "Free and open source"]
            st.html('<div class="split-text"><h2>See the end from the opening.</h2>'
                    '<p>Upload any chess game. The app looks only at the opening - the first 10 to 30 moves - and '
                    'gives the chance that White wins, Black wins, or the game is drawn.</p>'
                    '<p>For players with five or more games in the file, it also names their playing style.</p>'
                    '<h3>The details</h3><ul>' + "".join(f"<li>{esc(d)}</li>" for d in details) + '</ul></div>')

    # ---- 2) numbers strip (all read from results/ and the saved model) ----
    tr = get_models()[(10, True)]["test_result"]
    conf = results_csv("confidence_bands.csv").set_index("model_at_least_sure")
    cut = results_csv("models_by_cutoff.csv").set_index("cutoff_move")
    n_styles = int(results_csv("clustering_profiles.csv")["name"].notna().sum())
    nums = [(pct(tr["test_acc"]), "Accuracy at move 10", f"test · always White {pct(tr['test_acc_always_white'])}"),
            (pct(conf.loc[">= 80%", "test_acc"]), "When the model is sure",
             f"at least 80% sure ({conf.loc['>= 80%', 'share_of_test']:.0%} of test games) · "
             f"always White {pct(conf.loc['>= 80%', 'test_acc_always_white_same_games'])}"),
            (pct(cut.loc[30, "cv_acc_mean"]), "By move 30", f"train CV · always White {pct(cut.loc[30, 'cv_always_white'])}"),
            (str(n_styles), "Playing styles", "K-Means clustering")]
    st.html('<div class="numbers">' + "".join(
        f'<div><div class="n">{n}</div><div class="l">{esc(l)}</div><div class="s">{esc(sub)}</div></div>'
        for n, l, sub in nums) + '</div>')

    # ---- 3) how it works: four steps with line-art pieces ----
    steps = [("pawn", "Upload your games"), ("knight", "We read the first moves"),
             ("bishop", "Stockfish and our model judge the position"), ("king", "See who is likely to win")]
    st.html('<div class="home-how"><h2>How it works</h2><div class="how-cards">' + "".join(
        f'<div class="how-card">{piece_icon(kind, 70)}<div class="no">0{i + 1}</div><p>{esc(t)}</p></div>'
        for i, (kind, t) in enumerate(steps)) + '</div></div>')

    # ---- 4) ready to play? ----
    st.html('<div class="ready"><h2>Ready to play?</h2><span class="rule"></span></div>')
    with st.container(key="ready_buttons", horizontal=True, horizontal_alignment="center", gap="medium"):
        st.button("Predict a game", key="home_predict", on_click=go, args=("predict",))
        st.button("How it works", key="home_model", on_click=go, args=("model",))
    site_foot()


# =====================================================================
# PAGE 2 - PREDICT (everything the old app does)
# =====================================================================
def engine_strip(engine, final):
    if engine is not None:
        name = engine.id.get("name", "Stockfish")
        extra = ("" if "Stockfish 19" in name else
                 "<span>The models were trained with Stockfish 19; this server has a different version, "
                 "so its scores can differ a little.</span>")
        eng = f"<span>ENGINE &nbsp;<b>{esc(name)}</b>, depth 10</span>{extra}"
    else:
        eng = ("<span>ENGINE &nbsp;<b>Stockfish not found</b> - using the models without Stockfish "
               "(about 2-3 points less accurate)</span>")
    tr = final.get("test_result", {})
    test = (f"<span>TEST ACCURACY (MOVE 10) &nbsp;<b>{pct(tr['test_acc'])}</b> on {tr['n_test']:,} unseen games · "
            f"always guessing White {pct(tr['test_acc_always_white'])}</span>") if tr else ""
    st.html(f'<div class="engine">{eng}{test}</div>')


def how_it_works(final):
    tr = final.get("test_result", {})
    foot = (f"<b>Test result (move 10):</b> accuracy {pct(tr['test_acc'])} on {tr['n_test']:,} unseen 2025 blitz games; "
            f"always guessing White: {pct(tr['test_acc_always_white'])}. " if tr else "")
    steps = [("Read", "The app reads each game in the file or the pasted moves."),
             ("Cut", f"It looks at the game only up to the <b>latest cutoff it has reached</b> "
                     f"({', '.join(map(str, CUTOFFS))} moves per side). Nothing after the cutoff is used."),
             ("Measure", "It makes the same features used in training: ratings, time control, clocks, the board, "
                         "and a <b>Stockfish</b> evaluation of the board at the cutoff."),
             ("Predict", "The model for that cutoff gives the chance of each result: White win, draw, Black win.")]
    body = "".join(f'<div><div class="no">0{i + 1}</div><div class="t">{t}</div><p>{d}</p></div>'
                   for i, (t, d) in enumerate(steps))
    st.html(f'<div class="paper-band"><h2>How it works</h2><div class="steps">{body}</div>'
            f'<div class="foot">{foot}Trained on Lichess blitz games played on 1 January 2025. Draws are rare '
            f'(about 4%), so the model almost never picks "draw" as most likely.</div></div>')


def chances_cell(r):
    if pd.isna(r["p_white"]):
        return '<td class="dim">–</td>'
    w, d, b = r["p_white"] * 100, r["p_draw"] * 100, r["p_black"] * 100
    return (f'<td class="chance"><div class="bar"><span class="w" style="width:{w:.1f}%"></span>'
            f'<span class="d" style="width:{d:.1f}%"></span><span class="b" style="width:{b:.1f}%"></span></div>'
            f'<div class="pcts"><span>{w:.0f}%</span><span>{d:.0f}%</span><span>{b:.0f}%</span></div></td>')


def predictions_table(df, show):
    rows = []
    for i, r in df.iterrows():
        s = show.iloc[i]
        we = "?" if np.isnan(r["white_elo"]) else int(r["white_elo"])
        be = "?" if np.isnan(r["black_elo"]) else int(r["black_elo"])
        rows.append(
            f'<tr><td class="num dim">{int(r["game"])}</td>'
            f'<td>{esc(r["white"])} <span class="elo">{we}</span></td>'
            f'<td>{esc(r["black"])} <span class="elo">{be}</span></td>'
            f'<td class="num">{s["Moves"]}</td><td>{esc(s["Result"])}</td>'
            f'<td class="num">{esc(s["Move used"])}</td><td class="num">{esc(s["Stockfish"])}</td>'
            f'{chances_cell(r)}<td>{esc(s["Most likely"])}</td><td class="dim">{esc(s["Note"])}</td></tr>')
    head = ("<tr><th>Game</th><th>White</th><th>Black</th><th>Moves</th><th>Result</th>"
            "<th title='The cutoff the model used (moves per side)'>Move used</th>"
            "<th title='Stockfish score at that move, in pawns (+ means White is better)'>Stockfish</th>"
            "<th>Chances · White / Draw / Black</th><th>Most likely</th><th>Note</th></tr>")
    st.html(legend([("White win", "background:#f5f5f5"), ("Draw", "background:#8a8a8a"),
                    ("Black win", "background:#000;border:1px solid #8a8a8a")]) +
            f'<div class="ptable-wrap"><table class="ptable"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>')


def one_game(df, ok):
    st.html(section_title("04", "Look at one game", "Pick a game to see the board the model judged."))
    choices = df.loc[ok, "game"].tolist()
    pick = st.selectbox("Game", choices, format_func=lambda i: f"Game {i}: {df.loc[i - 1, 'white']} vs {df.loc[i - 1, 'black']}")
    r = df.loc[pick - 1]
    left, right = st.columns([1, 1.15], gap="large")
    colors = {"square light": "#ececec", "square dark": "#6b6c72", "margin": "#111216", "coord": "#8a8a8a"}
    with left:
        # st.markdown (not st.html) because st.html removes <svg> pictures
        st.markdown(chess.svg.board(chess.Board(r["fen_cutoff"]), size=400, colors=colors) +
                    f'<div class="board-cap">Board after move {int(r["cutoff_move"])} (the position the model judged).</div>',
                    unsafe_allow_html=True)
    with right:
        sf = (f'<div class="kv"><span>Stockfish at move {int(r["cutoff_move"])}</span>'
              f'<b>{r["stockfish_pawns"]:+.2f} pawns</b></div>' if r["used_stockfish"] else "")
        st.html(f'<div class="game-side"><div class="big3">'
                f'<div><div class="n">{r["p_white"]:.0%}</div><div class="l">White win</div></div>'
                f'<div><div class="n">{r["p_draw"]:.0%}</div><div class="l">Draw</div></div>'
                f'<div><div class="n">{r["p_black"]:.0%}</div><div class="l">Black win</div></div></div>'
                f'<div class="kv"><span>White</span><b>{esc(r["white"])}</b></div>'
                f'<div class="kv"><span>Black</span><b>{esc(r["black"])}</b></div>{sf}'
                f'<div class="kv"><span>Actual result in the file</span><b>{esc(r["result"])}</b></div>'
                f'<p class="small-note" style="margin-top:12px">+ means White is better. The model never sees the '
                f'moves after the cutoff.</p></div>')


# one short line per style group, written from results/clustering_profiles.csv:
# Developer = most legal moves (38.0) and many pieces out, castles least often (38% of games) after Tactical;
# Early castler = castles in 75% of games and earliest (ply 15.6), fewest captures (1.37);
# Tactical = most captures (2.05) and biggest material swings (4.36), fewest pieces developed (2.74).
STYLE_ONE_LINE = {"Developer": "Gets the pieces out fast and keeps many moves open.",
                  "Early castler": "Castles early and keeps the opening quiet.",
                  "Tactical": "Trades early, with big swings in material."}


def style_groups(text):
    style = get_style_model()
    if style is None:
        return
    st.html(section_title("05", "Player style groups",
                          f"For every player with {style['min_games']} or more games in this file (only games longer "
                          f"than 10 moves count). The groups were found with K-Means clustering on "
                          f"{style['n_players']:,} Lichess blitz players, using only how they play the first 10 moves "
                          "(castling, captures, material swings, development, mobility, opening variety) - not ratings "
                          "or results. The groups overlap, so treat this as a rough description of style."))
    sty = run_styles(text)
    if len(sty) == 0:
        st.html(note_box(f"No player has {style['min_games']} or more games (longer than 10 moves) in this file, "
                         "so no style group can be given. Upload more games of the same player."))
        return
    cards = "".join(f'<div class="card"><div class="name">{esc(name)}</div>'
                    f'<div class="tag">{int(r["n_games"])} games</div><div class="grp">{esc(r["group_name"])}</div>'
                    f'<p class="one">{esc(STYLE_ONE_LINE.get(r["group_name"], ""))}</p>'
                    f'<p class="full">{esc(r["group_meaning"])}</p></div>' for name, r in sty.iterrows())
    st.html(f'<div class="players">{cards}</div>')
    who = st.selectbox("Pick a player to compare their numbers with their group and with all players",
                       sty.index.tolist())
    r = sty.loc[who]
    centres = style["scaler"].inverse_transform(style["kmeans"].cluster_centers_)   # group averages
    feats = style["features"]
    body = "".join(f'<tr><td>{esc(STYLE_LABELS[c])}</td><td class="you">{r[c]:.2f}</td>'
                   f'<td>{centres[int(r["group"])][j]:.2f}</td><td class="dim">{style["scaler"].mean_[j]:.2f}</td></tr>'
                   for j, c in enumerate(feats))
    st.html(f'<p><b>{esc(who)}</b> is in the group <b>{esc(r["group_name"])}</b>: {esc(r["group_meaning"])}</p>'
            f'<div class="table-wrap"><table class="ctable"><thead><tr><th>Style feature (first 10 moves)</th>'
            f'<th>{esc(who)}</th><th>{esc(r["group_name"])} group average</th><th>All players average</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div>')
    if not r["opening_variety_known"]:
        st.html('<p class="small-note">This player\'s games have no opening (ECO) code, so opening variety was set to the average.</p>')


def predict_section(engine):
    """Input, ratings, predictions, one game, style groups. Returns early when there is nothing to show."""
    st.html(section_title("01", "Your games", "Upload a PGN file (one or many games) or paste moves. For each game "
                          "the app shows the chances of White win / draw / Black win, judged from the first 10-30 moves only."))
    tab_up, tab_paste = st.tabs(["Upload .pgn file", "Paste moves"])
    with tab_up:
        up = st.file_uploader("PGN file from Lichess or chess.com (one or many games)", type=["pgn", "txt"])
    with tab_paste:
        pasted = st.text_area("Paste a PGN, or just the moves (for example: 1. e4 e5 2. Nf3 Nc6 ...)", height=140)

    b1, b2, _ = st.columns([1.3, 1.1, 3])
    if b1.button("Try sample games", key="sample_on"):
        st.session_state["use_sample"] = True
    if st.session_state.get("use_sample") and b2.button("Clear sample", key="sample_off"):
        st.session_state["use_sample"] = False
        st.rerun()

    if up is not None:
        text = up.getvalue().decode("utf-8", errors="replace")
    elif pasted and pasted.strip():
        text = pasted
    elif st.session_state.get("use_sample"):
        text = read_text("data", "demo_games.pgn")
        st.html('<p class="small-note">Showing the sample games (data/demo_games.pgn: 18 real Lichess blitz games). '
                'Upload a file or paste moves to use your own.</p>')
    else:
        text = ""
    if not text.strip():
        st.html(note_box("Upload a file, paste moves, or press <b>TRY SAMPLE GAMES</b> to start."))
        return

    games = read_games(text)
    if not games:
        st.html(note_box("No games could be read. Please check the file or the pasted moves."))
        return
    if len(games) >= MAX_GAMES:
        st.html(note_box(f"Only the first {MAX_GAMES} games are used."))
    infos = [game_info(g) for g in games]

    # ---------- ratings that are missing ----------
    missing = [i for i, inf in enumerate(infos) if np.isnan(inf["white_elo"]) or np.isnan(inf["black_elo"])]
    typed = {}
    if missing:
        st.html(section_title("02", "Type the missing ratings",
                              "These games have no ratings in the file. Type each player's rating "
                              "(Lichess blitz rating if you know it)."))
        for i in missing:
            c0, c1, c2 = st.columns([1.2, 2, 2], vertical_alignment="bottom")
            c0.html(f'<p class="caps" style="font-size:.75rem;color:#8a8a8a;margin-bottom:10px">Game {i + 1}</p>')
            w0 = 1500 if np.isnan(infos[i]["white_elo"]) else int(infos[i]["white_elo"])
            b0 = 1500 if np.isnan(infos[i]["black_elo"]) else int(infos[i]["black_elo"])
            we = c1.number_input(f"White: {infos[i]['white']}", 400, 3500, w0, 1, key=f"we_{i}_{hash(text)}")
            be = c2.number_input(f"Black: {infos[i]['black']}", 400, 3500, b0, 1, key=f"be_{i}_{hash(text)}")
            typed[i] = (float(we), float(be))

    # ---------- predict ----------
    with st.spinner(f"Analysing {len(games)} game(s)..."):
        df = run_predictions(text, tuple(sorted(typed.items())), engine is not None)
    ok = df["p_white"].notna()
    n_ok = int(ok.sum())

    st.html(section_title("03", "Predictions"))
    st.html(f'<div class="stats"><div><div class="n">{len(df)}</div><div class="l">Games read</div></div>'
            f'<div><div class="n">{n_ok}</div><div class="l">Games predicted</div></div>'
            f'<div><div class="n">{int(df["note"].str.startswith("too short").sum())}</div>'
            f'<div class="l">Too short (10 moves or fewer)</div></div></div>')

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
        st.html(f'<p class="small-note">— {esc(n)}</p>')

    # same table as app.py (also used for the CSV download)
    DASH = "–"
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
    predictions_table(df, show)

    conf = results_csv("confidence_bands.csv")
    if conf is not None:
        r70 = conf[conf["model_at_least_sure"] == ">= 70%"]
        if len(r70):
            st.html(f'<p class="small-note" style="margin-top:12px">How much to trust it: on unseen test games, when the '
                    f'move-10 model was at least 70% sure it was right {r70["test_acc"].iloc[0]:.0%} of the time; over all '
                    f'games it was right {conf["test_acc"].iloc[0]:.0%} of the time (always guessing White: '
                    f'{conf["test_acc_always_white_same_games"].iloc[0]:.0%}).</p>')
    st.download_button("Download predictions (CSV)", show.to_csv(index=False).encode("utf-8"),
                       file_name="chess_predictions.csv", mime="text/csv")

    if n_ok:
        one_game(df, ok)
    style_groups(text)


def page_predict():
    models = get_models()
    engine = get_engine()
    final = models[(10, True)]
    hero("The predictor", "See the end from the opening")
    engine_strip(engine, final)
    predict_section(engine)
    how_it_works(final)
    site_foot()


# =====================================================================
# PAGE 3 - THE MODEL (all numbers read from results/ or the saved model)
# =====================================================================
def page_model():
    hero("The model", "What it learned, and how well")
    final = get_models()[(10, True)]
    tr = final["test_result"]
    conf = results_csv("confidence_bands.csv").set_index("model_at_least_sure")
    cut = results_csv("models_by_cutoff.csv").set_index("cutoff_move")
    hist = results_csv("history_cv.csv").set_index(["features", "model"])
    sfcv = results_csv("stockfish_cv.csv").set_index(["features", "model"])
    prof = results_csv("clustering_profiles.csv")
    gap = results_csv("accuracy_by_rating_gap.csv").set_index("rating_gap")
    dec = results_csv("decisive_only.csv").iloc[0]
    sf_gain = sfcv.loc[("with Stockfish", "HistGradientBoosting"), "cv_acc_mean"] - \
        sfcv.loc[("without Stockfish", "HistGradientBoosting"), "cv_acc_mean"]
    groups = prof[prof["name"].notna()]
    grp_txt = ", ".join(f"<b>{esc(r['name'])}</b> {r['share']:.0%}" for _, r in groups.iterrows())
    vcheck = results_csv("stockfish_version_check.csv")
    vc = vcheck[vcheck["sf19_white_pct"].notna()] if vcheck is not None else None

    cards = [
        ("The question",
         "<p>Looking only at the <b>first 10 moves</b> of an online blitz game, can a computer tell who will win: "
         "White, Black, or a draw? The same question is also asked after 15, 20, 25 and 30 moves.</p>"),
        ("The data",
         "<p>Real games from the official <b>Lichess</b> database (via Hugging Face): blitz games played on "
         "<b>1 January 2025</b>.</p><ul><li><b>623,722</b> games in the training file</li>"
         f"<li>A separate, later file of the same day kept <b>untouched</b>; {tr['n_test']:,} of its games "
         "were used once, at the very end, as the test.</li></ul>"),
        ("The features",
         "<p>Ratings, time control, clocks, board features (castling, development, captures, material, mobility "
         "...) and the <b>Stockfish score</b> of the board at the cutoff. <b>Nothing after the cutoff</b> is used - "
         "checked on 1,913 real games cut after move 10.</p>"
         f"<p>Stockfish added about <b>{sf_gain * 100:.1f} points</b> of accuracy (train CV).</p>"),
        ("The model",
         "<p><b>Soft voting</b> of Logistic Regression + Histogram Gradient Boosting: each model gives chances, and "
         "the votes are averaged.</p><p>One model per cutoff (move 10, 15, 20, 25, 30), chosen and tuned with "
         "5-fold cross-validation on training data only.</p>"),
        ("The results",
         f'<div class="big">{pct(tr["test_acc"])} <small>vs {pct(tr["test_acc_always_white"])} always White</small></div>'
         f"<p>Move 10, test on {tr['n_test']:,} unseen games (macro-F1 {tr['test_macroF1']:.2f}).</p><ul>"
         f"<li>When at least 70% sure: <b>{pct(conf.loc['>= 70%', 'test_acc'])}</b> right "
         f"({conf.loc['>= 70%', 'share_of_test']:.0%} of games); at least 80% sure: "
         f"<b>{pct(conf.loc['>= 80%', 'test_acc'])}</b> ({conf.loc['>= 80%', 'share_of_test']:.0%}) - test</li>"
         f"<li>Rating gap 200+: <b>{pct(gap.loc['200+', 'test_acc_model'])}</b> (test)</li>"
         f"<li>Decisive games only: <b>{pct(dec['test_acc'])}</b> vs {pct(dec['test_acc_always_white'])} (test)</li>"
         f"<li>By move 30: <b>{pct(cut.loc[30, 'cv_acc_mean'])}</b> vs {pct(cut.loc[30, 'cv_always_white'])} (train CV)</li></ul>"),
        ("Player styles",
         f"<p><b>K-Means</b> with 3 groups on {int(groups['players'].sum()):,} players with at least 5 games, using only "
         f"how they play the first 10 moves (no ratings, no results): {grp_txt}.</p>"
         "<p>Hierarchical clustering and DBSCAN were also tried; the groups overlap (silhouette 0.17).</p>"),
        ("What did not help",
         "<p><b>Player history</b> - results from each player's earlier games that day (checked for leakage: "
         "0 mismatches in 20,000 recomputed numbers).</p>"
         f"<p>Train CV: {hist.loc[('base + Stockfish', 'HistGradientBoosting'), 'cv_acc_mean']:.2%} without it, "
         f"{hist.loc[('base + Stockfish + history', 'HistGradientBoosting'), 'cv_acc_mean']:.2%} with it - no gain, "
         "so it was dropped.</p>"),
        ("Limits",
         f"<ul><li>It <b>never predicts a draw</b> ({tr['predicted_draws']} draws predicted on the test set). Only "
         "about 4% of games are draws, and after 10 moves they look like normal games.</li>"
         "<li>Only <b>one day</b> of data.</li>"
         "<li>The live server has <b>Stockfish 17.1</b>; training used Stockfish 19"
         + (f" (on {len(vc)} demo games the most likely result changed in {int(vc['most_likely_changed'].sum())})"
            if vc is not None else "") + ".</li></ul>"),
        ("Tech stack",
         "<p>Python · python-chess · pandas · NumPy · scikit-learn · Stockfish · Streamlit · three.js · GitHub · "
         "Streamlit Community Cloud</p>"),
    ]
    html = "".join(f'<div class="card"><h3><span class="no">{i + 1:02d}</span>{esc(t)}</h3>{b}</div>'
                   for i, (t, b) in enumerate(cards))
    st.html(f'<div class="cards">{html}</div>')

    # charts
    st.html(section_title("", "In two charts"))
    rows_cut = [(int(m), float(r["cv_acc_mean"]), float(r["cv_always_white"])) for m, r in cut.iterrows()]
    order = ["all games", ">= 60%", ">= 70%", ">= 80%"]
    labels = {"all games": "All games", ">= 60%": "≥ 60% sure", ">= 70%": "≥ 70% sure", ">= 80%": "≥ 80% sure"}
    rows_conf = [(labels[k], float(conf.loc[k, "share_of_test"]), float(conf.loc[k, "test_acc"]),
                  float(conf.loc[k, "test_acc_always_white_same_games"])) for k in order]
    lg = legend([("Model", "background:#f5f5f5"), ("Always guessing White", "background:#8a8a8a")])
    lg_line = legend([("Model", "border-top:2px solid #f5f5f5;height:0;width:18px"),
                      ("Always guessing White", "border-top:2px dashed #8a8a8a;height:0;width:18px")])
    st.markdown(f'<div class="chart-grid">'
            f'<div class="chart-box"><h3>Accuracy by cutoff move</h3>{lg_line}{chart_cutoff(rows_cut)}'
            f'<p class="chart-cap">Train CV (5-fold). The more moves the model sees, the better it does. '
            f'Move 10 on the untouched test file: {pct(tr["test_acc"])}.</p></div>'
            f'<div class="chart-box"><h3>When the model is sure</h3>{lg}{chart_confidence(rows_conf)}'
            f'<p class="chart-cap">Move-10 model, test set. The more sure it is, the more often it is right - '
            f'but it is very sure on only a few games.</p></div></div>', unsafe_allow_html=True)
    site_foot()


# =====================================================================
# PAGE 4 - HOW IT'S DIFFERENT
# =====================================================================
def page_different():
    hero("How it's different", "Three other projects, side by side")
    projects = [
        ("Fan, Kuang & Lin", "Stanford CS229, 2013",
         "FIDE tournament games; about half of them draws.",
         "Player ratings only.", "3 (White / draw / Black)", "Hidden Markov Model",
         "55.64% (85.73% when draws are left out)"),
        ("Zeng", "CONF-SEML 2026",
         "3,196 endgame positions (UCI King-Rook vs King-Pawn).",
         "The pieces of an endgame whose result is already fixed by chess theory.", "2 (win / no win)",
         "Decision tree", "99.37%"),
        ("oindrillac / Chess-outcome-prediction", "GitHub",
         "About 20,000 Lichess games from Kaggle.", "Where the pieces stand on the board.", "Game result",
         "CNN / SVM", "No accuracy reported"),
        ("This project", "CSET-211, 2026",
         "623,722 real Lichess blitz games (1 January 2025) + an untouched test file.",
         "Only the first 10-30 moves: ratings, time control, clocks, board, Stockfish score at the cutoff.",
         "3 (White / draw / Black)", "Soft voting: Logistic Regression + Gradient Boosting",
         "58.1% at move 10 vs 49.6% always White (test); 71.9% by move 30 (train CV)"),
    ]
    keys = ["Data", "What it sees", "Classes", "Method", "Reported result"]
    head = "<tr><th></th>" + "".join(
        f'<th class="{"ours" if i == 3 else ""}">{esc(p[0])}<br><span style="letter-spacing:.1em">{esc(p[1])}</span></th>'
        for i, p in enumerate(projects)) + "</tr>"
    body = "".join("<tr>" + f'<td class="k">{k}</td>' + "".join(
        f'<td class="{"ours" if i == 3 else ""}">{esc(p[j + 2])}</td>' for i, p in enumerate(projects)) + "</tr>"
        for j, k in enumerate(keys))
    st.html(section_title("01", "Side by side", "As reported by each project. The tasks are different, so the "
                          "accuracy numbers cannot be compared directly."))
    st.html(f'<div class="table-wrap"><table class="cmp"><thead>{head}</thead><tbody>{body}</tbody></table></div>')

    why = [
        ("Real games, mid-way", "We predict after move 10 of real online games, long before the end. "
                                "The endgame study works on positions whose result chess theory already knows."),
        ("Near-equal opponents", "Online matchmaking pairs players of similar rating: two thirds of our test games "
                                 "have a rating gap under 50 points, so ratings alone tell little."),
        ("No leakage", "Only what is known at the cutoff is used: no later moves, no final clocks, no result. "
                       "This was checked on 1,913 real games cut after move 10."),
        ("Honest baseline", "Every accuracy is shown next to 'always guess White' (49.6% on the test set) and with "
                            "macro-F1, and the final model was tested once on a file it had never seen."),
        ("Works on any game", "Upload any Lichess or chess.com PGN to the live app and get the chances for each "
                              "game, plus a style group for players with 5 or more games."),
    ]
    st.html(section_title("02", "Why ours is different, and harder"))
    st.html('<div class="why">' + "".join(
        f'<div><span class="no">0{i + 1}</span><div><h4>{esc(t)}</h4><p>{esc(d)}</p></div></div>'
        for i, (t, d) in enumerate(why)) + "</div>")
    st.html(note_box("<b>To be fair:</b> about half of Fan, Kuang & Lin's games were draws, so draws matter much more "
                     "in their task; our model never predicts a draw. Zeng's task is a different kind of problem: the "
                     "result of each position is fixed by chess theory, so a very high accuracy is expected. The GitHub "
                     "project tries a different idea (learning from where the pieces stand) and reports no accuracy, "
                     "so its numbers cannot be compared."))
    site_foot()


{"home": page_home, "predict": page_predict, "model": page_model, "different": page_different}[page]()
