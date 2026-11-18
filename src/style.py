"""Player style: one row per player, made from the first 10 moves of each of their games.

Used by src/clustering.py (to build the groups), by app.py (to put a player from an uploaded file
into a group) and by the notebook. Ratings and results are NOT used.
"""
import numpy as np
import pandas as pd

EARLY_PLIES = 20          # first 10 moves per side
MIN_GAMES = 5             # a player needs at least this many games to get a style group
STYLE_COLS = ["avg_castle_ply", "castle_rate", "avg_captures", "avg_material_swing",
              "avg_minors_developed", "avg_mobility", "opening_variety"]
STYLE_LABELS = {          # plain words for the app and the charts
    "avg_castle_ply": "castling move (ply; 21 = not castled)",
    "castle_rate": "castled in first 10 moves (share of games)",
    "avg_captures": "captures in first 10 moves",
    "avg_material_swing": "material swing in first 10 moves",
    "avg_minors_developed": "knights/bishops developed",
    "avg_mobility": "legal moves at move 10 (mobility)",
    "opening_variety": "opening variety (different ECO codes in first 5 games / 5)",
}


def long_table(df):
    """One row per (game, player): the player's own numbers in that game.
    df = game table with the columns made by src/features.py (white_*, black_*, eco, ...)."""
    order = np.arange(len(df))
    if "utc_date" in df and "utc_time" in df:          # play order, used for opening variety
        order = (df["utc_date"].astype(str) + " " + df["utc_time"].astype(str)).values
    parts = []
    for side in ["white", "black"]:
        parts.append(pd.DataFrame({
            "player": df[side].values,
            "order": order,
            "castle_ply": df[f"{side}_castle_ply"].fillna(EARLY_PLIES + 1).values,   # never castled -> 21
            "castled": df[f"{side}_castle_ply"].notna().astype(int).values,
            "captures": df[f"{side}_captures_early"].values,
            "material_swing": df["material_swing_early"].values,
            "minors_developed": df[f"{side}_minors_developed"].values,
            "mobility": df[f"{side}_mobility"].values,
            "eco": df["eco"].fillna("?").astype(str).values,
        }))
    return pd.concat(parts, ignore_index=True)


def player_table(df, min_games=MIN_GAMES):
    """Average style per player, only players with at least min_games games."""
    long = long_table(df).sort_values(["player", "order"], kind="stable")
    players = long.groupby("player").agg(
        n_games=("eco", "size"),
        avg_castle_ply=("castle_ply", "mean"),
        castle_rate=("castled", "mean"),
        avg_captures=("captures", "mean"),
        avg_material_swing=("material_swing", "mean"),
        avg_minors_developed=("minors_developed", "mean"),
        avg_mobility=("mobility", "mean"),
        opening_variety=("eco", lambda s: s.head(5).nunique() / 5),     # first 5 games, same as midterm
    )
    return players[players["n_games"] >= min_games]


def style_for_games(games, bundle):
    """Used by the app. games = python-chess games from the uploaded file.
    Returns one row per player with >= MIN_GAMES usable games and their style group.
    A game is usable if it lasts MORE than 10 moves per side (same rule as the training data)."""
    from src.features import extract_game_features_v2          # imported here so style.py stays light
    rows = []
    for g in games:
        if sum(1 for _ in g.mainline_moves()) <= EARLY_PLIES:
            continue
        rows.append(extract_game_features_v2(g, early_plies=EARLY_PLIES))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    eco_known = df["eco"].notna() & (df["eco"].astype(str).str.strip() != "")
    players = player_table(df, bundle["min_games"])
    players = players[~players.index.isin(["?", "", "None"])]        # games without player names
    if len(players) == 0:
        return pd.DataFrame()
    # opening variety needs the ECO code; if a player's games have none, use the average player's value
    long = long_table(df.assign(eco=df["eco"].where(eco_known)))
    no_eco = long.groupby("player")["eco"].apply(lambda s: (s == "?").any())
    players["opening_variety_known"] = ~no_eco.reindex(players.index).fillna(True).values
    mean_variety = bundle["scaler"].mean_[bundle["features"].index("opening_variety")]
    players.loc[~players["opening_variety_known"], "opening_variety"] = mean_variety
    return assign_groups(players, bundle).sort_values("n_games", ascending=False)


def assign_groups(players, bundle):
    """Put players (output of player_table) into the saved K-Means groups.
    Returns a copy with columns group (number), group_name, group_meaning."""
    out = players.copy()
    if len(out) == 0:
        return out.assign(group=[], group_name=[], group_meaning=[])
    X = bundle["scaler"].transform(out[bundle["features"]].astype(float))
    out["group"] = bundle["kmeans"].predict(X)
    out["group_name"] = out["group"].map(bundle["names"])
    out["group_meaning"] = out["group"].map(bundle["meanings"])
    return out
