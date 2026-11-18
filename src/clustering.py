"""Step 5: group players by playing style (clustering) on the 2025 blitz data.

Run from the project folder:   python src/clustering.py            (scores only, to choose k)
                               python src/clustering.py --k 3      (final groups, saves model + charts)
One row per player with >= 5 games (src/style.py). Only style features; NO ratings, NO results.
 1. K-Means for k = 2..8 on all players: elbow (inertia) + silhouette
 2. On a random 3,000 players: hierarchical clustering (Ward, dendrogram) and DBSCAN, compared with K-Means
 3. Profiles of the chosen groups + PCA plot; K-Means model + scaler saved for the app
Outputs: results/clustering_*.csv / *.png, models/style_kmeans.joblib
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.style import player_table, STYLE_COLS, STYLE_LABELS, MIN_GAMES

RS = 42
DATA = os.path.join(ROOT, "data", "games_2025_01_blitz.csv.gz")
RES = os.path.join(ROOT, "results")
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cd6", "#6b6b6b"]
N_SMALL = 3000                      # players used for hierarchical + DBSCAN (they are slow on many points)
SIL_SAMPLE = 20000                  # silhouette on a random 20,000 players (exact one needs n*n distances)

# names + meanings of the final groups (filled in after looking at the profiles; group number -> text)
# (k = 3 chosen: clearest elbow bend, silhouette 3 ~ 4, same k as the midterm; names from results/clustering_profiles.csv)
NAMES = {0: "Developer", 1: "Early castler", 2: "Tactical"}
MEANINGS = {
    0: "Brings knights and bishops out and keeps many moves available; often leaves the king in the centre "
       "for the first 10 moves. Few early captures.",
    1: "Puts the king safe early: castles in most games, and sooner than others. Quiet openings with the "
       "fewest captures and smallest material swings.",
    2: "Sharp, tactical starts: the most captures and the biggest material swings in the first 10 moves, "
       "fewer pieces developed, and rarely castles early.",
}


def load_players():
    # white_elo / black_elo / result are loaded ONLY to describe the groups at the end (not used to make them)
    cols = ["white", "black", "white_elo", "black_elo", "result", "utc_date", "utc_time", "eco", "material_swing_early"] + \
           [f"{s}_{c}" for s in ["white", "black"] for c in ["castle_ply", "captures_early", "minors_developed", "mobility"]]
    games = pd.read_csv(DATA, usecols=cols)
    return games, player_table(games, MIN_GAMES)


def kmeans_scores(X):
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=SIL_SAMPLE, random_state=RS)
        rows.append({"k": k, "inertia": round(km.inertia_, 1), "silhouette": round(sil, 4),
                     "smallest_group_share": round(np.bincount(km.labels_).min() / len(X), 4)})
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "clustering_kmeans_scores.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t["k"], t["inertia"], "o-", color=COLORS[0], lw=2)
    ax[0].set(title="Elbow method: inertia (lower = tighter groups)", xlabel="k (number of groups)", ylabel="inertia")
    ax[1].plot(t["k"], t["silhouette"], "o-", color=COLORS[1], lw=2)
    ax[1].set(title="Silhouette score (higher = better separated)", xlabel="k (number of groups)", ylabel="silhouette")
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle(f"K-Means on {len(X):,} players (2025 blitz, >= {MIN_GAMES} games each)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "clustering_elbow_silhouette.png"), dpi=150)
    plt.close(fig)
    return t


def compare_methods(X, km, k, variety):
    """Hierarchical (Ward) and DBSCAN on a random 3,000 players, compared with K-Means on the same players."""
    rng = np.random.RandomState(RS)
    idx = rng.choice(len(X), size=min(N_SMALL, len(X)), replace=False)
    Xs = X[idx]
    vs = np.asarray(variety)[idx]          # opening variety only takes 5 values (0.2 ... 1.0)

    def variety_purity(lab):
        """Share of (non-noise) players whose group has ONE opening-variety value. 1.0 = the groups are just
        the 5 opening-variety levels (a side effect of that feature, not a real style)."""
        lab = np.asarray(lab)
        m = lab != -1
        if not m.any():
            return np.nan
        t = pd.crosstab(lab[m], vs[m])
        return round(t.max(axis=1).sum() / m.sum(), 4)
    km_lab = km.predict(Xs)
    def largest(lab):          # share of the 3,000 players in the biggest group (noise not counted as a group)
        lab = np.asarray(lab)
        lab = lab[lab != -1]
        return round(np.bincount(lab - lab.min()).max() / len(Xs), 4) if len(lab) else 0.0

    rows = [{"method": f"K-Means (k={k}, fitted on all players)", "groups": k, "largest_group_share": largest(km_lab),
             "noise_share": 0.0,
             "silhouette": round(silhouette_score(Xs, km_lab), 4), "agreement_with_kmeans_ARI": 1.0,
             "same_opening_variety_in_group": variety_purity(km_lab)}]

    Z = linkage(Xs, method="ward")
    hc_lab = fcluster(Z, t=k, criterion="maxclust")
    rows.append({"method": f"Hierarchical, Ward (cut into {k} groups)", "groups": k,
                 "largest_group_share": largest(hc_lab), "noise_share": 0.0,
                 "silhouette": round(silhouette_score(Xs, hc_lab), 4),
                 "agreement_with_kmeans_ARI": round(adjusted_rand_score(km_lab, hc_lab), 4),
                 "same_opening_variety_in_group": variety_purity(hc_lab)})
    fig, ax = plt.subplots(figsize=(11, 4.5))
    dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, color_threshold=Z[-(k - 1), 2], above_threshold_color="#888888")
    ax.axhline(Z[-(k - 1), 2], color="#555555", ls="--", lw=1)
    ax.set(title=f"Hierarchical clustering (Ward) of {len(Xs):,} random players - dashed line cuts it into {k} groups",
           xlabel="groups of players (number in brackets = players in that branch)", ylabel="distance (Ward)")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "clustering_dendrogram.png"), dpi=150)
    plt.close(fig)

    min_samples = 2 * X.shape[1]          # common rule of thumb: 2 x number of features
    dist = np.sort(NearestNeighbors(n_neighbors=min_samples).fit(Xs).kneighbors(Xs)[0][:, -1])
    eps_list = [0.6, 0.8, 1.0, 1.2, 1.5]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dist, color=COLORS[0], lw=2)
    for e in eps_list:
        ax.axhline(e, color="#999999", ls=":", lw=1)
        ax.text(5, e, f"eps={e}", va="bottom", fontsize=8, color="#555555")
    ax.set(title=f"DBSCAN: distance to {min_samples}th nearest neighbour (sorted)",
           xlabel="players (sorted)", ylabel="distance")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "clustering_dbscan_kdistance.png"), dpi=150)
    plt.close(fig)
    for e in eps_list:
        lab = DBSCAN(eps=e, min_samples=min_samples).fit_predict(Xs)
        groups = len(set(lab)) - (1 if -1 in lab else 0)
        inl = lab != -1
        sil = silhouette_score(Xs[inl], lab[inl]) if groups >= 2 else np.nan
        rows.append({"method": f"DBSCAN (eps={e}, min_samples={min_samples})", "groups": groups,
                     "largest_group_share": largest(lab), "noise_share": round((~inl).mean(), 4),
                     "silhouette": round(sil, 4) if sil == sil else np.nan,
                     "agreement_with_kmeans_ARI": round(adjusted_rand_score(km_lab, lab), 4),
                     "same_opening_variety_in_group": variety_purity(lab)})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(RES, "clustering_method_comparison.csv"), index=False)
    print(t.to_string(index=False))
    return t


def profiles(players, games, k):
    """Average style per group (and, ONLY for describing the groups afterwards, rating and score)."""
    p = players.groupby("group")[STYLE_COLS + ["n_games"]].mean()
    p.insert(0, "players", players["group"].value_counts().sort_index())
    p.insert(1, "share", (p["players"] / p["players"].sum()).round(3))
    # describe-only columns (NOT used to make the groups): average rating and average score of the players
    side = pd.concat([pd.DataFrame({"player": games["white"], "elo": games["white_elo"],
                                    "score": games["result"].map({"1-0": 1, "1/2-1/2": .5, "0-1": 0})}),
                      pd.DataFrame({"player": games["black"], "elo": games["black_elo"],
                                    "score": games["result"].map({"1-0": 0, "1/2-1/2": .5, "0-1": 1})})])
    per = side.groupby("player")[["elo", "score"]].mean()
    j = players[["group"]].join(per)
    p["describe_only_avg_rating"] = j.groupby("group")["elo"].mean().round(0)
    p["describe_only_avg_score"] = j.groupby("group")["score"].mean().round(3)
    allrow = players[STYLE_COLS + ["n_games"]].mean()
    p.loc["all players"] = [len(players), 1.0] + allrow.tolist() + [round(per.loc[players.index, "elo"].mean()),
                                                                     round(per.loc[players.index, "score"].mean(), 3)]
    p = p.round(3)
    p.insert(0, "name", [NAMES.get(g, "") for g in p.index])
    p.to_csv(os.path.join(RES, "clustering_profiles.csv"), index_label="group")
    print(p.T.to_string())
    return p


def pca_plot(X, players, k):
    pca = PCA(n_components=2, random_state=RS).fit(X)
    rng = np.random.RandomState(RS)
    idx = rng.choice(len(X), size=min(6000, len(X)), replace=False)       # 6,000 dots so the plot stays readable
    pts = pca.transform(X[idx])
    lab = players["group"].values[idx]
    fig, ax = plt.subplots(figsize=(8, 6))
    for g in range(k):
        m = lab == g
        ax.scatter(pts[m, 0], pts[m, 1], s=9, alpha=.55, color=COLORS[g], edgecolors="none",
                   label=f"{g}: {NAMES.get(g, 'group ' + str(g))}")
    ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)",
           title=f"Player style groups (K-Means, k={k}) - PCA view of 6,000 random players")
    ax.legend(markerscale=2.5, frameon=False, loc="best")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "clustering_pca.png"), dpi=150)
    plt.close(fig)
    load = pd.DataFrame(pca.components_.T, index=STYLE_COLS, columns=["PC1", "PC2"]).round(3)
    print("PCA loadings:\n", load.to_string())
    return pca


def profile_chart(X, players, k):
    """Group centres in standard units (0 = average player): which features make each group different."""
    z = pd.DataFrame(X, columns=STYLE_COLS, index=players.index).groupby(players["group"]).mean()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / k
    x = np.arange(len(STYLE_COLS))
    for g in range(k):
        ax.bar(x + (g - (k - 1) / 2) * w, z.loc[g], width=w * .92, color=COLORS[g],
               label=f"{g}: {NAMES.get(g, 'group ' + str(g))}")
    ax.axhline(0, color="#333333", lw=1)
    ax.set_xticks(x, [STYLE_LABELS[c].split(" (")[0] for c in STYLE_COLS], rotation=20, ha="right")
    ax.set(ylabel="difference from average player\n(standard deviations)",
           title="What makes each style group different (0 = average player)")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "clustering_profiles.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None)
    k = ap.parse_args().k
    games, players = load_players()
    print(f"games {len(games):,}; players with >= {MIN_GAMES} games: {len(players):,}")
    players[STYLE_COLS + ["n_games"]].describe().round(3).to_csv(os.path.join(RES, "clustering_player_features.csv"))
    scaler = StandardScaler().fit(players[STYLE_COLS])
    X = scaler.transform(players[STYLE_COLS])
    print(pd.DataFrame(X, columns=STYLE_COLS).corr().round(2).to_string())
    if k is None:
        kmeans_scores(X)
        sys.exit(0)
    km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X)
    players["group"] = km.labels_
    compare_methods(X, km, k, players["opening_variety"].values)
    profiles(players, games, k)
    pca_plot(X, players, k)
    profile_chart(X, players, k)
    joblib.dump({"scaler": scaler, "kmeans": km, "features": STYLE_COLS, "k": k, "min_games": MIN_GAMES,
                 "names": NAMES, "meanings": MEANINGS, "n_players": len(players),
                 "data": "2025-01-01 Lichess blitz (games_2025_01_blitz.csv.gz)"},
                os.path.join(ROOT, "models", "style_kmeans.joblib"))
    print("saved models/style_kmeans.joblib")
