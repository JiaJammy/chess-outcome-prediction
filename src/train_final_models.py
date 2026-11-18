"""Step 4: final models (features = base + Stockfish; player history dropped because it did not help).

Run from the project folder:   python src/train_final_models.py
TRAIN = the 200,000 Stockfish-evaluated train games (data/sf_eval_train200k.csv.gz).
TEST  = 200,000 random games of the untouched part1 file with Stockfish (data/sf_eval_part1_200k.csv.gz),
        used ONCE at the end (3-class result, decisive-only result, rating-gap table, confidence table).

 1. 5-fold CV on the FULL train set: Logistic Regression, LDA, HGB, soft-voting ensemble of the three
 2. RandomizedSearchCV (5-fold, train only) for the winner; keep defaults if tuning does not beat them
 3. Refit on all train games, test ONCE on part1
 4. Decisive games only (draws removed): separate 2-class model of the same kind, CV on decisive train games,
    tested once on decisive part1 games, baseline = always White
 5. Accuracy by rating gap and by model confidence (test)
 6. One model per cutoff (moves 15, 20, 25, 30) trained on 100,000 train games each with the winning settings
    (5-fold CV reported; no test file for these cutoffs). Move 10 = the final model.
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform, randint, uniform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.features import add_model_columns, RATING_COLS_V2, TIME_COLS_V2, BOARD_COLS_V2, CLOCK_COLS_V2

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RS = 42
LABELS = {"0-1": 0, "1/2-1/2": 1, "1-0": 2}
CLASS_NAMES = ["black_win", "draw", "white_win"]
FEATURES = RATING_COLS_V2 + TIME_COLS_V2 + BOARD_COLS_V2 + CLOCK_COLS_V2 + ["sf_cp", "sf_expected"]
RES = os.path.join(ROOT, "results")
MOD = os.path.join(ROOT, "models")


def log(*a):
    print(*a, flush=True)


def candidates():
    imp = lambda: SimpleImputer(strategy="median")
    lr = make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=3000))
    lda = make_pipeline(imp(), StandardScaler(), LinearDiscriminantAnalysis())
    hgb = HistGradientBoostingClassifier(random_state=RS)
    vote = VotingClassifier([("lr", clone(lr)), ("lda", clone(lda)), ("hgb", clone(hgb))], voting="soft")
    return {"Logistic Regression": lr, "LDA": lda, "HistGradientBoosting": hgb, "Soft voting (LR+LDA+HGB)": vote}


SPACES = {
    "Logistic Regression": {"logisticregression__C": loguniform(1e-3, 100)},
    "LDA": {"lineardiscriminantanalysis__solver": ["lsqr"], "lineardiscriminantanalysis__shrinkage": uniform(0, 0.5)},
    "HistGradientBoosting": {"learning_rate": loguniform(0.02, 0.3), "max_iter": randint(100, 600),
                             "max_leaf_nodes": randint(15, 80), "min_samples_leaf": randint(20, 400),
                             "l2_regularization": loguniform(1e-3, 10), "max_features": uniform(0.4, 0.6)},
    "Soft voting (LR+LDA+HGB)": {"weights": [[1, 1, 1], [1, 1, 2], [1, 0, 1], [1, 0, 2], [2, 0, 1], [1, 1, 3], [0, 0, 1]],
                                 "hgb__learning_rate": loguniform(0.02, 0.3), "hgb__min_samples_leaf": randint(20, 400)},
}


def load_train():
    df = pd.read_csv(os.path.join(ROOT, "data/games_2025_01_blitz.csv.gz"))
    y_all = df["result"].map(LABELS).values
    itr, _ = train_test_split(np.arange(len(df)), test_size=0.2, random_state=RS, stratify=y_all)
    pick, _ = train_test_split(itr, train_size=200_000, random_state=RS, stratify=y_all[itr])
    df = df.iloc[pick].reset_index(drop=True)
    sf = pd.read_csv(os.path.join(ROOT, "data/sf_eval_train200k.csv.gz"))
    assert (sf["site"].values == df["site"].values).all()
    df[["sf_cp", "sf_expected"]] = sf[["sf_cp", "sf_expected"]].values
    return add_model_columns(df)


def load_test():
    te = pd.read_csv(os.path.join(ROOT, "data/games_2025_01_blitz_part1.csv.gz"))
    sf = pd.read_csv(os.path.join(ROOT, "data/sf_eval_part1_200k.csv.gz"))
    te = sf.merge(te, on="site", how="left")          # keeps the 200k sampled games, in sampled order
    assert te["result"].notna().all() and len(te) == 200_000
    return add_model_columns(te)


def cv_scores(model, X, y, cv):
    s = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "f1_macro"])
    return round(s["test_accuracy"].mean(), 4), round(s["test_accuracy"].std(), 4), round(s["test_f1_macro"].mean(), 4)


if __name__ == "__main__":
    t0 = time.time()
    tr = load_train()
    X, y = tr[FEATURES].astype(float), tr["result"].map(LABELS).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    log(f"train games: {len(y)}  always White (train): {(y == 2).mean():.4f}")

    # ---- 1. model choice on the full train set
    rows = []
    for name, m in candidates().items():
        t1 = time.time()
        acc, sd, f1 = cv_scores(m, X, y, cv)
        rows.append({"model": name, "cv_acc_mean": acc, "cv_acc_std": sd, "cv_macroF1_mean": f1,
                     "cv_always_white": round((y == 2).mean(), 4), "n_train": len(y), "secs": round(time.time() - t1)})
        log(rows[-1])
    cvt = pd.DataFrame(rows).sort_values("cv_acc_mean", ascending=False)
    cvt.to_csv(os.path.join(RES, "final_model_cv.csv"), index=False)
    winner = cvt.iloc[0]["model"]
    default_cv = float(cvt.iloc[0]["cv_acc_mean"])
    log("winner:", winner)

    # ---- 2. tuning (train only)
    search = RandomizedSearchCV(candidates()[winner], SPACES[winner], n_iter=20, cv=cv, scoring="accuracy",
                                random_state=RS, refit=False)
    search.fit(X, y)
    pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]] \
        .sort_values("rank_test_score").to_csv(os.path.join(RES, "final_model_tuning.csv"), index=False)
    use_tuned = round(search.best_score_, 4) > default_cv
    params = search.best_params_ if use_tuned else {}
    log(f"tuned CV {search.best_score_:.4f} vs default {default_cv:.4f} -> use tuned: {use_tuned} {params}")

    def make_final():
        m = candidates()[winner]
        if params:
            m.set_params(**params)
        return m

    # ---- 3. refit + test once
    final = make_final().fit(X, y)
    te = load_test()
    Xt, yt = te[FEATURES].astype(float), te["result"].map(LABELS).values
    proba = final.predict_proba(Xt)
    pred = proba.argmax(axis=1)
    test = {"model": winner, "tuned": bool(use_tuned), "params": json.dumps(params, default=float),
            "n_train": len(y), "n_test": len(yt), "test_acc": round(accuracy_score(yt, pred), 4),
            "test_macroF1": round(f1_score(yt, pred, average="macro"), 4),
            "test_acc_always_white": round((yt == 2).mean(), 4), "train_cv_acc": max(default_cv, round(search.best_score_, 4)),
            "predicted_draws": int((pred == 1).sum())}
    pd.DataFrame([test]).to_csv(os.path.join(RES, "final_model_test.csv"), index=False)
    pd.DataFrame(confusion_matrix(yt, pred), index=[f"true_{c}" for c in CLASS_NAMES],
                 columns=[f"pred_{c}" for c in CLASS_NAMES]).to_csv(os.path.join(RES, "final_model_confusion.csv"))
    log("TEST:", test)

    # ---- 5a. confidence bands (test)
    conf = proba.max(axis=1)
    bands = []
    for thr in [0.0, 0.6, 0.7, 0.8]:
        m = conf >= thr
        bands.append({"model_at_least_sure": "all games" if thr == 0 else f">= {int(thr * 100)}%",
                      "n_test_games": int(m.sum()), "share_of_test": round(m.mean(), 4),
                      "test_acc": round(accuracy_score(yt[m], pred[m]), 4),
                      "test_acc_always_white_same_games": round((yt[m] == 2).mean(), 4)})
    pd.DataFrame(bands).to_csv(os.path.join(RES, "confidence_bands.csv"), index=False)
    log(pd.DataFrame(bands).to_string(index=False))

    # ---- 5b. rating gap bands (test)
    gap = (te["white_elo"] - te["black_elo"]).abs().values
    gb = []
    for lo, hi, lab in [(0, 50, "0-50"), (50, 100, "50-100"), (100, 200, "100-200"), (200, 1e9, "200+")]:
        m = (gap >= lo) & (gap < hi)
        gb.append({"rating_gap": lab, "n_test_games": int(m.sum()), "share_of_test": round(m.mean(), 4),
                   "test_acc_always_white": round((yt[m] == 2).mean(), 4),
                   "test_acc_model": round(accuracy_score(yt[m], pred[m]), 4),
                   "test_macroF1_model": round(f1_score(yt[m], pred[m], average="macro"), 4)})
    pd.DataFrame(gb).to_csv(os.path.join(RES, "accuracy_by_rating_gap.csv"), index=False)
    log(pd.DataFrame(gb).to_string(index=False))

    # ---- 4. decisive games only (2 classes)
    dtr, dte = y != 1, yt != 1
    yb, ybt = (y[dtr] == 2).astype(int), (yt[dte] == 2).astype(int)      # 1 = White wins, 0 = Black wins
    bin_model = make_final()
    acc, sd, f1 = cv_scores(clone(bin_model), X[dtr], yb, cv)
    bin_model.fit(X[dtr], yb)
    pb = bin_model.predict(Xt[dte])
    dec = {"task": "decisive only (White win vs Black win)", "model": winner, "n_train": int(dtr.sum()),
           "train_cv_acc": acc, "train_cv_std": sd, "n_test": int(dte.sum()),
           "test_acc": round(accuracy_score(ybt, pb), 4), "test_macroF1": round(f1_score(ybt, pb, average="macro"), 4),
           "test_acc_always_white": round(ybt.mean(), 4)}
    pd.DataFrame([dec]).to_csv(os.path.join(RES, "decisive_only.csv"), index=False)
    log("DECISIVE:", dec)

    # ---- save move-10 model
    meta = {"features": FEATURES, "labels": CLASS_NAMES, "sklearn_version": sklearn.__version__,
            "stockfish_depth": 10, "trained_on": "Lichess 2025-01 blitz, 200k train games (HF file 0, 80% split rs=42)"}
    joblib.dump(dict(meta, model=final, early_plies=20, cutoff_move=10, test_result=test,
                     decisive_model=bin_model, decisive_result=dec), os.path.join(MOD, "model_move10.joblib"))

    # ---- 6. per-cutoff models
    pc = [{"cutoff_move": 10, "n_train": len(y), "cv_acc_mean": test["train_cv_acc"], "cv_always_white": round((y == 2).mean(), 4),
           "test_acc_part1": test["test_acc"], "test_acc_always_white_part1": test["test_acc_always_white"]}]
    for mv in [15, 20, 25, 30]:
        d = add_model_columns(pd.read_csv(os.path.join(ROOT, f"data/cutoff_{mv}_train100k.csv.gz")), early_plies=2 * mv)
        Xc, yc = d[FEATURES].astype(float), d["result"].map(LABELS).values
        acc, sd, f1 = cv_scores(make_final(), Xc, yc, cv)
        mc = make_final().fit(Xc, yc)
        joblib.dump(dict(meta, model=mc, early_plies=2 * mv, cutoff_move=mv,
                         cv_result={"cv_acc_mean": acc, "cv_acc_std": sd, "cv_macroF1_mean": f1}),
                    os.path.join(MOD, f"model_move{mv}.joblib"))
        pc.append({"cutoff_move": mv, "n_train": len(yc), "cv_acc_mean": acc, "cv_acc_std": sd, "cv_macroF1_mean": f1,
                   "cv_always_white": round((yc == 2).mean(), 4)})
        log(pc[-1])
    pd.DataFrame(pc).to_csv(os.path.join(RES, "models_by_cutoff.csv"), index=False)
    log(f"all done ({time.time() - t0:.0f} s)")
