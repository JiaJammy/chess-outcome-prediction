"""Choose, tune, test and save the FINAL move-10 model (main dataset: 2025 blitz).

Run from the project folder:   python src/train_final_model.py

Steps (the test set is used ONE time, at step 5):
 1. TRAIN data = the 80% train part of data/games_2025_01_blitz.csv.gz (same split as before, random_state 42).
    TEST data  = data/games_2025_01_blitz_part1.csv.gz, a different Lichess file that was never used
                 for any choice (the old 20% split was looked at during exploration, so it is not used here).
 2. Model choice: 5-fold cross-validation (accuracy) on a random 100,000 rows of TRAIN, same rows for every model.
 3. Tuning: RandomizedSearchCV (5-fold, accuracy) for the winner, on the same 100,000 TRAIN rows.
 4. Refit the tuned model on ALL train rows.
 5. Score the TEST set once: accuracy, macro-F1, always-White baseline, confusion matrix.
 6. Confidence table: accuracy on test games where the model is at least 60/70/80% sure.
 7. Save model + feature list + settings to models/final_model_move10.joblib.
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
from src.features import add_model_columns, RATING_COLS_V2, TIME_COLS_V2, BOARD_COLS_V2, CLOCK_COLS_V2, EARLY_PLIES

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

RS = 42
MAIN_FILE = os.path.join(ROOT, "data/games_2025_01_blitz.csv.gz")
TEST_FILE = os.path.join(ROOT, "data/games_2025_01_blitz_part1.csv.gz")
OUT_DIR = os.path.join(ROOT, "results")
MODEL_FILE = os.path.join(ROOT, "models/final_model_move10.joblib")
N_CV = 100_000
FEATURES = RATING_COLS_V2 + TIME_COLS_V2 + BOARD_COLS_V2 + CLOCK_COLS_V2
LABELS = {"0-1": 0, "1/2-1/2": 1, "1-0": 2}
CLASS_NAMES = ["black_win", "draw", "white_win"]


def prep(path):
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    df = add_model_columns(df, early_plies=EARLY_PLIES)
    return df[FEATURES].astype(float), df["result"].map(LABELS).values


def candidates():
    """Every model gets the same input. Models that cannot handle missing clocks get a median imputer
    (fitted inside each CV fold, so no leakage). Scaling for distance/linear models."""
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Logistic Regression": make_pipeline(imp(), StandardScaler(), LogisticRegression(max_iter=3000)),
        "LDA": make_pipeline(imp(), StandardScaler(), LinearDiscriminantAnalysis()),
        "Gaussian NB": make_pipeline(imp(), StandardScaler(), GaussianNB()),
        "Decision Tree": make_pipeline(imp(), DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, random_state=RS)),
        "KNN (k=50)": make_pipeline(imp(), StandardScaler(), KNeighborsClassifier(n_neighbors=50, n_jobs=-1)),
        "Random Forest": make_pipeline(imp(), RandomForestClassifier(300, min_samples_leaf=10, n_jobs=-1, random_state=RS)),
        "Extra Trees": make_pipeline(imp(), ExtraTreesClassifier(300, min_samples_leaf=10, n_jobs=-1, random_state=RS)),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RS),
    }


SEARCH_SPACES = {   # parameter ranges tried by RandomizedSearchCV (step name prefix for pipelines)
    "HistGradientBoosting": {"learning_rate": loguniform(0.02, 0.3), "max_iter": randint(100, 600),
                             "max_leaf_nodes": randint(15, 80), "min_samples_leaf": randint(20, 400),
                             "l2_regularization": loguniform(1e-3, 10), "max_features": uniform(0.4, 0.6)},
    "Logistic Regression": {"logisticregression__C": loguniform(1e-3, 100)},
    "LDA": {"lineardiscriminantanalysis__solver": ["lsqr"], "lineardiscriminantanalysis__shrinkage": uniform(0, 0.5)},
    "Random Forest": {"randomforestclassifier__min_samples_leaf": randint(2, 50),
                      "randomforestclassifier__max_features": uniform(0.1, 0.5)},
    "Extra Trees": {"extratreesclassifier__min_samples_leaf": randint(2, 50),
                    "extratreesclassifier__max_features": uniform(0.1, 0.6)},
    "Gaussian NB": {"gaussiannb__var_smoothing": loguniform(1e-10, 1e-2)},
    "Decision Tree": {"decisiontreeclassifier__max_depth": randint(3, 20),
                      "decisiontreeclassifier__min_samples_leaf": randint(10, 500)},
    "KNN (k=50)": {"kneighborsclassifier__n_neighbors": randint(10, 200)},
}


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

    # ---- 1. data
    X, y = prep(MAIN_FILE)
    itr, _old_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=RS, stratify=y)
    X_tr, y_tr = X.iloc[itr].reset_index(drop=True), y[itr]
    X_te, y_te = prep(TEST_FILE)
    print(f"train rows {len(y_tr)}, test rows (new file) {len(y_te)}", flush=True)

    # same 100k train rows for model choice and tuning
    cv_idx, _ = train_test_split(np.arange(len(y_tr)), train_size=N_CV, random_state=RS, stratify=y_tr)
    X_cv, y_cv = X_tr.iloc[cv_idx], y_tr[cv_idx]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

    # ---- 2. model choice by 5-fold CV on train
    cv_rows = []
    for name, model in candidates().items():
        t1 = time.time()
        s = cross_val_score(model, X_cv, y_cv, cv=cv, scoring="accuracy", n_jobs=1)
        cv_rows.append({"model": name, "cv_acc_mean": round(s.mean(), 4), "cv_acc_std": round(s.std(), 4),
                        "secs": round(time.time() - t1)})
        print(cv_rows[-1], flush=True)
    cv_table = pd.DataFrame(cv_rows).sort_values("cv_acc_mean", ascending=False)
    cv_table["cv_always_white"] = round((y_cv == 2).mean(), 4)
    cv_table.to_csv(os.path.join(OUT_DIR, "final_model_cv.csv"), index=False)
    winner = cv_table.iloc[0]["model"]
    print("winner by train CV:", winner, flush=True)

    # ---- 3. tuning on train only
    base = candidates()[winner]
    search = RandomizedSearchCV(base, SEARCH_SPACES[winner], n_iter=25,
                                cv=cv, scoring="accuracy", random_state=RS, n_jobs=1, refit=False)
    search.fit(X_cv, y_cv)
    best_params = search.best_params_
    tune = pd.DataFrame(search.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    tune.sort_values("rank_test_score").to_csv(os.path.join(OUT_DIR, "final_model_tuning.csv"), index=False)
    default_cv = float(cv_table.iloc[0]["cv_acc_mean"])
    print(f"tuned CV acc {search.best_score_:.4f} (default {default_cv:.4f}) params {best_params}", flush=True)
    # keep the default settings if tuning did not beat them on train CV
    use_tuned = search.best_score_ > default_cv
    final = candidates()[winner]
    if use_tuned:
        final.set_params(**best_params)

    # ---- 4. refit on all train rows
    final.fit(X_tr, y_tr)

    # ---- 5. test ONCE
    proba = final.predict_proba(X_te)
    pred = proba.argmax(axis=1)
    test = {"model": winner, "tuned": bool(use_tuned), "params": json.dumps(best_params if use_tuned else {}, default=float),
            "n_train": len(y_tr), "n_test": len(y_te),
            "test_acc": round(accuracy_score(y_te, pred), 4),
            "test_macroF1": round(f1_score(y_te, pred, average="macro"), 4),
            "test_acc_always_white": round((y_te == 2).mean(), 4),
            "train_cv_acc_default": default_cv, "train_cv_acc_tuned": round(search.best_score_, 4)}
    pd.DataFrame([test]).to_csv(os.path.join(OUT_DIR, "final_model_test.csv"), index=False)
    cm = pd.DataFrame(confusion_matrix(y_te, pred), index=[f"true_{c}" for c in CLASS_NAMES],
                      columns=[f"pred_{c}" for c in CLASS_NAMES])
    cm.to_csv(os.path.join(OUT_DIR, "final_model_confusion.csv"))
    print(test, "\n", cm, "\n", classification_report(y_te, pred, target_names=CLASS_NAMES, zero_division=0), flush=True)

    # ---- 6. confidence table
    conf = proba.max(axis=1)
    bands = []
    for thr in [0.0, 0.6, 0.7, 0.8]:
        m = conf >= thr
        bands.append({"model_at_least_sure": "all games" if thr == 0 else f">= {int(thr * 100)}%",
                      "n_test_games": int(m.sum()), "share_of_test": round(m.mean(), 4),
                      "test_acc": round(accuracy_score(y_te[m], pred[m]), 4) if m.any() else np.nan,
                      "test_acc_always_white_same_games": round((y_te[m] == 2).mean(), 4) if m.any() else np.nan})
    bands = pd.DataFrame(bands)
    bands.to_csv(os.path.join(OUT_DIR, "confidence_bands.csv"), index=False)
    print(bands, flush=True)

    # ---- 7. save
    joblib.dump({"model": final, "features": FEATURES, "early_plies": EARLY_PLIES, "labels": CLASS_NAMES,
                 "trained_on": "Lichess 2025-01 blitz (HF file 0), 80% train split rs=42",
                 "test_result": test, "sklearn_version": sklearn.__version__}, MODEL_FILE)
    print(f"saved {MODEL_FILE} ({time.time() - t0:.0f} s)")
