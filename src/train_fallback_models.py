"""Fallback models WITHOUT Stockfish (used by the app only if Stockfish cannot be found).

Run from the project folder:   python src/train_fallback_models.py
Same model type and settings as the final models (read from models/model_move10.joblib), same training games,
only the two Stockfish columns are left out. 5-fold CV on train is reported; the test file is NOT used.
Output: models/model_move{10,15,20,25,30}_nosf.joblib, results/fallback_models_cv.csv
"""
import json
import os
import sys

import joblib
import pandas as pd
import sklearn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.features import add_model_columns
from src.train_final_models import candidates, load_train, cv_scores, LABELS, CLASS_NAMES, FEATURES, RS
from sklearn.model_selection import StratifiedKFold

NOSF = [c for c in FEATURES if c not in ("sf_cp", "sf_expected")]

if __name__ == "__main__":
    final = joblib.load(os.path.join(ROOT, "models/model_move10.joblib"))
    winner = final["test_result"]["model"]
    params = json.loads(final["test_result"]["params"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)

    def make():
        m = candidates()[winner]
        if params:
            m.set_params(**params)
        return m

    rows = []
    for mv in [10, 15, 20, 25, 30]:
        if mv == 10:
            d = load_train()
        else:
            d = add_model_columns(pd.read_csv(os.path.join(ROOT, f"data/cutoff_{mv}_train100k.csv.gz")), early_plies=2 * mv)
        X, y = d[NOSF].astype(float), d["result"].map(LABELS).values
        acc, sd, f1 = cv_scores(make(), X, y, cv)
        m = make().fit(X, y)
        joblib.dump({"model": m, "features": NOSF, "labels": CLASS_NAMES, "early_plies": 2 * mv, "cutoff_move": mv,
                     "uses_stockfish": False, "sklearn_version": sklearn.__version__,
                     "cv_result": {"cv_acc_mean": acc, "cv_acc_std": sd, "cv_macroF1_mean": f1}},
                    os.path.join(ROOT, f"models/model_move{mv}_nosf.joblib"))
        rows.append({"cutoff_move": mv, "n_train": len(y), "cv_acc_mean": acc, "cv_acc_std": sd, "cv_macroF1_mean": f1,
                     "cv_always_white": round((y == 2).mean(), 4)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results/fallback_models_cv.csv"), index=False)
