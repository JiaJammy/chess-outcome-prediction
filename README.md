# Chess Game Outcome Prediction and Player Style Clustering (Lichess data)

CSET-211 Statistical Machine Learning project, Bennett University. Author: Jia Jain.

**Live app:** _link added after deployment_

## What it does
1. **Predicts the result** of a Lichess blitz game (White win / draw / Black win) using only the first
   10 moves. The model sees the ratings, the time control, the clocks, the board after move 10, and a
   Stockfish evaluation of that board. Nothing after move 10 is used.
   There are also models for 15, 20, 25 and 30 moves.
2. **Groups players by playing style** (K-Means clustering). It uses how they play their first 10 moves:
   castling, captures, material swings, development, mobility and opening variety. Ratings and results are not used.
3. **Web app** (Streamlit). Upload a PGN file (one or many games) or paste moves. For each game, the app uses
   the latest cutoff the game has reached and shows the chance of each result. For every player with 5 or more
   games in the file, it also shows their style group.

## Main results
Data: Lichess blitz games from 1 January 2025
([Hugging Face `Lichess/standard-chess-games`](https://huggingface.co/datasets/Lichess/standard-chess-games),
the official Lichess database). Models were chosen with 5-fold cross-validation on training data only.
The final model was then tested **once** on 200,000 games from a separate, untouched file.

| Result | Accuracy | Always guessing White | Macro-F1 |
|---|---|---|---|
| Final model, move 10 (test) | **58.1%** | 49.6% | 0.39 |
| Decisive games only, White vs Black (test) | 60.8% | 52.0% | 0.60 |
| Model at least 80% sure (test, 4% of games) | 88.0% | 58.1% | - |
| Rating gap 200+ (test) | 76.7% | 49.6% | 0.52 |
| Moves 15 / 20 / 25 / 30 (train CV) | 62.5% / 66.4% / 70.0% / 71.9% | 49.5% / 49.1% / 48.5% / 47.8% | 0.43 / 0.45 / 0.48 / 0.50 |
| Midterm games (2013), same split: before -> after (test) | 62.4% -> 66.2% | 51.0% | 0.43 -> 0.45 |

- The final model is soft voting (Logistic Regression + HistGradientBoosting) with Stockfish (depth 10).
  Stockfish adds about 1.8 points of accuracy. Features from a player's earlier games gave no gain.
- **Limitation:** the model never predicts a draw. Only about 4% of games are draws, and after 10 moves
  they look like normal games.
- Clustering of 83,870 players with K-Means (k = 3) gives three groups: **Developer**, **Early castler** and
  **Tactical**. The silhouette is low (0.17), so the styles overlap. Hierarchical clustering (Ward) and DBSCAN
  were also compared; DBSCAN found no real groups.

All tables are in `results/` (CSV + PNG). The notebook `notebook/chess_midterm_project_v2.ipynb` explains every step.

## Run the app on your computer
1. Install Python 3.13, then in the project folder run:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
2. Open http://localhost:8501 in a browser. Try `data/demo_games.pgn` or a file from `data/sample_pgn/`.
3. Stockfish (optional, but more accurate): install it ([stockfishchess.org](https://stockfishchess.org/download/)) so the
   command `stockfish` works. You can also put the Windows exe at
   `tools/stockfish/stockfish/stockfish-windows-x86-64-universal.exe`.
   Without Stockfish, the app uses models trained without it (about 2-3 points less accurate) and says so.

## Run the notebook
`pip install notebook`, open `notebook/chess_midterm_project_v2.ipynb` and click Run All (about 1-2 minutes).
Part 1 is the midterm; Part 2 (sections 9-20) covers the final project. Slow steps were run as scripts in `src/`;
the notebook loads their saved results. Without the full 2025 data, the notebook uses the 50,000-game sample in
`data/sample/`, so some numbers will be a little different.

## Rebuild everything from the raw data
The big data files are not on GitHub. To rebuild them (run from the project folder, in this order):

| Step | Command | Makes |
|---|---|---|
| 1 | `python src/download_data.py` | `data/hf/*.parquet` (about 2 GB raw Lichess games) |
| 2 | `python src/build_dataset.py` | `data/games_2025_01_blitz.csv.gz` (training file) |
| 3 | `python src/build_dataset.py data/hf/lichess_2025_01_part1.parquet data/games_2025_01_blitz_part1.csv.gz` | untouched test file |
| 4 | `python src/build_dataset.py data/hf/lichess_2013_01.parquet data/games_2013_01_all.csv.gz --no-filter` | 2013 midterm month |
| 5 | `python src/cutoff_experiment.py` | accuracy by cutoff |
| 6 | `python src/stockfish_experiment.py` | Stockfish on 200,000 training games (about 40 min with 8 engines) |
| 7 | `python src/stockfish_runs.py` | Stockfish on the test games and per-cutoff training games (about 1.5 h) |
| 8 | `python src/player_history.py` then `python src/history_experiment.py` | player-history test (no gain) |
| 9 | `python src/train_final_models.py` | final models `models/model_move{10..30}.joblib` + test results |
| 10 | `python src/train_fallback_models.py` | models without Stockfish (`*_nosf.joblib`) |
| 11 | `python src/clustering.py --k 3` | style groups `models/style_kmeans.joblib` + charts |

Stockfish 19 was used for all evaluations.

## Folders
- `src/`: all code (features, data building, experiments, training, clustering, prediction)
- `app.py`: the Streamlit app
- `models/`: saved models (joblib)
- `results/`: result tables and charts
- `notebook/`: the notebook
- `data/`: small data only (2013 midterm data, a 50,000-game 2025 sample, demo and sample PGN files)
