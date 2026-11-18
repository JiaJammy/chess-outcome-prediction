"""Download the raw Lichess games used in this project (official Lichess database, as parquet on Hugging Face).

Run from the project folder:   python src/download_data.py
Saves to data/hf/ (about 2 GB). Then build the game tables with src/build_dataset.py (see README).
"""
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://huggingface.co/datasets/Lichess/standard-chess-games/resolve/main/data/"
FILES = {
    "lichess_2025_01_part0.parquet": "year=2025/month=01/train-00000-of-00072.parquet",   # main data (train)
    "lichess_2025_01_part1.parquet": "year=2025/month=01/train-00001-of-00072.parquet",   # untouched test file
    "lichess_2013_01.parquet": "year=2013/month=01/train-00000-of-00001.parquet",         # midterm month
}

if __name__ == "__main__":
    out_dir = os.path.join(ROOT, "data", "hf")
    os.makedirs(out_dir, exist_ok=True)
    for name, path in FILES.items():
        out = os.path.join(out_dir, name)
        if os.path.exists(out):
            print("already there:", name)
            continue
        print("downloading", name, "...", flush=True)
        urllib.request.urlretrieve(BASE + path, out + ".part")
        os.replace(out + ".part", out)
        print("  saved", out, f"({os.path.getsize(out) / 1e6:.0f} MB)")
