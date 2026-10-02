"""Compute expected digests from the ORIGINAL Parquet, independently of the .tmx writer.

The point is that the Julia side is checked against numbers derived from the source data and not from the same code path that produced the file. Reading the .tmx back in Python would only prove the
writer agrees with itself.

    python research/b0-boundary/digest.py > research/b0-boundary/expected.txt

One line per year: year, rows, total popcount, malware count.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402

from tmcyber import lamda  # noqa: E402


def main(root):
    years = lamda.discover(root)
    cols = lamda.feature_names(years[0].train)
    for yf in years:
        rows = 0
        popcount = 0
        malware = 0
        for path in (yf.train, yf.test):          # same order build_tmx writes: train then test
            X, meta = lamda.read_features(path, cols)
            rows += X.shape[0]
            popcount += int(X.sum())
            malware += int(meta["label"].astype(int).sum())
        print(f"{yf.year} {rows} {popcount} {malware}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "data/lamda/raw"))
