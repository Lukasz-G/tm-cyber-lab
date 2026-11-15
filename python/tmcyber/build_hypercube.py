"""Convert the Hypercube Drebin features to packed `.tmx` matrices.

    python -I python/tmcyber/build_hypercube.py data/hypercube/raw/hypercube/hypercube-v1.1 data/hypercube/tmx

Source: the Hypercube dataset of Chow et al., *Breaking Out from the Tesseract* (arXiv:2506.23814),
release v1.1 (Drebin features re-extracted in July 2025). Google Play applications, 2021-2023,
sampled monthly at 10% malware. Only `hypercube_drebin.json` (a JSON object mapping sha256 to a
sparse token dictionary) and `hypercube_metadata.json` are read. The release also contains two
pickle files with other feature spaces; they are not used, and nothing here unpickles anything.

Decisions, each fixed before looking at any model:

* **Label.** Malware if `vt_detection >= 4`, benign if `vt_detection == 0`, as in the authors'
  loader and in LAMDA. The release contains no sample with 1 to 3 detections.
* **Time.** `gp_date`, the Google Play date. `dex_date` is a file timestamp set at build time
  (often 1980 or 1981) and is not used.
* **Splits.** The pool is every sample dated 2021. Each month from 2022-01 to 2023-12 is one test
  file. This mirrors the APIGraph and AndroZoo layout (train on the pool, test on every later month).
* **Feature selection.** The rule that produced LAMDA's 4,561-feature baseline: scikit-learn's
  `VarianceThreshold(0.001)` over the binary token matrix, which keeps a token when
  p(1 - p) > 0.001, with p its share of rows. It is fitted on the pool only, so no information from
  2022-2023 enters the vocabulary. Columns are in sorted token order.
* **Values.** A token is present (1) when its value is nonzero.

The JSON is 1.6 GB, so it is streamed twice with ijson: once to count tokens over the pool, once to
build the rows. Samples present in the metadata but missing from the features are dropped and
counted.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ijson
import numpy as np
import pyarrow as pa

from tmcyber import tmx

POOL_YEAR = 2021
SERIES = [(y, m) for y in (2022, 2023) for m in range(1, 13)]
VT_MALWARE = 4
VARIANCE_THRESHOLD = 0.001


def load_metadata(path: Path) -> dict:
    """sha256 -> (label, year, month, vt, gp_date), for labelled samples only."""
    rows = json.load(open(path, encoding="utf-8"))
    meta, dropped_grey = {}, 0
    for e in rows:
        vt = float(e["vt_detection"])
        if vt == 0:
            label = False
        elif vt >= VT_MALWARE:
            label = True
        else:
            dropped_grey += 1
            continue
        d = date.fromisoformat(str(e["gp_date"])[:10])
        meta[e["sha256"]] = (label, d.year, d.month, int(vt), d.isoformat())
    return meta, len(rows), dropped_grey


def group_of(year: int, month: int):
    if year == POOL_YEAR:
        return "train"
    if (year, month) in SERIES:
        return f"{year}-{month:02d}"
    return None


def stream(path: Path):
    with open(path, "rb") as f:
        yield from ijson.kvitems(f, "", use_float=True)


def main(src_dir: str, out_dir: str) -> int:
    t0 = time.time()
    src, out = Path(src_dir), Path(out_dir)
    feats_path = src / "hypercube_drebin.json"
    meta, n_meta, n_grey = load_metadata(src / "hypercube_metadata.json")
    print(f"metadata: {n_meta:,} rows, {len(meta):,} labelled, {n_grey:,} with 1-3 detections dropped")

    # Pass 1: token document frequency over the pool.
    counts: Counter = Counter()
    n_pool, n_seen, n_unknown = 0, 0, 0
    for sha, toks in stream(feats_path):
        info = meta.get(sha)
        if info is None:
            n_unknown += 1
            continue
        n_seen += 1
        if group_of(info[1], info[2]) == "train":
            n_pool += 1
            counts.update(t for t, v in toks.items() if v)
    var = {t: (c / n_pool) * (1 - c / n_pool) for t, c in counts.items()}
    names = sorted(t for t, v in var.items() if v > VARIANCE_THRESHOLD)
    col = {t: i for i, t in enumerate(names)}
    print(f"pass 1: {n_seen:,} samples with features ({n_unknown:,} feature entries without metadata), "
          f"pool {n_pool:,} rows, {len(counts):,} distinct pool tokens -> {len(names):,} kept "
          f"[{time.time() - t0:.0f} s]")
    del counts, var

    # Pass 2: sparse rows over the kept tokens, grouped by split.
    groups = defaultdict(list)
    for sha, toks in stream(feats_path):
        info = meta.get(sha)
        if info is None:
            continue
        g = group_of(info[1], info[2])
        if g is None:
            continue
        idx = np.fromiter(sorted(col[t] for t, v in toks.items() if v and t in col), dtype=np.int32)
        groups[g].append((info[4], sha, info, idx))
    missing = len(meta) - sum(len(v) for v in groups.values())
    print(f"pass 2: rows grouped; {missing:,} labelled samples have no features [{time.time() - t0:.0f} s]")

    out.mkdir(parents=True, exist_ok=True)
    tmx.write_features(out / "features.arrow", names)
    width = len(names)
    total_set, total_rows = 0, 0
    for g in ["train"] + [f"{y}-{m:02d}" for y, m in SERIES]:
        rows = sorted(groups.get(g, []), key=lambda r: (r[0], r[1]))
        n = len(rows)
        X = np.zeros((n, width), dtype=bool)
        for i, (_, _, _, idx) in enumerate(rows):
            X[i, idx] = True
        label = np.array([r[2][0] for r in rows], dtype=bool)
        table = pa.table({
            "label": pa.array(label, type=pa.bool_()),
            "year": pa.array([r[2][1] for r in rows], type=pa.int16()),
            "month": pa.array([0 if g == "train" else r[2][2] for r in rows], type=pa.int8()),
            "split": pa.array(["train" if g == "train" else "test"] * n, type=pa.string()),
            "family": pa.array(["unknown" if r[2][0] else "benign" for r in rows], type=pa.string()),
            "family_index": pa.array([-1 if r[2][0] else 0 for r in rows], type=pa.int32()),
            "vt_detection": pa.array([r[2][3] for r in rows], type=pa.int16()),
            "hash": pa.array([r[1] for r in rows], type=pa.string()),
            "date": pa.array([r[0] for r in rows], type=pa.string()),
        })
        tmx.write_tmx(out / f"{g}.tmx", X)
        tmx.write_meta(out / f"{g}.meta.arrow", table)
        s = int(X.sum())
        total_set += s
        total_rows += n
        print(f"  {g:>7}: {n:>7,} rows, {int(label.sum()):>6,} malware ({100 * label.mean():.1f}%), "
              f"density {100 * s / max(1, n * width):.2f}%")
        del X
    print(f"width {width:,}; mean density {100 * total_set / (total_rows * width):.2f}%; "
          f"{total_rows:,} rows; {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    for stream_ in (sys.stdout, sys.stderr):
        try:
            stream_.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
