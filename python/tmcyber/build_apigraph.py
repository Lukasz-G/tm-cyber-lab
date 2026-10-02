"""Convert the APIGraph / AndroZoo Drebin features to packed `.tmx` matrices.

    python -m tmcyber.build_apigraph data/apigraph/data/gen_apigraph_drebin data/apigraph/tmx/apigraph
    python -m tmcyber.build_apigraph data/apigraph/data/gen_androzoo_drebin data/apigraph/tmx/androzoo

Source: the Drebin feature extraction released with Chen et al., *Continuous Learning for Android
Malware Detection* (USENIX Security 2023). Note what this is and is not: the APIGraph authors release
only MD5 hashes ("for security and copyright reasons"), so these features are a **third party's
extraction** of the APIGraph sample set, not the APIGraph authors' own. Any cross-dataset claim has to
say so.

Two things about the format that are easy to get wrong, and one of them silently produces nonsense:

* **`y_train` holds multi-class family labels, not a binary flag.** Zero is benign; every positive
  value is a family index, and `y_mal_family` names the malware samples in order. Testing `y == 1`
  looks reasonable and selects one arbitrary family — on the 2012 pool that is 179 samples out of
  3,061 actual malware. Binary detection is `y > 0`.
* The per-month files and the training pool are distinguished only by the filename: a pool spans
  months (`2012-01to2012-12_selected.npz`), a month does not (`2013-01_selected.npz`).

Layout written: one `.tmx` + `.meta.arrow` per month, plus `train.tmx`/`train.meta.arrow` for the
pool and a shared `features.arrow`. Split names follow the source protocol — train on the pool, test
on every subsequent month — which is NOT LAMDA's AnoShift arrangement and should not be forced into
it.
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

from tmcyber import tmx

MONTH_RE = re.compile(r"^(\d{4})-(\d{2})_selected\.npz$")
POOL_RE = re.compile(r"^(\d{4})-(\d{2})to(\d{4})-(\d{2})_selected\.npz$")


def feature_names(src: Path) -> list:
    """Feature names in column order, from the pool's `_selected_training_features.json`.

    The JSON is a name -> column-index mapping, so it must be inverted, not iterated.
    """
    cands = sorted(glob.glob(str(src / "*_selected_training_features.json")))
    full = [c for c in cands if "to" in Path(c).name and "full" not in Path(c).name]
    pick = full[-1] if full else cands[-1]
    mapping = json.load(open(pick))
    if not isinstance(mapping, dict):
        raise RuntimeError(f"{pick}: expected a name->index mapping, got {type(mapping).__name__}")
    names = [None] * len(mapping)
    for name, idx in mapping.items():
        names[int(idx)] = name
    if any(n is None for n in names):
        raise RuntimeError(f"{pick}: index mapping has gaps")
    return names


def write_one(npz_path: Path, out_stem: Path, year: int, month: int, split: str, n_cols: int):
    f = np.load(npz_path, allow_pickle=True)
    X = f["X_train"]
    y = f["y_train"]
    if X.shape[1] != n_cols:
        raise RuntimeError(f"{npz_path}: width {X.shape[1]}, expected {n_cols}")
    if not set(np.unique(X).tolist()) <= {0, 1}:
        raise RuntimeError(f"{npz_path}: features are not binary")

    fam_idx = y.astype(np.int64)
    label = fam_idx > 0                      # zero is benign; any positive value is a family
    families = f["y_mal_family"] if "y_mal_family" in f else None
    fam_names = np.full(len(y), "benign", dtype=object)
    if families is not None:
        if len(families) != int(label.sum()):
            raise RuntimeError(f"{npz_path}: {len(families)} family names for {int(label.sum())} malware")
        fam_names[label] = [str(s) for s in families]

    header = tmx.write_tmx(f"{out_stem}.tmx", X.astype(bool))
    table = pa.table({
        "label": pa.array(label, type=pa.bool_()),
        "year": pa.array(np.full(len(y), year, dtype=np.int16), type=pa.int16()),
        "month": pa.array(np.full(len(y), month, dtype=np.int8), type=pa.int8()),
        "split": pa.array([split] * len(y), type=pa.string()),
        "family": pa.array(fam_names.tolist(), type=pa.string()),
        "family_index": pa.array(fam_idx, type=pa.int32()),
        "vt_detection": pa.array(np.full(len(y), -1, dtype=np.int16), type=pa.int16()),
    })
    tmx.write_meta(f"{out_stem}.meta.arrow", table)
    return header, int(label.sum())


def main(src_dir: str, out_dir: str) -> int:
    src, out = Path(src_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = feature_names(src)
    tmx.write_features(out / "features.arrow", names)
    print(f"features.arrow: {len(names)} names, e.g. {names[0]}")

    files = sorted(src.glob("*_selected.npz"))
    pools = [p for p in files if POOL_RE.match(p.name)]
    months = [p for p in files if MONTH_RE.match(p.name)]
    if len(pools) != 1:
        raise RuntimeError(f"expected exactly one training pool, found {[p.name for p in pools]}")

    m = POOL_RE.match(pools[0].name)
    h, mal = write_one(pools[0], out / "train", int(m.group(1)), 0, "train", len(names))
    print(f"  train  {pools[0].name}: {h.n_rows:>7,} rows, {mal:>6,} malware "
          f"({100*mal/h.n_rows:.1f}%)")

    total = h.n_rows
    for p in months:
        mm = MONTH_RE.match(p.name)
        year, month = int(mm.group(1)), int(mm.group(2))
        h, mal = write_one(p, out / f"{year}-{month:02d}", year, month, "test", len(names))
        total += h.n_rows
        print(f"  month  {year}-{month:02d}: {h.n_rows:>7,} rows, {mal:>6,} malware "
              f"({100*mal/max(1,h.n_rows):.1f}%)")
    print(f"total rows written: {total:,} across {len(months)} months plus the pool")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
