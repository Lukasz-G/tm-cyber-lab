"""Convert LAMDA Parquet to the packed `.tmx` matrices the Julia side trains on.

    python -m tmcyber.build_tmx data/lamda/raw data/lamda/tmx

One `.tmx` plus one `.meta.arrow` per year, and a single shared `features.arrow`. Per year, not one combined file because months never span years, so a per-month slice only ever needs one
year loaded, and a 100k x 4,561 year is a comfortable 57 MB packed while the whole corpus would be
574 MB. All 24 released files share the same feature column order, which is checked here and not assumed.

Column-name note: the Parquet calls the VirusTotal count `vt_count`; the sidecar stores it as
`vt_detection`, which is the name the dataset's own metadata and paper use, and the name the matrix
format specifies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

from tmcyber import lamda, tmx


def convert_year(yf, cols, out_dir: Path, verbose: bool = True):
    """Write one year's train and test portions into a single .tmx, train rows first."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for kind, path in (("train", yf.train), ("test", yf.test)):
        feats, meta = lamda.read_features(path, cols)
        meta = meta.copy()
        meta["file_split"] = kind
        parts.append((feats, meta))

    X = np.vstack([p[0] for p in parts])
    import pandas as pd

    meta = pd.concat([p[1] for p in parts], ignore_index=True)

    stem = out_dir / f"{yf.year}"
    header = tmx.write_tmx(f"{stem}.tmx", X)

    months = meta["year_month"].astype(str).str.slice(5, 7).astype(np.int8)
    table = pa.table({
        "label": pa.array(meta["label"].to_numpy().astype(bool), type=pa.bool_()),
        "year": pa.array(np.full(len(meta), yf.year, dtype=np.int16), type=pa.int16()),
        "month": pa.array(months.to_numpy(), type=pa.int8()),
        "split": pa.array([lamda.split_of(yf.year)] * len(meta), type=pa.string()),
        "file_split": pa.array(meta["file_split"].to_numpy(), type=pa.string()),
        "family": pa.array(meta["family"].astype(str).to_numpy(), type=pa.string()),
        "vt_detection": pa.array(meta["vt_count"].to_numpy().astype(np.int16), type=pa.int16()),
        "hash": pa.array(meta["hash"].astype(str).to_numpy(), type=pa.string()),
    })
    tmx.write_meta(f"{stem}.meta.arrow", table)

    if verbose:
        mb = Path(f"{stem}.tmx").stat().st_size / 1e6
        print(f"  {yf.year}: {header.n_rows:>7,} x {header.n_cols} -> {mb:6.1f} MB  "
              f"malware {int(meta['label'].sum()):>6,}  months {sorted(set(months))}")
    return header


def main(raw_root: str, out_root: str) -> int:
    years = lamda.discover(raw_root)
    cols = lamda.feature_names(years[0].train)
    for yf in years:
        for p in (yf.train, yf.test):
            if lamda.feature_names(p) != cols:
                raise RuntimeError(f"{p} has a different feature column order")

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    tmx.write_features(out / "features.arrow", cols)
    print(f"features.arrow: {len(cols)} names")

    total = 0
    for yf in years:
        h = convert_year(yf, cols, out)
        tmx.check_consistent(out / f"{yf.year}.tmx", out / f"{yf.year}.meta.arrow",
                             out / "features.arrow")
        total += h.n_rows
    print(f"total rows written: {total:,}")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/lamda/raw"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/lamda/tmx"
    raise SystemExit(main(raw, dst))
