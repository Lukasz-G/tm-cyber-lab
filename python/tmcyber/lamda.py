"""LAMDA: file layout, metadata, and the AnoShift-style splits.

The released baseline variant is per-year Parquet under `Baseline/<year>/<year>_{train,test}.parquet`,
each row carrying `hash`, `label`, `family`, `vt_count`, `year_month` and `feat_0..feat_4560`.

Two things worth knowing before using this:

* `year_month` is in the Parquet itself, so the month-level analysis needs no join against
  `metadata.csv`. That matters because the reference explanation-drift script reads monthwise `.npz`
  files from a path local to the authors' machine (`/home/shared-datasets/...npz_monthwise_Final`)
  which is **not** part of the release. Monthly splits here are therefore derived by us from the
  released `year_month`, not the authors' own monthly files.
* 2015 is absent by construction — AndroZoo lacks the hashes.

`metadata.csv` (124 MB) carries the per-sample `vt_detection` and an `added` timestamp for the whole
corpus, which is the raw material for separating label drift from data drift. It is not needed for
training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_ROOT = Path("data/lamda/raw")

META_COLUMNS = ("hash", "label", "family", "vt_count", "year_month")
FEATURE_PREFIX = "feat_"

# AnoShift-style splits as the paper defines them. FAR is deliberately not treated as one block:
# 2023-2025 malware counts collapse (7,892 / 794 / 23 against ~45,000 benign per year), which is
# antivirus label lag, not drift, so the usable window ends at 2022 and every FAR figure is
# reported per year.
TRAIN_YEARS = (2013, 2014)
NEAR_YEARS = (2016, 2017)
FAR_YEARS = (2018, 2019, 2020, 2021, 2022)
LABEL_LAG_YEARS = (2023, 2024, 2025)


@dataclass(frozen=True)
class YearFiles:
    year: int
    train: Path
    test: Path


def discover(root: str | Path = DEFAULT_ROOT, variant: str = "Baseline") -> list:
    """Every year present on disk, sorted. Raises if a year has only one of its two files."""
    base = Path(root) / variant
    if not base.is_dir():
        raise FileNotFoundError(f"{base} not found — has the dataset been downloaded?")
    out = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        year = int(d.name)
        tr, te = d / f"{year}_train.parquet", d / f"{year}_test.parquet"
        missing = [p.name for p in (tr, te) if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{d}: missing {missing}")
        out.append(YearFiles(year, tr, te))
    return out


def feature_names(path: str | Path) -> list:
    """Feature column names in column order. The order is the matrix's column order."""
    names = pq.ParquetFile(str(path)).schema_arrow.names
    return [n for n in names if n.startswith(FEATURE_PREFIX)]


def read_meta(path: str | Path):
    """Just the metadata columns — cheap, because Parquet is columnar and there are 4,561 others."""
    return pq.read_table(str(path), columns=list(META_COLUMNS)).to_pandas()


def read_features(path: str | Path, columns: list | None = None):
    """The feature block as a numpy bool array, plus the metadata frame, in file order.

    Reading 4,561 int8 columns for a year is a few hundred MB; the caller is expected to convert to
    packed `.tmx` and work from that afterwards.
    """
    import numpy as np

    cols = columns if columns is not None else feature_names(path)
    tbl = pq.read_table(str(path), columns=list(META_COLUMNS) + list(cols))
    meta = tbl.select(list(META_COLUMNS)).to_pandas()
    feats = tbl.select(list(cols)).to_pandas().to_numpy(dtype=np.int8)
    return feats.astype(bool), meta


def split_of(year: int) -> str:
    """Which AnoShift split a year belongs to. `label_lag` is named, not lumped into `far`."""
    if year in TRAIN_YEARS:
        return "train"
    if year in NEAR_YEARS:
        return "near"
    if year in FAR_YEARS:
        return "far"
    if year in LABEL_LAG_YEARS:
        return "label_lag"
    raise ValueError(f"year {year} is not in any known LAMDA split")
