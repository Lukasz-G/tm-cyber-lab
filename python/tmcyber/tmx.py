"""Write and read the `.tmx` packed binary matrix -- the Python/Julia boundary.

Format spec: docs/matrix-format.md. The on-disk bit layout is the layout TMCore's `TMInput`
uses internally, so the Julia side loads a row by reinterpreting words, not by
converting bits.

Nothing here knows about LAMDA. Dataset-specific work belongs in lamda.py.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

MAGIC = b"TMCYBERX"
VERSION = 1
HEADER_BYTES = 64
HEADER_STRUCT = "<8sIIQQ32x"


@dataclass(frozen=True)
class TmxHeader:
    version: int
    flags: int
    n_rows: int
    n_cols: int

    @property
    def words_per_row(self) -> int:
        return (self.n_cols + 63) // 64


def _pack(rows: np.ndarray) -> np.ndarray:
    """Pack a bool matrix into little-endian uint64 words, row-major, LSB first.

    Padding bits in the final word of each row are zero. TMCore requires clause include-masks
    to be zero there too, so a writer that leaves garbage in the padding produces a file that
    trains differently -- hence the explicit zero pad instead of trusting the input's dtype.
    """
    if rows.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {rows.shape}")
    rows = np.ascontiguousarray(rows, dtype=bool)
    n_rows, n_cols = rows.shape
    pad = (-n_cols) % 64
    if pad:
        rows = np.hstack([rows, np.zeros((n_rows, pad), dtype=bool)])
    packed = np.packbits(rows, axis=1, bitorder="little")
    return packed.view("<u8")


def write_tmx(path: str | Path, rows: np.ndarray) -> TmxHeader:
    """Write a bool matrix as `.tmx`. Returns the header that was written."""
    path = Path(path)
    n_rows, n_cols = rows.shape
    words = _pack(rows)
    header = struct.pack(HEADER_STRUCT, MAGIC, VERSION, 0, n_rows, n_cols)
    assert len(header) == HEADER_BYTES, len(header)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        words.tofile(f)
    return TmxHeader(VERSION, 0, n_rows, n_cols)


def read_header(path: str | Path) -> TmxHeader:
    with open(path, "rb") as f:
        raw = f.read(HEADER_BYTES)
    if len(raw) < HEADER_BYTES:
        raise ValueError(f"{path}: shorter than a {HEADER_BYTES}-byte header")
    magic, version, flags, n_rows, n_cols = struct.unpack(HEADER_STRUCT, raw)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic}, expected {MAGIC}")
    if version != VERSION:
        raise ValueError(f"{path}: version {version}, this reader speaks {VERSION}")
    return TmxHeader(version, flags, n_rows, n_cols)


def read_tmx(path: str | Path) -> np.ndarray:
    """Read a `.tmx` back into a bool matrix.

    Used for the round-trip check and for Python-side baselines that want the same bits the
    Tsetlin machine saw. Training reads these files from Julia.
    """
    h = read_header(path)
    words = np.fromfile(path, dtype="<u8", offset=HEADER_BYTES)
    expected = h.n_rows * h.words_per_row
    if words.size != expected:
        raise ValueError(f"{path}: {words.size} words, expected {expected}")
    bits = np.unpackbits(
        words.reshape(h.n_rows, h.words_per_row).view(np.uint8),
        axis=1,
        bitorder="little",
    )
    return bits[:, : h.n_cols].astype(bool)


def write_meta(path: str | Path, table: pa.Table) -> None:
    """Write an Arrow sidecar. One row per matrix row, in matrix order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with ipc.new_file(str(path), table.schema) as w:
        w.write_table(table)


def read_meta(path: str | Path) -> pa.Table:
    with ipc.open_file(str(path)) as r:
        return r.read_all()


def write_features(path: str | Path, names: list) -> None:
    """Write the feature-name sidecar, one row per matrix column, in column order.

    Not optional: without it no clause literal can be checked against ground-truth indicators,
    which is the one interpretability measurement that is stronger in this domain than in the
    image and text tasks the sibling project used.
    """
    write_meta(path, pa.table({"name": pa.array(list(names), type=pa.string())}))


REQUIRED_META_COLUMNS = ("label", "year", "month", "split")


def check_consistent(
    tmx_path: str | Path,
    meta_path: str | Path,
    features_path: str | Path | None = None,
) -> TmxHeader:
    """Assert the invariants a Julia reader is allowed to assume. Cheap; run after every write."""
    h = read_header(tmx_path)
    meta = read_meta(meta_path)
    if meta.num_rows != h.n_rows:
        raise ValueError(f"metadata has {meta.num_rows} rows, matrix has {h.n_rows}")
    for col in REQUIRED_META_COLUMNS:
        if col not in meta.column_names:
            raise ValueError(f"metadata is missing required column {col}")
    if features_path is not None:
        feats = read_meta(features_path)
        if feats.num_rows != h.n_cols:
            raise ValueError(
                f"feature names have {feats.num_rows} rows, matrix has {h.n_cols} columns"
            )
    return h


def self_test(tmp_dir: str | Path) -> None:
    """Round-trip check. B0 requires the boundary to be bit-exact before anything is trained."""
    rng = np.random.default_rng(0)
    tmp_dir = Path(tmp_dir)
    for n_rows, n_cols in ((1, 1), (3, 64), (5, 65), (7, 4561), (2, 127)):
        rows = rng.random((n_rows, n_cols)) < 0.05
        p = tmp_dir / f"rt_{n_rows}x{n_cols}.tmx"
        write_tmx(p, rows)
        back = read_tmx(p)
        if back.shape != rows.shape or not np.array_equal(back, rows):
            raise AssertionError(f"round trip failed at {n_rows}x{n_cols}")
        h = read_header(p)
        assert (h.n_rows, h.n_cols) == (n_rows, n_cols)
    print("tmx round trip: ok")
