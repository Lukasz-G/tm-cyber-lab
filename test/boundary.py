"""Write a fixture for the cross-language boundary check, then verify the Julia side read it.

    python test/boundary.py write <dir>      # writes .tmx + sidecars + expected.txt
    julia --project=. test/boundary.jl <dir> # reads them and writes actual.txt
    python test/boundary.py verify <dir>     # compares

B0 requires the Python/Julia boundary to round-trip bit-exactly before anything is trained. A
Python-only round trip (tmcyber.tmx.self_test) does not establish that: it checks the writer
against itself. This checks the writer against the reader that actually feeds the machine.

expected.txt has one line per row: the sorted 1-based indices of that row's set bits, space
separated. That is a representation both languages can produce independently, and it catches the
failure that matters -- an off-by-one or endianness disagreement in the bit layout, which would
silently train a different model, not raise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from tmcyber import tmx  # noqa: E402

# Windows consoles default to cp1252, which cannot print a path containing a non-Latin-1
# character. A diagnostic that crashes while reporting success is worse than no diagnostic.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# 4561 is LAMDA's baseline feature count -- not a multiple of 64, which is the case that catches
# padding bugs. 1 and 64 are the degenerate widths.
SHAPES = ((1, 1), (3, 64), (5, 65), (9, 4561))
DENSITY = 0.05  # Drebin rows are sparse; exercise the layout the machine will actually see


def row_lines(rows: np.ndarray) -> list:
    return [" ".join(str(i + 1) for i in np.flatnonzero(r)) for r in rows]


def write(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260918)
    manifest = []
    for n_rows, n_cols in SHAPES:
        rows = rng.random((n_rows, n_cols)) < DENSITY
        rows[0, 0] = True                  # first bit of first word
        rows[-1, n_cols - 1] = True        # last real bit, next to the zero padding
        stem = f"b_{n_rows}x{n_cols}"
        tmx.write_tmx(out / f"{stem}.tmx", rows)
        meta = pa.table({
            "label": pa.array(rows[:, 0], type=pa.bool_()),
            "year": pa.array([2013 + (i % 10) for i in range(n_rows)], type=pa.int16()),
            "month": pa.array([1 + (i % 12) for i in range(n_rows)], type=pa.int8()),
            "split": pa.array(["train"] * n_rows, type=pa.string()),
            "family": pa.array([""] * n_rows, type=pa.string()),
            "vt_detection": pa.array([0] * n_rows, type=pa.int16()),
        })
        tmx.write_meta(out / f"{stem}.meta.arrow", meta)
        tmx.write_features(out / f"{stem}.features.arrow", [f"f{j}" for j in range(n_cols)])
        tmx.check_consistent(out / f"{stem}.tmx",
                             out / f"{stem}.meta.arrow",
                             out / f"{stem}.features.arrow")
        (out / f"{stem}.expected.txt").write_text("\n".join(row_lines(rows)) + "\n", encoding="utf-8")
        manifest.append(stem)
    (out / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} fixtures to {out}")


def verify(out: Path) -> int:
    stems = (out / "manifest.txt").read_text(encoding="utf-8").split()
    failures = 0
    for stem in stems:
        exp = (out / f"{stem}.expected.txt").read_text(encoding="utf-8").splitlines()
        actual_path = out / f"{stem}.actual.txt"
        if not actual_path.exists():
            print(f"FAIL {stem}: {actual_path.name} missing -- did the Julia side run?")
            failures += 1
            continue
        act = actual_path.read_text(encoding="utf-8").splitlines()
        if exp == act:
            print(f"ok   {stem}: {len(exp)} rows match")
        else:
            failures += 1
            print(f"FAIL {stem}: {sum(a != b for a, b in zip(exp, act))} of {len(exp)} rows differ")
            for i, (a, b) in enumerate(zip(exp, act)):
                if a != b:
                    print(f"       row {i + 1} expected {a[:80]!r}")
                    print(f"       row {i + 1} actual   {b[:80]!r}")
                    break
    print("boundary check:", "ok" if failures == 0 else f"{failures} failed")
    return 1 if failures else 0


def main(argv: list) -> int:
    if len(argv) != 3 or argv[1] not in ("write", "verify"):
        print(__doc__)
        return 2
    if argv[1] == "write":
        write(Path(argv[2]))
        return 0
    return verify(Path(argv[2]))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
