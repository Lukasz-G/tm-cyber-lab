"""TM-Cyber: Tsetlin machines on Android malware concept drift.

Python owns datasets, splits, baselines and statistics. Julia owns Tsetlin training, clause
inspection and drift measurement. The boundary is the binarised feature matrix -- see
docs/matrix-format.md.
"""

from tmcyber.tmx import (  # noqa: F401
    TmxHeader,
    check_consistent,
    read_header,
    read_meta,
    read_tmx,
    write_features,
    write_meta,
    write_tmx,
)

__all__ = [
    "TmxHeader",
    "check_consistent",
    "read_header",
    "read_meta",
    "read_tmx",
    "write_features",
    "write_meta",
    "write_tmx",
]
