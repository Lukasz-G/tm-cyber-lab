# `.tmx` — the Python/Julia boundary

The project boundary is the **binarized feature matrix**. Feature extraction, dataset handling,
baselines and statistics are Python; Tsetlin training, clause inspection and drift measurement are
Julia. Nothing crosses that line except a matrix, its row metadata, and a trained model.

`.tmx` exists so that crossing is a `reinterpret` rather than a conversion: the on-disk bit layout is
the layout `TMCore`'s `TMInput` already uses internally.

## Why not Arrow, Parquet or `Serialization` for the matrix itself

- Julia `Serialization` survives neither a Julia upgrade nor a struct rename. A format that cannot
  survive a Julia upgrade is not a format.
- Parquet and Arrow are good columnar formats and wrong for this: a booleanized example is consumed
  as one packed row, and a columnar reader would have to transpose a million rows to hand back
  something `TMInput` accepts.
- A fixed header plus raw packed words is readable in fifty lines in any language, which is the same
  reasoning behind tm-lab's model format.

Row **metadata** is a different problem — mixed types, string family names, queried by year and month
— so it goes in a sidecar Arrow file, which is what Arrow is for.

## Layout

Little-endian throughout. One file holds one matrix.

### Header — exactly 64 bytes

| offset | size | field | value |
|---|---|---|---|
| 0 | 8 | magic | ASCII `TMCYBERX` |
| 8 | 4 | version | `UInt32` = 1 |
| 12 | 4 | flags | `UInt32` = 0, reserved |
| 16 | 8 | n_rows | `UInt64` |
| 24 | 8 | n_cols | `UInt64`, feature bits per row, before negation |
| 32 | 32 | reserved | zero |

### Body

`n_rows * cld(n_cols, 64)` `UInt64` words, row-major. Within a row, feature `i` (1-based) lives at
bit `(i - 1) % 64` of word `(i - 1) ÷ 64 + 1`, least significant bit first.

**Padding bits in a row's final word are always zero.** `TMCore` requires clause include-masks to be
zero there too, so padding can never manufacture a literal miss. A writer that leaves garbage in the
padding produces a file that trains differently.

In numpy that layout is exactly:

```python
np.packbits(rows, axis=1, bitorder="little").view("<u8")
```

with `rows` a `bool` array padded on the right to a multiple of 64 columns.

## Sidecar metadata

`<name>.meta.arrow`, one row per matrix row, same order. Columns used by this project:

| column | type | notes |
|---|---|---|
| `label` | `bool` | `true` = malware. LAMDA: benign at `vt_detection == 0`, malware at `>= 4` |
| `year` | `int16` | |
| `month` | `int8` | 1-12. Required: the explanation-drift measurement is month over month |
| `split` | `string` | `train`, `iid`, `near`, `far` |
| `family` | `string` | AVClass2 family, empty for benign |
| `vt_detection` | `int16` | kept for the label-drift analysis, not for thresholding at read time |

Feature **names** go in `<name>.features.arrow`, one row per column of the matrix, in column order,
with at least a `name` column. Without it no interpretability claim can be checked against
ground-truth indicators, so it is not optional.

## Invariants a reader may assume, and a writer must guarantee

1. `n_cols` is the true feature count; the file may be longer than `n_cols` bits per row only by
   zero padding inside the last word.
2. Row order in `.tmx`, `.meta.arrow` and any derived prediction file is identical.
3. `.features.arrow` has exactly `n_cols` rows.
4. A file is never rewritten in place. Deriving a subset or a resampled matrix writes a new file.
