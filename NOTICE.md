# Third-party notices

TM-Cyber is MIT-licensed (see [LICENSE](LICENSE)). It consumes and cites the projects below. All
code dependencies are MIT, so the licenses are compatible — but compatibility is not attribution.

## The rule

Any source file that transfers code, memory layout, or a non-obvious algorithm from an upstream
project carries the upstream copyright notice **inline, in that file**, in addition to this one. A
file that merely implements something described in a paper does not.

This project is expected to stay on the citation side of that line: it uses the Tsetlin stack as a
library rather than transferring its internals. The one place to watch is the packed matrix format
in `docs/matrix-format.md` and its readers, which deliberately reproduce Tsetlin.jl's chunked input
bit layout so that loading is a reinterpret rather than a conversion. That is a layout convention
rather than transferred code, and it is noted here so the judgement is on the record.

## Code

### tm-lab — MIT

- Copyright (c) 2026 Lukasz Gagala
- `github.com/Lukasz-G/tm-lab`
- **Role:** the Tsetlin stack. `TMCore` (clause evaluator, feedback, training, model format) and
  `TMBoolean` (booleanization encoders) are consumed as libraries, cloned at a pinned commit by
  `julia/bootstrap.jl`. Not vendored into this repository's history.

### Tsetlin.jl — MIT

- Copyright (c) 2024-2026 Artem Hnilov
- `github.com/BooBSD/Tsetlin.jl`
- **Role:** upstream of tm-lab's memory layout and inner loop, and the origin of the chunked input
  bit layout that `.tmx` reproduces on disk. Reached here transitively; tm-lab carries the inline
  notices.

### FuzzyPatternTM — MIT

- Copyright (c) 2024-2026 Artem Hnilov
- `github.com/BooBSD/FuzzyPatternTM`
- **Role:** reference semantics for the Fuzzy-Pattern TM, and the source of the IMDb hyperparameter
  settings used as a starting point here, IMDb being the closest published analogue to this
  dataset's shape. Read, not copied.

## Datasets

Licensed by their authors, not by this project. Check each before redistributing anything derived
from it; derived matrices are gitignored and are not published from this repository.

### LAMDA

- IQSeC-Lab, University of Texas at El Paso, and the University of Edinburgh
- `huggingface.co/datasets/IQSeC-Lab/LAMDA`, DOI 10.57967/hf/5563, arXiv:2505.18551
- Built on AndroZoo, whose own access terms apply upstream of it. Labels come from VirusTotal
  verdicts aggregated by AVClass2.

### APIGraph

- Zhang et al., CCS 2020. Used for the cross-dataset check.

## Papers

Cited, not licensed:

- LAMDA — arXiv:2505.18551
- Fuzzy-Pattern Tsetlin Machine — arXiv:2508.08350
- Graph Tsetlin Machine — arXiv:2507.14874
- TESSERACT — Pendlebury et al., USENIX Security 2019
- Drebin — Arp et al., NDSS 2014
- Drift forensics of malware classifiers — Chow et al., AISec 2023
- CADE — Yang et al., USENIX Security 2021
- DroidEvolver — Xu et al., EuroS&P 2019
