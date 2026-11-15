"""The Python arms of b2-corpora: 2s, 4 and 0. Design, arms and conditions: the header of run.jl.

  python -X utf8 research/b2-corpora/arms.py <lamda|apigraph|androzoo|hypercube> [--smoke] [--months N]
                 [--arms 2s,4,0,timing] [--conds B0,U,O] [--jobs J] [--cfg matched|S100] [--runs4 R]

Reads what run.jl exported under data/b2-corpora[-smoke]/<corpus>/: the month draws and each month's
training rows (months.json: the B0 cut, and per seed the U and O resample), each fixed model's clauses
per configuration, condition and pool seed (<cfg>/<cond>/pool<p>/model.json), its AUC and scores per
month (month-*.json), and the exact per-row attributions of the arm-4 months (phi-*.bin). Writes, beside
run.jl's outputs, one file per configuration, condition and arm:
  <corpus>-<cfg>-<cond>-pairs-2s.csv, <corpus>-<cfg>-<cond>-py-months-2s.csv, <corpus>-<cfg>-<cond>-arm4.csv,
  <corpus>-mlp-<cond>-pairs-0.csv, <corpus>-mlp-<cond>-py-months-0.csv, and for `timing`
  <corpus>-<cfg>-<cond>-timing-py.csv: arm 2s on the arm-4 months in one process, against the exact
  attribution's one-thread time on the same rows. Run `timing` with nothing else on the machine.

  2s  the fixed TM, shap.KernelExplainer at nsamples=100 with l1_reg="num_features(10)" passed
      explicitly, on the same background and explained rows as arm 2; consecutive months; both pool seeds.
  4   three months, both pool seeds, 5 runs each of: 4a nsamples=100 as 2s; 4f the same seeds with
      l1_reg=False; 4c nsamples=1000 as 2s. 4b is 4a's runs against each other. A k at which the exact
      attribution has fewer than k non-zero features is not compared.
  0   the released LAMDA protocol: the authors' MLP refitted on the month's training rows of the
      condition (run r uses seed r+1's resample), KernelExplainer over its class-1 probability as in 2s,
      3 runs per month. The loss is unweighted in every condition.

shap is pinned to 0.49.1: KernelExplainer's l1_reg default and its output shape both changed across
versions, and the start-up check refuses any other version. The tm-lab checkout is checked against the
commit the TM rows carry. Every sampled run is timed on one thread (BLAS and torch pinned to one thread
per worker); --jobs runs workers in parallel processes, and every output row records the worker's peak
resident memory.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from itertools import combinations  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(HERE))

from tmcyber.tmx import read_header  # noqa: E402

import stats  # noqa: E402

TM_LAB_COMMIT = "43dba5f"
SHAP_VERSION = "0.49.1"
L1_REG = "num_features(10)"
TM_PROV = dict(tm_lab=TM_LAB_COMMIT, ceiling="LiteralCapped", parallel="none")
NA_PROV = dict(tm_lab="na", ceiling="na", parallel="na")
KS = (100, 1000)
AUC_MIN = 0.6
NSAMPLES = 100
NSAMPLES_BIG = 1000
RUNS4 = 5
RUNS0 = 3
CONDS = ("B0", "U", "O")

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def check_versions() -> None:
    import shap
    if shap.__version__ != SHAP_VERSION:
        raise SystemExit(f"shap {shap.__version__} installed; b2-corpora is defined at shap {SHAP_VERSION}")
    try:
        head = subprocess.run(["git", "-C", str(ROOT / "vendor" / "tm-lab"), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = ""
    if not head.startswith(TM_LAB_COMMIT):
        raise SystemExit(f"vendor/tm-lab is at '{head}', expected {TM_LAB_COMMIT}")


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB. A worker runs several jobs, so the value it
    reports for a job is the peak over that job and every earlier one: an upper bound for the class."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0     # Linux reports KB
    except ImportError:
        import psutil
        mi = psutil.Process().memory_info()
        return getattr(mi, "peak_wset", mi.rss) / 2**20


# --- resumable units and the memory check --------------------------------------------------------

# Peak resident memory per job class in MB, the starting values for the memory check: the largest peak
# a worker reported in the smoke of 2026-11-13, rounded up (AndroZoo arm 0 had not run; it is set high).
# The value used is the larger of this and every peak recorded since in peak-ram.jsonl.
PEAK_MB_DEFAULT = {
    "py-2s": {"apigraph": 300, "hypercube": 400, "lamda": 400, "androzoo": 900},
    "py-4a": {"apigraph": 500, "hypercube": 1100, "lamda": 1200, "androzoo": 3900},
    "py-4f": {"apigraph": 500, "hypercube": 1100, "lamda": 1200, "androzoo": 3900},
    "py-4c": {"apigraph": 500, "hypercube": 1100, "lamda": 1200, "androzoo": 3900},
    "py-0": {"apigraph": 900, "hypercube": 1500, "lamda": 1300, "androzoo": 4000},
}
MEM_FACTOR = 1.25
MEM_RECHECK_S = 30
RAMP_S = 60          # a job started this recently may not have reached its peak yet; its peak is reserved


def sha_hex(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sidecar(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def write_atomic(path: Path, write) -> Path:
    """Write through a temporary file and a rename, then the .sha256 sidecar the same way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        write(fh)
    os.replace(tmp, path)
    stmp = sidecar(path).with_name(sidecar(path).name + ".tmp")
    stmp.write_text(f"{sha_hex(path)}  {path.name}\n", encoding="utf-8")
    os.replace(stmp, sidecar(path))
    return path


def verified(path: Path) -> bool:
    s = sidecar(path)
    if not (path.is_file() and s.is_file()):
        return False
    return s.read_text(encoding="utf-8").split()[0] == sha_hex(path)


def save_unit(path: Path, result: dict) -> None:
    """One unit's result: arrays as .npz members, everything else as one JSON member."""
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    rest = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    write_atomic(path, lambda fh: np.savez(fh, _meta=np.array(json.dumps(rest)), **arrays))


def load_unit(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = json.loads(str(z["_meta"]))
        out.update({k: z[k] for k in z.files if k != "_meta"})
    out["loaded"] = True
    return out


def mem_available_mb() -> float:
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    import psutil
    return psutil.virtual_memory().available / 2**20


def peak_table(dirs) -> dict:
    best = {}
    for d in dirs:
        f = Path(d) / "peak-ram.jsonl"
        if f.is_file():
            for line in f.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                best[r["class"]] = max(best.get(r["class"], 0.0), float(r["peak_mb"]))
    return best


def run_units(pool_exec, fn, jobs, unit_of, class_of, need_of, max_inflight, log):
    """Run `fn` over `jobs`. A job whose unit file verifies is loaded, not run. The others start only
    when available memory covers MEM_FACTOR x their class's peak plus the peaks of jobs started in the
    last RAMP_S seconds; otherwise the scheduler waits and re-checks. Each worker writes its own unit."""
    from collections import deque
    from concurrent.futures import FIRST_COMPLETED, wait
    results, pending, inflight = [], deque(), {}
    loaded = 0
    for j in jobs:
        u = unit_of(j)
        if verified(u):
            results.append(load_unit(u))
            loaded += 1
        else:
            pending.append(j)
    log(f"  units: {loaded} loaded (verified), {len(pending)} to compute")
    while pending or inflight:
        while pending and len(inflight) < max_inflight:
            j = pending[0]
            need = need_of(class_of(j))
            recent = sum(nd for t0, nd in inflight.values() if time.time() - t0 < RAMP_S)
            avail = mem_available_mb()
            if avail >= MEM_FACTOR * need + recent:
                inflight[pool_exec.submit(fn, j)] = (time.time(), need)
                pending.popleft()
            else:
                log(f"  memory: {class_of(j)} needs {need:.0f} MB x {MEM_FACTOR} + {recent:.0f} MB reserved, "
                    f"{avail:.0f} MB available; waiting")
                if not inflight:
                    time.sleep(MEM_RECHECK_S)
                break
        if inflight:
            done, _ = wait(list(inflight), timeout=MEM_RECHECK_S, return_when=FIRST_COMPLETED)
            for f in done:
                results.append(f.result())
                del inflight[f]
    return results


# --- data -------------------------------------------------------------------------------------

def data_dir(corpus: str, smoke: bool) -> Path:
    return ROOT / "data" / ("b2-corpora-smoke" if smoke else "b2-corpora") / corpus


def out_dir(smoke: bool) -> Path:
    return HERE / "smoke" if smoke else HERE


def read_rows(path: Path, rows) -> np.ndarray:
    """Rows of a .tmx by 1-based index, repeats allowed, as a uint8 0/1 matrix in the order given."""
    h = read_header(path)
    wpr = h.words_per_row
    mm = np.memmap(path, dtype="<u8", mode="r", offset=64, shape=(h.n_rows, wpr))
    idx = np.asarray(rows, dtype=np.int64) - 1
    words = np.ascontiguousarray(mm[idx])
    bits = np.unpackbits(words.view(np.uint8), axis=1, bitorder="little")[:, : h.n_cols]
    return bits.astype(np.uint8)


_CACHE: dict = {}


def months_of(ddir: Path) -> dict:
    key = ("months", str(ddir))
    if key not in _CACHE:
        with open(ddir / "months.json", encoding="utf-8") as fh:
            _CACHE[key] = json.load(fh)
    return _CACHE[key]


def month_rec(ddir: Path, label: str) -> dict:
    return next(m for m in months_of(ddir)["months"] if m["label"] == label)


def tmx_path(ddir: Path, rec: dict) -> Path:
    return ROOT / months_of(ddir)["dir"] / f"{rec['stem']}.tmx"


def model_dir(ddir: Path, cfg: str, cond: str, pool: int) -> Path:
    return ddir / cfg / cond / f"pool{pool}"


def model_of(ddir: Path, cfg: str, cond: str, pool: int) -> dict:
    key = ("model", str(ddir), cfg, cond, pool)
    if key not in _CACHE:
        with open(model_dir(ddir, cfg, cond, pool) / "model.json", encoding="utf-8") as fh:
            _CACHE[key] = json.load(fh)
    return _CACHE[key]


def month_json(ddir: Path, cfg: str, cond: str, pool: int, label: str) -> dict:
    with open(model_dir(ddir, cfg, cond, pool) / f"month-{label}.json", encoding="utf-8") as fh:
        return json.load(fh)


def training_rows(rec: dict, cond: str, seed: int) -> tuple:
    """Malware and goodware row indices of one month's training set under a condition, for seed `seed`,
    rebuilt from run.jl's export: B0 is the cut itself; U replaces the majority with its drawn subset;
    O repeats every minority row `repeat` times and appends the extra minority rows."""
    mal, good = list(rec["train"]["B0"]["mal"]), list(rec["train"]["B0"]["good"])
    if cond == "B0":
        return mal, good, 1
    r = rec["train"][f"{cond}/{seed}"]
    malmin = r["minority"] == "malware"
    if cond == "U":
        return ((mal, list(r["majority_rows"]), 1) if malmin else (list(r["majority_rows"]), good, 1))
    if cond == "O":
        if malmin:
            return mal * r["repeat"] + list(r["extra_rows"]), good, r["repeat"]
        return mal, good * r["repeat"] + list(r["extra_rows"]), r["repeat"]
    raise ValueError(cond)


# --- the TM scorer ------------------------------------------------------------------------------

def build_scorer(model: dict):
    """The FPTM clause vote of the explained class over a dense 0/1 matrix: each clause votes
    max(0, ceiling - misses), signed by its polarity bank. The same arithmetic as the Julia
    evaluator, checked against the exported Julia scores before anything is explained."""
    width = model["width"]
    clauses = [(np.asarray(c["features"], dtype=np.int64) - 1, np.asarray(c["negated"], dtype=bool),
                int(c["ceiling"]), int(c["sign"])) for c in model["clauses"]]

    def score(X):
        X = np.asarray(X).reshape(-1, width)
        out = np.zeros(X.shape[0], dtype=np.float64)
        for f, neg, ceil, sgn in clauses:
            if f.size == 0:
                out += sgn * ceil
                continue
            vals = X[:, f].astype(bool)
            missed = np.where(neg, vals, ~vals).sum(axis=1)
            out += sgn * np.maximum(0, ceil - missed)
        return out

    return score


def port_check(score, bg, ex, mj, where: str) -> None:
    d = max(np.abs(score(bg) - np.asarray(mj["score_bg"])).max(),
            np.abs(score(ex) - np.asarray(mj["score_ex"])).max())
    if d != 0.0:
        raise AssertionError(f"{where}: Python clause vote differs from Julia scores by {d:g}")


def vote_bound(model: dict) -> float:
    """No exact Shapley value of the TM score can exceed the sum of the clause ceilings in absolute
    value, so a sampled row beyond it is outside the range of the quantity being estimated."""
    return float(sum(int(c["ceiling"]) for c in model["clauses"]))


def out_of_range(phi: np.ndarray, bound: float) -> int:
    return int((np.abs(phi).max(axis=1) > bound).sum())


# --- KernelExplainer ----------------------------------------------------------------------------

def kernel_phi(f, bg: np.ndarray, ex: np.ndarray, nsamples: int, seed: int, l1_reg=L1_REG,
               output: int | None = None) -> tuple:
    """Signed per-row attributions (n x M), wall-clock seconds, the l1_reg shap used, the number of
    warnings shap raised, and the number of explained rows whose regression failed.

    The rows are explained one by one with KernelExplainer.explain, in order, after one np.random.seed:
    that is the loop of shap 0.49.1's shap_values, so a run in which no row fails consumes the same random
    stream and returns the same values. A row whose solve raises numpy.linalg.LinAlgError ("SVD did not
    converge", "Singular matrix") gets NaN for every feature and is counted; the other rows are kept."""
    import shap

    np.random.seed(seed)
    t0 = time.time()
    rows, failed = [], 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        expl = shap.KernelExplainer(f, bg)
        for i in range(ex.shape[0]):
            try:
                rows.append(np.asarray(expl.explain(ex[i:i + 1, :], nsamples=nsamples, silent=True,
                                                    l1_reg=l1_reg), dtype=np.float64))
            except np.linalg.LinAlgError:
                rows.append(None)
                failed += 1
    secs = time.time() - t0
    shape = next((r.shape for r in rows if r is not None),
                 (ex.shape[1], expl.D) if getattr(expl, "vector_out", False) else (ex.shape[1],))
    vals = np.stack([r if r is not None else np.full(shape, np.nan) for r in rows])
    if output is not None:
        if vals.ndim != 3 or vals.shape[0] != ex.shape[0]:
            raise ValueError(f"unexpected shap output shape {vals.shape}")
        vals = vals[:, :, output]
    elif vals.ndim != 2:
        raise ValueError(f"unexpected shap output shape {vals.shape}")
    return vals.astype(np.float64), secs, str(getattr(expl, "l1_reg", "?")), len(caught), failed


def row_mean_abs(phi: np.ndarray) -> np.ndarray:
    """Importance = mean of |phi| over the explained rows that did not fail (NaN rows are skipped); all
    NaN when every row failed."""
    ok = ~np.isnan(phi).any(axis=1)
    if not ok.any():
        return np.full(phi.shape[1], np.nan)
    return np.abs(phi[ok]).mean(axis=0)


def varying(bg: np.ndarray, ex: np.ndarray) -> np.ndarray:
    """shap's M per explained row: features on which the row differs from at least one background row."""
    lo, hi = bg.min(axis=0), bg.max(axis=0)
    return ((ex < lo) | (ex > hi) | (lo != hi)[None, :]).sum(axis=1)


def test_rng(corpus: str, cond: str, cfg: str, pool, test: str) -> np.random.Generator:
    """One bootstrap stream per test, seeded by a stable hash of what the test is (as in decide.py), so an
    interval does not depend on which conditions or pool seeds the same process handled before it."""
    import zlib
    return np.random.default_rng([20261112, zlib.crc32(f"{corpus}|{cond}|{cfg}|{pool}|{test}".encode("utf-8"))])


def result_line(**kv) -> str:
    """A RESULT line, which tools/fanout.sh carries into the results log that survives the box."""
    return "RESULT " + " ".join(f"{k}={v}" for k, v in kv.items())


def nanmean(xs) -> float:
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    return float(xs.mean()) if xs.size else float("nan")


# --- workers ------------------------------------------------------------------------------------

def _tm_inputs(ddir, cfg, cond, pool, label):
    rec = month_rec(ddir, label)
    path = tmx_path(ddir, rec)
    bg, ex = read_rows(path, rec["bg"]), read_rows(path, rec["ex"])
    mj = month_json(ddir, cfg, cond, pool, label)
    model = model_of(ddir, cfg, cond, pool)
    score = build_scorer(model)
    port_check(score, bg, ex, mj, f"{cfg} {cond} pool{pool} {label}")
    return rec, bg, ex, mj, score, model


def work_2s(args):
    ddir, cfg, cond, pool, label, seed, unit = args
    rec, bg, ex, mj, score, model = _tm_inputs(ddir, cfg, cond, pool, label)
    phi, secs, l1, nwarn, failed = kernel_phi(score, bg, ex, NSAMPLES, seed)
    out = dict(arm="2s", cfg=cfg, cond=cond, pool=pool, label=label, imp=row_mean_abs(phi),
                failed_rows=failed, all_failed=failed == ex.shape[0],
                secs=secs, l1=l1, nwarn=nwarn, auc=mj["auc"], m_exact=int(mj["m"]),
                M_eff=float(varying(bg, ex).mean()), oor=out_of_range(phi, vote_bound(model)),
                N=model["N"], share=model["share"], repeat=model["repeat"],
                seconds_exact=mj.get("seconds_exact"), threads_exact=mj.get("threads_exact"),
                seconds_exact_1thread=mj.get("seconds_exact_1thread"), rss=peak_rss_mb())
    save_unit(unit, out)
    return out


def work_4(args):
    ddir, cfg, cond, pool, label, variant, run, seed, unit = args
    rec, bg, ex, mj, score, model = _tm_inputs(ddir, cfg, cond, pool, label)
    ns = NSAMPLES_BIG if variant == "4c" else NSAMPLES
    l1 = False if variant == "4f" else L1_REG
    phi, secs, l1used, nwarn, failed = kernel_phi(score, bg, ex, ns, seed, l1_reg=l1)
    Mr = varying(bg, ex)
    out = dict(cfg=cfg, cond=cond, pool=pool, label=label, variant=variant, run=run, phi=phi,
                secs=secs, l1=l1used, nwarn=nwarn, frac=float(np.mean(ns / (2 * Mr + 2048))),
                oor=out_of_range(phi, vote_bound(model)), rss=peak_rss_mb(),
                failed_rows=failed, all_failed=failed == ex.shape[0])
    save_unit(unit, out)
    return out


def work_0(args):
    ddir, cond, label, run, idx, unit = args
    import torch
    torch.set_num_threads(1)
    rec = month_rec(ddir, label)
    mal, good, reps = training_rows(rec, cond, run + 1)
    path = tmx_path(ddir, rec)
    Xtr = read_rows(path, mal + good)
    ytr = np.concatenate([np.ones(len(mal), dtype=np.int64), np.zeros(len(good), dtype=np.int64)])
    Xte = read_rows(path, rec["test"])
    yte = np.asarray(rec["test_label"], dtype=bool)
    bg, ex = read_rows(path, rec["bg"]), read_rows(path, rec["ex"])
    torch.manual_seed(1000 * run + idx)
    np.random.seed(1000 * run + idx)
    t0 = time.time()
    model = train_chen_mlp(Xtr, ytr)
    t_train = time.time() - t0
    f = make_predict(model)
    a = auc_of(f(Xte)[:, 1], yte)
    phi, secs, l1, nwarn, failed = kernel_phi(f, bg, ex, NSAMPLES, 500_000 + 1000 * run + idx, output=1)
    n = len(mal) + len(good)
    out = dict(arm="0", cond=cond, label=label, run=run, imp=row_mean_abs(phi), secs=secs,
                failed_rows=failed, all_failed=failed == ex.shape[0],
                t_train=t_train, l1=l1, nwarn=nwarn, auc=a, M_eff=float(varying(bg, ex).mean()),
                oor=out_of_range(phi, 1.0), N=n, share=len(mal) / n, repeat=reps, rss=peak_rss_mb())
    save_unit(unit, out)
    return out


# --- the released MLP, transcribed in research/b0-noise-floor/run.py from the authors' script ---

def _torch_mlp(input_dim):
    import torch.nn as nn
    import torch.nn.functional as F

    class ChenEncoderMLP(nn.Module):
        def __init__(self, input_dim, num_classes=2):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512), nn.ReLU(),
                nn.Linear(512, 384), nn.ReLU(),
                nn.Linear(384, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(128, 100), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(100, 100), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(100, num_classes),
            )

        def forward(self, x, return_prob=False):
            logits = self.classifier(self.encoder(x))
            return F.softmax(logits, dim=1) if return_prob else logits

    return ChenEncoderMLP(input_dim)


def train_chen_mlp(X: np.ndarray, y: np.ndarray, epochs: int = 20):
    """20 epochs, batch 64, shuffled, Adam lr 1e-3, unweighted cross-entropy, as released. Rows stay
    uint8 and are converted per batch, so a large training set does not need a float copy."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    model = _torch_mlp(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    n = len(yt)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, 64):
            b = perm[s:s + 64]
            opt.zero_grad()
            crit(model(Xt[b].float()), yt[b]).backward()
            opt.step()
    return model


def make_predict(model):
    import torch

    def f(x):
        model.eval()
        with torch.no_grad():
            out = []
            x = np.asarray(x)
            for s in range(0, x.shape[0], 8192):
                xb = torch.from_numpy(np.ascontiguousarray(x[s:s + 8192], dtype=np.float32))
                out.append(model(xb, return_prob=True).numpy())
            return np.vstack(out)
    return f


def auc_of(s: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    r = rankdata(s)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def kendall_dist(a, b, ia, ib) -> float:
    from scipy.stats import kendalltau
    u = sorted(set(a.tolist()) | set(b.tolist()))
    tau, _ = kendalltau([float(ia[i]) for i in u], [float(ib[i]) for i in u])
    return 1.0 if tau is None or np.isnan(tau) else (1 - tau) / 2


# --- driver -------------------------------------------------------------------------------------

PAIR_COLS = ("corpus,cfg,S,matched,condition,share,repeat,N_a,N_b,arm,pool_seed,refit_seed,k,from,to,anchor,"
             "jaccard,kendall,inter,m_a,m_b,m_ab,E,c,guard_ok,support_ok,tm_lab,ceiling,parallel")

MONTH_COLS = ("corpus,arm,cfg,S,condition,N,share,repeat,pool_seed,run,month,auc,guard_ok,m,m_exact,M_eff,"
              "nsamples,l1_reg,warnings,failed_rows,rows_out_of_range,seconds,seconds_train,threads,seconds_exact,"
              "threads_exact,seconds_exact_1thread,peak_rss_mb,shap,tm_lab,ceiling,parallel")


def pair_line(corpus, cfg, S, matched, cond, share, repeat, Na, Nb, arm, ps, rs, k, fa, fb, a, b, guard,
              prov, extra_ok=True) -> str:
    st = stats.pairstats(a, b, k)
    kd = kendall_dist(stats.topk(a, k), stats.topk(b, k), a, b)
    ok = st["support_ok"] and extra_ok
    return (f"{corpus},{cfg},{S},{str(matched).lower()},{cond},{share:.4f},{repeat},{Na},{Nb},{arm},{ps},{rs},"
            f"{k},{fa},{fb},{fb},{st['jaccard']:.6f},{kd:.6f},{st['inter']},{st['m_a']},{st['m_b']},"
            f"{st['m_ab']},{st['E']:.6f},{st['c']:.6f},{str(guard).lower()},{str(ok).lower()},"
            f"{prov['tm_lab']},{prov['ceiling']},{prov['parallel']}")


def consecutive(labels):
    """Pairs of labels that are calendar-consecutive."""
    def nxt(l):
        y, m = int(l[:4]), int(l[5:7])
        return f"{y + (m == 12)}-{(m % 12) + 1:02d}"
    s = set(labels)
    return [(l, nxt(l)) for l in sorted(labels) if nxt(l) in s]


def main(argv) -> int:
    import shap
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", choices=("lamda", "apigraph", "androzoo", "hypercube"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--months", type=int, default=0, help="limit to the first N usable months")
    ap.add_argument("--arms", default="2s,4,0")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--cfg", default="matched", choices=("matched", "S100"))
    ap.add_argument("--runs4", type=int, default=RUNS4, help="sampled runs per arm-4 variant (5)")
    a = ap.parse_args(argv)
    check_versions()
    arms = set(a.arms.split(","))
    conds = [c for c in a.conds.split(",") if c]
    ddir = data_dir(a.corpus, a.smoke)
    odir = out_dir(a.smoke)
    odir.mkdir(parents=True, exist_ok=True)
    M = months_of(ddir)
    if not str(M.get("tm_lab", "")).startswith(TM_LAB_COMMIT):
        raise SystemExit(f"the exported models carry tm-lab {M.get('tm_lab')}, expected {TM_LAB_COMMIT}")
    cfg = f"S{M['s_matched']}" if a.cfg == "matched" else "S100"
    S = int(cfg[1:])
    matched = S == M["s_matched"]
    pools = list(range(1, M["npool"] + 1))
    runs0 = min(RUNS0, M["nrefit"])
    series = [m for m in M["months"] if m["base"]]
    if a.months:
        series = series[: a.months]
    labels = [m["label"] for m in series]
    idx_of = {m["label"]: i for i, m in enumerate(M["months"])}
    arm4 = [m["label"] for m in series if m["arm4"]]
    if a.smoke:
        arm4 = arm4[:1]
    print(f"b2-corpora python arms  corpus {a.corpus}  cfg {cfg}  conditions {conds}  arms {sorted(arms)}  "
          f"months {len(labels)}  arm-4 months {arm4}  pools {pools}  arm-0 runs {runs0}  jobs {a.jobs}")
    print(f"shap {shap.__version__}  l1_reg {L1_REG}  numpy {np.__version__}  tm-lab {TM_LAB_COMMIT}  "
          f"one thread per worker")
    sys.stdout.flush()

    class AtomicCSV:
        """Collects a CSV and writes it through a temporary file, a rename and a .sha256 sidecar."""
        def __init__(self, path, cols):
            self.path, self.parts = path, [cols + "\n"]

        def write(self, text):
            self.parts.append(text)

        def close(self):
            data = "".join(self.parts).encode("utf-8")
            write_atomic(self.path, lambda fh: fh.write(data))

    def open_csv(name, cols):
        return AtomicCSV(odir / name, cols)

    units = ddir / "units" / "py"
    peaks = peak_table([ddir.parent, ddir.parent.parent / "b2-corpora", ddir.parent.parent / "b2-corpora-smoke"])
    need_of = lambda cls: max(PEAK_MB_DEFAULT[cls][a.corpus], peaks.get(f"{cls}-{a.corpus}", 0.0))  # noqa: E731
    classes = ({"py-2s"} if arms & {"2s", "timing"} else set()) | \
              ({"py-4a", "py-4f", "py-4c"} if "4" in arms else set()) | ({"py-0"} if "0" in arms else set())
    if classes:
        top = max(need_of(c) for c in classes)
        while (avail := mem_available_mb()) < MEM_FACTOR * top:
            print(f"memory: largest job class needs {top:.0f} MB x {MEM_FACTOR}, {avail:.0f} MB available; waiting")
            sys.stdout.flush()
            time.sleep(MEM_RECHECK_S)
        print(f"memory: {mem_available_mb():.0f} MB available; peaks used " +
              ", ".join(f"{c} {need_of(c):.0f} MB" for c in sorted(classes)))
    log = lambda msg: (print(msg), sys.stdout.flush())  # noqa: E731
    computed_rss = {}

    def note_rss(cls, res):
        for r in res:
            if not r.get("loaded"):
                computed_rss[cls] = max(computed_rss.get(cls, 0.0), float(r["rss"]))

    def month_line(r, arm, cfg_, S_, ps, run, prov, t_train=""):
        return (f"{a.corpus},{arm},{cfg_},{S_},{r['cond']},{r['N']},{r['share']:.4f},{r['repeat']},{ps},{run},"
                f"{r['label']},{r['auc']:.6f},{str(r['auc'] >= AUC_MIN).lower()},"
                f"{int((r['imp'] > stats.TINY).sum())},{r.get('m_exact', '')},{r['M_eff']:.1f},{NSAMPLES},"
                f"{r['l1']},{r['nwarn']},{r.get('failed_rows', 0)},{r['oor']},{r['secs']:.3f},{t_train},1,"
                f"{r.get('seconds_exact') or ''},"
                f"{r.get('threads_exact') or ''},{r.get('seconds_exact_1thread') or ''},{r['rss']:.0f},"
                f"{shap.__version__},{prov['tm_lab']},{prov['ceiling']},{prov['parallel']}\n")

    # ---- timing: arm 2s on the arm-4 months, one process, nothing else running in this script ----
    if "timing" in arms:
        for cond in conds:
            fh = open_csv(f"{a.corpus}-{cfg}-{cond}-timing-py.csv", "corpus,cfg,condition,pool_seed,month,"
                          "seconds_kernel_100,seconds_exact,threads_exact,seconds_exact_1thread,M_eff,"
                          "peak_rss_mb,shap")
            for p in pools:
                for l in arm4:
                    r = work_2s((ddir, cfg, cond, p, l, 1_000_000 + 1000 * p + idx_of[l],
                                 units / "timing" / cfg / cond / f"2s-pool{p}-{l}.npz"))
                    fh.write(f"{a.corpus},{cfg},{cond},{100 + p},{l},{r['secs']:.3f},{r['seconds_exact']},"
                             f"{r['threads_exact']},{r['seconds_exact_1thread']},{r['M_eff']:.1f},"
                             f"{r['rss']:.0f},{shap.__version__}\n")
                    print(f"timing {cfg} {cond} pool {100 + p} {l}: KernelExplainer at 100 {r['secs']:.2f}s, "
                          f"exact {r['seconds_exact_1thread']:.3f}s on one thread")
                    sys.stdout.flush()
            fh.close()

    with ProcessPoolExecutor(max_workers=a.jobs) as pool_exec:
        # ---- arm 2s ----
        if "2s" in arms:
            jobs = [(ddir, cfg, c, p, l, 1_000_000 + 1000 * p + idx_of[l], units / cfg / c / f"2s-pool{p}-{l}.npz")
                    for c in conds for p in pools for l in labels]
            log("arm 2s:")
            res = run_units(pool_exec, work_2s, jobs, lambda j: j[-1], lambda j: "py-2s", need_of, a.jobs, log)
            note_rss("py-2s", res)
            got = {(r["cond"], r["pool"], r["label"]): r for r in res}
            for c in conds:
                mon = open_csv(f"{a.corpus}-{cfg}-{c}-py-months-2s.csv", MONTH_COLS)
                for r in res:
                    if r["cond"] == c:
                        mon.write(month_line(r, "2s", cfg, S, 100 + r["pool"], "", TM_PROV))
                mon.close()
                pairs = open_csv(f"{a.corpus}-{cfg}-{c}-pairs-2s.csv", PAIR_COLS)
                for p in pools:
                    for la, lb in consecutive(labels):
                        ra, rb = got[(c, p, la)], got[(c, p, lb)]
                        g = ra["auc"] >= AUC_MIN and rb["auc"] >= AUC_MIN
                        for k in KS:
                            # the header's support guard is on the exact attribution of the same model-months
                            ex_ok = ra["m_exact"] >= k and rb["m_exact"] >= k
                            pairs.write(pair_line(a.corpus, cfg, S, matched, c, ra["share"], ra["repeat"],
                                                  ra["N"], rb["N"], "2s", 100 + p, "", k, la, lb, ra["imp"],
                                                  rb["imp"], g, TM_PROV, extra_ok=ex_ok) + "\n")
                pairs.close()
                rc = [r for r in res if r["cond"] == c]
                print(f"arm 2s {c}: {len(rc)} model-months, {np.mean([r['secs'] for r in rc]):.1f}s each "
                      f"(one thread per worker, {a.jobs} workers), peak {max(r['rss'] for r in rc):.0f} MB; "
                      f"rows out of range {sum(r['oor'] for r in rc)} of {100 * len(rc)}")
                for p in pools:
                    rp = [r for r in rc if r["pool"] == p]
                    js = [stats.pairstats(got[(c, p, la)]["imp"], got[(c, p, lb)]["imp"], 100)["jaccard"]
                          for la, lb in consecutive(labels)]
                    print(result_line(corpus=a.corpus, cfg=cfg, condition=c, arm="2s", pool_seed=100 + p, k=100,
                                      jaccard_mean=f"{nanmean(js):.4f}", pairs=len(js), units=len(rp),
                                      failed_rows=sum(int(r.get("failed_rows", 0)) for r in rp),
                                      units_all_failed=sum(1 for r in rp if r.get("all_failed")),
                                      loaded=sum(1 for r in rp if r.get("loaded")),
                                      seconds_mean=f"{np.mean([r['secs'] for r in rp]):.2f}",
                                      peak_mb=f"{max(r['rss'] for r in rp):.0f}", N=rp[0]["N"],
                                      share=f"{rp[0]['share']:.2f}", repeat=rp[0]["repeat"], l1_reg=L1_REG,
                                      shap=SHAP_VERSION, **TM_PROV))
            sys.stdout.flush()

        # ---- arm 4 ----
        if "4" in arms and arm4:
            jobs = []
            for c in conds:
                for p in pools:
                    for l in arm4:
                        for r in range(a.runs4):
                            base = 2_000_000 + 1000 * p + 10 * idx_of[l] + r
                            u = units / cfg / c
                            jobs += [(ddir, cfg, c, p, l, "4a", r, base, u / f"4-pool{p}-{l}-4a-r{r}.npz"),
                                     (ddir, cfg, c, p, l, "4f", r, base, u / f"4-pool{p}-{l}-4f-r{r}.npz"),
                                     (ddir, cfg, c, p, l, "4c", r, base + 500_000, u / f"4-pool{p}-{l}-4c-r{r}.npz")]
            # the expensive 4c jobs first, so the cheap ones fill the tail
            jobs.sort(key=lambda j: j[5] != "4c")
            log("arm 4:")
            res = run_units(pool_exec, work_4, jobs, lambda j: j[-1], lambda j: f"py-{j[5]}", need_of, a.jobs, log)
            for v in ("4a", "4f", "4c"):
                note_rss(f"py-{v}", [r for r in res if r["variant"] == v])
            for c in conds:
                rows = open_csv(f"{a.corpus}-{cfg}-{c}-arm4.csv", "corpus,cfg,S,condition,N,share,repeat,"
                                "pool_seed,month,measure,variant,k,value,lo,hi,n,tm_lab,ceiling,parallel")
                for p in pools:
                    model = model_of(ddir, cfg, c, p)
                    width = model["width"]
                    for l in arm4:
                        mj = month_json(ddir, cfg, c, p, l)
                        exact = np.fromfile(model_dir(ddir, cfg, c, p) / f"phi-{l}.bin", dtype="<f8")
                        exact = exact.reshape(-1, width)
                        exact_imp = np.asarray(mj["exact_importance"])
                        m_exact = int((exact_imp > stats.TINY).sum())
                        got = {v: sorted([r for r in res if r["cond"] == c and r["pool"] == p
                                          and r["label"] == l and r["variant"] == v], key=lambda r: r["run"])
                               for v in ("4a", "4f", "4c")}
                        runs = {v: np.stack([r["phi"] for r in got[v]]) for v in got}

                        def w(measure, variant, k, value, lo="", hi="", n=""):
                            fmt = (lambda x: f"{x:.6f}" if isinstance(x, float) else str(x))
                            rows.write(f"{a.corpus},{cfg},{S},{c},{model['N']},{model['share']:.4f},"
                                       f"{model['repeat']},{100 + p},{l},{measure},{variant},{k},{fmt(value)},"
                                       f"{fmt(lo)},{fmt(hi)},{n},{TM_PROV['tm_lab']},{TM_PROV['ceiling']},"
                                       f"{TM_PROV['parallel']}\n")

                        w("exact_support", "exact", "", float(m_exact))
                        for k in KS:
                            if m_exact < k:
                                # fewer than k non-zero exact features: the top-k set is partly ties at zero
                                for v in ("4a", "4f", "4c"):
                                    w("c_vs_exact", v, k, float("nan"), n=0)
                                    w("jaccard_vs_exact", v, k, float("nan"), n=0)
                                w("c_between", "4b", k, float("nan"), n=0)
                                w("jaccard_between", "4b", k, float("nan"), n=0)
                                continue
                            for v in ("4a", "4f", "4c"):
                                sts = [stats.pairstats(row_mean_abs(ph), exact_imp, k) for ph in runs[v]]
                                ok = [s for s in sts if s["support_ok"]]
                                w("c_vs_exact", v, k, nanmean([s["c"] for s in ok]), n=len(ok))
                                w("jaccard_vs_exact", v, k, nanmean([s["jaccard"] for s in ok]), n=len(ok))
                            imps = [row_mean_abs(ph) for ph in runs["4a"]]
                            sts = [stats.pairstats(imps[i], imps[j], k)
                                   for i, j in combinations(range(len(imps)), 2)]
                            ok = [s for s in sts if s["support_ok"]]
                            w("c_between", "4b", k, nanmean([s["c"] for s in ok]), n=len(ok))
                            w("jaccard_between", "4b", k, nanmean([s["jaccard"] for s in ok]), n=len(ok))
                        for v in ("4a", "4f", "4c"):
                            keep = ~np.isnan(runs[v]).any(axis=(0, 2))       # rows no run of v failed on
                            fv = sum(int(r.get("failed_rows", 0)) for r in got[v])
                            w("failed_rows", v, "", float(fv), n=len(got[v]))
                            if keep.any():
                                b2, var = stats.bias_variance_rows(runs[v][:, keep], exact[keep])
                                share, lo, hi, used = stats.bias_share(
                                    b2, var, test_rng(a.corpus, c, cfg, 100 + p, f"{l}-bias-{v}"))
                            else:
                                share = lo = hi = float("nan")
                                used = 0
                            w("bias_share", v, "", share, lo, hi, used)
                            print(result_line(corpus=a.corpus, cfg=cfg, condition=c, arm="4", variant=v,
                                              pool_seed=100 + p, month=l, bias_share=f"{share:.4f}",
                                              lo=f"{lo:.4f}", hi=f"{hi:.4f}", rows=used, runs=len(got[v]),
                                              seconds_mean=f"{nanmean([r['secs'] for r in got[v]]):.1f}",
                                              peak_mb=f"{max(r['rss'] for r in got[v]):.0f}", failed_rows=fv,
                                              N=model["N"],
                                              share=f"{model['share']:.2f}", repeat=model["repeat"],
                                              l1_reg=got[v][0]["l1"], shap=SHAP_VERSION, **TM_PROV))
                            w("budget_fraction", v, "", nanmean([r["frac"] for r in got[v]]), n=len(got[v]))
                            w("seconds_per_run", v, "", nanmean([r["secs"] for r in got[v]]), n=len(got[v]))
                            w("peak_rss_mb", v, "", float(max(r["rss"] for r in got[v])), n=len(got[v]))
                            w("rows_out_of_range", v, "", nanmean([r["oor"] for r in got[v]]), n=len(got[v]))
                            w("l1_reg", v, "", got[v][0]["l1"], n=len(got[v]))
                        keep = ~(np.isnan(runs["4a"]).any(axis=(0, 2)) | np.isnan(runs["4c"]).any(axis=(0, 2)))
                        if keep.any():
                            d, lo, hi = stats.budget_difference(runs["4a"][:, keep], runs["4c"][:, keep], exact[keep],
                                                                100, test_rng(a.corpus, c, cfg, 100 + p, f"{l}-budget"))
                        else:
                            d = lo = hi = float("nan")
                        w("budget_difference_c", "4c-4a", 100, d, lo, hi, int(keep.sum()))
                        print(result_line(corpus=a.corpus, cfg=cfg, condition=c, arm="4", variant="4c-4a",
                                          pool_seed=100 + p, month=l, k=100, budget_difference_c=f"{d:.4f}",
                                          lo=f"{lo:.4f}", hi=f"{hi:.4f}", N=model["N"],
                                          share=f"{model['share']:.2f}", repeat=model["repeat"],
                                          shap=SHAP_VERSION, **TM_PROV))
                rows.close()
                for v in ("4a", "4f", "4c"):
                    rc = [r for r in res if r["cond"] == c and r["variant"] == v]
                    print(f"arm 4 {c} {v}: {len(rc)} runs, {np.mean([r['secs'] for r in rc]):.1f}s each, "
                          f"peak {max(r['rss'] for r in rc):.0f} MB")
            sys.stdout.flush()

        # ---- arm 0 ----
        if "0" in arms:
            jobs = [(ddir, c, l, r, idx_of[l], units / "mlp" / c / f"0-{l}-r{r}.npz")
                    for c in conds for l in labels for r in range(runs0)]
            log("arm 0:")
            res = run_units(pool_exec, work_0, jobs, lambda j: j[-1], lambda j: "py-0", need_of, a.jobs, log)
            note_rss("py-0", res)
            got = {(r["cond"], r["label"], r["run"]): r for r in res}
            ok = lambda r: r["auc"] >= AUC_MIN  # noqa: E731
            for c in conds:
                mon = open_csv(f"{a.corpus}-mlp-{c}-py-months-0.csv", MONTH_COLS)
                for r in res:
                    if r["cond"] == c:
                        mon.write(month_line(r, "0", "mlp", 0, "", r["run"], NA_PROV, f"{r['t_train']:.3f}"))
                mon.close()
                pairs = open_csv(f"{a.corpus}-mlp-{c}-pairs-0.csv", PAIR_COLS)
                for k in KS:
                    for run in range(runs0):
                        for la, lb in consecutive(labels):
                            ra, rb = got[(c, la, run)], got[(c, lb, run)]
                            pairs.write(pair_line(a.corpus, "mlp", 0, "na", c, ra["share"], ra["repeat"],
                                                  ra["N"], rb["N"], "0", "", run, k, la, lb, ra["imp"],
                                                  rb["imp"], ok(ra) and ok(rb), NA_PROV) + "\n")
                    for l in labels:
                        for r1, r2 in combinations(range(runs0), 2):
                            ra, rb = got[(c, l, r1)], got[(c, l, r2)]
                            pairs.write(pair_line(a.corpus, "mlp", 0, "na", c, ra["share"], ra["repeat"],
                                                  ra["N"], rb["N"], "0w", "", f"{r1}-{r2}", k, l, l, ra["imp"],
                                                  rb["imp"], ok(ra) and ok(rb), NA_PROV) + "\n")
                pairs.close()
                rc = [r for r in res if r["cond"] == c]
                print(f"arm 0 {c}: {len(rc)} MLP runs, train {np.mean([r['t_train'] for r in rc]):.1f}s, "
                      f"explain {np.mean([r['secs'] for r in rc]):.1f}s each (one thread per worker), "
                      f"peak {max(r['rss'] for r in rc):.0f} MB; rows out of range "
                      f"{sum(r['oor'] for r in rc)} of {100 * len(rc)}")
                js = [stats.pairstats(got[(c, la, run)]["imp"], got[(c, lb, run)]["imp"], 100)["jaccard"]
                      for run in range(runs0) for la, lb in consecutive(labels)]
                print(result_line(corpus=a.corpus, cfg="mlp", condition=c, arm="0", k=100,
                                  jaccard_mean=f"{nanmean(js):.4f}", pairs=len(js), units=len(rc),
                                  failed_rows=sum(int(r.get("failed_rows", 0)) for r in rc),
                                  units_all_failed=sum(1 for r in rc if r.get("all_failed")),
                                  loaded=sum(1 for r in rc if r.get("loaded")),
                                  train_seconds_mean=f"{np.mean([r['t_train'] for r in rc]):.1f}",
                                  explain_seconds_mean=f"{np.mean([r['secs'] for r in rc]):.1f}",
                                  peak_mb=f"{max(r['rss'] for r in rc):.0f}",
                                  N=f"{min(r['N'] for r in rc)}-{max(r['N'] for r in rc)}",
                                  share=f"{rc[0]['share']:.2f}", repeat=rc[0]["repeat"], l1_reg=L1_REG,
                                  shap=SHAP_VERSION, **NA_PROV))
    if computed_rss:
        with open(ddir.parent / "peak-ram.jsonl", "a", encoding="utf-8") as fh:
            for cls, mb in sorted(computed_rss.items()):
                fh.write(json.dumps({"class": f"{cls}-{a.corpus}", "peak_mb": mb, "smoke": a.smoke,
                                     "jobs": a.jobs, "date": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        print("peak RSS per job class (MB): " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(computed_rss.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
