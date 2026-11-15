"""Statistics shared by arms.py and decide.py for b2-corpora.

Everything that turns attributions into the quantities named in run.jl's header lives here, so the
Python arms and the decision step use one implementation, and the self-test in decide.py exercises the
same functions the run uses.

  chance level   two top-k sets drawn uniformly from supports A and B share a feature of A n B with
                 probability (k/|A|)(k/|B|), so E = k^2 |A n B| / (|A| |B|); c = (|inter| - E) / (k - E)
  bootstrap      moving blocks of length 3 over the ordered anchors of one year, 2,000 resamples,
                 percentile 95% interval of the mean
  year rule      a year decides only with at least 6 usable anchors; a condition holds for a corpus
                 when it holds in more than half of its deciding years
  bias share     per explained row, bias^2 = |mean over runs - exact|^2 - (across-run variance)/R and
                 variance = across-run variance (summed over features, ddof 1); the row's share is
                 bias^2 / (bias^2 + variance), and the statistic is the mean share over rows, rows
                 resampled for the interval
"""

from __future__ import annotations

import numpy as np

TINY = 1e-12
BLOCK = 3
RESAMPLES = 2000
MIN_ANCHORS = 6
WITHIN = 0.05


# --- top-k overlap and its chance correction ----------------------------------------------------

def topk(imp: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest values; ties at the boundary go to the lower index. Used only on
    attributions computed in Python or compared within Python, never against a Julia top-k set."""
    k = min(k, imp.size)
    order = np.lexsort((np.arange(imp.size), -imp))
    return order[:k]


def chance_E(k: int, ma: int, mb: int, mab: int) -> float:
    if ma <= 0 or mb <= 0:
        return float("nan")
    return k * k * mab / (ma * mb)


def corrected(inter: float, k: int, E: float) -> float:
    if not np.isfinite(E) or k - E < 1e-9:
        return float("nan")
    return (inter - E) / (k - E)


def pairstats(a: np.ndarray, b: np.ndarray, k: int) -> dict:
    sa, sb = a > TINY, b > TINY
    ma, mb, mab = int(sa.sum()), int(sb.sum()), int((sa & sb).sum())
    ta, tb = set(topk(a, k).tolist()), set(topk(b, k).tolist())
    inter = len(ta & tb)
    jac = 1 - inter / len(ta | tb)
    E = chance_E(k, ma, mb, mab)
    return dict(jaccard=jac, inter=inter, m_a=ma, m_b=mb, m_ab=mab, E=E,
                c=corrected(inter, k, E), support_ok=(ma >= k and mb >= k))


# --- moving-block bootstrap ---------------------------------------------------------------------

def block_bootstrap(values, rng: np.random.Generator, block: int = BLOCK,
                    resamples: int = RESAMPLES) -> tuple:
    """Percentile 95% interval of the mean of an ordered series, by a moving-block bootstrap."""
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    b = min(block, n)
    starts = np.arange(n - b + 1)
    nblocks = -(-n // b)
    picks = rng.choice(starts, size=(resamples, nblocks))
    idx = (picks[:, :, None] + np.arange(b)[None, None, :]).reshape(resamples, -1)[:, :n]
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --- the year rule ------------------------------------------------------------------------------

def by_year(series: dict) -> dict:
    """{anchor 'YYYY-MM': value} -> {year: [values in calendar order]}."""
    out: dict = {}
    for a in sorted(series):
        out.setdefault(int(a[:4]), []).append(series[a])
    return out


def joint(*series: dict) -> dict:
    """Anchors present in every series, as tuples of values."""
    keys = set(series[0])
    for s in series[1:]:
        keys &= set(s)
    return {a: tuple(s[a] for s in series) for a in sorted(keys)}


def decide_years(series: dict, condition, rng: np.random.Generator) -> dict:
    """Apply `condition(mean, lo, hi) -> bool` per year to an anchored scalar series.
    Returns per-year rows and the corpus verdict: True / False by majority of deciding years,
    None when no year decides."""
    rows = []
    for year, vals in sorted(by_year(series).items()):
        if len(vals) < MIN_ANCHORS:
            rows.append(dict(year=year, n=len(vals), decides=False))
            continue
        m, lo, hi = block_bootstrap(vals, rng)
        rows.append(dict(year=year, n=len(vals), decides=True, mean=m, lo=lo, hi=hi,
                         holds=bool(condition(m, lo, hi))))
    deciding = [r for r in rows if r["decides"]]
    verdict = None if not deciding else sum(r["holds"] for r in deciding) * 2 > len(deciding)
    return dict(years=rows, verdict=verdict, deciding=len(deciding))


def decide_years_multi(series: dict, conditions, rng: np.random.Generator) -> dict:
    """As decide_years for a series of value tuples with one condition per component; a year holds
    when every component's condition holds on its own interval. Resampling uses the same rng
    stream per component, so components are bootstrapped with separate block draws."""
    rows = []
    for year, vals in sorted(by_year(series).items()):
        if len(vals) < MIN_ANCHORS:
            rows.append(dict(year=year, n=len(vals), decides=False))
            continue
        parts = []
        for j, cond in enumerate(conditions):
            m, lo, hi = block_bootstrap([v[j] for v in vals], rng)
            parts.append(dict(mean=m, lo=lo, hi=hi, holds=bool(cond(m, lo, hi))))
        rows.append(dict(year=year, n=len(vals), decides=True, parts=parts,
                         holds=all(p["holds"] for p in parts)))
    deciding = [r for r in rows if r["decides"]]
    verdict = None if not deciding else sum(r["holds"] for r in deciding) * 2 > len(deciding)
    return dict(years=rows, verdict=verdict, deciding=len(deciding))


def within(lo: float, hi: float, tol: float = WITHIN) -> bool:
    return -tol <= lo and hi <= tol


def all_seeds(verdicts) -> object:
    """A condition holds for a corpus only when it holds for every pool seed. None if any seed has
    no deciding year."""
    vs = list(verdicts)
    if any(v is None for v in vs):
        return None
    return all(vs)


# --- arm 4: bias in attribution space and the budget comparison ---------------------------------

def bias_variance_rows(runs: np.ndarray, exact: np.ndarray) -> tuple:
    """runs: R x n x M sampled phi; exact: n x M. Per row: bias^2 (unbiased for |E[phi] - exact|^2)
    and the single-run variance, both summed over features. Each row is first divided by its largest
    absolute value across the runs and the exact values. The share below is invariant to that scale,
    and KernelExplainer at small budgets returns rows with values near 1e77, whose squares would
    otherwise overflow."""
    R = runs.shape[0]
    s = np.maximum(np.abs(runs).max(axis=(0, 2)), np.abs(exact).max(axis=1))
    s = np.where(s > 0, s, 1.0)
    runs = runs / s[None, :, None]
    exact = exact / s[:, None]
    mean = runs.mean(axis=0)
    var = runs.var(axis=0, ddof=1).sum(axis=1)
    bias2 = ((mean - exact) ** 2).sum(axis=1) - var / R
    return bias2, var


def row_shares(bias2: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Per explained row, bias^2 / (bias^2 + variance); NaN for a row the estimator reproduces
    exactly in every run (both terms zero)."""
    den = bias2 + var
    out = np.full(bias2.shape, np.nan)
    ok = den > 0
    out[ok] = bias2[ok] / den[ok]
    return out


def bias_share(bias2: np.ndarray, var: np.ndarray, rng: np.random.Generator,
               resamples: int = RESAMPLES) -> tuple:
    """Mean over explained rows of the per-row bias share, with a percentile interval over rows
    resampled. Returns (share, lo, hi, rows used)."""
    sh = row_shares(bias2, var)
    sh = sh[np.isfinite(sh)]
    n = sh.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    idx = rng.integers(0, n, size=(resamples, n))
    boots = sh[idx].mean(axis=1)
    return (float(sh.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), n)


def _boot_topk(abs_phi: np.ndarray, W: np.ndarray, k: int) -> tuple:
    """abs_phi: n x M; W: B x n resampling weights. Returns per-resample top-k index sets (B x k)
    and supports (B)."""
    imp = W @ abs_phi / W.sum(axis=1, keepdims=True)
    sup = (imp > TINY)
    kk = min(k, imp.shape[1])
    order = np.argpartition(-imp, kk - 1, axis=1)[:, :kk]
    return order, sup


def _boot_c(ta, sa, tb, sb, k):
    B = ta.shape[0]
    out = np.empty(B)
    for i in range(B):
        inter = np.intersect1d(ta[i], tb[i], assume_unique=True).size
        ma, mb = int(sa[i].sum()), int(sb[i].sum())
        mab = int((sa[i] & sb[i]).sum())
        out[i] = corrected(inter, k, chance_E(k, ma, mb, mab)) if (ma >= k and mb >= k) else np.nan
    return out


def budget_difference(runs_small: np.ndarray, runs_big: np.ndarray, exact: np.ndarray, k: int,
                      rng: np.random.Generator, resamples: int = RESAMPLES) -> tuple:
    """Interval of mean c(big run vs exact) - mean c(small run vs exact), explained rows resampled.
    runs_*: R x n x M signed phi; exact: n x M."""
    n = exact.shape[0]
    idx = rng.integers(0, n, size=(resamples, n))
    W = np.zeros((resamples, n))
    np.add.at(W, (np.repeat(np.arange(resamples), n), idx.ravel()), 1.0)
    W = np.vstack([np.ones((1, n)), W])          # row 0 is the point estimate
    te, se = _boot_topk(np.abs(exact), W, k)

    def mean_c(runs):
        cs = [_boot_c(*_boot_topk(np.abs(r), W, k), te, se, k) for r in runs]
        return np.nanmean(np.vstack(cs), axis=0)

    d = mean_c(runs_big) - mean_c(runs_small)
    boots = d[1:]
    return float(d[0]), float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))
