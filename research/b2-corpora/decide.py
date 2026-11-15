"""Statistics and conditions of b2-corpora, applied to the outputs of run.jl and arms.py.

  python -X utf8 research/b2-corpora/decide.py <lamda|apigraph|androzoo|hypercube> [--smoke]
  python -X utf8 research/b2-corpora/decide.py --selftest

The conditions are those fixed in run.jl's header, at k=100:
  F1  c(2) <= 0.30 and c(2r) - c(2) >= 0.30, both by the interval bound (upper bound of c(2), lower
      bound of the difference), per year, majority of deciding years, every pool seed.
  F2  lower bound of the bias share of 4a > 0.25; per arm-4 month, majority of months, every pool seed.
  F3  the interval of c(4c vs exact) - c(4a vs exact) inside [-0.05, 0.05]; as F2.
  Unresolved  c(1) - c(2) and c(1) - c(3) both with intervals inside [-0.05, 0.05]; as F1.
  P1  F2 does not hold.
  P2  Jaccard distance J(2s) - J(2) >= 0.30 by the lower bound; as F1.
  P3  c(2) - c(3) > 0 by the lower bound; as F1.
Everything is evaluated separately for each training condition (B0, U, O), and every condition is
reported. A pair enters only with guard_ok (AUC >= 0.6 on both model-months) and support_ok (m >= k on
both sides; for arm 2s also the exact support of the same model-months). Arms with several seeds per
anchor (3, 1, 3n, 1n, 0, 0w) enter as the mean over seeds. The decomposition 3-2, 2s-2 and 0-2s is
taken over the months both arms have, as the conditions are. Every RESULT line carries tm-lab, ceiling
policy and training parallel mode, and inputs from another tm-lab commit are refused.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stats  # noqa: E402

K = 100
SEED = 20261112
CONDS = ("B0", "U", "O")
PROV = dict(tm_lab="43dba5f", ceiling="LiteralCapped", parallel="none")
# malware share and repeat factor of every training set under each condition (B0 is exactly 1:9, so O
# repeats each minority row 9 times and draws no extra rows)
COND_PROV = {"B0": (0.10, 1), "U": (0.50, 1), "O": (0.50, 9)}


def test_rng(corpus: str, cond: str, cfg: str, pool: str, test: str) -> np.random.Generator:
    """One bootstrap stream per test, seeded by a stable hash of what the test is, so an interval does
    not depend on which other tests ran before it or on which files were present."""
    import zlib
    return np.random.default_rng([SEED, zlib.crc32(f"{corpus}|{cond}|{cfg}|{pool}|{test}".encode("utf-8"))])


def n_range(rows: list) -> str:
    ns = [int(r[k]) for r in rows for k in ("N_a", "N_b") if r.get(k, "") not in ("", None)]
    return f"{min(ns)}-{max(ns)}" if ns else "-"

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


# --- the conditions, shared by the run and the self-test ----------------------------------------

def cond_F1(c2: dict, c2r: dict, rng) -> dict:
    s = {a: (v[0], v[1] - v[0]) for a, v in stats.joint(c2, c2r).items()}
    return stats.decide_years_multi(s, (lambda m, lo, hi: hi <= 0.30,
                                        lambda m, lo, hi: lo >= 0.30), rng)


def cond_unresolved(c1: dict, c2: dict, c3: dict, rng) -> dict:
    s = {a: (v[0] - v[1], v[0] - v[2]) for a, v in stats.joint(c1, c2, c3).items()}
    return stats.decide_years_multi(s, (lambda m, lo, hi: stats.within(lo, hi),
                                        lambda m, lo, hi: stats.within(lo, hi)), rng)


def cond_P2(j2s: dict, j2: dict, rng) -> dict:
    s = {a: v[0] - v[1] for a, v in stats.joint(j2s, j2).items()}
    return stats.decide_years(s, lambda m, lo, hi: lo >= 0.30, rng)


def cond_P3(c2: dict, c3: dict, rng) -> dict:
    s = {a: v[0] - v[1] for a, v in stats.joint(c2, c3).items()}
    return stats.decide_years(s, lambda m, lo, hi: lo > 0.0, rng)


def cond_F2(months: list) -> object:
    """months: [(lo, hi)] of the 4a bias share, one per arm-4 month."""
    if not months:
        return None
    return sum(lo > 0.25 for lo, _ in months) * 2 > len(months)


def cond_F3(months: list) -> object:
    """months: [(lo, hi)] of c(4c) - c(4a), one per arm-4 month."""
    if not months:
        return None
    return sum(stats.within(lo, hi) for lo, hi in months) * 2 > len(months)


# --- reading ------------------------------------------------------------------------------------

UNVERIFIED: list = []


def verified(path: Path) -> bool:
    """The file and its .sha256 sidecar exist and agree."""
    import hashlib
    side = path.with_name(path.name + ".sha256")
    if not (path.is_file() and side.is_file()):
        return False
    return side.read_text(encoding="utf-8").split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list:
    """Rows of a results file, read only when its checksum verifies; anything else is listed and skipped."""
    if not path.is_file():
        return []
    if not verified(path):
        UNVERIFIED.append(path.name)
        return []
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def series(pairs: list, cfg: str, cond: str, arm: str, pool: str = "", metric: str = "c",
           k: int = K) -> dict:
    """{anchor: value} for usable pairs of one configuration and training condition, the mean over
    seeds where an anchor has several."""
    acc = defaultdict(list)
    for r in pairs:
        if r["cfg"] != cfg or r["condition"] != cond or r["arm"] != arm or int(r["k"]) != k:
            continue
        if pool and r["pool_seed"] != pool:
            continue
        if r["guard_ok"] != "true" or r["support_ok"] != "true":
            continue
        v = float(r[metric])
        if np.isfinite(v):
            acc[r["anchor"]].append(v)
    return {a: float(np.mean(v)) for a, v in acc.items()}


def check_c(pairs: list) -> int:
    """Recompute E and c from the stored ingredients; return the number of rows that disagree."""
    bad = 0
    for r in pairs:
        k = int(r["k"])
        E = stats.chance_E(k, int(r["m_a"]), int(r["m_b"]), int(r["m_ab"]))
        c = stats.corrected(int(r["inter"]), k, E)
        stored = float(r["c"])
        if np.isfinite(c) or np.isfinite(stored):
            if not (np.isfinite(c) and np.isfinite(stored) and abs(c - stored) < 1e-5):
                bad += 1
    return bad


def fmt_verdict(v) -> str:
    return "no deciding year" if v is None else ("HOLDS" if v else "does not hold")


def year_lines(res: dict) -> list:
    out = []
    for y in res["years"]:
        if not y["decides"]:
            out.append(f"      {y['year']}: {y['n']} anchors, does not decide")
        elif "parts" in y:
            p = "; ".join(f"{q['mean']:+.3f} [{q['lo']:+.3f}, {q['hi']:+.3f}]" for q in y["parts"])
            out.append(f"      {y['year']}: {y['n']} anchors  {p}  -> {'holds' if y['holds'] else 'no'}")
        else:
            out.append(f"      {y['year']}: {y['n']} anchors  {y['mean']:+.3f} "
                       f"[{y['lo']:+.3f}, {y['hi']:+.3f}]  -> {'holds' if y['holds'] else 'no'}")
    return out


# --- EXPLORATORY: the drift-vs-lag curve, from the saved attributions (no pass/fail) -----------------

LAGS = (1, 2, 3, 6, 12)


def _month_index(label: str) -> int:
    return int(label[:4]) * 12 + int(label[5:7]) - 1


def exact_vectors(ddir: Path, cfg: str, cond: str, pool: int) -> dict:
    """{month: arm-2 exact importance} of one fixed model, from verified month units only."""
    out = {}
    udir = ddir / "units" / cfg / cond
    if not udir.is_dir():
        return out
    for f in sorted(udir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].json")):
        b = f.with_name(f.stem + ".imp.bin")
        if not (verified(f) and verified(b)):
            UNVERIFIED.append(str(f.relative_to(ddir)))
            continue
        import json
        u = json.loads(f.read_text(encoding="utf-8"))
        key = f"2:{pool}"
        if key not in u["keys"]:
            continue
        j, w = u["keys"].index(key), int(u["width"])
        raw = np.frombuffer(b.read_bytes(), dtype="<f8")
        out[u["month"]] = raw[j * w:(j + 1) * w].copy()
    return out


def sampled_vectors(ddir: Path, cfg: str, cond: str, pool: int) -> dict:
    """{month: arm-2s KernelExplainer importance} of one fixed model, from verified arms.py units only."""
    out = {}
    udir = ddir / "units" / "py" / cfg / cond
    if not udir.is_dir():
        return out
    for f in sorted(udir.glob(f"2s-pool{pool}-*.npz")):
        if not verified(f):
            UNVERIFIED.append(str(f.relative_to(ddir)))
            continue
        with np.load(f, allow_pickle=False) as z:
            out[f.stem.split("-", 2)[2]] = z["imp"].copy()
    return out


def lag_curve(corpus: str, ddir: Path, pairs: list, jm: list, cfgs: list, matched: str, conds: list,
              pools: list, p, ptxt: str) -> None:
    p("\nEXPLORATORY (no pass/fail): drift against lag, J and c at k=100 between months t and t+L, with the arm-2r "
      "same-month floor; pairs may span calendar gaps (missing = months absent in between)")
    p(f"  {'cfg':6} {'cnd':3} {'arm':3} {'pool':4} {'L':>3} {'n':>4} {'missing':>7} {'J':>6} {'J lo':>6} {'J hi':>6} "
      f"{'c':>6} {'c lo':>6} {'c hi':>6} | {'2r J':>6} {'2r lo':>6} {'2r hi':>6}")
    guard = {(r["cfg"], r["condition"], r["seed"], r["month"]): r["guard_ok"] == "true"
             for r in jm if r["model"] == "pool"}
    for cond in conds:
        for cfg in cfgs:
            for pool in pools:
                pi = int(pool) - 100
                ex = exact_vectors(ddir, cfg, cond, pi)
                arms_ = [("2", ex)]
                if cfg == matched:
                    arms_.append(("2s", sampled_vectors(ddir, cfg, cond, pi)))
                f2r = series(pairs, cfg, cond, "2r", pool, "jaccard")
                fm, flo, fhi = (stats.block_bootstrap([f2r[x] for x in sorted(f2r)],
                                                      test_rng(corpus, cond, cfg, pool, "lag-floor-2r"))
                                if f2r else (float("nan"),) * 3)
                for arm, vec in arms_:
                    months = sorted(vec, key=_month_index)
                    for L in LAGS:
                        js, cs, missing = [], [], 0
                        for a in months:
                            b = next((m for m in months if _month_index(m) == _month_index(a) + L), None)
                            if b is None:
                                continue
                            if not (guard.get((cfg, cond, pool, a), False) and guard.get((cfg, cond, pool, b), False)):
                                continue
                            st = stats.pairstats(vec[a], vec[b], K)
                            ok = st["support_ok"]
                            if arm == "2s":   # the support guard is on the exact attribution as well
                                ok = ok and a in ex and b in ex and (ex[a] > stats.TINY).sum() >= K \
                                    and (ex[b] > stats.TINY).sum() >= K
                            if not ok:
                                continue
                            js.append(st["jaccard"])
                            cs.append(st["c"])
                            missing += sum(1 for t in range(_month_index(a) + 1, _month_index(b))
                                           if not any(_month_index(m) == t for m in months))
                        if not js:
                            continue
                        mj, lj, hj = stats.block_bootstrap(js, test_rng(corpus, cond, cfg, pool, f"lag-{arm}-L{L}-J"))
                        mc, lc, hc = stats.block_bootstrap(cs, test_rng(corpus, cond, cfg, pool, f"lag-{arm}-L{L}-c"))
                        p(f"  {cfg:6} {cond:3} {arm:3} {pool:4} {L:3d} {len(js):4d} {missing:7d} {mj:6.3f} {lj:6.3f} "
                          f"{hj:6.3f} {mc:6.3f} {lc:6.3f} {hc:6.3f} | {fm:6.3f} {flo:6.3f} {fhi:6.3f}")
                        print(f"RESULT corpus={corpus} cfg={cfg} condition={cond} exploratory=lag arm={arm} "
                              f"pool_seed={pool} k={K} L={L} n={len(js)} missing_months={missing} "
                              f"jaccard={mj:.4f} jaccard_lo={lj:.4f} jaccard_hi={hj:.4f} c={mc:.4f} c_lo={lc:.4f} "
                              f"c_hi={hc:.4f} floor_2r={fm:.4f} floor_2r_lo={flo:.4f} floor_2r_hi={fhi:.4f} "
                              f"share={COND_PROV[cond][0]:.2f} repeat={COND_PROV[cond][1]} {ptxt}", file=sys.stderr)


def run(corpus: str, smoke: bool) -> int:
    d = HERE / "smoke" if smoke else HERE
    jp = read_csv(d / f"{corpus}-pairs.csv")
    many = lambda pattern: [r for f in sorted(d.glob(pattern)) for r in read_csv(f)]  # noqa: E731
    pp = many(f"{corpus}-*-pairs-2s.csv") + many(f"{corpus}-mlp-*-pairs-0.csv")
    a4 = many(f"{corpus}-*-arm4.csv")
    jm = read_csv(d / f"{corpus}-months.csv")
    pm = many(f"{corpus}-*-py-months-2s.csv") + many(f"{corpus}-mlp-*-py-months-0.csv")
    tp = many(f"{corpus}-*-timing-py.csv")
    planted = read_csv(d / f"{corpus}-planted.csv")
    if not jp:
        print(f"no {corpus}-pairs.csv in {d}")
        return 1
    pairs = jp + pp
    out = []
    p = out.append
    bad = check_c(pairs)
    p(f"b2-corpora decision  corpus {corpus}  k={K}  pairs {len(jp)} julia + {len(pp)} python  "
      f"c recomputed: {bad} disagree")
    p(f"input files read only when their .sha256 verifies; unverified and skipped: "
      f"{', '.join(UNVERIFIED) if UNVERIFIED else 'none'}")
    cfgs = sorted({r["cfg"] for r in jp}, key=lambda s: int(s[1:]))
    matched = next(r["cfg"] for r in jp if r["matched"] == "true")
    pools = sorted({r["pool_seed"] for r in jp if r["arm"] == "2"})
    conds = [c for c in CONDS if any(r["condition"] == c for r in jp)]
    labs = {r["tm_lab"] for r in jp} | {r["tm_lab"] for r in pp if r["tm_lab"] != "na"}
    if labs != {PROV["tm_lab"]}:
        print(f"tm-lab commits in the inputs: {sorted(labs)}, expected {PROV['tm_lab']}")
        return 1
    p(f"configurations {cfgs} (matched {matched})  pool seeds {pools}  training conditions {conds}")
    p(f"tm-lab {PROV['tm_lab']}  ceiling {PROV['ceiling']}  training parallel :{PROV['parallel']}  "
      f"every condition reported")
    ptxt = " ".join(f"{k}={v}" for k, v in PROV.items())

    arms = ("2", "2r", "P", "5", "3", "1", "3n", "1n", "2s", "0", "0w")
    verdicts = {}
    for cond in conds:
        p(f"\n==== training condition {cond} ====")
        # --- descriptive: every arm over the series, block bootstrap over the ordered anchors ---
        p("Arm means over the series (usable pairs; 95% moving-block interval, block 3, 2,000 resamples)")
        p(f"  {'cfg':6} {'arm':4} {'pool':5} {'k':>5} {'J':>7} {'J lo':>7} {'J hi':>7} {'c':>7} "
          f"{'c lo':>7} {'c hi':>7} {'n':>4}")
        for cfg in cfgs + ["mlp"]:
            for arm in arms:
                for pool in (pools if arm in ("2", "2r", "P", "5", "2s") else [""]):
                    for k in (100, 1000):
                        sj = series(pairs, cfg, cond, arm, pool, "jaccard", k)
                        if not sj:
                            continue
                        sc = series(pairs, cfg, cond, arm, pool, "c", k)
                        mj, lj, hj = stats.block_bootstrap(
                            [sj[a] for a in sorted(sj)], test_rng(corpus, cond, cfg, pool, f"desc-{arm}-J-k{k}"))
                        mc, lc, hc = (stats.block_bootstrap(
                            [sc[a] for a in sorted(sc)], test_rng(corpus, cond, cfg, pool, f"desc-{arm}-c-k{k}"))
                            if sc else (float("nan"),) * 3)
                        rows_ = [r for r in pairs if r["cfg"] == cfg and r["condition"] == cond
                                 and r["arm"] == arm and int(r["k"]) == k and (not pool or r["pool_seed"] == pool)]
                        sh_, rp_ = COND_PROV.get(cond, (float("nan"), 0))
                        p(f"  {cfg:6} {arm:4} {pool or '-':5} {k:5d} {mj:7.3f} {lj:7.3f} {hj:7.3f} "
                          f"{mc:7.3f} {lc:7.3f} {hc:7.3f} {len(sj):4d}")
                        print(f"RESULT corpus={corpus} cfg={cfg} condition={cond} arm={arm} "
                              f"pool_seed={pool or '-'} k={k} jaccard={mj:.4f} jaccard_lo={lj:.4f} "
                              f"jaccard_hi={hj:.4f} c={mc:.4f} c_lo={lc:.4f} c_hi={hc:.4f} n={len(sj)} "
                              f"N={n_range(rows_)} share={sh_:.2f} repeat={rp_} {ptxt}", file=sys.stderr)

        p("Decomposition in Jaccard distance at k=100, matched configuration: each difference over the "
          "months both arms have (joint anchors), with its interval")
        for pool in pools:
            J2 = series(pairs, matched, cond, "2", pool, "jaccard")
            J3 = series(pairs, matched, cond, "3", "", "jaccard")
            J2s = series(pairs, matched, cond, "2s", pool, "jaccard")
            J0 = series(pairs, "mlp", cond, "0", "", "jaccard")
            parts = []
            for name, (A, B) in (("refitting 3-2", (J3, J2)), ("estimator 2s-2", (J2s, J2)),
                                 ("protocol 0-2s", (J0, J2s))):
                jt = stats.joint(A, B)
                dif = [v[0] - v[1] for _, v in sorted(jt.items())]
                m, lo, hi = stats.block_bootstrap(dif, test_rng(corpus, cond, matched, pool, f"decomp-{name}"))
                parts.append(f"{name} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] n={len(dif)}")
                print(f"RESULT corpus={corpus} cfg={matched} condition={cond} pool_seed={pool} k={K} "
                      f"difference={name.split()[1]} jaccard={m:.4f} lo={lo:.4f} hi={hi:.4f} n={len(dif)} "
                      f"share={COND_PROV[cond][0]:.2f} repeat={COND_PROV[cond][1]} {ptxt}", file=sys.stderr)
            m2 = np.mean(list(J2.values())) if J2 else float("nan")
            p(f"  pool {pool}: data (2) {m2:.3f} n={len(J2)}; " + "; ".join(parts))

        # --- conditions ---
        p("Conditions at k=100, per pool seed; a corpus verdict needs every pool seed")
        for cfg in cfgs:
            f1, un, p3, p2 = [], [], [], []
            for pool in pools:
                c2 = series(pairs, cfg, cond, "2", pool)
                c2r = series(pairs, cfg, cond, "2r", pool)
                c3 = series(pairs, cfg, cond, "3")
                c1 = series(pairs, cfg, cond, "1")
                r = cond_F1(c2, c2r, test_rng(corpus, cond, cfg, pool, "F1"))
                f1.append(r["verdict"])
                p(f"  {cfg} pool {pool}  F1: {fmt_verdict(r['verdict'])} ({r['deciding']} deciding years)"
                  f"   parts: c(2) <= 0.30 upper bound; c(2r)-c(2) >= 0.30 lower bound")
                out.extend(year_lines(r))
                r = cond_unresolved(c1, c2, c3, test_rng(corpus, cond, cfg, pool, "Unresolved"))
                un.append(r["verdict"])
                p(f"  {cfg} pool {pool}  Unresolved: {fmt_verdict(r['verdict'])} "
                  f"({r['deciding']} deciding years)   parts: c(1)-c(2); c(1)-c(3)")
                out.extend(year_lines(r))
                r = cond_P3(c2, c3, test_rng(corpus, cond, cfg, pool, "P3"))
                p3.append(r["verdict"])
                p(f"  {cfg} pool {pool}  P3 c(2)-c(3) > 0: {fmt_verdict(r['verdict'])} "
                  f"({r['deciding']} deciding years)")
                out.extend(year_lines(r))
                if cfg == matched:
                    r = cond_P2(series(pairs, cfg, cond, "2s", pool, "jaccard"),
                                series(pairs, cfg, cond, "2", pool, "jaccard"), test_rng(corpus, cond, cfg, pool, "P2"))
                    p2.append(r["verdict"])
                    p(f"  {cfg} pool {pool}  P2 J(2s)-J(2) >= 0.30: {fmt_verdict(r['verdict'])} "
                      f"({r['deciding']} deciding years)")
                    out.extend(year_lines(r))
            verdicts[(cond, cfg, "F1")] = stats.all_seeds(f1)
            verdicts[(cond, cfg, "Unresolved")] = stats.all_seeds(un)
            verdicts[(cond, cfg, "P3")] = stats.all_seeds(p3)
            if cfg == matched:
                verdicts[(cond, cfg, "P2")] = stats.all_seeds(p2)

        f2, f3 = [], []
        for pool in pools:
            rows = [r for r in a4 if r["pool_seed"] == pool and r["condition"] == cond and r["cfg"] == matched]
            bs = [(float(r["lo"]), float(r["hi"])) for r in rows
                  if r["measure"] == "bias_share" and r["variant"] == "4a"]
            bd = [(float(r["lo"]), float(r["hi"])) for r in rows if r["measure"] == "budget_difference_c"]
            f2.append(cond_F2(bs))
            f3.append(cond_F3(bd))
            for r in rows:
                if r["measure"] in ("bias_share", "budget_difference_c", "rows_out_of_range"):
                    lohi = (f" [{float(r['lo']):.3f}, {float(r['hi']):.3f}]" if r["lo"] else "")
                    p(f"  {matched} pool {pool} {r['month']} {r['measure']} {r['variant']}: "
                      f"{float(r['value']):.3f}{lohi}")
            p(f"  {matched} pool {pool}  F2: {fmt_verdict(cond_F2(bs))} ({len(bs)} months)   "
              f"F3: {fmt_verdict(cond_F3(bd))} ({len(bd)} months)")
        verdicts[(cond, matched, "F2")] = stats.all_seeds(f2)
        verdicts[(cond, matched, "F3")] = stats.all_seeds(f3)
        f2v = verdicts[(cond, matched, "F2")]
        verdicts[(cond, matched, "P1")] = None if f2v is None else not f2v

        # --- positive control P (null player): (a) planted features carry exactly zero attribution;
        # (b) with a planted support of at least 100, J(P) >= 1/3. J(P) minus the arm-2r floor beside. ---
        p("Positive control P (top-20 exact features set to 0 in explained AND background rows, pool seed "
          f"{pools[0] if pools else '-'}): (a) max |phi| of planted features = 0; (b) J(P) >= 1/3 when the "
          "planted support >= 100")
        for cfg in cfgs:
            pr = [r for r in planted if r["cfg"] == cfg and r["condition"] == cond]
            jrow = [r for r in jp if r["cfg"] == cfg and r["condition"] == cond and r["arm"] == "P"
                    and int(r["k"]) == K and (not pools or r["pool_seed"] == pools[0])]
            if not pr or not jrow:
                verdicts[(cond, cfg, "positive control")] = None
                p(f"  {cfg}: no planted unit or no P pair")
                continue
            maxabs, sup = float(pr[0]["max_abs_phi_planted"]), int(pr[0]["support_planted"])
            # J from the integer intersection, not the 6-decimal column: J = 1/3 exactly at |A n B| = 80
            inter = int(jrow[0]["inter"])
            jv = 1 - inter / (2 * K - inter)
            a_ok = maxabs == 0.0
            # J >= 1/3  <=>  i / (2k - i) <= 2/3  <=>  5 i <= 4 k, tested in integers
            b_ok = (5 * inter <= 4 * K) if sup >= K else None
            ok = a_ok and (b_ok is not False)
            j2r = series(pairs, cfg, cond, "2r", pools[0] if pools else "", "jaccard")
            m2r, lo2r, hi2r = (stats.block_bootstrap([j2r[x] for x in sorted(j2r)],
                                                     test_rng(corpus, cond, cfg, pools[0] if pools else "", "P-floor"))
                               if j2r
                               else (float("nan"),) * 3)
            verdicts[(cond, cfg, "positive control")] = ok
            p(f"  {cfg} {pr[0]['month']}: (a) max |phi| {maxabs:g} -> {'holds' if a_ok else 'FAILS'}; "
              f"(b) planted support {sup}, J(P) {jv:.3f} -> "
              f"{'not applicable' if b_ok is None else ('holds' if b_ok else 'FAILS')}; "
              f"J(P) - arm 2r floor {jv - m2r:+.3f} (arm 2r {m2r:.3f} [{lo2r:.3f}, {hi2r:.3f}], {len(j2r)} months)")
            print(f"RESULT corpus={corpus} cfg={cfg} condition={cond} control=null_player month={pr[0]['month']} "
                  f"max_abs_phi_planted={maxabs:g} support_planted={sup} J_P={jv:.4f} J_2r={m2r:.4f} "
                  f"J_P_minus_2r={jv - m2r:.4f} a={'PASS' if a_ok else 'FAIL'} "
                  f"b={'NA' if b_ok is None else ('PASS' if b_ok else 'FAIL')} N={n_range(jrow)} "
                  f"share={COND_PROV[cond][0]:.2f} repeat={COND_PROV[cond][1]} {ptxt}", file=sys.stderr)

        # --- explained rows on which KernelExplainer's regression raised LinAlgError (phi NaN there) ---
        fr = {"2s": sum(int(r.get("failed_rows") or 0) for r in pm if r["arm"] == "2s"
                        and r["condition"] == cond and r["cfg"] == matched),
              "0": sum(int(r.get("failed_rows") or 0) for r in pm if r["arm"] == "0" and r["condition"] == cond)}
        for v in ("4a", "4f", "4c"):
            fr[v] = int(sum(float(r["value"]) for r in a4 if r["measure"] == "failed_rows" and r["variant"] == v
                            and r["condition"] == cond and r["cfg"] == matched))
        p("KernelExplainer rows failed (LinAlgError; NaN, every quantity from the remaining rows): " +
          ", ".join(f"{k} {v}" for k, v in fr.items()))
        print(f"RESULT corpus={corpus} cfg={matched} condition={cond} " +
              " ".join(f"failed_rows_{k}={v}" for k, v in fr.items()) +
              f" share={COND_PROV[cond][0]:.2f} repeat={COND_PROV[cond][1]} {ptxt}", file=sys.stderr)

        # --- cost of exact against sampled attribution, same rows ---
        p("Seconds per model-month, fixed model, the same 100 explained rows and 100 background rows")
        ex = [r for r in jm if r["model"] == "pool" and r["cfg"] == matched and r["condition"] == cond]
        if ex:
            p(f"  exact, {ex[0]['threads']} threads, every month: "
              f"{np.mean([float(r['seconds_attr']) for r in ex]):.3f} s (n={len(ex)})")
        tc = [r for r in tp if r["condition"] == cond and r["cfg"] == matched]
        if tc:
            p(f"  arm-4 months, one process each: exact on 1 thread "
              f"{np.mean([float(r['seconds_exact_1thread']) for r in tc]):.3f} s, KernelExplainer at "
              f"nsamples=100 {np.mean([float(r['seconds_kernel_100']) for r in tc]):.3f} s (n={len(tc)})")
        sa = [r for r in pm if r["arm"] == "2s" and r["condition"] == cond and r["cfg"] == matched]
        if sa:
            p(f"  KernelExplainer at nsamples=100 during the arm-2s run (parallel workers, one thread each): "
              f"{np.mean([float(r['seconds']) for r in sa]):.3f} s (n={len(sa)})")
            oor = sum(int(r["rows_out_of_range"]) for r in sa)
            p(f"  arm 2s rows with an attribution beyond the sum of clause ceilings: {oor} of {100 * len(sa)}")

    ddir = HERE.parents[1] / "data" / ("b2-corpora-smoke" if smoke else "b2-corpora") / corpus
    lag_curve(corpus, ddir, pairs, jm, cfgs, matched, conds, pools, p, ptxt)

    p("\nCorpus verdicts, per training condition")
    for (cond, cfg, name), v in sorted(verdicts.items()):
        p(f"  {corpus} {cond} {cfg} {name}: {fmt_verdict(v)}")
        print(f"RESULT corpus={corpus} condition={cond} cfg={cfg} test={name} verdict={fmt_verdict(v)} "
              f"share={COND_PROV[cond][0]:.2f} repeat={COND_PROV[cond][1]} {ptxt}", file=sys.stderr)

    text = "\n".join(out)
    print(text)
    with open(d / f"{corpus}-decision.txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return 0


# --- self-test ----------------------------------------------------------------------------------

def _anchors(years, per_year=12):
    return [f"{y}-{m:02d}" for y in years for m in range(1, per_year + 1)]


def _series(anchors, centre, noise, rng):
    return {a: float(centre + rng.normal(0, noise)) for a in anchors}


def selftest() -> int:
    rng = np.random.default_rng(1)
    A = _anchors((2016, 2017, 2018))
    checks = []

    def check(name, got, want):
        ok = got == want
        checks.append(ok)
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: got {got}, want {want}")

    # F1 fires: low exact overlap, high row-draw floor
    check("F1 fires", cond_F1(_series(A, 0.10, 0.03, rng), _series(A, 0.65, 0.03, rng), rng)["verdict"], True)
    check("F1 silent", cond_F1(_series(A, 0.70, 0.03, rng), _series(A, 0.75, 0.03, rng), rng)["verdict"], False)
    check("F1 silent when only one part holds",
          cond_F1(_series(A, 0.10, 0.03, rng), _series(A, 0.20, 0.03, rng), rng)["verdict"], False)
    # Unresolved fires: arms 1, 2, 3 indistinguishable
    check("Unresolved fires", cond_unresolved(_series(A, 0.50, 0.004, rng), _series(A, 0.50, 0.004, rng),
                                              _series(A, 0.50, 0.004, rng), rng)["verdict"], True)
    check("Unresolved silent", cond_unresolved(_series(A, 0.30, 0.02, rng), _series(A, 0.70, 0.02, rng),
                                               _series(A, 0.50, 0.02, rng), rng)["verdict"], False)
    check("P2 fires", cond_P2(_series(A, 0.85, 0.03, rng), _series(A, 0.30, 0.03, rng), rng)["verdict"], True)
    check("P2 silent", cond_P2(_series(A, 0.40, 0.03, rng), _series(A, 0.30, 0.03, rng), rng)["verdict"], False)
    check("P3 fires", cond_P3(_series(A, 0.70, 0.03, rng), _series(A, 0.40, 0.03, rng), rng)["verdict"], True)
    check("P3 silent", cond_P3(_series(A, 0.40, 0.03, rng), _series(A, 0.40, 0.03, rng), rng)["verdict"], False)

    # the year rule: 5 anchors do not decide; majority of deciding years
    short = _anchors((2016,), per_year=5)
    check("5 anchors do not decide",
          cond_F1(_series(short, 0.1, 0.01, rng), _series(short, 0.7, 0.01, rng), rng)["verdict"], None)
    two_of_three = {**_series(_anchors((2016, 2017)), 0.10, 0.02, rng), **_series(_anchors((2018,)), 0.70, 0.02, rng)}
    check("majority of deciding years",
          cond_F1(two_of_three, _series(A, 0.70, 0.02, rng), rng)["verdict"], True)
    one_of_three = {**_series(_anchors((2016,)), 0.10, 0.02, rng), **_series(_anchors((2017, 2018)), 0.70, 0.02, rng)}
    check("minority of deciding years",
          cond_F1(one_of_three, _series(A, 0.75, 0.02, rng), rng)["verdict"], False)
    check("every pool seed", stats.all_seeds([True, False]), False)
    check("a seed without a deciding year", stats.all_seeds([True, None]), None)

    # F2 on synthetic attributions: R runs x n rows x M features
    R, n, M = 5, 100, 400
    exact = rng.normal(0, 1, (n, M)) * (rng.random((n, M)) < 0.2)
    biased = exact[None] + 0.5 + rng.normal(0, 0.1, (R, n, M))
    noisy = exact[None] + rng.normal(0, 1.0, (R, n, M))
    def f2(runs):
        months = []
        for _ in range(3):
            b2, var = stats.bias_variance_rows(runs, exact)
            _, lo, hi, _ = stats.bias_share(b2, var, rng, resamples=500)
            months.append((lo, hi))
        return cond_F2(months)
    check("F2 fires on a biased estimator", f2(biased), True)
    check("F2 silent on an unbiased noisy estimator", f2(noisy), False)

    # F3: same distribution at both budgets -> within 0.05; a budget that removes the noise -> not
    exact_pos = np.abs(rng.normal(0, 1, (n, M))) * (rng.random((n, M)) < 0.5)
    near = exact_pos[None] + rng.normal(0, 1e-3, (R, n, M))
    far = rng.normal(0, 1.0, (R, n, M))
    def f3(small, big):
        d, lo, hi = stats.budget_difference(small, big, exact_pos, 100, rng, resamples=300)
        return cond_F3([(lo, hi)] * 3)
    check("F3 fires when the larger budget does not help", f3(near, near + rng.normal(0, 1e-3, near.shape)), True)
    check("F3 silent when the larger budget moves toward exact", f3(far, near), False)

    # chance level: identical supports of size m give E = k^2/m
    check("E for identical supports", round(stats.chance_E(100, 400, 400, 400), 6), 25.0)
    check("c of identical top-k sets", stats.corrected(100, 100, 25.0), 1.0)
    check("c at chance", stats.corrected(25, 100, 25.0), 0.0)

    ok = all(checks)
    print(f"selftest: {sum(checks)} of {len(checks)} checks pass")
    return 0 if ok else 1


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", choices=("lamda", "apigraph", "androzoo", "hypercube"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.corpus:
        ap.error("corpus required")
    return run(a.corpus, a.smoke)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
