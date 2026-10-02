# QUESTION:  LAMDA reports Jaccard ~0.9 between consecutive months over top-100 SHAP features and
#            reads it as explanation drift. What is the NOISE FLOOR of that measurement — the Jaccard
#            between two independent runs of the same protocol on the SAME month?
# SURPRISE:  yes, and decisively either way. Their pipeline runs three times per month but only ever
#            compares consecutive months within a run; between-run agreement within a month is one
#            line away and was never computed. If two runs on identical data already disagree at
#            ~0.9, the published curve measures the estimator and not the malware.
# ARMS:      three, and the decomposition is the point:
#              (a) between-run, within month, model REFIT each run   -> refit + explainer noise
#              (b) between-run, within month, model held FIXED       -> explainer noise alone
#              (c) between consecutive months                        -> their reported quantity
#            (c) is meaningless without (a); (b) separates the two causes inside (a). A two-arm
#            version of this experiment would not be interpretable.
# PASS/FAIL: not a pass/fail. The decision rule is stated in advance:
#              if (a) >= ~0.8 and comparable to (c)  -> the published drift is largely estimator
#                                                        variance; that is the B2 headline
#              if (a) ~ 0 while (c) ~ 0.9            -> the churn is real; B2 becomes the exact
#                                                        decomposition instead, and we say so
#              if (b) << (a)                          -> retraining per month, not sampling, drives it
#
#   python research/b0-noise-floor/run.py [--months N] [--runs R] [--topk K]
#
# Faithfulness. The MLP, its training loop, the explainer, the sample budgets and the importance
# aggregation are transcribed from the authors' released script
# (code/section_4_concept_drift_analysis/4_5_shap_explanation_monthly_lamda.py):
# ChenEncoderMLP 4561-512-384-256-128 + 100-100-2 head with dropout 0.2, 20 epochs, batch 64,
# Adam lr 1e-3, CrossEntropyLoss; shap.KernelExplainer over softmax probabilities with 100 background
# rows, 100 explained rows and nsamples=100; importance = mean over explained rows of |shap| for
# class 1.
#
# One deviation, unavoidable: their monthwise .npz files are read from a path local to their machine
# and are not in the release, so month splits are derived here from the released `year_month` column.
# The noise floor is a WITHIN-month quantity, so it is unaffected by whether our month boundaries
# match theirs exactly.

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

import numpy as np  # noqa: E402
import shap  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.optim as optim  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from tmcyber import lamda  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 100          # their nsamples, and their background / explained row counts


# --- transcribed from the authors' script -----------------------------------------------------
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


def train_chen_mlp(X, y, epochs=20):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    model = ChenEncoderMLP(X.shape[1]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
    return model


def make_shap_predict(model):
    def f(x):
        model.eval()
        with torch.no_grad():
            return model(torch.tensor(x, dtype=torch.float32).to(DEVICE),
                         return_prob=True).cpu().numpy()
    return f
# ---------------------------------------------------------------------------------------------


def top_indices(model, X_train, X_test, topk, seed):
    """Their attribution step, verbatim in structure. `seed` varies only the explainer's sampling."""
    np.random.seed(seed)
    explainer = shap.KernelExplainer(make_shap_predict(model), X_train[:N_SAMPLES])
    vals = explainer.shap_values(X_test[:N_SAMPLES], nsamples=N_SAMPLES, silent=True)
    imp = np.mean([np.abs(s[:, 1]) for s in vals], axis=0)
    if imp.shape[0] != X_train.shape[1]:
        raise RuntimeError(f"importance has shape {imp.shape}, expected {X_train.shape[1]} features")
    return np.argsort(imp)[-topk:][::-1], imp


def jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    u = len(sa | sb)
    return 1.0 if u == 0 else 1 - len(sa & sb) / u


def kendall(a, b, imp_a, imp_b):
    """Their Kendall distance: (1 - tau)/2 over the union of the two top-k sets' rankings."""
    union = sorted(set(a.tolist()) | set(b.tolist()))
    ra = [float(imp_a[i]) for i in union]
    rb = [float(imp_b[i]) for i in union]
    tau, _ = kendalltau(ra, rb)
    return 1.0 if tau is None or np.isnan(tau) else (1 - tau) / 2


def month_data(by_year, cols, year, month):
    """That month's rows from the released train and test portions of its year."""
    out = {}
    for kind, path in (("train", by_year[year].train), ("test", by_year[year].test)):
        X, meta = lamda.read_features(path, cols)
        sel = meta["year_month"].astype(str).str.slice(5, 7).astype(int).to_numpy() == month
        out[kind] = (X[sel].astype(np.float32), meta["label"].to_numpy().astype(np.int64)[sel])
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/lamda/raw")
    ap.add_argument("--months", type=int, default=8, help="how many months to probe")
    ap.add_argument("--runs", type=int, default=3, help="independent runs per month (theirs: 3)")
    ap.add_argument("--topk", type=int, default=100)
    a = ap.parse_args(argv)

    years = lamda.discover(a.root)
    by_year = {y.year: y for y in years}
    cols = lamda.feature_names(by_year[2013].train)

    # Consecutive, well-populated months so that arm (c) is a fair rendering of their curve.
    # 2016-02 onward is the first long run of full months in the corpus.
    candidates = [(2016, m) for m in range(2, 13)] + [(2017, m) for m in (1, 2, 3, 4, 5)]
    chosen = candidates[: a.months]
    print(f"device {DEVICE}   months {chosen}   runs/month {a.runs}   topk {a.topk}   "
          f"nsamples {N_SAMPLES}")

    tops, imps = {}, {}
    fixed_tops, fixed_imps = {}, {}
    for (year, month) in chosen:
        d = month_data(by_year, cols, year, month)
        ntr, nte = len(d["train"][1]), len(d["test"][1])
        if ntr < N_SAMPLES or nte < N_SAMPLES:
            print(f"  {year}-{month:02d}: SKIP (train {ntr}, test {nte}; needs {N_SAMPLES} of each)")
            continue
        t0 = time.time()
        for r in range(a.runs):
            torch.manual_seed(1000 * r + month)
            np.random.seed(1000 * r + month)
            model = train_chen_mlp(d["train"][0], d["train"][1])
            idx, imp = top_indices(model, d["train"][0], d["test"][0], a.topk, seed=r)
            tops[(year, month, r)] = idx
            imps[(year, month, r)] = imp
            if r == 0:
                # arm (b): same trained model, explainer re-run with different sampling
                for e in range(1, a.runs):
                    idx_e, imp_e = top_indices(model, d["train"][0], d["test"][0], a.topk,
                                               seed=500 + e)
                    fixed_tops[(year, month, e)] = idx_e
                    fixed_imps[(year, month, e)] = imp_e
        print(f"  {year}-{month:02d}: train {ntr:>6} test {nte:>6}   "
              f"{a.runs} runs + {a.runs-1} fixed-model re-explains in {time.time()-t0:.0f}s")

    months_done = sorted({(y, m) for (y, m, _) in tops})
    if len(months_done) < 2:
        print("not enough months completed to compare")
        return 1

    def stats(vals):
        v = np.array(vals, dtype=float)
        return v.mean(), v.std(), v.min(), v.max()

    print(f"\n{'arm':<46}{'mean':>8}{'sd':>8}{'min':>8}{'max':>8}{'n':>5}")

    a_j, a_k = [], []
    for (y, m) in months_done:
        for r1, r2 in itertools.combinations(range(a.runs), 2):
            a_j.append(jaccard(tops[(y, m, r1)], tops[(y, m, r2)]))
            a_k.append(kendall(tops[(y, m, r1)], tops[(y, m, r2)],
                               imps[(y, m, r1)], imps[(y, m, r2)]))
    mu, sd, lo, hi = stats(a_j)
    print(f"{'(a) same month, between runs, model refit':<46}{mu:>8.3f}{sd:>8.3f}{lo:>8.3f}{hi:>8.3f}{len(a_j):>5}")

    b_j = []
    for (y, m) in months_done:
        for e in range(1, a.runs):
            if (y, m, e) in fixed_tops:
                b_j.append(jaccard(tops[(y, m, 0)], fixed_tops[(y, m, e)]))
    if b_j:
        mu_b, sd_b, lo_b, hi_b = stats(b_j)
        print(f"{'(b) same month, same model, explainer re-run':<46}{mu_b:>8.3f}{sd_b:>8.3f}{lo_b:>8.3f}{hi_b:>8.3f}{len(b_j):>5}")

    c_j, c_k = [], []
    for (y1, m1), (y2, m2) in zip(months_done, months_done[1:]):
        for r in range(a.runs):
            c_j.append(jaccard(tops[(y1, m1, r)], tops[(y2, m2, r)]))
            c_k.append(kendall(tops[(y1, m1, r)], tops[(y2, m2, r)],
                               imps[(y1, m1, r)], imps[(y2, m2, r)]))
    mu_c, sd_c, lo_c, hi_c = stats(c_j)
    print(f"{'(c) consecutive months, within a run [THEIRS]':<46}{mu_c:>8.3f}{sd_c:>8.3f}{lo_c:>8.3f}{hi_c:>8.3f}{len(c_j):>5}")

    print(f"\nKendall distance: (a) {np.mean(a_k):.3f}   (c) {np.mean(c_k):.3f}")
    print(f"\npublished reference: Jaccard 'close to 0.9' between consecutive months, top-100")
    print(f"noise floor as a fraction of their signal: {np.mean(a_j)/max(1e-9, mu_c):.2f}")

    print(f"\nRESULT noise_floor_jaccard={np.mean(a_j):.4f} "
          f"explainer_only_jaccard={np.mean(b_j) if b_j else float('nan'):.4f} "
          f"consecutive_month_jaccard={mu_c:.4f} "
          f"noise_kendall={np.mean(a_k):.4f} month_kendall={np.mean(c_k):.4f} "
          f"months={len(months_done)} runs={a.runs} topk={a.topk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
