"""Stage 1 (E4): does the MA-BBOB failure pre-screen transfer to new problems?

The frozen child-failure probe of experiment 1 (``child_failed_gen_mean_L21``,
trained on MA-BBOB LLaMEA children, used in the in-loop pre-screen) is read
out, without retraining, on the code activations of the step 5d screens
(``11_transfer_analyze.py --stage acts``: teacher-forced, mean over the code
tokens, layers 15 and 21). Reference: the same readout on the MA-BBOB corpus
of step 5b (same representation, different from the in-loop one, which
averages over all generated tokens in the real prompt).

Per problem: AUROC of the frozen probe for "the algorithm fails", the
pre-screen at the probe's own threshold (share skipped, precision, share
of failures caught), and two references: a probe refit on the problem's own
activations (5-fold CV, ridge as in experiment 1; the attainable AUROC) and
the code length. The probe's threshold does not transfer (the
representation differs from the in-loop one, even on MA-BBOB), so the
pre-screen is also evaluated at a recalibrated threshold that skips the
same share as in the loop (21%). Writes ``results/prescreen-transfer/summary.md``.
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MI_ROOT = HERE.parents[1]
sys.path[:0] = [str(MI_ROOT), str(HERE.parent), str(HERE)]

from common.paths import data_dir, results_dir
from common.prescreen import load_probe
from exp_config import EXPERIMENT

PROBE = MI_ROOT / "exp01_probing/results/main/probes/child_failed_gen_mean_L21.npz"
LAYER = 21
SKIP = 0.21  # share of evaluations the in-loop pre-screen skipped (exp01)


def frozen_scores(probe, X):
    z = (X - probe["mean"]) / probe["scale"]
    return z @ probe["coef"] + probe["intercept"]


def cv_auroc(X, y, seed=0):
    if y.sum() < 5 or (1 - y).sum() < 5:
        return np.nan
    s = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1000.0))
        m.fit(X[tr], y[tr])
        s[te] = m.decision_function(X[te])
    return roc_auc_score(y, s)


def boot_auc(y, s, rng, n=2000):
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(roc_auc_score(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def datasets():
    """(name, X at LAYER, failed, code length, condition) per problem."""
    an = importlib.import_module("11_transfer_analyze")
    for tag, name in [("transfer-bragg", "Bragg"), ("transfer-packing", "packing")]:
        proc = data_dir(EXPERIMENT, "processed", tag)
        a = np.load(proc / "acts.npz")
        keys = pd.read_csv(proc / "acts_rows.csv")
        df = an.load(tag)
        df = keys.merge(df, on=["prompt", "k", "condition"], how="left")
        X = a["acts"][:, list(a["layers"]).index(LAYER)]
        ok = np.isfinite(X).all(1)
        yield (
            name,
            X[ok],
            df.failed.to_numpy(bool)[ok],
            df.code.str.len().to_numpy()[ok],
            df.condition.to_numpy()[ok],
        )
    proc = data_dir(EXPERIMENT, "processed", "corpus")
    a = np.load(proc / "acts.npz")
    rows = pd.read_csv(proc / "rows.csv")
    X = a["acts"][:, list(a["layers"]).index(LAYER)]
    ok = np.isfinite(X).all(1) & rows.failed.notna().to_numpy()
    code_len = rows.get("code_len", pd.Series(np.nan, index=rows.index))
    yield (
        "MA-BBOB corpus (reference)",
        X[ok],
        rows.failed.to_numpy(bool)[ok],
        code_len.to_numpy()[ok],
        rows.source.to_numpy()[ok],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()
    probe = load_probe(PROBE)
    thr = float(probe["threshold"])
    rng = np.random.default_rng(0)
    out = []
    for name, X, y, length, cond in datasets():
        s = frozen_scores(probe, X.astype(np.float32))
        auc = roc_auc_score(y, s)
        lo, hi = boot_auc(y, s, rng)
        flag = s > thr
        none = cond == "none"
        out.append(
            {
                "problem": name,
                "n": len(y),
                "failure rate": y.mean(),
                "frozen AUROC": f"{auc:.2f} ({lo:.2f} to {hi:.2f})",
                "frozen AUROC, unsteered": (
                    f"{roc_auc_score(y[none], s[none]):.2f}"
                    if none.sum() > 10 and 0 < y[none].mean() < 1
                    else "-"
                ),
                "skipped at threshold": flag.mean(),
                "precision": y[flag].mean() if flag.any() else np.nan,
                "failures caught": flag[y].mean() if y.any() else np.nan,
                f"precision, top {SKIP:.0%} skipped": y[
                    s >= np.quantile(s, 1 - SKIP)
                ].mean(),
                "refit AUROC (CV)": cv_auroc(X, y.astype(int)),
                "code length AUROC": (
                    roc_auc_score(y, length) if np.isfinite(length).all() else np.nan
                ),
            }
        )
        print(out[-1], flush=True)
    res = pd.DataFrame(out)
    dest = results_dir(EXPERIMENT, "prescreen-transfer")
    dest.mkdir(parents=True, exist_ok=True)
    res.to_csv(dest / "summary.csv", index=False)
    fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
    lines = [
        "# Failure pre-screen on new problems (frozen MA-BBOB probe)",
        "",
        f"Probe `{PROBE.name}`, threshold {thr:.3f} (score > threshold is "
        "skipped). Activations: teacher-forced code mean at layer "
        f"{LAYER}.",
        "",
        "| " + " | ".join(res.columns) + " |",
        "|" + "---|" * len(res.columns),
    ]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for _, r in res.iterrows()]
    (dest / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
