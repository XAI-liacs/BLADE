"""Step 6: do the activations add information beyond the baselines?

``03_train_probes.py`` compares probes and baselines side by side, and picks
the best layer after seeing the test scores. This script answers two
follow-up questions without that bias:

1. **Incremental value.** Leave-one-run-out cross-validation. Inside every
   training split, the probe layer is chosen by an inner ``GroupKFold`` over
   the training runs only, and the inner out-of-fold predictions of the
   probe and of the code bag-of-words are stacked with the metadata in a
   small second-level model. Compared models (all out-of-fold):

   * ``M``: metadata (operator, generation, parent fitness / failure /
     over-budget status, prompt length);
   * ``M+T``: metadata + text (code TF-IDF score, entropy, log-prob,
     generated length). The text exists only after generation, so it is
     the fair reference for ``gen_mean``; for the prompt positions compare
     ``M+P`` with ``M``;
   * ``P``: the probe alone (nested layer choice);
   * ``M+P`` and ``M+T+P``.

   ``delta`` = (with probe) - (without probe), with a 95% bootstrap CI over
   runs (resampling whole LLaMEA runs).

2. **Over-budget confound.** A child usually inherits its parent's
   evaluation loop, and so its over-budget status. ``strata.csv`` scores
   the same out-of-fold predictions within strata of
   ``parent_over_budget`` (parent within budget / no working parent);
   inside a stratum the parent's status cannot explain the prediction.

Writes ``results/<tag>/followup/``: ``incremental.csv``, ``strata.csv``,
``layers.csv`` (layer chosen per outer fold) and ``oof_<target>_<pos>.csv``.
"""

import argparse
import importlib
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common.paths import data_dir, results_dir
from exp_config import EXPERIMENT

probes = importlib.import_module("03_train_probes")

TARGETS = ("child_failed", "over_budget", "improved", "child_fitness")
MODELS = ("M", "M+T", "P", "M+P", "M+T+P")
UNCERTAINTY = ["mean_entropy", "mean_logprob", "n_gen_tokens"]


def tfidf():
    return TfidfVectorizer(
        token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=2, max_features=5000
    )


def metric(kind, y, pred) -> float:
    ok = np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    if kind == "binary":
        return (
            probes.score(kind, y, pred)["auroc"] if len(np.unique(y)) == 2 else np.nan
        )
    return probes.score(kind, y, pred)["spearman"] if len(y) > 2 else np.nan


def inner_oof(X, y, groups, kind, n_splits, transform=None):
    _, oof = probes.cross_val(X, y, groups, kind, n_splits, transform)
    return np.where(np.isfinite(oof), oof, np.nanmean(oof))


def fit_predict(X_train, y_train, X_test, kind, transform=None):
    if transform is not None:
        vec = transform()
        X_train, X_test = vec.fit_transform(X_train), vec.transform(X_test)
        model = RidgeCV(alphas=probes.ALPHAS)
    else:
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=probes.ALPHAS))
    target = 2.0 * y_train - 1.0 if kind == "binary" else y_train
    return model.fit(X_train, target).predict(X_test)


def stack_predict(F_train, y_train, F_test, kind):
    if kind == "binary":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        return model.fit(F_train, y_train).decision_function(F_test)
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=probes.ALPHAS))
    return model.fit(F_train, y_train).predict(F_test)


def outer_fold(train, test, X_all, meta_X, unc, code, y, groups, kind, cand, n_inner):
    """One leave-one-run-out split; returns test predictions per model."""
    with threadpool_limits(int(os.environ.get("FOLD_THREADS", 8))):
        g, yt = groups[train], y[train]
        n_in = min(n_inner, len(np.unique(g)))
        # Inner layer choice on the training runs only.
        best = (-np.inf, None, None)
        for li in cand:
            oof = inner_oof(X_all[train, li], yt, g, kind, n_in)
            s = metric(kind, yt, oof)
            if np.isfinite(s) and s > best[0]:
                best = (s, li, oof)
        _, li, p_train = best
        p_test = fit_predict(X_all[train, li], yt, X_all[test, li], kind)
        b_train = inner_oof(code[train], yt, g, kind, n_in, tfidf)
        b_test = fit_predict(code[train], yt, code[test], kind, tfidf)

        feats = {
            "M": (meta_X[train], meta_X[test]),
            "T": (
                np.column_stack([b_train, unc[train]]),
                np.column_stack([b_test, unc[test]]),
            ),
            "P": (p_train[:, None], p_test[:, None]),
        }
        preds = {"P": p_test}
        for name in ("M", "M+T", "M+P", "M+T+P"):
            parts = name.split("+")
            F_tr = np.column_stack([feats[p][0] for p in parts])
            F_te = np.column_stack([feats[p][1] for p in parts])
            preds[name] = stack_predict(F_tr, yt, F_te, kind)
        return test, preds, li


def bootstrap_delta(kind, y, a, b, groups, n_boot, rng):
    runs = np.unique(groups)
    by_run = {r: np.flatnonzero(groups == r) for r in runs}
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([by_run[r] for r in rng.choice(runs, len(runs))])
        deltas.append(metric(kind, y[idx], b[idx]) - metric(kind, y[idx], a[idx]))
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    return lo, hi


def bootstrap_ci(kind, y, pred, groups, n_boot, rng):
    runs = np.unique(groups)
    by_run = {r: np.flatnonzero(groups == r) for r in runs}
    vals = [
        metric(kind, y[idx], pred[idx])
        for idx in (
            np.concatenate([by_run[r] for r in rng.choice(runs, len(runs))])
            for _ in range(n_boot)
        )
    ]
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--positions", nargs="*", default=list(probes.POSITIONS))
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--n-inner", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    proc = data_dir(EXPERIMENT, "processed", args.tag)
    meta = pd.read_csv(proc / "meta.csv", dtype={"record_id": str, "run": str})
    acts = np.load(proc / "activations.npz")
    layers = acts["layers"]
    codes = {}
    with open(proc / "code.jsonl") as f:
        for line in f:
            c = json.loads(line)
            codes[c["record_id"]] = c["code"]
    out = results_dir(EXPERIMENT, args.tag) / "followup"
    out.mkdir(exist_ok=True)

    inc_rows, strata_rows, layer_rows = [], [], []
    for target in args.targets:
        kind, row_filter, _ = probes.TARGETS[target]
        mask = row_filter(meta).to_numpy()
        sub = meta[mask].reset_index(drop=True)
        y = sub[target].to_numpy(dtype=float)
        groups = sub["run"].to_numpy()
        meta_X = probes.meta_features(meta, target)[mask]
        pob = sub["parent_over_budget"]
        meta_X = np.column_stack([meta_X, pob.fillna(0), pob.isna()])
        unc = sub[UNCERTAINTY].to_numpy(dtype=float)
        code = np.array([codes.get(r, "") for r in sub["record_id"]], dtype=object)
        idx = np.flatnonzero(mask)
        for pos in args.positions:
            X_all = acts[pos][idx].astype(np.float32)
            cand = range(0, len(layers), args.layer_stride)
            splits = list(LeaveOneGroupOut().split(X_all, y, groups))
            print(f"== {target} / {pos}: n={len(y)}, {len(splits)} outer folds")
            results = Parallel(n_jobs=args.n_jobs)(
                delayed(outer_fold)(
                    tr,
                    te,
                    X_all,
                    meta_X,
                    unc,
                    code,
                    y,
                    groups,
                    kind,
                    cand,
                    args.n_inner,
                )
                for tr, te in splits
            )
            oof = {m: np.full(len(y), np.nan) for m in MODELS}
            for k, (te, preds, li) in enumerate(results):
                for m, p in preds.items():
                    oof[m][te] = p
                layer_rows.append(
                    {
                        "target": target,
                        "position": pos,
                        "fold": k,
                        "run": groups[te][0],
                        "layer": int(layers[li]),
                    }
                )
            df = sub[["record_id", "run", "operator", "parent_over_budget", target]]
            df = df.assign(**{f"pred_{m}": oof[m] for m in MODELS})
            df.to_csv(out / f"oof_{target}_{pos}.csv", index=False)

            base = {"target": target, "kind": kind, "position": pos, "n": len(y)}
            scores = {m: metric(kind, y, oof[m]) for m in MODELS}
            for m in MODELS:
                inc_rows.append({**base, "model": m, "score": scores[m]})
            for a, b in (("M", "M+P"), ("M+T", "M+T+P"), ("M+T", "P")):
                lo, hi = bootstrap_delta(
                    kind, y, oof[a], oof[b], groups, args.n_boot, rng
                )
                inc_rows.append(
                    {
                        **base,
                        "model": f"{b} - {a}",
                        "score": scores[b] - scores[a],
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
            print("   " + "  ".join(f"{m} {s:.3f}" for m, s in scores.items()))

            if target == "over_budget":
                for name, sel in {
                    "parent within budget": pob == 0,
                    "no working parent": pob.isna(),
                    "parent over budget": pob == 1,
                }.items():
                    sel = sel.to_numpy()
                    if len(np.unique(y[sel])) < 2:  # empty or one class
                        print(f"   {name:22s} n={int(sel.sum()):3d} skipped")
                        continue
                    row = {
                        "position": pos,
                        "stratum": name,
                        "n": int(sel.sum()),
                        "positives": int(y[sel].sum()),
                    }
                    for m in MODELS:
                        row[m] = metric(kind, y[sel], oof[m][sel])
                    for m in ("P", "M+T"):
                        lo, hi = bootstrap_ci(
                            kind, y[sel], oof[m][sel], groups[sel], args.n_boot, rng
                        )
                        row[f"{m}_ci_low"], row[f"{m}_ci_high"] = lo, hi
                    strata_rows.append(row)
                    print(
                        f"   {name:22s} n={row['n']:3d} pos={row['positives']:3d} "
                        + " ".join(f"{m} {row[m]:.3f}" for m in MODELS)
                    )

    pd.DataFrame(inc_rows).to_csv(out / "incremental.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(out / "layers.csv", index=False)
    if strata_rows:
        pd.DataFrame(strata_rows).to_csv(out / "strata.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
