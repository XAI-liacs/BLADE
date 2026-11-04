"""Step 3: layer-wise linear probes, controls and baselines.

For every (target, position, layer): a standardised ridge probe evaluated
with cross-validation grouped by LLaMEA run (no solution of a test run is
seen in training), plus the same probe on shuffled labels as a control.
Binary targets are scored by AUROC of the ridge decision value, regression
targets by Spearman rho and R^2.

Baselines that use no activations (same folds):

* ``meta``: operator, generation, parent fitness / failure, prompt length;
* ``uncertainty``: mean token entropy, mean log-prob, generated length
  (available after generation, like ``gen_mean``);
* ``code_bow``: TF-IDF of the generated code (after generation).

Writes to ``results/<tag>/``: ``probe_scores.csv``, ``baselines.csv`` and
``oof_<target>_<position>.csv`` (out-of-fold predictions at the best layer,
used for the pre-screening analysis in the notebook).
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import data_dir, results_dir
from exp_config import EXPERIMENT

ALPHAS = np.logspace(-1, 6, 15)
POSITIONS = ("last", "prompt_mean", "gen_mean")

# name -> (kind, row filter, question)
TARGETS = {
    "parent_fitness": (
        "regression",
        lambda m: m["parent_fitness"].notna(),
        "Does the model represent how good the parent algorithm is?",
    ),
    "child_failed": (
        "binary",
        lambda m: m["operator"].notna(),
        "Does the model know its code will crash or time out?",
    ),
    "over_budget": (
        "binary",
        lambda m: m["child_failed"] == 0,
        "Does the model know its (running) code overspends the budget?",
    ),
    "improved": (
        "binary",
        lambda m: m["improved"].notna(),
        "Does the model know whether its mutation will improve the parent?",
    ),
    "child_fitness": (
        "regression",
        lambda m: m["child_fitness"].notna(),
        "Does the model know how good its (working) algorithm will be?",
    ),
}


def score(kind, y, pred) -> dict:
    if kind == "binary":
        return {"auroc": roc_auc_score(y, pred)}
    if np.std(pred) == 0:  # constant prediction carries no ranking
        return {"spearman": np.nan, "r2": r2_score(y, pred)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # constant labels within a fold
        rho = spearmanr(y, pred).statistic
    return {"spearman": rho, "r2": r2_score(y, pred)}


def cross_val(X, y, groups, kind, n_splits, transform=None):
    """Returns (per-fold score dicts, out-of-fold predictions)."""
    oof = np.full(len(y), np.nan)
    folds = []
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
        y_train = y[train]
        if kind == "binary" and len(np.unique(y_train)) < 2:
            continue
        if kind == "binary" and len(np.unique(y[test])) < 2:
            continue
        if transform is not None:
            vec = transform()
            X_train = vec.fit_transform(X[train])
            X_test = vec.transform(X[test])
            model = RidgeCV(alphas=ALPHAS)
        else:
            X_train, X_test = X[train], X[test]
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        target = 2.0 * y_train - 1.0 if kind == "binary" else y_train
        model.fit(X_train, target)
        oof[test] = model.predict(X_test)
        folds.append(score(kind, y[test], oof[test]))
    return folds, oof


def summarise(folds: list[dict]) -> dict:
    if not folds:
        return {}
    out = {"n_folds": len(folds)}
    for metric in folds[0]:
        vals = np.array([f[metric] for f in folds], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-nan folds
            out[metric] = np.nanmean(vals)
            out[f"{metric}_std"] = np.nanstd(vals)
    return out


def meta_features(meta: pd.DataFrame, target: str) -> np.ndarray:
    cols = pd.get_dummies(meta["operator"], prefix="op").astype(float)
    cols["generation"] = meta["generation"].fillna(0)
    cols["parent_failed"] = meta["parent_failed"].astype(float)
    cols["n_prompt_tokens"] = meta["n_prompt_tokens"]
    if target != "parent_fitness":
        cols["parent_fitness"] = meta["parent_fitness"].fillna(0)
        cols["parent_fitness_missing"] = meta["parent_fitness"].isna().astype(float)
    return cols.to_numpy(dtype=float)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    proc = data_dir(EXPERIMENT, "processed", args.tag)
    meta = pd.read_csv(proc / "meta.csv", dtype={"record_id": str, "run": str})
    acts = np.load(proc / "activations.npz")
    layers = acts["layers"]
    codes = {}
    with open(proc / "code.jsonl") as f:
        for line in f:
            c = json.loads(line)
            codes[c["record_id"]] = c["code"]
    meta = meta[~meta["steered"]].copy() if "steered" in meta else meta
    out = results_dir(EXPERIMENT, args.tag)

    probe_rows, base_rows = [], []
    for target in args.targets:
        kind, row_filter, question = TARGETS[target]
        mask = row_filter(meta).to_numpy()
        sub = meta[mask]
        y = sub[target].to_numpy(dtype=float)
        groups = sub["run"].to_numpy()
        n_splits = min(args.n_splits, len(np.unique(groups)))
        if n_splits < 2 or (kind == "binary" and len(np.unique(y)) < 2):
            print(f"skip {target}: n={len(y)}, runs={n_splits}, classes={np.unique(y)}")
            continue
        info = {"target": target, "kind": kind, "n": len(y)}
        if kind == "binary":
            info["base_rate"] = float(y.mean())
        print(f"\n== {target} ({question}) n={len(y)}")

        idx = np.flatnonzero(mask)
        baselines = {
            "meta": (meta_features(meta, target)[mask], None),
            "uncertainty": (
                sub[["mean_entropy", "mean_logprob", "n_gen_tokens"]].to_numpy(),
                None,
            ),
            "code_bow": (
                np.array([codes.get(r, "") for r in sub["record_id"]], dtype=object),
                lambda: TfidfVectorizer(
                    token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=2, max_features=5000
                ),
            ),
        }
        for name, (X, transform) in baselines.items():
            try:
                folds, _ = cross_val(X, y, groups, kind, n_splits, transform)
            except ValueError as e:  # e.g. empty TF-IDF vocabulary
                print(f"  baseline {name:12s} failed: {e}")
                continue
            row = {**info, "baseline": name, **summarise(folds)}
            base_rows.append(row)
            print(f"  baseline {name:12s} {_fmt(row, kind)}")

        y_shuffled = rng.permutation(y)
        for pos in POSITIONS:
            X_all = acts[pos][idx]
            ok = np.isfinite(X_all).all(axis=(1, 2))
            best = (-np.inf, None, None)
            for li in range(0, len(layers), args.layer_stride):
                X = X_all[ok, li].astype(np.float32)
                folds, oof = cross_val(X, y[ok], groups[ok], kind, n_splits)
                ctrl, _ = cross_val(X, y_shuffled[ok], groups[ok], kind, n_splits)
                row = {**info, "position": pos, "layer": int(layers[li])}
                row.update(summarise(folds))
                row.update({f"control_{k}": v for k, v in summarise(ctrl).items()})
                probe_rows.append(row)
                key = row.get("auroc", row.get("spearman", -np.inf))
                if np.isfinite(key) and key > best[0]:
                    best = (key, int(layers[li]), oof)
            if best[2] is not None:
                oof_df = sub[ok][["record_id", "run", "operator", target]].copy()
                oof_df["pred"] = best[2]
                oof_df["layer"] = best[1]
                oof_df.to_csv(out / f"oof_{target}_{pos}.csv", index=False)
                print(f"  probe {pos:12s} best layer {best[1]:3d}: {best[0]:.3f}")

    pd.DataFrame(probe_rows).to_csv(out / "probe_scores.csv", index=False)
    pd.DataFrame(base_rows).to_csv(out / "baselines.csv", index=False)
    print(f"\nwrote {out}")


def _fmt(row, kind):
    if "n_folds" not in row:
        return "(not enough folds with both classes)"
    if kind == "binary":
        return f"AUROC {row['auroc']:.3f} ± {row['auroc_std']:.3f}"
    return f"rho {row['spearman']:.3f} ± {row['spearman_std']:.3f}"


if __name__ == "__main__":
    main()
