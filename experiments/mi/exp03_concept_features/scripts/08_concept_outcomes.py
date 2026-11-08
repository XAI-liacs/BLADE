"""Step 8: do concept features predict failure or fitness beyond the labels?

The concept labels (step 3) say *which* concepts an algorithm uses; the
SAE features (steps 4 and 7) say how strongly the model represents them
while writing. If the features carry information about the outcome that
the labels do not, the representation tracks something about the
implementation (how the concept is realised) rather than its presence.

Targets (from experiment 1): ``child_failed`` (all algorithms),
``over_budget`` (working algorithms) and ``child_fitness`` (working
algorithms, Spearman). Outer cross-validation is ``GroupKFold`` over
LLaMEA runs. High-dimensional blocks enter a second-level model as
inner out-of-fold scores (inner ``GroupKFold`` on the training runs):

* ``M``: metadata (operator, generation, parent fitness / failure /
  over-budget, prompt length), as in experiment 1;
* ``L``: the 18 concept labels + log code length;
* ``C``: the concept features that survive line masking (step 7: best
  line-masked feature per concept, at least 0.05 above the
  ``other_labels`` baseline), log1p of their per-solution max;
* ``A``: all selected concept features (top 5 per concept and layer),
  as one inner-CV ridge score;
* ``S``: the whole SAE code (log1p per-solution max over code tokens of
  every feature active in >= 0.5% of solutions), reduced to
  ``--svd`` components per layer by a truncated SVD on all solutions
  (label-free, so no outcome leaks into it; a ridge on the 27k sparse
  columns takes minutes per fit), one ridge score per layer;
* ``T``: code TF-IDF ridge score (the strongest text baseline of
  experiment 1).

Compared: ``M+L`` vs ``M+L+C`` / ``M+L+A`` / ``M+L+S``, and ``M+L+T`` vs
``M+L+T+C`` / ``M+L+T+S``; ``delta`` with a 95% bootstrap CI over runs.

Per feature (``C``) it also reports, within the solutions that use the
concept, the association of the feature's activation with each target:
does a more strongly represented concept go with better or worse code?

Writes ``results/<tag>/concept_outcomes.csv``,
``concept_outcome_features.csv`` and ``concept_outcomes.md``.
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp03_concept_features
sys.path.insert(0, str(HERE))

from common.paths import data_dir, results_dir
from exp_config import EXPERIMENT, SOURCE_EXPERIMENT

concept_labels = importlib.import_module("03_concept_labels")
step4 = importlib.import_module("04_concept_features")

sys.path.insert(0, str(HERE.parents[1] / "exp01_probing" / "scripts"))
# only TARGETS and meta_features are used (its exp_config import resolves to
# this experiment's, which it needs only in its own main)
probes = importlib.import_module("03_train_probes")

TARGETS = ("child_failed", "over_budget", "child_fitness")
ALPHAS = np.logspace(-1, 6, 15)
COMPARISONS = (
    ("M+L", "M+L+C"),
    ("M+L", "M+L+A"),
    ("M+L", "M+L+S"),
    ("M+L+T", "M+L+T+C"),
    ("M+L+T", "M+L+T+S"),
)
MODELS = ("M", "M+L", "M+L+C", "M+L+A", "M+L+S", "M+L+T", "M+L+T+C", "M+L+T+S")


def metric(kind, y, p):
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if kind == "binary":
        return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return spearmanr(y, p)[0] if len(y) > 2 and np.std(p) > 0 else np.nan


def ridge_score(X_tr, y_tr, X_te, kind, transform=None):
    if transform is not None:
        vec = transform()
        X_tr, X_te = vec.fit_transform(X_tr), vec.transform(X_te)
    elif not sparse.issparse(X_tr):
        sc = StandardScaler().fit(X_tr)
        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
    target = 2.0 * y_tr - 1.0 if kind == "binary" else y_tr
    return RidgeCV(alphas=ALPHAS).fit(X_tr, target).predict(X_te)


def inner_oof(X, y, groups, kind, n_splits, transform=None):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(np.zeros(len(y)), y, groups):
        oof[te] = ridge_score(X[tr], y[tr], X[te], kind, transform)
    return oof


def tfidf():
    return TfidfVectorizer(
        token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=2, max_features=5000
    )


def stack(F_tr, y_tr, F_te, kind):
    if kind == "binary":
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        return m.fit(F_tr, y_tr).decision_function(F_te)
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    return m.fit(F_tr, y_tr).predict(F_te)


def outer_fold(tr, te, blocks, y, groups, kind, n_inner):
    """``blocks``: name -> (array, kind) with kind 'dense' (enters the
    stacker directly) or 'score'/'text' (inner-CV ridge score)."""
    with threadpool_limits(int(os.environ.get("FOLD_THREADS", 4))):
        g, yt = groups[tr], y[tr]
        feats = {}
        for name, (X, how) in blocks.items():
            if how == "dense":
                feats[name] = (X[tr], X[te])
                continue
            transform = tfidf if how == "text" else None
            cols_tr, cols_te = [], []
            for Xi in X if isinstance(X, list) else [X]:
                cols_tr.append(inner_oof(Xi[tr], yt, g, kind, n_inner, transform))
                cols_te.append(ridge_score(Xi[tr], yt, Xi[te], kind, transform))
            feats[name] = (np.column_stack(cols_tr), np.column_stack(cols_te))
        preds = {}
        for model in MODELS:
            parts = model.split("+")
            F_tr = np.column_stack([feats[p][0] for p in parts])
            F_te = np.column_stack([feats[p][1] for p in parts])
            preds[model] = stack(F_tr, yt, F_te, kind)
        return te, preds


def run_bootstrap(groups, n_boot, rng):
    runs = np.unique(groups)
    by_run = {r: np.flatnonzero(groups == r) for r in runs}
    for _ in range(n_boot):
        yield np.concatenate([by_run[r] for r in rng.choice(runs, len(runs))])


def robust_features(tag, margin):
    km = pd.read_csv(results_dir(EXPERIMENT, tag) / "keyword_masking.csv")
    km = km[km.auroc_line - km.auroc_other_labels >= margin]
    return km.loc[km.groupby("concept").auroc_line.idxmax()].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="k100")
    ap.add_argument("--targets", nargs="*", default=list(TARGETS))
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--min-active", type=float, default=0.005)
    ap.add_argument("--svd", type=int, default=256)
    ap.add_argument("--n-outer", type=int, default=10)
    ap.add_argument("--n-inner", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    t0 = time.time()

    enc = step4.load_encodings(args.tag)
    ids = enc["record_id"]
    labels = pd.read_csv(data_dir(EXPERIMENT, "processed") / "labels.csv")
    labels = labels.set_index("record_id").loc[ids]
    meta, codes = [], {}
    for tag in sorted(labels.tag.unique()):
        proc = data_dir(SOURCE_EXPERIMENT, "processed", tag)
        m = pd.read_csv(proc / "meta.csv", dtype={"record_id": str, "run": str})
        meta.append(m.assign(run=tag + "/" + m.run))
        with open(proc / "code.jsonl") as f:
            for line in f:
                c = json.loads(line)
                codes[c["record_id"]] = c["code"]
    meta = pd.concat(meta).set_index("record_id").loc[ids].reset_index()
    n_tok = {}
    for f in data_dir(EXPERIMENT, "processed", args.tag).glob("shard-*.npz"):
        z = np.load(f)
        n_tok.update(zip(z["record_id"], z["n_code_tokens"]))
    length = np.log([n_tok[r] for r in ids])

    concepts = list(concept_labels.CONCEPTS)
    L = np.column_stack([labels[concepts].to_numpy(float), length])
    robust = robust_features(args.tag, args.margin)
    print("robust features:")
    print(robust[["concept", "layer", "feature", "auroc_line"]].to_string(index=False))
    C = np.column_stack(
        [
            np.log1p(enc[(r.layer, "max")][:, r.feature].toarray().ravel())
            for r in robust.itertuples()
        ]
    )
    sel = pd.read_csv(results_dir(EXPERIMENT, args.tag) / "concept_features.csv")
    A = np.column_stack(
        [
            np.log1p(enc[(r.layer, "max")][:, r.feature].toarray().ravel())
            for r in sel.drop_duplicates(["layer", "feature"]).itertuples()
        ]
    )
    S = []
    for layer in enc["layers"]:
        X = enc[(layer, "max")].tocsr()
        keep = np.flatnonzero(np.asarray((X > 0).mean(0)).ravel() >= args.min_active)
        X = X[:, keep].astype(np.float32)
        X.data = np.log1p(X.data)
        svd = TruncatedSVD(args.svd, random_state=0)
        S.append(svd.fit_transform(X))
        print(
            f"L{layer}: {len(keep)} SAE features active in >= {args.min_active}, "
            f"SVD-{args.svd} explains {svd.explained_variance_ratio_.sum():.2f}",
            flush=True,
        )
    code = np.array([codes.get(r, "") for r in ids], dtype=object)

    out = results_dir(EXPERIMENT, args.tag)
    inc_rows, feat_rows = [], []
    for target in args.targets:
        kind, row_filter, _ = probes.TARGETS[target]
        rows = row_filter(meta).to_numpy()
        sub = meta[rows].reset_index(drop=True)
        y = sub[target].to_numpy(float)
        groups = sub.run.to_numpy()
        M = probes.meta_features(meta, target)[rows]
        pob = sub.parent_over_budget
        M = np.column_stack([M, pob.fillna(0), pob.isna()])
        blocks = {
            "M": (M, "dense"),
            "L": (L[rows], "dense"),
            "C": (C[rows], "dense"),
            "A": (A[rows], "score"),
            "S": ([X[rows] for X in S], "score"),
            "T": (code[rows], "text"),
        }
        splits = list(GroupKFold(args.n_outer).split(M, y, groups))
        print(f"== {target}: n={len(y)}, {len(splits)} outer folds", flush=True)
        res = Parallel(n_jobs=args.n_jobs)(
            delayed(outer_fold)(tr, te, blocks, y, groups, kind, args.n_inner)
            for tr, te in splits
        )
        oof = {m: np.full(len(y), np.nan) for m in MODELS}
        for te, preds in res:
            for m, p in preds.items():
                oof[m][te] = p
        pd.DataFrame(
            {"record_id": sub.record_id, "run": groups, target: y}
            | {f"pred_{m}": oof[m] for m in MODELS}
        ).to_csv(out / f"oof_outcome_{target}.csv", index=False)
        boots = list(run_bootstrap(groups, args.n_boot, rng))
        for m in MODELS:
            vals = [metric(kind, y[i], oof[m][i]) for i in boots]
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            inc_rows.append(
                {
                    "target": target,
                    "n": len(y),
                    "base_rate": float(y.mean()) if kind == "binary" else np.nan,
                    "model": m,
                    "score": metric(kind, y, oof[m]),
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
        for a, b in COMPARISONS:
            d = [
                metric(kind, y[i], oof[b][i]) - metric(kind, y[i], oof[a][i])
                for i in boots
            ]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            inc_rows.append(
                {
                    "target": target,
                    "n": len(y),
                    "model": f"{b} - {a}",
                    "score": metric(kind, y, oof[b]) - metric(kind, y, oof[a]),
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
        for row in inc_rows[-len(MODELS) - len(COMPARISONS) :]:
            print(
                f"  {row['model']:20s} {row['score']:+.3f} "
                f"[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]",
                flush=True,
            )

        # per robust feature, within the solutions that use the concept
        for j, r in enumerate(robust.itertuples()):
            uses = labels[r.concept].to_numpy()[rows] == 1
            x, yy, gg = C[rows][uses, j], y[uses], groups[uses]
            stat = lambda i: (
                metric(kind, yy[i], x[i])
                if kind == "binary"
                else spearmanr(yy[i], x[i])[0]
            )
            vals = [stat(i) for i in run_bootstrap(gg, args.n_boot, rng)]
            lo, hi = np.nanpercentile(vals, [2.5, 97.5])
            feat_rows.append(
                {
                    "target": target,
                    "concept": r.concept,
                    "layer": r.layer,
                    "feature": r.feature,
                    "n_uses": int(uses.sum()),
                    "stat": "auroc" if kind == "binary" else "spearman",
                    "value": stat(np.arange(len(yy))),
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
        print(f"  [{time.time() - t0:.0f} s]", flush=True)

    inc = pd.DataFrame(inc_rows)
    feats = pd.DataFrame(feat_rows)
    inc.to_csv(out / "concept_outcomes.csv", index=False)
    feats.to_csv(out / "concept_outcome_features.csv", index=False)
    md = [
        "# Concept features vs outcome",
        "",
        "Out-of-fold scores (AUROC for `child_failed` / `over_budget`,",
        "Spearman for `child_fitness`), 95% bootstrap CI over runs. Blocks:",
        "M metadata, L concept labels + length, C line-mask-robust concept",
        "features, A all selected concept features, S whole SAE code, T code",
        "TF-IDF.",
        "",
        to_markdown(inc.drop(columns="base_rate")),
        "",
        "## Robust concept features within the solutions that use the concept",
        "",
        to_markdown(feats),
        "",
    ]
    (out / "concept_outcomes.md").write_text("\n".join(md))
    print(f"wrote {out} in {time.time() - t0:.0f} s")


def to_markdown(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
