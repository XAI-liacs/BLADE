"""Step 4: which SAE features encode each metaheuristic concept?

For every layer and concept (labels from ``03_concept_labels.py``) the
LLaMEA runs are split in two halves (``discovery`` and ``validation``;
whole runs, so related algorithms of one lineage stay together):

* every feature is scored on the discovery half by the AUROC of its
  per-solution max activation for the concept label;
* the best features are re-scored on the validation half (``auroc_val``),
  with ``p_active_pos`` / ``p_active_neg`` (how often the feature fires in
  algorithms with / without the concept);
* an L1 logistic regression on all features (discovery half) gives the
  validation AUROC of a sparse combination (``multi_auroc_val``).

``05_feature_contexts.py`` renders the top-activating contexts of the
selected features.

Writes ``results/<tag>/concept_features.csv`` and ``concept_decoding.csv``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp03_concept_features

from common.paths import data_dir, results_dir
from exp_config import EXPERIMENT, MODEL
from records import load_records


def load_encodings(tag: str):
    """Merges the shards: record ids, per-layer sparse matrices, top examples."""
    files = sorted(data_dir(EXPERIMENT, "processed", tag).glob("shard-*.npz"))
    zs = [np.load(f) for f in files]
    layers = [int(l) for l in zs[0]["layers"]]
    d_sae = int(zs[0]["d_sae"])
    ids = np.concatenate([z["record_id"] for z in zs])
    out = {"record_id": ids, "layers": layers, "d_sae": d_sae}
    offsets = np.cumsum([0] + [len(z["record_id"]) for z in zs])
    for l in layers:
        for m in ("max", "mean", "count"):
            mats = [
                sparse.csr_matrix(
                    (
                        z[f"L{l}_{m}_data"].astype(np.float32),
                        z[f"L{l}_{m}_indices"],
                        z[f"L{l}_{m}_indptr"],
                    ),
                    shape=(len(z["record_id"]), d_sae),
                )
                for z in zs
            ]
            out[(l, m)] = sparse.vstack(mats).tocsc()
        val = np.concatenate([z[f"L{l}_top_val"] for z in zs], 1)
        row = np.concatenate(
            [
                np.where(z[f"L{l}_top_row"] >= 0, z[f"L{l}_top_row"] + o, -1)
                for z, o in zip(zs, offsets)
            ],
            1,
        )
        tok = np.concatenate([z[f"L{l}_top_tok"] for z in zs], 1)
        order = np.argsort(-val, 1)
        take = lambda a: np.take_along_axis(a, order, 1)
        out[(l, "top")] = (take(val), take(row), take(tok))
    return out


def column_ranks(X, rows, chunk=4096):
    """Ranks of every feature column over ``rows`` (dense, chunked)."""
    X = X[rows]
    R = np.empty(X.shape, np.float32)
    for a in range(0, X.shape[1], chunk):
        R[:, a : a + chunk] = rankdata(X[:, a : a + chunk].toarray(), axis=0)
    return R


def auroc_all(R, y):
    """AUROC of every feature column for binary labels ``y`` (from ranks)."""
    n1 = y.sum()
    n0 = len(y) - n1
    return (y @ R - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="k100")
    ap.add_argument("--min-prevalence", type=float, default=0.02)
    ap.add_argument("--n-features", type=int, default=5)
    ap.add_argument("--min-active", type=float, default=0.01)
    args = ap.parse_args()

    enc = load_encodings(args.tag)
    labels = pd.read_csv(data_dir(EXPERIMENT, "processed") / "labels.csv")
    labels = labels.set_index("record_id").loc[enc["record_id"]]
    concepts = [
        c
        for c in labels.columns.drop("tag")
        if args.min_prevalence <= labels[c].mean() <= 1 - args.min_prevalence
    ]
    recs = load_records(sorted(labels.tag.unique()))
    run = recs.set_index("record_id").loc[enc["record_id"], "run"].to_numpy()
    runs = np.unique(run)
    rng = np.random.default_rng(0)
    disc_runs = set(rng.permutation(runs)[: len(runs) // 2])
    disc = np.array([r in disc_runs for r in run])
    val = ~disc
    print(f"{len(run)} solutions, {len(runs)} runs; concepts: {concepts}")

    feat_rows, dec_rows = [], []
    for layer in enc["layers"]:
        X = enc[(layer, "max")].tocsr()
        R = column_ranks(X, disc)
        Xv = X[val]
        active = np.asarray((X > 0).mean(0)).ravel()
        cols = np.flatnonzero(active >= args.min_active)
        lr_X = X[:, cols].copy()
        lr_X.data = np.log1p(lr_X.data)
        for c in concepts:
            y = labels[c].to_numpy()
            a = auroc_all(R, y[disc].astype(np.float32))
            best = np.argsort(-a)[: args.n_features]
            for rank, f in enumerate(best):
                xv = Xv[:, f].toarray().ravel()
                on = xv > 0
                row = {
                    "layer": layer,
                    "concept": c,
                    "rank": rank,
                    "feature": int(f),
                    "auroc_disc": float(a[f]),
                    "auroc_val": roc_auc_score(y[val], xv),
                    "p_active_pos": float(on[y[val] == 1].mean()),
                    "p_active_neg": float(on[y[val] == 0].mean()),
                }
                feat_rows.append(row)
            lr = LogisticRegression(penalty="l1", C=0.05, solver="liblinear")
            lr.fit(lr_X[disc], y[disc])
            p = lr.decision_function(lr_X[val])
            dec_rows.append(
                {
                    "layer": layer,
                    "concept": c,
                    "prevalence": float(y.mean()),
                    "best_feature_auroc_val": feat_rows[-args.n_features]["auroc_val"],
                    "multi_auroc_val": roc_auc_score(y[val], p),
                    "multi_n_features": int((lr.coef_ != 0).sum()),
                }
            )
            print(
                f"L{layer} {c:24s} best f{best[0]} val {feat_rows[-args.n_features]['auroc_val']:.3f}"
                f"  multi {dec_rows[-1]['multi_auroc_val']:.3f}"
                f" ({dec_rows[-1]['multi_n_features']} feats)",
                flush=True,
            )
    out = results_dir(EXPERIMENT, args.tag)
    pd.DataFrame(feat_rows).to_csv(out / "concept_features.csv", index=False)
    pd.DataFrame(dec_rows).to_csv(out / "concept_decoding.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
