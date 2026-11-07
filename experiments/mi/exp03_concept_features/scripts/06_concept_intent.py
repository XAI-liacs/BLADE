"""Step 6: does the model know which concepts it will write before it writes?

At the last prompt token (before any code is generated) the SAE code of
the residual stream is compared with the concepts that end up in the
child. Most children inherit their parent's concepts, so the reference is
a model that knows the parent's concepts and the operator:

* ``B``: parent concept labels + operator (logistic regression);
* ``S``: the last-token SAE code at one layer (L2 logistic regression);
* ``B+S``: both.

Scores are AUROCs from 5-fold cross-validation grouped by run. They are
reported for all children (``present``) and for the children whose parent
lacks the concept (``adopted``: does the model plan to *introduce* it?).
For ``adopted`` the code-side concept feature of step 4 (rank 0, same
layer) is also scored on its own at the last prompt token
(``feature``): is the concept feature already active before writing?

Writes ``results/<tag>/concept_intent.csv``.
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp03_concept_features
sys.path.insert(0, str(HERE))

from common.paths import data_dir, results_dir
from exp_config import EXPERIMENT
from records import load_records

concept_labels = importlib.import_module("03_concept_labels")


def last_token_codes(tag: str):
    files = sorted(data_dir(EXPERIMENT, "processed", tag).glob("shard-*.npz"))
    zs = [np.load(f) for f in files]
    ids = np.concatenate([z["record_id"] for z in zs])
    d_sae = int(zs[0]["d_sae"])
    out = {}
    for l in zs[0]["layers"]:
        idx = np.concatenate([z[f"L{l}_last_idx"] for z in zs])
        val = np.concatenate([z[f"L{l}_last_val"] for z in zs]).astype(np.float32)
        rows = np.repeat(np.arange(len(ids)), idx.shape[1])
        X = sparse.csr_matrix(
            (np.log1p(val.ravel()), (rows, idx.ravel())), shape=(len(ids), d_sae)
        )
        out[int(l)] = X
    return ids, out


def cv_scores(X, y, groups, C, n_splits=5):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        model = LogisticRegression(C=C, max_iter=2000)
        oof[te] = model.fit(X[tr], y[tr]).decision_function(X[te])
    return oof


def auroc(y, p):
    ok = np.isfinite(p)
    return roc_auc_score(y[ok], p[ok]) if len(np.unique(y[ok])) == 2 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="k100")
    ap.add_argument("--min-positives", type=int, default=40)
    ap.add_argument("--C", type=float, default=0.05)
    args = ap.parse_args()

    ids, codes = last_token_codes(args.tag)
    labels = pd.read_csv(data_dir(EXPERIMENT, "processed") / "labels.csv")
    labels = labels.set_index("record_id").loc[ids]
    concepts = [c for c in concept_labels.CONCEPTS if c in labels.columns]
    recs = load_records(sorted(labels.tag.unique())).set_index("record_id").loc[ids]
    keep = ((recs.operator != "init") & (recs.parent_code != "")).to_numpy()
    parent = pd.DataFrame(
        [concept_labels.labels(c) for c in recs.parent_code[keep]],
        index=recs.index[keep],
    )
    op = OneHotEncoder().fit_transform(recs.loc[keep, ["operator"]]).toarray()
    B = np.column_stack([parent[concepts].to_numpy(), op])
    groups = recs.loc[keep, "run"].to_numpy()
    feats = pd.read_csv(results_dir(EXPERIMENT, args.tag) / "concept_features.csv")
    print(f"{keep.sum()} children with a parent")

    rows = []
    for c in concepts:
        y_all = labels.loc[keep, c].to_numpy()
        lacks = parent[c].to_numpy() == 0
        if y_all.sum() < args.min_positives or y_all[lacks].sum() < args.min_positives:
            continue
        p_B = cv_scores(B, y_all, groups, C=1.0)
        for layer, X in codes.items():
            Xk = X[keep]
            p_S = cv_scores(Xk, y_all, groups, args.C)
            BS = sparse.hstack([sparse.csr_matrix(B), Xk]).tocsr()
            p_BS = cv_scores(BS, y_all, groups, args.C)
            # adopted: children whose parent lacks the concept, refitted
            y_ad, g_ad = y_all[lacks], groups[lacks]
            q_B = cv_scores(B[lacks], y_ad, g_ad, C=1.0)
            q_S = cv_scores(Xk[lacks], y_ad, g_ad, args.C)
            q_BS = cv_scores(BS[lacks], y_ad, g_ad, args.C)
            f = feats.query("layer == @layer and concept == @c and rank == 0")
            fid = int(f.feature.iloc[0]) if len(f) else None
            f_act = Xk[lacks][:, fid].toarray().ravel() if fid is not None else None
            row = {
                "concept": c,
                "layer": layer,
                "n": int(keep.sum()),
                "prevalence": float(y_all.mean()),
                "n_adopted": int(y_ad.sum()),
                "n_lacks": int(lacks.sum()),
                "present_B": auroc(y_all, p_B),
                "present_S": auroc(y_all, p_S),
                "present_B+S": auroc(y_all, p_BS),
                "adopted_B": auroc(y_ad, q_B),
                "adopted_S": auroc(y_ad, q_S),
                "adopted_B+S": auroc(y_ad, q_BS),
                "feature": fid,
                "adopted_feature": auroc(y_ad, f_act) if fid is not None else np.nan,
                "feature_active_last": (
                    float((f_act > 0).mean()) if fid is not None else np.nan
                ),
            }
            rows.append(row)
            print(
                f"{c:24s} L{layer:<2d} present B {row['present_B']:.3f} "
                f"S {row['present_S']:.3f} B+S {row['present_B+S']:.3f} | "
                f"adopted ({row['n_adopted']}/{row['n_lacks']}) B {row['adopted_B']:.3f} "
                f"S {row['adopted_S']:.3f} B+S {row['adopted_B+S']:.3f} "
                f"f {row['adopted_feature']:.3f}",
                flush=True,
            )
    out = results_dir(EXPERIMENT, args.tag)
    pd.DataFrame(rows).to_csv(out / "concept_intent.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
