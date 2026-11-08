"""Step 11: which failure types does the SAE code predict?

``child_failed`` lumps together very different errors. The logs keep the
error message of every failing algorithm (``metadata["true_error"]`` for
pre-screened ones), so failures are split by exception type: the
``--min-count`` most common types each become a target (that type vs
working algorithms; other failures are left out).

Per type, with ``GroupKFold`` over runs (out of fold):

* ``T``: code TF-IDF, L2 logistic regression;
* ``S``: SAE code at ``--layer`` (log1p per-solution max, max-abs scaled),
  L1 logistic regression at ``--C``;
* ``T+S``: second-level logistic regression on both out-of-fold scores
  (grouped CV again); ``delta`` = T+S - T with a 95% bootstrap CI over
  runs.

The L1 probe refitted on all rows names the features of each type
(largest positive coefficients, with their most common max tokens).

Writes ``results/<tag>/failure_types.csv``, ``failure_type_features.csv``
and ``failure_types.md``.
"""

import argparse
import importlib
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MaxAbsScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp03_concept_features
sys.path.insert(0, str(HERE))

from common.paths import data_dir, results_dir
from common.runs import iter_run_dirs, load_run
from exp_config import EXPERIMENT, SOURCE_EXPERIMENT
from records import load_records

step4 = importlib.import_module("04_concept_features")
step5 = importlib.import_module("05_feature_contexts")
step10 = importlib.import_module("10_sparse_failure_probe")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inconsistent values")


def error_type(message: str) -> str:
    if "timed out" in message.lower():
        return "Timeout"
    found = re.findall(r"\b([A-Z]\w*(?:Error|Exception))\b", message)
    return found[-1] if found else "other"


def load_errors(tags) -> dict[str, str]:
    """record id -> error message of the generated algorithm ("" if none)."""
    out = {}
    for tag in tags:
        for run_dir in iter_run_dirs(data_dir(SOURCE_EXPERIMENT, "raw", tag)):
            solutions, records = load_run(run_dir)
            for rec in records:
                sol = solutions.get(rec.get("solution_id")) or {}
                meta = sol.get("metadata") or {}
                err = meta.get("true_error") if "true_fitness" in meta else None
                out[rec["record_id"]] = str(err or sol.get("error") or "")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="k100")
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--min-count", type=int, default=60)
    ap.add_argument("--min-active", type=float, default=0.005)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-named", type=int, default=8)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = results_dir(EXPERIMENT, args.tag)

    enc = step4.load_encodings(args.tag)
    ids = enc["record_id"]
    labels = pd.read_csv(data_dir(EXPERIMENT, "processed") / "labels.csv")
    tags = sorted(labels.tag.unique())
    meta = pd.concat(
        [
            pd.read_csv(
                data_dir(SOURCE_EXPERIMENT, "processed", t) / "meta.csv",
                dtype={"record_id": str, "run": str},
            ).assign(run=lambda m, t=t: t + "/" + m.run)
            for t in tags
        ]
    ).set_index("record_id")
    meta = meta.loc[ids]
    errors = load_errors(tags)
    failed = meta.child_failed.to_numpy() == 1
    etype = np.array(
        [error_type(errors.get(r, "")) if f else "" for r, f in zip(ids, failed)]
    )
    codes = {}
    for t in tags:
        with open(data_dir(SOURCE_EXPERIMENT, "processed", t) / "code.jsonl") as fh:
            for line in fh:
                c = json.loads(line)
                codes[c["record_id"]] = c["code"]
    code = np.array([codes.get(r, "") for r in ids], dtype=object)
    groups = meta.run.to_numpy()
    counts = Counter(etype[failed])
    print("failure types:", dict(counts.most_common()))
    (out / "failure_type_counts.json").write_text(
        json.dumps(dict(counts.most_common()), indent=1)
    )
    types = [t for t, n in counts.most_common() if n >= args.min_count]

    X = enc[(args.layer, "max")].tocsr()
    cols = np.flatnonzero(np.asarray((X > 0).mean(0)).ravel() >= args.min_active)
    X = X[:, cols].astype(np.float64)
    X.data = np.log1p(X.data)
    val_top, row_top, tok_top = enc[(args.layer, "top")]
    ctx = step5.Contexts(load_records(tags))

    rows, feat_rows, md = [], [], ["# Failure types", ""]
    for t in types:
        keep = (~failed) | (etype == t)
        y = (etype[keep] == t).astype(int)
        g = groups[keep]
        Xk, ck = X[keep], code[keep]
        p_T = np.full(len(y), np.nan)
        for tr, te in GroupKFold(args.n_splits).split(ck, y, g):
            vec = TfidfVectorizer(
                token_pattern=r"[A-Za-z_][A-Za-z0-9_]*", min_df=2, max_features=5000
            )
            m = LogisticRegression(max_iter=2000, class_weight="balanced")
            m.fit(vec.fit_transform(ck[tr]), y[tr])
            p_T[te] = m.decision_function(vec.transform(ck[te]))
        p_S, n_feat = step10.l1_oof(Xk, y, g, args.C, args.n_splits)
        p_TS = step10.stacked_oof(np.column_stack([p_T, p_S]), y, g, args.n_splits)
        d, lo, hi = step10.delta_ci(y, p_T, p_TS, g, args.n_boot, rng)
        row = {
            "type": t,
            "n_type": int(y.sum()),
            "n_working": int((y == 0).sum()),
            "auroc_T": step10.auroc(y, p_T),
            "auroc_S": step10.auroc(y, p_S),
            "auroc_T+S": step10.auroc(y, p_TS),
            "delta": d,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_features_S": n_feat,
        }
        rows.append(row)
        print(
            f"{t:22s} n={y.sum():4d}  T {row['auroc_T']:.3f}  S {row['auroc_S']:.3f}"
            f"  T+S {row['auroc_T+S']:.3f}  delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]",
            flush=True,
        )
        # named features: L1 probe on all rows of this type vs working
        sc = MaxAbsScaler().fit(Xk)
        m = LogisticRegression(penalty="l1", C=args.C, solver="liblinear")
        m.fit(sc.transform(Xk), y)
        coef = m.coef_[0]
        md += [
            f"## {t} (n = {y.sum()}): T {row['auroc_T']:.3f}, S "
            f"{row['auroc_S']:.3f}, T+S {row['auroc_T+S']:.3f}",
            "",
            "Example error: `"
            + next(errors[r] for r, e in zip(ids, etype) if e == t)
            .replace("\n", " ")
            .replace("`", "'")[-220:]
            + "`",
            "",
        ]
        for j in np.argsort(-coef)[: args.n_named]:
            if coef[j] <= 0:
                break
            f = int(cols[j])
            x = Xk[:, j].toarray().ravel()
            ex = [
                (v, ids[r], int(k))
                for v, r, k in zip(val_top[f], row_top[f], tok_top[f])
                if r >= 0
            ]
            common = Counter(ctx.token(rid, k).strip() for _, rid, k in ex)
            common = common.most_common(3)
            fr = {
                "type": t,
                "layer": args.layer,
                "feature": f,
                "coef": float(coef[j]),
                "auroc": step10.auroc(y, x),
                "p_active_type": float((x[y == 1] > 0).mean()),
                "p_active_work": float((x[y == 0] > 0).mean()),
                "top_tokens": " | ".join(f"{a!r}:{n}" for a, n in common),
            }
            feat_rows.append(fr)
            md.append(
                f"- **f{f}** coef {fr['coef']:+.2f}, AUROC {fr['auroc']:.3f}, active "
                f"{fr['p_active_type']:.2f} / {fr['p_active_work']:.2f}; max tokens "
                + ", ".join(f"`{a!r}` ×{n}" for a, n in common)
            )
            for v, rid, k in ex[:2]:
                md.append(f"  - {v:.1f}: `{ctx(rid, k)}`")
        md.append("")

    df = pd.DataFrame(rows)
    df.to_csv(out / "failure_types.csv", index=False)
    pd.DataFrame(feat_rows).to_csv(out / "failure_type_features.csv", index=False)
    md = md[:2] + [step10.to_markdown(df), ""] + md[2:]
    (out / "failure_types.md").write_text("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
