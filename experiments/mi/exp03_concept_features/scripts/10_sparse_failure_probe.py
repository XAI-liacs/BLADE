"""Step 10: name the SAE features behind the failure signal, and check it.

Step 8 found that the SAE code predicts ``child_failed`` beyond metadata,
concept labels and code text (``M+L+T+S`` vs ``M+L+T``); step 9 found no
single strong feature. This step

1. **Strata.** Re-scores step 8's out-of-fold predictions within strata
   where an obvious explanation is held fixed: without surrogate
   algorithms (93% fail), without surrogate or restart, per operator,
   and by whether the parent failed. ``delta`` = AUROC(M+L+T+S) -
   AUROC(M+L+T) with a 95% bootstrap CI over runs.
2. **Sparse probes.** L1 logistic regressions on the SAE code alone
   (log1p per-solution max, features active in >= 0.5% of solutions,
   max-abs scaled), per layer and on all layers concatenated, over a
   grid of ``C``: out-of-fold AUROC (``GroupKFold`` over runs) against
   the number of features used. Each sparse score is then stacked with
   step 8's out-of-fold ``M+L+T`` prediction in a second-level logistic
   regression (again grouped CV): how few features recover the gain?
3. **Named features.** The probe at ``--name-C`` on the best layer
   (all data): every non-zero feature with its coefficient, univariate
   AUROC, activity in failing / working code and most common max tokens.

Writes ``results/<tag>/failure_strata.csv``, ``failure_sparse.csv``,
``failure_probe_features.csv`` and ``failure_probe.md``.
"""

import argparse
import importlib
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MaxAbsScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp03_concept_features
sys.path.insert(0, str(HERE))

from common.paths import data_dir, results_dir
from exp_config import EXPERIMENT, SOURCE_EXPERIMENT
from records import load_records

step4 = importlib.import_module("04_concept_features")
step5 = importlib.import_module("05_feature_contexts")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Inconsistent values")

C_GRID = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1)


def auroc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def boot_runs(groups, n_boot, rng):
    runs = np.unique(groups)
    by_run = {r: np.flatnonzero(groups == r) for r in runs}
    for _ in range(n_boot):
        yield np.concatenate([by_run[r] for r in rng.choice(runs, len(runs))])


def delta_ci(y, a, b, groups, n_boot, rng):
    d = [auroc(y[i], b[i]) - auroc(y[i], a[i]) for i in boot_runs(groups, n_boot, rng)]
    return auroc(y, b) - auroc(y, a), *np.nanpercentile(d, [2.5, 97.5])


def l1_oof(X, y, groups, C, n_splits):
    oof = np.full(len(y), np.nan)
    n_feat = []
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        sc = MaxAbsScaler().fit(X[tr])
        m = LogisticRegression(penalty="l1", C=C, solver="liblinear")
        m.fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.decision_function(sc.transform(X[te]))
        n_feat.append(int((m.coef_ != 0).sum()))
    return oof, float(np.mean(n_feat))


def stacked_oof(F, y, groups, n_splits):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(F, y, groups):
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        m = LogisticRegression(max_iter=1000).fit((F[tr] - mu) / sd, y[tr])
        oof[te] = m.decision_function((F[te] - mu) / sd)
    return oof


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="k100")
    ap.add_argument("--min-active", type=float, default=0.005)
    ap.add_argument("--n-splits", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--name-C", type=float, default=0.01)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = results_dir(EXPERIMENT, args.tag)

    enc = step4.load_encodings(args.tag)
    ids = enc["record_id"]
    oof8 = pd.read_csv(out / "oof_outcome_child_failed.csv").set_index("record_id")
    oof8 = oof8.loc[ids]
    labels = pd.read_csv(data_dir(EXPERIMENT, "processed") / "labels.csv")
    labels = labels.set_index("record_id").loc[ids]
    meta = pd.concat(
        [
            pd.read_csv(
                data_dir(SOURCE_EXPERIMENT, "processed", t) / "meta.csv",
                dtype={"record_id": str, "run": str},
            )
            for t in sorted(labels.tag.unique())
        ]
    ).set_index("record_id")
    meta = meta.loc[ids]
    y = oof8.child_failed.to_numpy().astype(int)
    groups = oof8.run.to_numpy()
    base, full = oof8["pred_M+L+T"].to_numpy(), oof8["pred_M+L+T+S"].to_numpy()

    # 1. strata -----------------------------------------------------------
    strata = {
        "all": np.ones(len(y), bool),
        "no surrogate": labels.surrogate.to_numpy() == 0,
        "no surrogate/restart": (labels.surrogate == 0).to_numpy()
        & (labels.restart == 0).to_numpy(),
        "parent worked": (meta.parent_failed == False).to_numpy()  # noqa: E712
        & meta.operator.ne("init").to_numpy(),
        "parent failed": (meta.parent_failed == True).to_numpy(),  # noqa: E712
    }
    for op in sorted(meta.operator.dropna().unique()):
        strata[f"operator={op}"] = (meta.operator == op).to_numpy()
    strata_rows = []
    for name, m in strata.items():
        d, lo, hi = delta_ci(y[m], base[m], full[m], groups[m], args.n_boot, rng)
        strata_rows.append(
            {
                "stratum": name,
                "n": int(m.sum()),
                "fail_rate": float(y[m].mean()),
                "auroc_MLT": auroc(y[m], base[m]),
                "auroc_MLTS": auroc(y[m], full[m]),
                "delta": d,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
        print(
            f"{name:24s} n={m.sum():5d} fail {y[m].mean():.3f} "
            f"{strata_rows[-1]['auroc_MLT']:.3f} -> {strata_rows[-1]['auroc_MLTS']:.3f} "
            f"delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}]",
            flush=True,
        )
    pd.DataFrame(strata_rows).to_csv(out / "failure_strata.csv", index=False)

    # 2. sparse probes ------------------------------------------------------
    mats, cols = {}, {}
    for layer in enc["layers"]:
        X = enc[(layer, "max")].tocsr()
        keep = np.flatnonzero(np.asarray((X > 0).mean(0)).ravel() >= args.min_active)
        X = X[:, keep].astype(np.float64)
        X.data = np.log1p(X.data)
        mats[layer], cols[layer] = X, keep
    mats["all"] = sparse.hstack([mats[l] for l in enc["layers"]]).tocsr()
    jobs = [(k, C) for k in mats for C in C_GRID]
    res = Parallel(n_jobs=args.n_jobs)(
        delayed(l1_oof)(mats[k], y, groups, C, args.n_splits) for k, C in jobs
    )
    sparse_rows = []
    for (k, C), (oof, nf) in zip(jobs, res):
        st = stacked_oof(np.column_stack([base, oof]), y, groups, args.n_splits)
        d, lo, hi = delta_ci(y, base, st, groups, args.n_boot, rng)
        sparse_rows.append(
            {
                "layer": k,
                "C": C,
                "n_features": nf,
                "auroc_S": auroc(y, oof),
                "auroc_MLT+S": auroc(y, st),
                "delta_vs_MLT": d,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
        print(
            f"L{k} C={C}: {nf:6.1f} feats, S {auroc(y, oof):.3f}, "
            f"M+L+T+S {auroc(y, st):.3f} (delta {d:+.3f} [{lo:+.3f}, {hi:+.3f}])",
            flush=True,
        )
    sp = pd.DataFrame(sparse_rows)
    sp.to_csv(out / "failure_sparse.csv", index=False)

    # 3. named features at --name-C on the best single layer ------------------
    per_layer = sp[(sp.layer != "all") & (sp.C == args.name_C)]
    layer = int(per_layer.loc[per_layer["auroc_MLT+S"].idxmax(), "layer"])
    X = mats[layer]
    sc = MaxAbsScaler().fit(X)
    m = LogisticRegression(penalty="l1", C=args.name_C, solver="liblinear")
    m.fit(sc.transform(X), y)
    nz = np.flatnonzero(m.coef_[0])
    val_top, row_top, tok_top = enc[(layer, "top")]
    ctx = step5.Contexts(load_records(sorted(labels.tag.unique())))
    feat_rows, md_feats = [], []
    for j in nz[np.argsort(-np.abs(m.coef_[0][nz]))]:
        f = int(cols[layer][j])
        x = X[:, j].toarray().ravel()
        ex = [
            (v, ids[r], int(t))
            for v, r, t in zip(val_top[f], row_top[f], tok_top[f])
            if r >= 0
        ]
        common = Counter(ctx.token(rid, t).strip() for _, rid, t in ex).most_common(3)
        row = {
            "layer": layer,
            "feature": f,
            "coef": float(m.coef_[0][j]),
            "auroc": auroc(y, x),
            "p_active_fail": float((x[y == 1] > 0).mean()),
            "p_active_work": float((x[y == 0] > 0).mean()),
            "fail_rate_top": float(np.mean([y[ids == rid][0] for _, rid, _ in ex])),
            "top_tokens": " | ".join(f"{t!r}:{n}" for t, n in common),
        }
        feat_rows.append(row)
        md_feats.append(
            f"- **f{f}** coef {row['coef']:+.2f}, AUROC {row['auroc']:.3f}, active "
            f"{row['p_active_fail']:.2f} / {row['p_active_work']:.2f} "
            f"(failing / working); top examples fail {row['fail_rate_top']:.2f}; "
            "max tokens " + ", ".join(f"`{t!r}` ×{n}" for t, n in common)
        )
        for v, rid, t in ex[:3]:
            md_feats.append(f"  - {v:.1f}: `{ctx(rid, t)}`")
    fr = pd.DataFrame(feat_rows)
    fr.to_csv(out / "failure_probe_features.csv", index=False)
    print(fr.drop(columns="layer").round(3).to_string(index=False))

    md = [
        "# Failure probe on the SAE code",
        "",
        "## Gain of the SAE code over metadata + labels + text, by stratum",
        "",
        to_markdown(pd.DataFrame(strata_rows)),
        "",
        "## Sparse (L1) probes: out-of-fold AUROC vs number of features",
        "",
        to_markdown(sp),
        "",
        f"## Features of the L1 probe at C = {args.name_C}, layer {layer} "
        "(all data)",
        "",
        *md_feats,
        "",
    ]
    (out / "failure_probe.md").write_text("\n".join(md))
    print(f"wrote {out}")


def to_markdown(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
