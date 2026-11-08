"""Step 9: which SAE features predict that an algorithm fails?

Step 8 shows that the whole SAE code predicts ``child_failed`` well beyond
the concept labels and the code text, while the concept features do not.
This step looks for the features that carry it: every feature is scored
by the AUROC of its per-solution max for ``child_failed`` on the
discovery half of the runs (the split of step 4) and the best ones, in
both directions (active in failing / in working code), are re-scored on
the validation half and rendered with their top tokens and contexts.

Writes ``results/<tag>/failure_features.csv`` and
``failure_features.md``.
"""

import argparse
import importlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp03_concept_features
sys.path.insert(0, str(HERE))

from common.paths import data_dir, results_dir
from exp_config import EXPERIMENT, SOURCE_EXPERIMENT
from records import load_records

step4 = importlib.import_module("04_concept_features")
step5 = importlib.import_module("05_feature_contexts")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="k100")
    ap.add_argument("--n-features", type=int, default=10)
    ap.add_argument("--min-active", type=float, default=0.01)
    ap.add_argument("--n-contexts", type=int, default=4)
    args = ap.parse_args()

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
    y = meta.child_failed.to_numpy(float)
    run = meta.run.to_numpy()
    rng = np.random.default_rng(0)
    runs = np.unique(run)
    disc_runs = set(rng.permutation(runs)[: len(runs) // 2])
    disc = np.array([r in disc_runs for r in run])
    val = ~disc
    print(f"{len(y)} solutions, failure rate {y.mean():.3f}")

    recs = load_records(tags)
    ctx = step5.Contexts(recs)
    rows, md = [], ["# Failure-predictive SAE features", ""]
    for layer in enc["layers"]:
        X = enc[(layer, "max")].tocsr()
        active = np.asarray((X > 0).mean(0)).ravel()
        R = step4.column_ranks(X, disc)
        a = step4.auroc_all(R, y[disc].astype(np.float32))
        a[active < args.min_active] = 0.5
        Xv = X[val]
        val_top, row_top, tok_top = enc[(layer, "top")]
        md += [f"## L{layer}", ""]
        for direction, order in (
            ("fails", np.argsort(-a)),
            ("works", np.argsort(a)),
        ):
            for rank, f in enumerate(order[: args.n_features]):
                xv = Xv[:, f].toarray().ravel()
                on = xv > 0
                ex = [
                    (v, ids[r], int(t))
                    for v, r, t in zip(val_top[f], row_top[f], tok_top[f])
                    if r >= 0
                ]
                common = Counter(ctx.token(rid, t).strip() for _, rid, t in ex)
                common = common.most_common(3)
                # failure rate of the top-example solutions
                fail_top = float(np.mean([meta.child_failed[rid] for _, rid, _ in ex]))
                row = {
                    "layer": layer,
                    "direction": direction,
                    "rank": rank,
                    "feature": int(f),
                    "auroc_disc": float(a[f]),
                    "auroc_val": roc_auc_score(y[val], xv),
                    "p_active_fail": float(on[y[val] == 1].mean()),
                    "p_active_work": float(on[y[val] == 0].mean()),
                    "fail_rate_top": fail_top,
                    "top_tokens": " | ".join(f"{t!r}:{n}" for t, n in common),
                }
                rows.append(row)
                md.append(
                    f"- **f{f}** ({direction}) AUROC {row['auroc_val']:.3f} (val), "
                    f"active {row['p_active_fail']:.2f} in failing / "
                    f"{row['p_active_work']:.2f} in working code; top examples "
                    f"fail {fail_top:.2f}; max tokens "
                    + ", ".join(f"`{t!r}` ×{n}" for t, n in common)
                )
                for v, rid, t in ex[: args.n_contexts]:
                    md.append(
                        f"  - {v:.1f} ({'fail' if meta.child_failed[rid] else 'ok'}):"
                        f" `{ctx(rid, t)}`"
                    )
            md.append("")
        print(
            pd.DataFrame(rows)
            .query("layer == @layer")[
                [
                    "direction",
                    "feature",
                    "auroc_val",
                    "p_active_fail",
                    "p_active_work",
                    "top_tokens",
                ]
            ]
            .round(3)
            .to_string(index=False),
            flush=True,
        )
    out = results_dir(EXPERIMENT, args.tag)
    pd.DataFrame(rows).to_csv(out / "failure_features.csv", index=False)
    (out / "failure_features.md").write_text("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
