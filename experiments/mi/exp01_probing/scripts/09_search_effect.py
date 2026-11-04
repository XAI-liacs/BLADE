"""Step 9: does acting on the pre-screen harm the search? (pooled seeds)

Pools pre-screen campaigns (default ``prescreen`` + ``prescreen2``, 36
seeds, each run under ``shadow`` and ``screened``) and tests the
pre-registered hypothesis:

* endpoint: best fitness LLaMEA could select in a run (rejected algorithms
  are invisible to it);
* non-inferiority margin: 0.10 AOCC;
* test: one-sided paired t-test of ``screened - shadow`` against ``-margin``
  at alpha = 0.05, i.e. the lower end of the two-sided 90% CI must lie
  above ``-margin``. A paired Wilcoxon test is reported as a check.

The margin and the number of seeds were fixed from a power analysis on the
first 12 seeds (``--power-from``) before the extra seeds were run; this
script recomputes that analysis so the report can show it.

Writes ``results/<out-tag>/``: ``search_effect.csv``, ``power.csv`` and
``search_runs.csv``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import data_dir, results_dir
from exp_config import EXPERIMENT


def run_table(tags: list[str]) -> pd.DataFrame:
    frames = []
    for tag in tags:
        m = pd.read_csv(data_dir(EXPERIMENT, "processed", tag) / "meta.csv")
        frames.append(m.assign(campaign=tag))
    m = pd.concat(frames).sort_values(["run", "query_idx"])
    m["visible_fitness"] = m["child_fitness"].where(m["visible_failed"] == 0)
    m["eval_idx"] = m.groupby(["campaign", "run"]).cumcount() + 1
    m["best_so_far"] = m.groupby(["campaign", "run"])["visible_fitness"].cummax()
    runs = (
        m.groupby(["campaign", "seed", "condition"])
        .agg(
            n=("record_id", "size"),
            best=("visible_fitness", "max"),
            failure_rate=("child_failed", "mean"),
            flagged=("prescreen_flagged", "mean"),
        )
        .reset_index()
    )
    curves = m[["campaign", "seed", "condition", "eval_idx", "best_so_far"]]
    return runs, curves


def paired_power(n, delta, sd, alpha=0.05, one_sided=False):
    """Power of a paired t-test for a true mean difference ``delta``."""
    df, nc = n - 1, delta / (sd / np.sqrt(n))
    if one_sided:
        return 1 - stats.nct.cdf(stats.t.ppf(1 - alpha, df), df, nc)
    tc = stats.t.ppf(1 - alpha / 2, df)
    return 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tags", nargs="*", default=["prescreen", "prescreen2"])
    ap.add_argument("--power-from", default="prescreen")
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--out-tag", default="prescreen")
    args = ap.parse_args()
    out = results_dir(EXPERIMENT, args.out_tag)

    # Power analysis from the first campaign only (as done before the rest).
    first, _ = run_table([args.power_from])
    w0 = first.pivot(index="seed", columns="condition", values="best")
    sd = float((w0["screened"] - w0["shadow"]).std())
    power_rows = []
    for n in (12, 24, 36, 48, 72):
        mde = next(
            d for d in np.arange(0.005, 1, 0.005) if paired_power(n, d, sd) >= 0.8
        )
        power_rows.append(
            {
                "n_seeds": n,
                "sd_diff": sd,
                "mde_80": mde,
                "ni_power_margin_0.05": paired_power(n, 0.05, sd, one_sided=True),
                "ni_power_margin_0.10": paired_power(n, 0.10, sd, one_sided=True),
            }
        )
    power = pd.DataFrame(power_rows)
    power.to_csv(out / "power.csv", index=False)
    print(f"power analysis from {args.power_from} (sd of paired diff {sd:.3f}):")
    print(power.round(3).to_string(index=False))

    runs, curves = run_table(args.tags)
    runs.to_csv(out / "search_runs.csv", index=False)
    curves.to_csv(out / "search_curves.csv", index=False)
    w = runs.pivot(index="seed", columns="condition", values="best").dropna()
    d = w["screened"] - w["shadow"]
    n = len(d)
    se = d.std() / np.sqrt(n)
    t90 = stats.t.ppf(0.95, n - 1)
    lo, hi = d.mean() - t90 * se, d.mean() + t90 * se
    t_ni = (d.mean() + args.margin) / se
    p_ni = 1 - stats.t.cdf(t_ni, n - 1)
    fr = runs.pivot(index="seed", columns="condition", values="failure_rate").loc[
        w.index
    ]
    res = {
        "n_seeds": n,
        "shadow_mean": w["shadow"].mean(),
        "screened_mean": w["screened"].mean(),
        "shadow_median": w["shadow"].median(),
        "screened_median": w["screened"].median(),
        "mean_diff": d.mean(),
        "ci90_low": lo,
        "ci90_high": hi,
        "margin": args.margin,
        "p_noninferiority": p_ni,
        "non_inferior": bool(lo > -args.margin),
        "p_wilcoxon_two_sided": stats.wilcoxon(d).pvalue,
        "p_ttest_two_sided": stats.ttest_1samp(d, 0).pvalue,
        "corr_between_conditions": np.corrcoef(w["shadow"], w["screened"])[0, 1],
        "failure_rate_shadow": fr["shadow"].mean(),
        "failure_rate_screened": fr["screened"].mean(),
        "p_failure_rate_wilcoxon": stats.wilcoxon(fr["screened"] - fr["shadow"]).pvalue,
        "evaluations_skipped": runs.loc[
            runs["condition"] == "screened", "flagged"
        ].mean(),
    }
    pd.DataFrame([res]).to_csv(out / "search_effect.csv", index=False)
    print("\nsearch effect (pooled):")
    for k, v in res.items():
        print(f"  {k:26s} {v:.4f}" if isinstance(v, float) else f"  {k:26s} {v}")


if __name__ == "__main__":
    main()
