"""Pre-registered analysis of the confirmation campaign (``llamea-confirm``).

Implements ``paper_gecco2027/PREREG_confirm.md`` exactly:

* primary: ``adaptive`` > ``control`` on the best AOCC after 200 evaluated
  algorithms, one-sided paired t-test over the 24 seeds (316-339),
  alpha 0.05, with the mean difference, the 95% bootstrap CI (10,000
  resamples over seeds) and the number of seeds won;
* secondary: four two-sided paired t-tests, Holm-corrected over the four
  (adaptive vs adaptive_prompt, adaptive vs temp11, adaptive_prompt vs
  control, temp11 vs control);
* descriptive: the same comparisons at 100 evaluations and on the area
  under the best-so-far curve, failure rates, the adaptive dose and the
  share of prompted children, and the pooled 40-seed estimate of adaptive
  vs control (exploratory seeds 300-315 from ``results/budget200`` plus
  the confirmatory seeds).

Reads the raw logs directly (``data/raw/llamea-confirm/shard-*/run-*``),
so it does not depend on step 6. Writes ``results/llamea-confirm/
confirm_per_run.csv``, ``confirm_tests.csv`` and ``confirm.md``.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TAG = "llamea-confirm"
SEEDS = list(range(316, 340))
ARMS = ["control", "adaptive", "adaptive_prompt", "temp11"]
SECONDARY = [
    ("adaptive", "adaptive_prompt"),
    ("adaptive", "temp11"),
    ("adaptive_prompt", "control"),
    ("temp11", "control"),
]
RUN_RE = re.compile(r"run-LLaMEA-(.+?)-Qwen-.*-(\d+)$")


def load() -> pd.DataFrame:
    rows = []
    for d in sorted((ROOT / "data/raw" / TAG).glob("shard-*/run-*")):
        arm, seed = RUN_RE.match(d.name).groups()
        for i, line in enumerate((d / "log.jsonl").read_text().splitlines()):
            x = json.loads(line)
            m = x.get("metadata") or {}
            try:
                f = float(x["fitness"])
            except (TypeError, ValueError):
                f = float("-inf")
            rows.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "idx": i,
                    "aocc": f if np.isfinite(f) else np.nan,
                    "failed": not np.isfinite(f) or bool(x.get("error")),
                    "dose": m.get("adaptive_dose"),
                    "prompted": bool(m.get("prompt_suffix")),
                }
            )
    return pd.DataFrame(rows)


def per_run(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, seed), g in s.groupby(["arm", "seed"]):
        g = g.sort_values("idx")
        r = {"arm": arm, "seed": seed, "n": len(g)}
        for b in (100, 200):
            f = g[g.idx < b].aocc
            f = f.fillna(f.min())
            r[f"best{b}"] = f.max()
            r[f"auc{b}"] = f.cummax().mean()
        r["failure_rate"] = g.failed.mean()
        dose = pd.to_numeric(g.dose, errors="coerce")
        r["mean_dose"] = dose.mean()
        r["prompted_share"] = g.prompted.mean()
        rows.append(r)
    return pd.DataFrame(rows)


def boot_ci(d, rng, n=10000):
    m = rng.choice(d, (n, len(d)), replace=True).mean(1)
    return np.quantile(m, 0.025), np.quantile(m, 0.975)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def compare(w, a, b, metric, rng, alternative):
    d = (w[(metric, a)] - w[(metric, b)]).dropna().to_numpy()
    lo, hi = boot_ci(d, rng)
    t = stats.ttest_rel(
        w[(metric, a)].loc[w.index],
        w[(metric, b)].loc[w.index],
        alternative=alternative,
    )
    return {
        "a": a,
        "b": b,
        "metric": metric,
        "n": len(d),
        "diff": d.mean(),
        "lo95": lo,
        "hi95": hi,
        "wins": int((d > 0).sum()),
        "t": t.statistic,
        "p": t.pvalue,
        "alternative": alternative,
    }


def main():
    rng = np.random.default_rng(0)
    s = load()
    r = per_run(s)
    assert (r.n == 200).all() and len(r) == 96, r[r.n != 200]
    assert sorted(r.seed.unique()) == SEEDS and sorted(r.arm.unique()) == sorted(ARMS)
    dest = ROOT / "results" / TAG
    dest.mkdir(parents=True, exist_ok=True)
    r.to_csv(dest / "confirm_per_run.csv", index=False)
    w = r.pivot_table(index="seed", columns="arm")
    tests = [
        compare(w, "adaptive", "control", "best200", rng, "greater")
        | {"role": "primary"}
    ]
    sec = [
        compare(w, a, b, "best200", rng, "two-sided") | {"role": "secondary"}
        for a, b in SECONDARY
    ]
    for x, ph in zip(sec, holm([x["p"] for x in sec])):
        x["p_holm"] = ph
    tests += sec
    for metric in ("best100", "auc200", "auc100"):
        for a, b in [("adaptive", "control")] + SECONDARY:
            tests.append(
                compare(w, a, b, metric, rng, "two-sided") | {"role": "descriptive"}
            )
    # pooled 40 seeds: exploratory 300-315 + confirmatory
    ex = pd.read_csv(ROOT / "results/budget200/per_run.csv")
    ex = ex[ex.arm.isin(["adaptive", "control"])].pivot_table(
        index="seed", columns="arm", values="best200"
    )
    cf = w["best200"][["adaptive", "control"]]
    pooled = pd.concat([ex, cf])
    d = (pooled.adaptive - pooled.control).dropna().to_numpy()
    lo, hi = boot_ci(d, rng)
    tp = stats.ttest_1samp(d, 0.0)
    tests.append(
        {
            "a": "adaptive",
            "b": "control",
            "metric": "best200 pooled 300-339",
            "n": len(d),
            "diff": d.mean(),
            "lo95": lo,
            "hi95": hi,
            "wins": int((d > 0).sum()),
            "t": tp.statistic,
            "p": tp.pvalue,
            "alternative": "two-sided",
            "role": "descriptive (pooled)",
        }
    )
    t = pd.DataFrame(tests)
    t.to_csv(dest / "confirm_tests.csv", index=False)

    means = (
        r.groupby("arm")[
            [
                "best100",
                "best200",
                "auc200",
                "failure_rate",
                "mean_dose",
                "prompted_share",
            ]
        ]
        .mean()
        .loc[ARMS]
    )
    p = t.iloc[0]
    verdict = "SUPPORTED" if p.p < 0.05 else "NOT SUPPORTED"
    md = [
        "# Confirmation of the adaptive dial (pre-registered)",
        "",
        f"96 runs (4 arms x 24 seeds 316-339) x 200 algorithms, all complete.",
        "",
        f"**Primary (H1, adaptive > control, best AOCC @200, one-sided paired t):"
        f" {verdict}.** diff {p['diff']:+.3f} [95% CI {p.lo95:+.3f}, {p.hi95:+.3f}],"
        f" t = {p.t:.2f}, p = {p.p:.4f}, adaptive wins {p.wins}/{p.n} seeds.",
        "",
        "## Secondary (two-sided paired t, Holm over four)",
        "",
    ]
    md += [
        f"* {x.a} vs {x.b}: {x['diff']:+.3f} [{x.lo95:+.3f}, {x.hi95:+.3f}],"
        f" wins {x.wins}/{x.n}, p = {x.p:.4f}, Holm {x.p_holm:.4f}"
        for _, x in t[t.role == "secondary"].iterrows()
    ]
    md += ["", "## Arm means", "", "```", means.round(3).to_string(), "```", ""]
    md += ["## Descriptive comparisons", "", "```"]
    md += [
        t[t.role.str.startswith("descriptive")]
        .drop(columns=["p_holm"], errors="ignore")
        .round(4)
        .to_string(index=False),
        "```",
        "",
    ]
    (dest / "confirm.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(main())
