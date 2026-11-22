"""Pre-registered analysis of the dial + temperature arm (``llamea-combo``).

Implements ``paper_gecco2027/PREREG_combo.md``:

* primary: ``adaptive_temp11`` > ``control`` on the best AOCC after 200
  evaluated algorithms, one-sided paired t-test over seeds 316-339,
  alpha 0.05, with the mean difference, the 95% bootstrap CI (10,000
  resamples over seeds) and the number of seeds won;
* secondary: ``adaptive_temp11`` vs ``adaptive`` and vs ``temp11``,
  two-sided paired t-tests, Holm-corrected over the two;
* descriptive: the interaction contrast (combo - adaptive - temp11 +
  control), the same comparisons at 100 evaluations and on the area under
  the best-so-far curve, combo vs ``adaptive_prompt``, failure and timeout
  rates, the adaptive dose.

The four other arms come from ``llamea-confirm`` (same seeds). Reuses the
loader of ``18_confirm_analyze.py``. Writes ``results/llamea-combo/
combo_per_run.csv``, ``combo_tests.csv`` and ``combo.md``. Families and
diversity: ``06_llamea_analyze.py --tag llamea-combo`` (descriptive).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("confirm", HERE / "18_confirm_analyze.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

TAG = "llamea-combo"
COMBO = "adaptive_temp11"
SECONDARY = [(COMBO, "adaptive"), (COMBO, "temp11")]
DESCRIPTIVE = [(COMBO, "control")] + SECONDARY + [(COMBO, "adaptive_prompt")]


def load(tag: str) -> pd.DataFrame:
    C.TAG = tag
    return C.load()


def timeouts(tag: str) -> pd.DataFrame:
    rows = []
    for d in sorted((ROOT / "data/raw" / tag).glob("shard-*/run-*")):
        arm, seed = C.RUN_RE.match(d.name).groups()
        n = t = 0
        for line in (d / "log.jsonl").read_text().splitlines():
            n += 1
            t += "timed out" in (json.loads(line).get("error") or "")
        rows.append({"arm": arm, "seed": int(seed), "timeout_rate": t / n})
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(0)
    r = pd.concat([C.per_run(load("llamea-confirm")), C.per_run(load(TAG))])
    to = pd.concat([timeouts("llamea-confirm"), timeouts(TAG)])
    r = r.merge(to, on=["arm", "seed"])
    arms = C.ARMS + [COMBO]
    assert (r.n == 200).all() and len(r) == 120, r[r.n != 200]
    assert sorted(r.seed.unique()) == C.SEEDS
    assert sorted(r.arm.unique()) == sorted(arms)
    dest = ROOT / "results" / TAG
    dest.mkdir(parents=True, exist_ok=True)
    r[r.arm == COMBO].to_csv(dest / "combo_per_run.csv", index=False)
    w = r.pivot_table(index="seed", columns="arm")
    tests = [
        C.compare(w, COMBO, "control", "best200", rng, "greater") | {"role": "primary"}
    ]
    sec = [
        C.compare(w, a, b, "best200", rng, "two-sided") | {"role": "secondary"}
        for a, b in SECONDARY
    ]
    for x, ph in zip(sec, C.holm([x["p"] for x in sec])):
        x["p_holm"] = ph
    tests += sec
    for metric in ("best200", "best100", "auc200", "auc100"):
        for a, b in DESCRIPTIVE:
            if metric == "best200" and (a, b) in [(COMBO, "control")] + SECONDARY:
                continue
            tests.append(
                C.compare(w, a, b, metric, rng, "two-sided") | {"role": "descriptive"}
            )
    # interaction: (combo - adaptive) - (temp11 - control)
    b = w["best200"]
    d = (b[COMBO] - b.adaptive - b.temp11 + b.control).to_numpy()
    lo, hi = C.boot_ci(d, rng)
    ti = stats.ttest_1samp(d, 0.0)
    tests.append(
        {
            "a": "combo-adaptive",
            "b": "temp11-control",
            "metric": "best200 interaction",
            "n": len(d),
            "diff": d.mean(),
            "lo95": lo,
            "hi95": hi,
            "wins": int((d > 0).sum()),
            "t": ti.statistic,
            "p": ti.pvalue,
            "alternative": "two-sided",
            "role": "descriptive (interaction)",
        }
    )
    t = pd.DataFrame(tests)
    t.to_csv(dest / "combo_tests.csv", index=False)

    cols = ["best100", "best200", "auc200", "failure_rate", "timeout_rate"]
    means = r.groupby("arm")[cols + ["mean_dose"]].mean().loc[arms]
    p = t.iloc[0]
    verdict = "SUPPORTED" if p.p < 0.05 else "NOT SUPPORTED"
    md = [
        "# The adaptive dial and temperature 1.1 together (pre-registered)",
        "",
        "24 runs (seeds 316-339) x 200 algorithms, paired with the four arms"
        " of `llamea-confirm`.",
        "",
        f"**Primary (H2, {COMBO} > control, best AOCC @200, one-sided paired t):"
        f" {verdict}.** diff {p['diff']:+.3f} [95% CI {p.lo95:+.3f}, {p.hi95:+.3f}],"
        f" t = {p.t:.2f}, p = {p.p:.4f}, wins {p.wins}/{p.n} seeds.",
        "",
        "## Secondary (two-sided paired t, Holm over two)",
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
    (dest / "combo.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(main())
