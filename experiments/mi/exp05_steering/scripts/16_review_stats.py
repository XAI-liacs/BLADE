"""Review follow-up: multiplicity and equivalence for the closed-loop arms.

Every steered or prompted LLaMEA arm is compared with control on the best
score of the run, on every problem (MA-BBOB at 100 and 200 evaluations,
Bragg, packing, BBOB f20, f2, f9): paired by seed, two-sided exact
sign-flip permutation test (all 2^n sign patterns, n <= 16 seeds), with
Holm's correction over the whole family and within each problem.

For the BBOB functions, where the arms end close together, the 90% CI of
each difference (bootstrap over seeds) gives the equivalence bound: the
largest effect the data do not exclude (TOST at alpha 0.05).

Writes ``results/review/multiplicity.csv`` and ``results/review/summary.md``.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
CONTROL = "control"


def runs() -> pd.DataFrame:
    """One row per (problem, arm, seed) with the best score of the run."""
    rows = []
    b = pd.read_csv(RES / "budget200/per_run.csv")
    b = b[b.arm != "adaptive_cal"]  # 8 seeds, compared with adaptive only
    for col, prob in [("best200", "MA-BBOB @200"), ("best100", "MA-BBOB @100")]:
        rows.append(
            b.assign(problem=prob, best=b[col])[["problem", "arm", "seed", "best"]]
        )
    for prob, tags in [
        ("Bragg", ["llamea-dial-bragg", "llamea-stage1-bragg"]),
        ("packing", ["llamea-dial-packing", "llamea-stage1-packing"]),
        ("BBOB f20", ["llamea-bbob-f20"]),
        ("BBOB f2", ["llamea-bbob-f2"]),
        ("BBOB f9", ["llamea-bbob-f9"]),
    ]:
        r = pd.concat([pd.read_csv(RES / t / "runs.csv") for t in tags])
        rows.append(
            r.assign(problem=prob, best=r.best_aocc)[["problem", "arm", "seed", "best"]]
        )
    return pd.concat(rows, ignore_index=True)


def sign_flip_p(d: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation p-value of mean(d) = 0."""
    n = len(d)
    obs = abs(d.mean())
    signs = np.array(list(itertools.product((-1, 1), repeat=n)))
    means = np.abs(signs @ d) / n
    return float((means >= obs - 1e-12).mean())


def boot_ci(d: np.ndarray, level: float, rng, n_boot=10000) -> tuple[float, float]:
    m = rng.choice(d, (n_boot, len(d)), replace=True).mean(1)
    a = (1 - level) / 2
    return float(np.quantile(m, a)), float(np.quantile(m, 1 - a))


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def main():
    rng = np.random.default_rng(0)
    r = runs()
    out = []
    for prob, g in r.groupby("problem", sort=False):
        w = g.pivot_table(index="seed", columns="arm", values="best")
        for arm in w.columns:
            if arm == CONTROL:
                continue
            d = (w[arm] - w[CONTROL]).dropna().to_numpy()
            lo95, hi95 = boot_ci(d, 0.95, rng)
            lo90, hi90 = boot_ci(d, 0.90, rng)
            out.append(
                {
                    "problem": prob,
                    "arm": arm,
                    "seeds": len(d),
                    "diff": d.mean(),
                    "lo95": lo95,
                    "hi95": hi95,
                    "lo90": lo90,
                    "hi90": hi90,
                    "equiv_bound": max(abs(lo90), abs(hi90)),
                    "wins": int((d > 0).sum()),
                    "p": sign_flip_p(d),
                }
            )
    m = pd.DataFrame(out)
    m["p_holm_all"] = holm(m.p.to_numpy())
    m["p_holm_problem"] = m.groupby("problem").p.transform(lambda s: holm(s.to_numpy()))
    dest = RES / "review"
    dest.mkdir(exist_ok=True)
    m.to_csv(dest / "multiplicity.csv", index=False)
    md = [
        "# Multiplicity and equivalence (review follow-up)",
        "",
        f"{len(m)} arm-vs-control comparisons of the best score of the run,"
        " paired by seed; exact sign-flip test; Holm over all and within"
        " each problem. equiv_bound: largest |difference| inside the 90%"
        " bootstrap CI (TOST, alpha 0.05).",
        "",
        "```",
        m.round(4).to_string(index=False),
        "```",
        "",
        "Survive Holm over all comparisons (p < 0.05):",
        "",
    ]
    md += [
        f"* {x.problem} {x.arm}: {x['diff']:+.3f}, p = {x.p:.4f}, Holm {x.p_holm_all:.3f}"
        for _, x in m[m.p_holm_all < 0.05].iterrows()
    ] or ["* none"]
    md += ["", "Survive Holm within the problem:", ""]
    md += [
        f"* {x.problem} {x.arm}: {x['diff']:+.3f}, p = {x.p:.4f}, Holm {x.p_holm_problem:.3f}"
        for _, x in m[m.p_holm_problem < 0.05].iterrows()
    ] or ["* none"]
    (dest / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[-40:]))


if __name__ == "__main__":
    sys.exit(main())
