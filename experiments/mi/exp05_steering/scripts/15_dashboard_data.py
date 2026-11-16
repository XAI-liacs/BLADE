"""Data files for the results dashboard (exp05 + exp01 probes).

Reads the results already written by 06_llamea_analyze.py (runs.csv,
solutions.csv per campaign), the transfer screens (11_transfer_analyze.py),
the step 5c dial screen, 14_bbob_test.py and experiment 1's probe scores,
and writes one JSON file per dataset, ``{"rows": [...], "meta": [...]}``,
to ``results/dashboard/``. ``meta`` carries this script's own text.

Problems and their campaigns (arms paired with control by seed):

* MA-BBOB: llamea-dial + llamea-stage1 (seeds 300-307), llamea-stage1-seeds
  (308-315), llamea-stage1-cal (adaptive_cal, 300-307);
* MA-BBOB, budget 200: llamea-ext200 + llamea-ext200-adaptive (300-307);
* Bragg / packing: llamea-dial-<p> + llamea-stage1-<p> (400-405);
* BBOB f20 20-D: llamea-bbob-f20 (500-509).

Usage (from exp05_steering): python scripts/15_dashboard_data.py
"""

import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RES = EXP / "results"
EXP01 = EXP.parent / "exp01_probing/results/main"
OUT = RES / "dashboard"

PROBLEMS = {
    "MA-BBOB": [
        "llamea-dial",
        "llamea-stage1",
        "llamea-stage1-seeds",
        "llamea-stage1-cal",
    ],
    "MA-BBOB (budget 200)": ["llamea-ext200", "llamea-ext200-adaptive"],
    "Bragg": ["llamea-dial-bragg", "llamea-stage1-bragg"],
    "Packing": ["llamea-dial-packing", "llamea-stage1-packing"],
    "BBOB f20 (20-D)": ["llamea-bbob-f20"],
}
METRICS = {
    "best": "best fitness",
    "auc_best": "area under the best-so-far curve",
    "failed": "share of algorithms that failed",
    "n_families": "distinct algorithm families",
    "div_concepts": "concept diversity",
    "exploration_pct": "exploration % (traced)",
    "intensification": "intensification (traced)",
}
CONCEPTS = [
    "population",
    "differential_evolution",
    "crossover",
    "gradient",
    "step_size_adaptation",
    "covariance",
    "local_search",
    "particle_swarm",
    "simulated_annealing",
    "restart",
    "elitism",
    "adaptive_parameters",
]
SCREENS = {
    "Bragg": "transfer-bragg",
    "Packing": "transfer-packing",
    "BBOB f20 (20-D)": "transfer-bbob-f20",
    "BBOB f2 (20-D)": "transfer-bbob-f2",
}
CODE = Path(__file__).read_text()


def load(problem):
    runs, sols = [], []
    for tag in PROBLEMS[problem]:
        d = RES / tag
        if not (d / "runs.csv").exists():
            continue
        runs.append(pd.read_csv(d / "runs.csv"))
        sols.append(pd.read_csv(d / "solutions.csv", low_memory=False))
    runs = pd.concat(runs).drop_duplicates(["arm", "seed"])
    sols = pd.concat(sols).drop_duplicates(["arm", "seed", "idx"])
    return runs.rename(columns={"best_aocc": "best"}), sols


def curve(g):
    f = g.sort_values("idx").aocc.astype(float)
    f = f.where(np.isfinite(f))
    return f.fillna(f.min()).cummax().to_numpy()


def boot(x, rng, n=4000):
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def r(v, k=4):
    return None if v is None or not np.isfinite(v) else round(float(v), k)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    run_rows, curve_rows, paired_rows, concept_rows = [], [], [], []
    for problem in PROBLEMS:
        runs, sols = load(problem)
        curves = {k: curve(g) for k, g in sols.groupby(["arm", "seed"])}
        runs["auc_best"] = [curves[(a, s)].mean() for a, s in zip(runs.arm, runs.seed)]
        for _, x in runs.iterrows():
            run_rows.append(
                {
                    "problem": problem,
                    "arm": x.arm,
                    "seed": int(x.seed),
                    "evaluations": int(x.n),
                }
                | {m: r(x.get(m, np.nan)) for m in METRICS}
            )
        # convergence: mean and quartiles of best-so-far over seeds
        for arm in runs.arm.unique():
            cs = [c for (a, _), c in curves.items() if a == arm]
            n = min(len(c) for c in cs)
            m = np.vstack([c[:n] for c in cs])
            for i in range(n):
                curve_rows.append(
                    {
                        "problem": problem,
                        "arm": arm,
                        "evaluation": i + 1,
                        "mean": r(m[:, i].mean()),
                        "q25": r(np.quantile(m[:, i], 0.25)),
                        "q75": r(np.quantile(m[:, i], 0.75)),
                        "seeds": len(cs),
                    }
                )
        # paired differences vs control, and explore - exploit
        piv = {
            m: runs.pivot(index="seed", columns="arm", values=m)
            for m in METRICS
            if m in runs
        }
        pairs = [(a, "control") for a in runs.arm.unique() if a != "control"]
        pairs += [
            (f"{p}explore", f"{p}exploit")
            for p in ("", "prompt_")
            if {f"{p}explore", f"{p}exploit"} <= set(runs.arm)
        ]
        if "adaptive_cal" in set(runs.arm) and "adaptive" in set(runs.arm):
            pairs.append(("adaptive_cal", "adaptive"))
        for a, b in pairs:
            for m, p in piv.items():
                if a not in p or b not in p:
                    continue
                x = (p[a] - p[b]).dropna().to_numpy()
                if len(x) < 3:
                    continue
                lo, hi = boot(x, rng)
                paired_rows.append(
                    {
                        "problem": problem,
                        "arm": a,
                        "versus": b,
                        "metric": m,
                        "seeds": len(x),
                        "diff": r(x.mean()),
                        "lo": r(lo),
                        "hi": r(hi),
                        "wins": int((x > 0).sum()),
                        "significant": bool(lo > 0 or hi < 0),
                    }
                )
        # concept shares among working algorithms
        if problem != "MA-BBOB (budget 200)":
            w = sols[~sols.failed.astype(bool)].copy()
            w["concepts"] = w.concepts.fillna("").str.split("|")
            for arm, g in w.groupby("arm"):
                for k in CONCEPTS:
                    concept_rows.append(
                        {
                            "problem": problem,
                            "arm": arm,
                            "concept": k,
                            "share": r(g.concepts.apply(lambda l: k in l).mean()),
                            "algorithms": len(g),
                        }
                    )
        print(problem, len(runs), "runs", flush=True)

    # f20: final best on the test instances
    t = pd.read_csv(RES / "llamea-bbob-f20/test.csv")
    test_rows = [
        {
            "problem": "BBOB f20 (20-D)",
            "arm": x.arm,
            "seed": int(x.seed),
            "train": r(x.train),
            "test": r(x.test),
            "gap": r(x.gap),
        }
        for _, x in t.iterrows()
    ]

    # mechanism: dial dose response on MA-BBOB (step 5c screen)
    d = pd.read_csv(RES / "dial/summary.csv")
    dose_rows = [
        {
            "direction": x.direction,
            "dose": r(x.dose),
            "samples": int(x.n),
            "failed": r(x.failed),
            "aocc": r(x.aocc),
            "exploration_pct": r(x.exploration_pct),
            "intensification": r(x.intensification),
        }
        for _, x in d.iterrows()
    ]
    # mechanism: dose slopes in the single-generation screens
    slope_rows = []
    s = pd.read_csv(RES / "dial/slopes.csv")
    for _, x in s[s.direction == "b_exploration_pct_resid@L27"].iterrows():
        for m in ("exploration_pct", "intensification"):
            slope_rows.append(
                {
                    "problem": "MA-BBOB",
                    "phase": "init",
                    "metric": m,
                    "slope": r(x[f"slope_{m}"]),
                    "lo": r(x[f"slope_{m}_lo"]),
                    "hi": r(x[f"slope_{m}_hi"]),
                    "samples": int(x.n),
                }
            )
    for problem, tag in SCREENS.items():
        f = RES / tag / "slopes.csv"
        if not f.exists():
            continue
        for _, x in pd.read_csv(f).query("phase == 'pooled'").iterrows():
            slope_rows.append(
                {
                    "problem": problem,
                    "phase": "init + refine",
                    "metric": x.metric,
                    "slope": r(x.slope),
                    "lo": r(x.lo),
                    "hi": r(x.hi),
                    "samples": int(x.n),
                }
            )
    # mechanism: experiment 1 probes per layer (linear probe vs shuffled control)
    p = pd.read_csv(EXP01 / "probe_scores.csv")
    probe_rows = []
    for _, x in p.iterrows():
        binary = x.kind == "binary"
        probe_rows.append(
            {
                "target": x.target,
                "position": x.position,
                "layer": int(x.layer),
                "score": r(x.auroc if binary else x.spearman),
                "control": r(x.control_auroc if binary else x.control_spearman),
                "measure": "AUROC" if binary else "Spearman",
                "samples": int(x.n),
            }
        )

    inputs = {
        "runs": "results/<campaign>/runs.csv, solutions.csv",
        "curves": "results/<campaign>/solutions.csv",
        "paired": "results/<campaign>/runs.csv, solutions.csv",
        "concepts": "results/<campaign>/solutions.csv",
        "test": "results/llamea-bbob-f20/test.csv",
        "dose": "results/dial/summary.csv",
        "slopes": "results/dial/slopes.csv, results/transfer-*/slopes.csv",
        "probes": "exp01_probing/results/main/probe_scores.csv",
    }
    for name, rows in [
        ("runs", run_rows),
        ("curves", curve_rows),
        ("paired", paired_rows),
        ("concepts", concept_rows),
        ("test", test_rows),
        ("dose", dose_rows),
        ("slopes", slope_rows),
        ("probes", probe_rows),
    ]:
        meta = [
            {
                "label": "15_dashboard_data.py",
                "value": CODE,
                "type": "code",
                "lang": "python",
            },
            {"label": "Input", "value": inputs[name]},
        ]
        (OUT / f"{name}.json").write_text(json.dumps({"rows": rows, "meta": meta}))
        print(name, len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
