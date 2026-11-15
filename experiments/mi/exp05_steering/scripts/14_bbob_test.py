"""Stage 2 (E1): the final best of each BBOB LLaMEA run on the test instances.

Per run of ``data/raw/<tag>`` (a ``llamea-bbob-f<fid>`` campaign): the
best algorithm (highest training AOCC over the whole run) is evaluated on
the test instances (15–30) with the same AOCC window as in the campaign,
in BLADE's sandbox. BLADE's ``Problem.__call__`` only evaluates on the
training instances, so a copy of the problem with the test instances as
its training set is used (timeout scaled with the number of instances).

Writes ``results/<tag>/test.csv`` (one row per run) and
``results/<tag>/test.md``: per arm, mean train and test AOCC and the gap
(train − test), and the paired test difference against control (95%
bootstrap CI over seeds). Usage (from anywhere):
python scripts/14_bbob_test.py --tag llamea-bbob-f20
"""

import argparse
import copy
import importlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MI_ROOT = HERE.parents[1]
sys.path[:0] = [str(MI_ROOT), str(HERE.parent), str(HERE)]

from common.paths import data_dir, results_dir
from common.runs import fitness_scalar, read_jsonl
from exp_config import EXPERIMENT, LLAMEA_CAMPAIGNS

from iohblade.solution import Solution


def best_per_run(tag: str) -> pd.DataFrame:
    rows = []
    for log in sorted(data_dir(EXPERIMENT, "raw", tag).glob("*/run-*/log.jsonl")):
        name = log.parent.name
        arm = name.split("-")[2]
        seed = int(re.search(r"-(\d+)(?:-\d+)?$", name).group(1))
        sols = list(read_jsonl(log))
        fit = np.array([fitness_scalar(s.get("fitness")) for s in sols])
        fit = np.where(np.isfinite(fit), fit, -np.inf)
        b = sols[int(np.argmax(fit))]
        rows.append(
            {
                "arm": arm,
                "seed": seed,
                "n": len(sols),
                "id": b["id"],
                "name": b["name"],
                "description": b.get("description", ""),
                "code": b["code"],
                "train": float(fit.max()),
            }
        )
    return pd.DataFrame(rows)


def test_problem(camp):
    problem = importlib.import_module("10_transfer").make_problem(camp.problem)
    test = copy.copy(problem)
    test.training_instances = list(problem.test_instances)
    scale = len(problem.test_instances) / len(problem.training_instances)
    test.eval_timeout = int(problem.eval_timeout * scale)
    return test


def ci(x, rng, n=4000):
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True, choices=list(LLAMEA_CAMPAIGNS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, help="first n runs only (smoke test)")
    ap.add_argument("--out", default="test", help="output file stem")
    args = ap.parse_args()
    camp = LLAMEA_CAMPAIGNS[args.tag]
    runs = best_per_run(args.tag)
    if args.limit:
        runs = runs.head(args.limit)
    problem = test_problem(camp)
    problem._ensure_env()

    def one(row):
        sol = Solution(code=row.code, name=row["name"], description=row.description)
        try:
            sol = problem(sol)
            fit = fitness_scalar(sol.fitness)
            err = sol.error or ""
        except Exception as e:  # infrastructure failure
            fit, err = np.nan, f"INFRA: {e!r}"
        md = sol.metadata
        print(f"{row.arm} {row.seed}: train {row.train:.3f} test {fit:.3f}", flush=True)
        return {
            "test": fit if np.isfinite(fit) and not err else np.nan,
            "test_error": err[:300],
            "test_final_regret": md.get("final_regret"),
        }

    with ThreadPoolExecutor(args.workers) as pool:
        out = list(pool.map(one, [r for _, r in runs.iterrows()]))
    problem.cleanup()
    runs = pd.concat([runs, pd.DataFrame(out)], axis=1)
    runs["gap"] = runs.train - runs.test
    dest = results_dir(EXPERIMENT, args.tag)
    dest.mkdir(parents=True, exist_ok=True)
    runs.drop(columns=["code"]).to_csv(dest / f"{args.out}.csv", index=False)

    rng = np.random.default_rng(0)
    arms = runs.groupby("arm")
    lines = [
        f"# {args.tag}: final best on the test instances",
        "",
        f"Best algorithm of each run (training AOCC), re-evaluated on "
        f"{len(problem.training_instances)} test instances. Mean over seeds; "
        "a run whose best fails on the test instances counts as missing.",
        "",
        "| arm | n | train | test | gap (train − test) | failed on test | "
        "test − control (paired) |",
        "|---|---|---|---|---|---|---|",
    ]
    ctrl = runs[runs.arm == "control"].set_index("seed").test
    for arm, g in arms:
        d = "–"
        if arm != "control":
            x = (g.set_index("seed").test - ctrl).dropna().to_numpy()
            if len(x):
                lo, hi = ci(x, rng)
                d = f"{x.mean():+.3f} ({lo:+.3f} to {hi:+.3f})"
        gap = g.gap.dropna().to_numpy()
        glo, ghi = ci(gap, rng) if len(gap) else (np.nan, np.nan)
        lines.append(
            f"| {arm} | {len(g)} | {g.train.mean():.3f} | {g.test.mean():.3f} | "
            f"{gap.mean():+.3f} ({glo:+.3f} to {ghi:+.3f}) | "
            f"{g.test.isna().sum()} | {d} |"
        )
    (dest / f"{args.out}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
