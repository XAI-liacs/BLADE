"""Smoke-test baseline for the failure pre-screen, and evaluation times.

A reviewer's first question about the probe pre-screen: would a cheap
smoke test (run the code briefly before the full evaluation) catch the
same failures for almost nothing? And does skipping 21% of the
evaluations save 21% of the time, if crashing code fails fast?

Every child of the 36 unscreened ("shadow") runs of the pre-registered
test (campaigns ``prescreen`` and ``prescreen2``; 3,600 algorithms, all
evaluated in the loop) is re-evaluated three times, in the original
sandbox (``ProbeConfig`` of those campaigns):

* ``full``: the original evaluation (10 instances, budget 2000*D,
  timeout 120 s), timed (wall clock, including the sandbox subprocess);
* ``smoke100``: one instance, 100 function evaluations, timeout 30 s;
* ``smoke1i``: one instance, the full budget, timeout 120 s.

Writes ``results/smoke-baseline/evals.csv`` (one row per child and variant).
``--analyze`` writes ``results/smoke-baseline/summary.csv``/``.md``: for each
screen (probe at its fixed threshold, the two smoke tests, and the probe
followed by a smoke test), the share of failures caught, of working
children lost, and the evaluation time spent and saved.

Usage:
  python scripts/10_smoke_test.py --workers 48
  python scripts/10_smoke_test.py --analyze
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))

from common.paths import data_dir, results_dir
from common.problems import BudgetEnforcedMABBOB
from exp_config import EXPERIMENT, PRESETS

from iohblade.solution import Solution

TAGS = ("prescreen", "prescreen2")
VARIANTS = {
    "full": {"instances": None, "budget_factor": None, "timeout": None},
    "smoke100": {"instances": 1, "budget_factor": 20, "timeout": 30},
    "smoke1i": {"instances": 1, "budget_factor": None, "timeout": None},
}


def load_children() -> pd.DataFrame:
    rows = []
    for tag in TAGS:
        for log in sorted(
            data_dir(EXPERIMENT, "raw", tag).glob("shard-*/run-*shadow*/log.jsonl")
        ):
            seed = int(log.parent.name.rsplit("-", 1)[1])
            for i, line in enumerate(log.read_text().splitlines()):
                d = json.loads(line)
                m = d.get("metadata", {})
                err = m.get("true_error", d.get("error", "")) or ""
                fit = m.get("true_fitness", d.get("fitness"))
                fit = fit if isinstance(fit, (int, float)) else float("nan")
                rows.append(
                    {
                        "tag": tag,
                        "seed": seed,
                        "idx": i,
                        "id": d["id"],
                        "operator": m.get("operator", d.get("operator")),
                        "code": d["code"],
                        "name": d.get("name", ""),
                        "logged_failed": int(bool(err) or not np.isfinite(fit)),
                        "logged_fitness": fit,
                        "probe_score": m.get("prescreen_score"),
                        "probe_flagged": int(bool(m.get("prescreen_flagged"))),
                    }
                )
    return pd.DataFrame(rows)


def make_problem(cfg, variant: str):
    v = VARIANTS[variant]
    inst = cfg.training_instances
    return BudgetEnforcedMABBOB(
        dims=cfg.dims,
        budget_factor=v["budget_factor"] or cfg.budget_factor,
        training_instances=inst[: v["instances"]] if v["instances"] else inst,
        eval_timeout=v["timeout"] or cfg.eval_timeout,
        dependencies=cfg.eval_dependencies,
        name=f"MA_BBOB_{variant}",
    )


def evaluate(problem, row) -> dict:
    s = Solution(code=row.code, name=row["name"])
    t0 = time.perf_counter()
    try:
        s = problem(s)
        err = s.error or ""
        fit = s.fitness
    except Exception as e:  # timeouts and sandbox errors count as failures
        err, fit = f"{type(e).__name__}: {e}", float("-inf")
    dt = time.perf_counter() - t0
    try:
        fit = float(fit)
    except (TypeError, ValueError):
        fit = float("nan")
    return {
        "id": row.id,
        "seconds": dt,
        "failed": int(bool(err) or not np.isfinite(fit)),
        "fitness": fit if np.isfinite(fit) else None,
        "error": err.splitlines()[-1][:200] if err else "",
    }


def run(workers: int, limit: int = 0):
    out = results_dir(EXPERIMENT, "smoke-baseline" + ("-test" if limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    df = load_children()
    if limit:
        df = df.sample(limit, random_state=0)
    print(f"{len(df)} children, failure rate {df.logged_failed.mean():.3f}", flush=True)
    df.drop(columns="code").to_csv(out / "children.csv", index=False)
    cfg = PRESETS["prescreen"]
    path = out / "evals.csv"
    done = set()
    if path.exists():
        prev = pd.read_csv(path)
        done = set(zip(prev.variant, prev.id))
    # one problem per variant; its sandbox is built once, before the threads
    problems = {v: make_problem(cfg, v) for v in VARIANTS}
    for p in problems.values():
        p._ensure_env()
    jobs = [(v, r) for v in VARIANTS for _, r in df.iterrows() if (v, r.id) not in done]
    print(f"{len(jobs)} evaluations to run with {workers} workers", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex, open(path, "a") as fh:
        if not done:
            fh.write("variant,id,seconds,failed,fitness,error\n")
        futs = {ex.submit(evaluate, problems[v], r): v for v, r in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            err = '"' + res["error"].replace('"', "'") + '"'
            fit = "" if res["fitness"] is None else f"{res['fitness']:.6f}"
            fh.write(
                f"{futs[fut]},{res['id']},{res['seconds']:.2f},{res['failed']},{fit},{err}\n"
            )
            fh.flush()
            if n % 200 == 0:
                print(
                    f"{n}/{len(jobs)} ({(time.time() - t0) / 60:.0f} min)", flush=True
                )


def analyze():
    out = results_dir(EXPERIMENT, "smoke-baseline")
    ch = pd.read_csv(out / "children.csv")
    ev = pd.read_csv(out / "evals.csv")
    w = ev.pivot_table(index="id", columns="variant", values=["seconds", "failed"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    d = ch.merge(w, left_on="id", right_index=True, how="inner")
    y = d.logged_failed.to_numpy().astype(bool)
    agree = (d.failed_full.astype(bool) == y).mean()
    t_full = d.seconds_full.to_numpy()
    total = t_full.sum()
    lines = [
        f"children: {len(d)}; logged failure rate {y.mean():.3f}; "
        f"re-evaluation agrees with the logged label for {agree:.3f}",
        f"full evaluation: median {np.median(t_full):.1f} s; failing children"
        f" {np.median(t_full[y]):.1f} s, working {np.median(t_full[~y]):.1f} s;"
        f" failing children use {t_full[y].sum() / total:.3f} of the total time",
    ]
    rows = []

    def screen(name, flag, cost):
        """flag: skip the full evaluation; cost: seconds spent on the screen."""
        flag = np.asarray(flag, bool)
        spent = cost.sum() + t_full[~flag].sum()
        rows.append(
            {
                "screen": name,
                "flagged": flag.mean(),
                "failures_caught": flag[y].mean(),
                "working_lost": flag[~y].mean(),
                "precision": y[flag].mean() if flag.any() else np.nan,
                "evals_skipped": flag.mean(),
                "time_screen": cost.sum() / total,
                "time_saved": 1 - spent / total,
            }
        )

    zero = np.zeros(len(d))
    probe = d.probe_flagged.to_numpy().astype(bool)
    s100 = d.failed_smoke100.to_numpy().astype(bool)
    s1i = d.failed_smoke1i.to_numpy().astype(bool)
    screen("none", np.zeros(len(d), bool), zero)
    screen("probe (fixed threshold)", probe, zero)
    screen("smoke test, 100 evaluations", s100, d.seconds_smoke100.to_numpy())
    screen("smoke test, 1 instance", s1i, d.seconds_smoke1i.to_numpy())
    # probe first; the smoke test only on children the probe lets through
    screen(
        "probe, then smoke 100",
        probe | s100,
        np.where(probe, 0.0, d.seconds_smoke100.to_numpy()),
    )
    screen("oracle (skip exactly the failures)", y, zero)
    res = pd.DataFrame(rows)
    res.to_csv(out / "summary.csv", index=False)
    # smoke-test failures that the full evaluation does not have, by error
    fp = d[s100 & ~y]
    lines.append(f"smoke100 false alarms: {len(fp)}")
    md = ["# Smoke-test baseline for the pre-screen", ""]
    md += [f"* {l}" for l in lines]
    md += ["", "```\n" + res.round(3).to_string(index=False) + "\n```"]
    errs = ev[(ev.variant == "smoke100") & ev.id.isin(fp.id)].error.str[:80]
    md += ["", "## smoke100 false alarms, most common errors", ""]
    md += [f"* {k} ({v})" for k, v in errs.value_counts().head(10).items()]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="test on a sample")
    a = ap.parse_args()
    analyze() if a.analyze else run(a.workers, a.limit)
