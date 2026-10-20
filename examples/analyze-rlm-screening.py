"""Summarise an RLM-screening experiment written by run-rlm-screening.py.

Per method (pooled over runs) it reports:

* real evaluations per run (rows in each run's ``log.jsonl``; candidates
  that ``threshold_reject``/``topk_filter`` skipped are never logged);
* failed evaluations;
* final best fitness per run (mean and std);
* for candidates with ``metadata["rlm_predicted_score"]`` and a finite
  fitness: Spearman and Pearson correlation between the prediction and the
  real fitness. Use the ``log-only`` arms for this: the filtering arms only
  log the candidates the RLM already preferred.

Usage:
    uv run python examples/analyze-rlm-screening.py results/rlm-screening-mabbob
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

RUN_DIR_RE = re.compile(r"^run-(?P<method>.+)-(?P<problem>[^-]+)-(?P<seed>\d+)$")


def load_runs(results_dir: Path) -> dict[str, list[list[dict]]]:
    """Maps method name to a list of runs, each a list of log rows."""
    runs = defaultdict(list)
    for log in sorted(results_dir.glob("run-*/log.jsonl")):
        m = RUN_DIR_RE.match(log.parent.name)
        method = m.group("method") if m else log.parent.name
        with open(log) as fh:
            runs[method].append([json.loads(line) for line in fh])
    return runs


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("results_dir", type=Path)
    args = p.parse_args()

    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"No run-*/log.jsonl under {args.results_dir}")

    header = (
        f"{'method':<28} {'runs':>4} {'evals/run':>9} {'failed':>7} "
        f"{'best mean':>10} {'best std':>9} {'n pred':>7} "
        f"{'spearman':>9} {'pearson':>8}"
    )
    print(header)
    print("-" * len(header))
    for method in sorted(runs):
        method_runs = runs[method]
        evals = [len(rows) for rows in method_runs]
        failed = sum(bool(r.get("error")) for rows in method_runs for r in rows)
        bests = []
        for rows in method_runs:
            fit = [as_float(r.get("fitness")) for r in rows]
            fit = [f for f in fit if math.isfinite(f)]
            bests.append(max(fit) if fit else float("nan"))
        pred, real = [], []
        for rows in method_runs:
            for r in rows:
                score = (r.get("metadata") or {}).get("rlm_predicted_score")
                f = as_float(r.get("fitness"))
                if score is not None and math.isfinite(f):
                    pred.append(float(score))
                    real.append(f)
        rho = r_p = float("nan")
        if len(pred) >= 3 and np.ptp(pred) > 0 and np.ptp(real) > 0:
            rho = spearmanr(pred, real).statistic
            r_p = pearsonr(pred, real).statistic
        print(
            f"{method:<28} {len(method_runs):>4} {np.mean(evals):>9.1f} "
            f"{failed:>7} {np.nanmean(bests):>10.4f} {np.nanstd(bests):>9.4f} "
            f"{len(pred):>7} {rho:>9.3f} {r_p:>8.3f}"
        )


if __name__ == "__main__":
    main()
