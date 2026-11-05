"""Step 2: join MI records with BLADE logs into one probing dataset.

One row per LLM query that produced a logged solution. Writes to
``data/processed/<tag>/``:

* ``meta.csv``: run, seed, operator, generation, fitness labels, and the
  output-uncertainty baselines (entropy, log-prob, token counts);
* ``activations.npz``: ``last``, ``prompt_mean``, ``gen_mean`` stacked as
  ``[n_rows, n_layers, d_model]`` (the model preset's ``store_dtype``) plus ``layers``;
* ``code.jsonl``: generated code per row (for the bag-of-words baseline).

Labels:

* ``parent_fitness``: fitness of the solution that was mutated;
* ``child_fitness``: fitness of the generated solution (nan if it failed);
* ``child_failed``: the generated code raised, timed out or had no code;
* ``over_budget``: the code ran but used more evaluations than allowed on
  at least one instance (original MA-BBOB scores this AOCC 0;
  ``BudgetEnforcedMABBOB`` truncates the run at the budget);
* ``improved``: the child beat its parent (failed children count as 0, a
  working child of a failed parent as 1);
* ``delta_fitness``: ``child_fitness - parent_fitness``;
* ``parent_over_budget``: the parent overspent (nan without a working
  parent). The feedback never says so, so the model can only see it in the
  parent's code; used to control the ``over_budget`` probe.

Pre-screen campaigns (``05_prescreen_generate.py``) also get ``condition``
(``shadow``/``screened``), ``prescreen_score`` and ``prescreen_flagged``.
The child labels there use the *true* evaluation result
(``metadata["true_*"]``), also for algorithms LLaMEA was shown as rejected;
``visible_failed`` is what LLaMEA saw. Parent labels use what LLaMEA saw.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import data_dir
from common.runs import (
    fitness_scalar,
    iter_run_dirs,
    load_arrays,
    load_run,
    prompt_text,
)
from exp_config import EXPERIMENT, OPERATORS

MUTATION_MARKER = "The selected solutions to update are"
# Errors caused by the machine, not by the generated code. Rows with these
# are dropped so they cannot leak into the failure labels.
INFRA_ERRORS = (
    "pthread_create failed",
    "Resource temporarily unavailable",
    "Disk quota exceeded",
    "No space left on device",
    "Failed to install the dependencies",
)


def detect_operator(prompt: str) -> str:
    if MUTATION_MARKER not in prompt:
        return "init"
    for name, text in OPERATORS.items():
        if text in prompt:
            return name
    return "other"


def _over_budget(sol: dict, n_scores: int) -> bool:
    """Whether the code used more evaluations than allowed on any instance."""
    meta = sol.get("metadata") or {}
    if "over_budget_instances" in meta:  # BudgetEnforcedMABBOB
        return meta["over_budget_instances"] > 0
    # Original MA-BBOB stops scoring at the first over-budget instance (AOCC 0).
    aucs = meta.get("aucs")
    return bool(aucs) and len(aucs) < n_scores and aucs[-1] == 0


def seed_from_run_dir(run_dir: Path) -> int:
    match = re.search(r"-(\d+)(?:-\d+)?$", run_dir.name)
    return int(match.group(1)) if match else -1


def run_condition(run_dir: Path) -> str:
    match = re.search(r"MA_BBOB_(shadow|screened)", run_dir.name)
    return match.group(1) if match else ""


def build(tag: str) -> pd.DataFrame:
    raw = data_dir(EXPERIMENT, "raw", tag)
    rows, arrays = [], {k: [] for k in ("last", "prompt_mean", "gen_mean")}
    codes, layers, n_infra = [], None, 0
    cfg = json.loads((raw / "config.json").read_text())
    n_scores = len(cfg["dims"]) * len(cfg["training_instances"])
    for run_dir in iter_run_dirs(raw):
        solutions, records = load_run(run_dir)
        run = str(run_dir.relative_to(raw))
        for rec in records:
            sol = solutions.get(rec.get("solution_id"))
            if sol is None or not rec.get("file"):
                continue  # query without a logged solution (e.g. budget hit)
            if any(e in (sol.get("error") or "") for e in INFRA_ERRORS):
                n_infra += 1
                continue
            arr = load_arrays(run_dir, rec)
            if layers is None:
                layers = rec["layers"]
            elif rec["layers"] != layers:
                raise ValueError(f"Inconsistent capture layers in {run_dir}")
            operator = detect_operator(prompt_text(rec))
            parents = [solutions.get(p) for p in rec.get("parent_ids", [])]
            parent_fits = [fitness_scalar(p["fitness"]) for p in parents if p]
            parent_fit = max(parent_fits) if parent_fits else np.nan
            smeta = sol.get("metadata") or {}
            screened = "true_fitness" in smeta  # 05_prescreen_generate.py
            visible_failed = bool(sol.get("error")) or np.isnan(
                fitness_scalar(sol["fitness"])
            )
            if screened:
                child_fit = fitness_scalar(smeta["true_fitness"])
                failed = bool(smeta.get("true_error")) or np.isnan(child_fit)
            else:
                child_fit = fitness_scalar(sol["fitness"])
                failed = visible_failed
            over_budget = (not failed) and _over_budget(sol, n_scores)
            parent = parents[0] if parents else None
            parent_works = parent is not None and not (
                parent.get("error") or np.isnan(fitness_scalar(parent["fitness"]))
            )
            has_parent = operator != "init" and bool(parent_fits)
            parent_ok = has_parent and not np.isnan(parent_fit)
            rows.append(
                {
                    "run": run,
                    "seed": seed_from_run_dir(run_dir),
                    "record_id": rec["record_id"],
                    "solution_id": sol["id"],
                    "query_idx": rec["query_idx"],
                    "generation": sol.get("generation"),
                    "operator": operator,
                    "parent_fitness": parent_fit if parent_ok else np.nan,
                    "parent_failed": bool(parents and parents[0].get("error")),
                    "child_fitness": child_fit,
                    "child_failed": int(failed),
                    "over_budget": int(over_budget),
                    # A working child of a failed parent counts as improved.
                    "improved": (
                        int((not failed) and (not parent_ok or child_fit > parent_fit))
                        if has_parent
                        else np.nan
                    ),
                    "delta_fitness": child_fit - parent_fit if parent_ok else np.nan,
                    "parent_over_budget": (
                        int(_over_budget(parent, n_scores)) if parent_works else np.nan
                    ),
                    "n_prompt_tokens": rec["n_prompt_tokens"],
                    "n_gen_tokens": rec["n_gen_tokens"],
                    "mean_entropy": rec["mean_entropy"],
                    "mean_logprob": rec["mean_logprob"],
                    "steered": rec.get("steering") is not None,
                }
            )
            if screened:
                rows[-1].update(
                    {
                        "condition": run_condition(run_dir),
                        "prescreen_score": smeta.get("prescreen_score"),
                        "prescreen_flagged": int(bool(smeta.get("prescreen_flagged"))),
                        "visible_failed": int(visible_failed),
                    }
                )
            for k in arrays:
                arrays[k].append(arr[k])
            codes.append({"record_id": rec["record_id"], "code": sol["code"]})

    if n_infra:
        print(f"dropped {n_infra} rows with infrastructure errors")
    if not rows:
        raise SystemExit(f"No MI records found under {raw}; run 01_generate.py.")
    meta = pd.DataFrame(rows)
    out = data_dir(EXPERIMENT, "processed", tag)
    meta.to_csv(out / "meta.csv", index=False)
    np.savez(
        out / "activations.npz",
        layers=np.array(layers),
        **{k: np.stack(v) for k, v in arrays.items()},
    )
    with open(out / "code.jsonl", "w") as f:
        for c in codes:
            f.write(json.dumps(c) + "\n")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="smoke")
    args = ap.parse_args()
    meta = build(args.tag)
    print(f"{len(meta)} rows from {meta['run'].nunique()} runs")
    print(meta["operator"].value_counts().to_string())
    print(
        meta[["child_failed", "improved", "parent_fitness", "child_fitness"]]
        .describe()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
