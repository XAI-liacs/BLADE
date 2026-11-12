"""Step 5d: does the exploration dial transfer to other BLADE problems?

Screen of the dial (``b_exploration_pct_resid@L27``, learned from MA-BBOB
code) on mini-Bragg photonics (``TracedPhotonics``) or unit-square circle
packing, n=26 (``SandboxedUnitSquarePacking``), see
``exp_config.TRANSFER_CAMPAIGNS``. Prompts are the ones LLaMEA builds for
the problem:

* ``--phase init``: the initial prompt, ``k_init`` samples per condition;
* ``--phase refine``: the refine prompt on each of the first ``n_parents``
  working unsteered initial algorithms, ``k_refine`` samples each
  (needs the init phase).

Conditions: ``none``, ``dial@<dose>`` and ``random@<dose>`` (one of the 8
random unit directions of ``05_llamea.dial_config``, random sign). The
sampling seed depends on the prompt and the replicate only (common random
numbers), so conditions are paired. One line per sample is appended to
``data/raw/<tag>/<phase>.jsonl``; rerun to resume::

    CUDA_VISIBLE_DEVICES=2 uv run python 10_transfer.py --tag transfer-bragg
"""

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MI_ROOT = HERE.parents[1]
sys.path.insert(0, str(MI_ROOT))
sys.path.insert(0, str(HERE.parent))

from common.models import MODELS
from common.paths import data_dir
from common.runs import read_jsonl
from exp_config import (
    CORPUS,
    EXPERIMENT,
    EXTRA_EVAL_DEPENDENCIES,
    MODEL,
    TRANSFER_CAMPAIGNS,
)

from iohblade.llm import HookedLLM
from iohblade.utils import NoCodeException

ROLE = "You are an excellent Python programmer."
REFINE = "Refine the strategy of the selected solution to improve it."
TEMPERATURE, TOP_P = 0.8, 0.95  # experiment 1's sampling settings


def make_problem(name: str):
    from common.transfer_problems import SandboxedUnitSquarePacking, TracedPhotonics

    if name == "bragg":
        return TracedPhotonics("bragg", extra_dependencies=EXTRA_EVAL_DEPENDENCIES)
    if name == "packing26":
        return SandboxedUnitSquarePacking(
            n_circles=26, extra_dependencies=EXTRA_EVAL_DEPENDENCIES
        )
    raise ValueError(name)


def dial(camp) -> dict:
    """Unit dial direction, its layer and scale, and the 8 random units
    (same generator as ``05_llamea.dial_config``)."""
    proc = data_dir(EXPERIMENT, "processed", CORPUS["tag"])
    info = json.loads((proc / "info.json").read_text())[camp.direction]
    unit = np.load(proc / "directions.npz")[camp.direction].astype(np.float32)
    rng = np.random.default_rng(54321)
    random = []
    for _ in range(8):
        r = rng.standard_normal(unit.shape).astype(np.float32)
        random.append(r / np.linalg.norm(r))
    return {
        "layer": info["layer"],
        "scale": info["residual_norm"],
        "unit": unit / np.linalg.norm(unit),
        "random": random,
    }


def conditions(camp) -> dict:
    out = {"none": ("none", 0.0)}
    for c in camp.doses:
        out[f"dial@{c:+g}"] = ("dial", float(c))
    out[f"random@{camp.random_dose:+g}"] = ("random", camp.random_dose)
    return out


def steer_vector(d: dict, kind: str, dose: float, seed: int):
    if kind == "none":
        return None, None
    if kind == "dial":
        return dose * d["scale"] * d["unit"], None
    k = seed % 8
    sign = 1.0 if (seed // 8) % 2 == 0 else -1.0
    return sign * dose * d["scale"] * d["random"][k], f"random{k}{'+-'[sign < 0]}"


def init_prompt(problem) -> list[dict]:
    text = ROLE + problem.task_prompt + problem.example_prompt + problem.format_prompt
    return [{"role": "user", "content": text}]


def refine_prompt(problem, parent: dict) -> list[dict]:
    """LLaMEA's ``construct_prompt`` with a one-algorithm population."""
    summary = f"{parent['name']}: {parent['description']} (Score: {parent['fitness']})"
    text = f"""{problem.task_prompt}
The current population of algorithms already evaluated (name, description, score) is:
{summary}

The selected solutions to update are:\n\n# 1
## Description:\n\t {parent['description']}
## With Code
```python
{parent['code']}
```
## Feedback:\n\t {parent['feedback']}\n
{REFINE}{problem.format_prompt}"""
    return [{"role": "user", "content": ROLE + text}]


def evaluate(problem, solution) -> dict:
    t0 = time.time()
    try:
        solution = problem(solution)
        fit = float(solution.fitness)
        err = solution.error or ""
    except Exception as e:  # infrastructure failure, not the code's
        return {"fitness": None, "error": f"INFRA: {e!r}"[:500], "infra": True}
    ok = bool(np.isfinite(fit) and not err)
    md = solution.metadata
    return {
        "fitness": fit if np.isfinite(fit) else None,
        "failed": not ok,
        "error": err[:500],
        "feedback": solution.feedback,
        "behaviour": md.get("behaviour") if ok else None,
        "final_regret": md.get("final_regret") if ok else None,
        "aocc_blade": md.get("aocc_blade") if ok else None,
        "over_budget_runs": md.get("over_budget_runs"),
        "runtime_s": md.get("runtime_s"),
        "eval_s": time.time() - t0,
        "infra": False,
    }


def tasks_for(camp, phase, problem, raw) -> list[tuple]:
    """(prompt id, session, k, condition, parent id) to sample."""
    conds = conditions(camp)
    if phase == "init":
        prompts = [("init", init_prompt(problem), None)]
        ks = range(camp.k_init)
    else:
        init = [r for r in read_jsonl(raw / "init.jsonl") if r["condition"] == "none"]
        good = sorted((r for r in init if not r.get("failed")), key=lambda r: r["k"])
        parents = good[: camp.n_parents]
        if len(parents) < camp.n_parents:
            print(f"only {len(parents)} working unsteered parents")
        prompts = [
            (f"refine-{p['k']}", refine_prompt(problem, p), p["k"]) for p in parents
        ]
        ks = range(camp.k_refine)
    return [(pid, s, k, c, par) for pid, s, par in prompts for k in ks for c in conds]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True, choices=list(TRANSFER_CAMPAIGNS))
    ap.add_argument("--phase", default="both", choices=["init", "refine", "both"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-workers", type=int, default=8)
    args = ap.parse_args()

    camp = TRANSFER_CAMPAIGNS[args.tag]
    raw = data_dir(EXPERIMENT, "raw", camp.tag)
    problem = make_problem(camp.problem)
    problem._ensure_env()
    d = dial(camp)
    conds = conditions(camp)
    preset = MODELS[MODEL]
    llm = HookedLLM(
        preset.hf_id,
        capture=False,
        device_map=args.device,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_new_tokens=camp.max_new_tokens,
        chat_template_kwargs=preset.chat_template_kwargs,
    )
    phases = ["init", "refine"] if args.phase == "both" else [args.phase]
    for phase in phases:
        path = raw / f"{phase}.jsonl"
        done = set()
        if path.exists():
            done = {
                (r["prompt"], r["k"], r["condition"])
                for r in read_jsonl(path)
                if not r.get("infra")
            }
        todo = [
            t
            for t in tasks_for(camp, phase, problem, raw)
            if t[:1] + t[2:4] not in done
        ]
        print(f"[{camp.tag}] {phase}: {len(todo)} to do ({len(done)} done)", flush=True)
        lock = threading.Lock()
        out = open(path, "a")

        def finish(row, solution):
            row.update(evaluate(problem, solution) if solution else {})
            with lock:
                out.write(json.dumps(row) + "\n")
                out.flush()

        pool = ThreadPoolExecutor(args.eval_workers)
        t0 = time.time()
        for i, (pid, session, k, cond, parent) in enumerate(todo):
            kind, dose = conds[cond]
            seed = zlib.crc32(f"{camp.tag}-{pid}-{k}".encode()) % (2**31 - 1)
            vec, rname = steer_vector(d, kind, dose, seed)
            if vec is not None:
                llm.set_steering({d["layer"]: vec.astype(np.float32)})
            np.random.seed(seed)  # HookedLLM draws its torch seed from numpy
            row = {
                "prompt": pid,
                "phase": phase,
                "parent_k": parent,
                "k": k,
                "condition": cond,
                "kind": kind,
                "dose": dose,
                "random_dir": rname,
                "seed": seed,
            }
            solution = None
            t1 = time.time()
            try:
                solution = llm.sample_solution(session)
                row.update(
                    name=solution.name,
                    code=solution.code,
                    description=solution.description,
                )
            except NoCodeException:
                row.update(fitness=None, failed=True, error="NoCodeException")
                row["infra"] = False
            except Exception as e:
                row.update(fitness=None, error=f"INFRA: {e!r}"[:500], infra=True)
            finally:
                llm.clear_steering()
            rec = llm.last_record if llm.records else None
            llm.records.clear()
            if rec is not None:
                row["n_gen_tokens"] = rec["n_gen_tokens"]
                row["mean_logprob"] = rec["mean_logprob"]
                llm.last_record = None
            row["gen_s"] = time.time() - t1
            pool.submit(finish, row, solution)
            if (i + 1) % 10 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(
                    f"[{camp.tag}] {phase}: {i + 1}/{len(todo)}, {rate * 60:.1f}/min",
                    flush=True,
                )
        pool.shutdown(wait=True)
        out.close()
    problem.cleanup()
    print(f"[{camp.tag}] done", flush=True)


if __name__ == "__main__":
    main()
