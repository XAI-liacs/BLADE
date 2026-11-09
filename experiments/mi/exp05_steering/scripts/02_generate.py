"""Step 2: sample and evaluate a child for every prompt x condition x replicate.

Conditions: ``none`` (no steering) and, for every direction of the
campaign (``directions.npz`` from ``01_prepare.py``) and every dose ``c``,
``<direction>@<c>``: the unit direction times ``c * scale`` is added to the
residual stream at its layer (``HookedLLM.set_steering``, generated tokens
only unless the campaign says ``positions="all"``).

As in experiment 2, the sampling seed depends only on the prompt and the
replicate, not on the condition (common random numbers), and every child
is evaluated on experiment 1's budget-enforced MA-BBOB problem in a
sandboxed subprocess. Generation runs on the GPU in the main thread,
evaluations in a thread pool. One line per sample is appended to
``data/raw/<tag>/shard-<k>.jsonl``; rerun to resume.
"""

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
from common.problems import BudgetEnforcedMABBOB, TracedMABBOB
from exp_config import CAMPAIGNS, EXPERIMENT, MODEL, SOURCE_EXPERIMENT

from iohblade.llm import HookedLLM
from iohblade.utils import NoCodeException


def load_source_config():
    from importlib import util

    path = MI_ROOT / SOURCE_EXPERIMENT / "exp_config.py"
    spec = util.spec_from_file_location("source_exp_config", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_seed(record_id: str, k: int) -> int:
    return zlib.crc32(f"{record_id}-{k}".encode()) % (2**31 - 1)


def conditions(camp, info) -> dict:
    """Condition name -> (direction, dose); ``none`` -> (None, 0).

    ``resample`` is unsteered with another seed: the natural variability
    that a steer's effect on the code is compared with.
    """
    out = {"none": (None, 0.0), "resample": (None, 0.0)}
    dirs = list(camp.directions) + [f"random{l}" for l in camp.random_layers]
    for d in dirs:
        if d not in info:
            continue
        for c in camp.doses:
            out[f"{d}@{c:+g}"] = (d, float(c))
    return out


def evaluate(problem, solution) -> dict:
    t0 = time.time()
    try:
        solution = problem(solution)
        fit = float(solution.fitness)
        err = solution.error or ""
    except Exception as e:  # infrastructure failure, not the code's
        return {"fitness": None, "error": f"INFRA: {e!r}"[:500], "infra": True}
    finite = np.isfinite(fit)
    return {
        "fitness": fit if finite else None,
        "failed": bool(not finite or err),
        "error": err[:500],
        "over_budget_instances": solution.metadata.get("over_budget_instances"),
        "behaviour": solution.metadata.get("behaviour") if finite and not err else None,
        "eval_s": time.time() - t0,
        "infra": False,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="pilot", choices=list(CAMPAIGNS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--eval-workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="prompts (smoke)")
    args = ap.parse_args()

    camp = CAMPAIGNS[args.tag]
    src = load_source_config().PRESETS["main"]
    preset = MODELS[MODEL]
    proc = data_dir(EXPERIMENT, "processed", camp.tag)
    info = json.loads((proc / "campaign.json").read_text())["directions"]
    dirs = np.load(proc / "directions.npz")
    conds = conditions(camp, info)
    queries = [json.loads(line) for line in open(proc / "queries.jsonl")]
    queries = queries[: args.limit]

    raw = data_dir(EXPERIMENT, "raw", camp.tag)
    raw.mkdir(parents=True, exist_ok=True)
    done = set()
    for path in sorted(raw.glob("shard-*.jsonl")):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:  # a line cut off by a stopped shard
                continue
            if not r.get("infra"):
                done.add((r["record_id"], r["condition"], r["k"]))
    tasks = [
        (q, c, k)
        for q in queries
        for k in range(camp.k)
        for c in conds
        if (q["record_id"], c, k) not in done
    ]
    tasks = tasks[args.shard :: args.n_shards]
    print(f"shard {args.shard}: {len(tasks)} samples to do ({len(done)} done)")
    if not tasks:
        return

    llm = HookedLLM(
        preset.hf_id,
        capture=False,
        device_map=args.device,
        temperature=src.temperature,
        top_p=src.top_p,
        max_new_tokens=camp.max_new_tokens,
        chat_template_kwargs=preset.chat_template_kwargs,
    )
    problem = (TracedMABBOB if camp.traced else BudgetEnforcedMABBOB)(
        dims=src.dims,
        budget_factor=src.budget_factor,
        training_instances=src.training_instances,
        eval_timeout=src.eval_timeout,
        dependencies=src.eval_dependencies,
        name="MA_BBOB_enforced",
    )
    problem._ensure_env()

    lock = threading.Lock()
    out = open(raw / f"shard-{args.shard}.jsonl", "a")

    def finish(row, solution):
        row.update(evaluate(problem, solution) if solution else {})
        with lock:
            out.write(json.dumps(row) + "\n")
            out.flush()

    pool = ThreadPoolExecutor(args.eval_workers)
    t0 = time.time()
    for i, (q, cond, k) in enumerate(tasks):
        direction, dose = conds[cond]
        norm = 0.0
        if direction is not None:
            d = info[direction]
            vec = dose * d["scale"] * dirs[direction]
            norm = float(np.linalg.norm(vec))
            llm.set_steering({d["layer"]: vec}, positions=camp.positions)
        seed = sample_seed(q["record_id"], k + (1000 if cond == "resample" else 0))
        np.random.seed(seed)  # HookedLLM draws its torch seed from numpy
        row = {
            "record_id": q["record_id"],
            "operator": q["operator"],
            "run": q["run"],
            "condition": cond,
            "direction": direction,
            "dose": dose,
            "steer_norm": norm,
            "k": k,
            "seed": seed,
        }
        solution = None
        t1 = time.time()
        try:
            solution = llm.sample_solution(q["session"])
            row.update(name=solution.name, code=solution.code)
        except NoCodeException:
            row.update(fitness=None, failed=True, error="NoCodeException", infra=False)
        except Exception as e:
            row.update(fitness=None, error=f"INFRA: {e!r}"[:500], infra=True)
        finally:
            llm.clear_steering()
        rec = llm.last_record if llm.records else None
        llm.records.clear()
        if rec is not None:
            row["response"] = rec.get("response")
            row["n_gen_tokens"] = rec["n_gen_tokens"]
            row["mean_logprob"] = rec["mean_logprob"]
            llm.last_record = None
        row["gen_s"] = time.time() - t1
        pool.submit(finish, row, solution)
        if (i + 1) % 10 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(
                f"shard {args.shard}: {i + 1}/{len(tasks)} generated, "
                f"{rate * 60:.1f}/min",
                flush=True,
            )
    pool.shutdown(wait=True)
    out.close()
    print(f"shard {args.shard}: done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
