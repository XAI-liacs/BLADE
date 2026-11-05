"""Step 4: sample and evaluate a child for every counterfactual prompt.

Reads ``data/processed/<tag>/cf_queries.jsonl`` from
``03_prepare_counterfactuals.py``. For every query, variant and replicate
``k`` it samples one response with the experiment 1 settings (Qwen3-8B,
T = 0.8, top-p = 0.95), extracts the code as LLaMEA does, and evaluates it
on the same budget-enforced MA-BBOB problem (sandboxed subprocess).

**Common random numbers.** The sampling seed depends only on the query and
``k``, not on the variant. If an edit does not change the next-token
distribution, the variant reproduces the ``full`` response token for token;
the length of the common prefix is a sensitive measure of influence.

The residual stream is captured at layers 9, 15 and 21, and the three
exported experiment 1 probes are read out: parent fitness (prompt mean,
L15), child failure (generated mean, L21) and over budget (generated mean,
L9). Arrays are not stored.

Generation runs on the GPU in the main thread; evaluations run in a thread
pool (each in its own sandboxed process). One line per sample is appended to
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
from common.paths import data_dir, results_dir
from common.prescreen import load_probe, probe_score
from common.problems import BudgetEnforcedMABBOB
from exp_config import EXPERIMENT, SOURCE_EXPERIMENT, load_source_config

from iohblade.llm import HookedLLM
from iohblade.utils import NoCodeException

PROBES = {  # readout name: (file, position, layer)
    "pf_L15": ("parent_fitness_prompt_mean_L15", "prompt_mean", 15),
    "cf_L21": ("child_failed_gen_mean_L21", "gen_mean", 21),
    "ob_L9": ("over_budget_gen_mean_L9", "gen_mean", 9),
}


def sample_seed(record_id: str, k: int) -> int:
    return zlib.crc32(f"{record_id}-{k}".encode()) % (2**31 - 1)


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
        "eval_s": time.time() - t0,
        "infra": False,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="cf")
    ap.add_argument("--k", type=int, default=4, help="replicates per variant")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--eval-workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="queries (smoke)")
    args = ap.parse_args()

    src = load_source_config().PRESETS["main"]
    preset = MODELS[src.model]
    proc = data_dir(EXPERIMENT, "processed", args.tag)
    queries = [json.loads(line) for line in open(proc / "cf_queries.jsonl")]
    queries = queries[args.shard :: args.n_shards][: args.limit]
    steer = np.load(proc / "cf_steering.npz")
    layer, alpha = int(steer["layer"]), float(steer["alpha"])
    vectors = {
        "up": alpha * steer["v"],
        "down": -alpha * steer["v"],
        "random": alpha * steer["random"],
    }
    probe_dir = results_dir(SOURCE_EXPERIMENT, "main", "probes")
    probes = {k: load_probe(probe_dir / f"{f}.npz") for k, (f, _, _) in PROBES.items()}

    out_path = data_dir(EXPERIMENT, "raw", args.tag) / f"shard-{args.shard}.jsonl"
    done = set()
    if out_path.exists():
        for line in open(out_path):
            r = json.loads(line)
            if not r.get("infra"):
                done.add((r["record_id"], r["variant"], r["k"]))
    tasks = [
        (q, v, k)
        for q in queries
        for k in range(args.k)
        for v in q["variants"]
        if (q["record_id"], v, k) not in done
    ]
    print(f"shard {args.shard}: {len(tasks)} samples to do ({len(done)} done)")
    if not tasks:
        return

    llm = HookedLLM(
        preset.hf_id,
        capture_layers=sorted({l for _, _, l in PROBES.values()}),
        device_map=args.device,
        temperature=src.temperature,
        top_p=src.top_p,
        max_new_tokens=src.max_new_tokens,
        chat_template_kwargs=preset.chat_template_kwargs,
    )
    problem = BudgetEnforcedMABBOB(
        dims=src.dims,
        budget_factor=src.budget_factor,
        training_instances=src.training_instances,
        eval_timeout=src.eval_timeout,
        dependencies=src.eval_dependencies,
        name="MA_BBOB_enforced",
    )
    problem._ensure_env()

    lock = threading.Lock()
    out = open(out_path, "a")

    def finish(row, solution):
        row.update(evaluate(problem, solution) if solution else {})
        with lock:
            out.write(json.dumps(row) + "\n")
            out.flush()

    pool = ThreadPoolExecutor(args.eval_workers)
    t0 = time.time()
    for i, (q, variant, k) in enumerate(tasks):
        spec = q["variants"][variant]
        if spec.get("steer"):
            llm.set_steering({layer: vectors[spec["steer"]]}, positions="all")
        seed = sample_seed(q["record_id"], k)
        np.random.seed(seed)  # HookedLLM draws its torch seed from numpy
        row = {
            "record_id": q["record_id"],
            "stratum": q["stratum"],
            "operator": q["operator"],
            "run": q["run"],
            "variant": variant,
            "k": k,
            "seed": seed,
        }
        solution = None
        try:
            solution = llm.sample_solution([{"role": "user", "content": spec["text"]}])
            row.update(name=solution.name, description=solution.description)
            row["code"] = solution.code
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
            arrays, layers = rec.get("arrays", {}), rec["layers"]
            for name, (_, pos, l) in PROBES.items():
                if pos in arrays and l in layers:
                    x = arrays[pos][layers.index(l)].astype(np.float32)
                    row[name] = (
                        probe_score(probes[name], x) if np.isfinite(x).all() else None
                    )
            llm.last_record = None
        pool.submit(finish, row, solution)
        if (i + 1) % 20 == 0:
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
