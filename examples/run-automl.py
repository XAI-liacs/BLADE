"""
Run AutoML experiments on OpenML tasks using LLaMEA with an Ollama-backed LLM.

- Loads an OpenML suite (classification by default) with optional filters.
- Supports sharding across multiple packets of tasks via --num-shards/--shard.
- Executes each task in its own process to isolate state and prevent nested parallelism.
- Ollama is assumed to run on localhost; you can override the port with OLLAMA_URL,
  e.g. OLLAMA_URL=http://127.0.0.1:11435.
- Results root via BLADE_RESULTS_DIR (default: ./results).
- Caches OpenML data under <RESULTS_ROOT>/openml_cache and writes logs/results under
  <RESULTS_ROOT>/results/automl-openml/<stamp>/<problem>.

- Example:
    export BLADE_RESULTS_DIR=/path/to/results
    export OLLAMA_URL=http://127.0.0.1:11434
    python examples/run-automl.py --suite amlb-classification-all --limit 4 --budget 20 --model qwen2.5-coder:32b
"""

import os
import sys
import warnings
import argparse
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
import openml

from iohblade.benchmarks import AutoML
from iohblade.loggers import ExperimentLogger
from iohblade.experiment import Experiment
from iohblade.llm import Ollama_LLM
from iohblade.methods import LLaMEA

# import path so we can run this from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
warnings.filterwarnings("ignore")

RESULTS_ROOT = os.getenv("BLADE_RESULTS_DIR", str((Path.cwd() / "results").resolve()))


def run_one_task(tid, results_root, model, ollama_url, budget, stamp):
    """
    Run a single OpenML task in its own process.
    """

    import os
    import warnings
    import openml

    warnings.filterwarnings("ignore")

    # Keep math libs single-threaded in each worker to avoid thread storms.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # OpenML cache lives under the shared results root
    openml.config.set_root_cache_directory(os.path.join(results_root, "openml_cache"))

    # Problem / logging directory
    prob = AutoML(openml_task_id=tid, name=f"AutoML-OpenML-{tid}")
    prob_dir = os.path.join(results_root, "results", "automl-openml", stamp, prob.name)
    os.makedirs(prob_dir, exist_ok=True)
    logger = ExperimentLogger(prob_dir)

    # Derive port from OLLAMA_URL (host is assumed local)
    parsed = urlparse(ollama_url)
    port = parsed.port or 11434

    llm = Ollama_LLM(
        model=model,
        port=port,
    )

    mutation_prompts = [
        "Refine the strategy of the selected solution to improve it.",  # small mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.",  # new random solution
        "Refine and simplify the selected algorithm to improve it.",  # simplify
    ]

    method = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA",
        mutation_prompts=mutation_prompts,
        n_parents=4,
        n_offspring=4,
        elitism=True,
        HPO=True,
        parallel_backend="threading",
    )

    # One method, one problem in this process
    Experiment(
        methods=[method],
        problems=[prob],
        runs=1,
        show_stdout=False,
        exp_logger=logger,
        budget=budget,
        n_jobs=1,
    )()

    return tid


def shard_list(lst, num_shards, shard_idx):
    """Return only the slice of lst that belongs to shard_idx (0-based)."""
    n = len(lst)
    base = n // num_shards
    rem = n % num_shards
    start = shard_idx * base + min(shard_idx, rem)
    end = start + base + (1 if shard_idx < rem else 0)
    return lst[start:end]


if __name__ == "__main__":
    # CLI for OpenML runs, defaults to the AMLB classification suite ('amlb-classification-all')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="How many tasks to run in parallel (processes).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only run the first N tasks of the suite.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the full task list into this many shards.",
    )
    parser.add_argument(
        "--shard", type=int, default=0, help="Which shard to run (0-based)."
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-coder:32b",
        help="Ollama model name (as seen in `ollama list`).",
    )
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument(
        "--suite",
        default="amlb-classification-all",
        help="OpenML suite ID or alias, e.g. 'amlb-classification-all'.",
    )
    parser.add_argument(
        "--skip",
        type=str,
        default="",
        help="Comma-separated OpenML task IDs to skip, e.g. 2073,359990",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the selected task IDs for this shard and exit.",
    )
    parser.add_argument(
        "--stamp",
        default=None,
        help="Shared run stamp so all shards write under the same folder.",
    )
    args = parser.parse_args()

    # One shared stamp per multi-shard run
    if args.stamp is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        stamp = args.stamp

    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    # openml_cache includes info about the datasets and tasks
    openml.config.set_root_cache_directory(os.path.join(RESULTS_ROOT, "openml_cache"))

    base_dir = os.path.join(RESULTS_ROOT, "results", "automl-openml", stamp)
    os.makedirs(base_dir, exist_ok=True)

    # Load tasks from OpenML suite
    clf_suite = openml.study.get_suite(args.suite)
    all_task_ids = clf_suite.tasks

    if args.skip:
        skip_ids = {int(x) for x in args.skip.split(",") if x.strip()}
        all_task_ids = [t for t in all_task_ids if t not in skip_ids]
    if args.limit:
        all_task_ids = all_task_ids[: args.limit]

    # Shard the task list (for multi-tmux splits)
    task_ids = shard_list(all_task_ids, args.num_shards, args.shard)
    if not task_ids:
        print("No tasks to run for this shard (empty task list after filtering/sharding).")
        raise SystemExit(0)

    print(
        f"Total tasks in suite: {len(all_task_ids)} | "
        f"Running shard {args.shard}/{args.num_shards} -> {len(task_ids)} tasks",
        flush=True,
    )
    print("Task IDs in this shard:", ",".join(map(str, task_ids)), flush=True)

    if args.list_tasks:
        raise SystemExit(0)

    # Cap concurrency to shard size
    max_workers = min(args.concurrency, len(task_ids)) or 1

    # Parallelize across tasks
    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(
                run_one_task,
                tid,
                RESULTS_ROOT,
                args.model,
                OLLAMA_URL,
                args.budget,
                stamp,
            ): tid
            for tid in task_ids
        }
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                fut.result()
                completed += 1
                print(
                    f"[{completed}/{len(task_ids)}] Task {tid} finished.",
                    flush=True,
                )
            except Exception as e:
                print(f"[ERROR] Task {tid} failed: {e}", flush=True)
