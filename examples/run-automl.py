"""
Run AutoML experiments on OpenML tasks using LLaMEA.

- Loads an OpenML suite (classification by default) with optional filters.
- Supports sharding across multiple packets of tasks via --num-shards/--shard.
- Executes each task in its own process to isolate state and prevent nested parallelism.
- Ollama is assumed to run on localhost; you can override the port with OLLAMA_URL,
  e.g. OLLAMA_URL=http://127.0.0.1:11435.
- Results root via BLADE_RESULTS_DIR (default: ./results).
- Caches OpenML data under <RESULTS_ROOT>/openml_cache and writes logs/results under
  <RESULTS_ROOT>/results/automl-openml-validation/<stamp>/<problem>.
- --search-eval-mode inner_val (default): candidates are scored on a validation part of
  the train fold, and the official test fold is only used for the final score of the
  best pipeline. official_test scores candidates on the official test folds (old behaviour).

- Example:
    export BLADE_RESULTS_DIR=/path/to/results
    export OLLAMA_URL=http://127.0.0.1:11434
    python examples/run-automl.py --tasks 2073 --budget 20 --model deepseek-chat \
        --search-eval-mode inner_val
"""

import os
import sys
import warnings
import argparse
import datetime
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse
import openml

# import path so we can run this from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from iohblade.benchmarks.automl.automl import AutoML, VALID_TABPFN_EVAL_SOURCES
from iohblade.loggers import ExperimentLogger
from iohblade.experiment import Experiment
from iohblade.llm import Ollama_LLM, DeepSeek_LLM, Gemini_LLM, OpenAI_LLM
from iohblade.methods.llamea import LLaMEA
from llamea.operator import Operator
from iohblade.tabpfn_utils import VALID_TABPFN_MODES, detect_tabpfn

RESULTS_ROOT = os.getenv("BLADE_RESULTS_DIR", str((Path.cwd() / "results").resolve()))
HEAVY_OPENML_TASKS = {
    3945, 7593, 10090, 168868, 168909,
    189354, 189355, 189356, 190412,
    359953, 359966, 359967, 359973,
    359976, 359985, 359989, 359994,
    360112, 360113, 360114, 360975
}

HEAVY_TIMEOUT_SEC = 4 * 3600   # 4 hours
NORMAL_TIMEOUT_SEC = 1 * 3600  # 1 hour


def make_deepseek_llm(model: str, temperature: float):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")
    return DeepSeek_LLM(api_key=api_key, model=model, temperature=temperature)


def make_gemini_llm(model: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    return Gemini_LLM(api_key=api_key, model=model)


def make_openai_llm(model: str, temperature: float):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI_LLM(api_key=api_key, model=model, temperature=temperature)


def get_openml_task_type(tid: int) -> str:
    task = openml.tasks.get_task(tid)
    task_type = str(task.task_type).lower()
    return "classification" if "classification" in task_type else "regression"


def unchecked_tabpfn_info(task_type: str) -> dict:
    return {
        "available": None,
        "version": None,
        "error": None,
        "package_available": None,
        "classifier_available": None,
        "classifier_error": None,
        "regressor_available": None,
        "regressor_error": None,
        "required_estimators": [],
        "task_type": task_type,
    }


def check_tabpfn_for_run(tabpfn_mode: str, require_tabpfn: bool, task_type: str):
    if tabpfn_mode == "off" and not require_tabpfn:
        return unchecked_tabpfn_info(task_type)

    info = detect_tabpfn(task_type)
    info["task_type"] = task_type
    if require_tabpfn and not info["available"]:
        raise RuntimeError(
            "--require-tabpfn was set, but TabPFN is not available for "
            f"{task_type}: {info.get('error')}"
        )
    if tabpfn_mode == "explicit" and not info["available"]:
        raise RuntimeError(
            "--tabpfn-mode explicit requires TabPFN to be importable for "
            f"{task_type}: {info.get('error')}"
        )
    return info


def tabpfn_signature_for_task(tabpfn_info: dict, task_type: str) -> str | None:
    if task_type == "regression":
        return tabpfn_info.get("regressor_signature")
    return tabpfn_info.get("classifier_signature")


def tabpfn_kwargs_for_task(tabpfn_info: dict, task_type: str) -> list[str] | None:
    if task_type == "regression":
        return tabpfn_info.get("regressor_kwargs")
    return tabpfn_info.get("classifier_kwargs")


def write_run_config(prob_dir: str, config: dict) -> None:
    config_path = os.path.join(prob_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def run_one_task(
    tid,
    results_root,
    model,
    ollama_url,
    budget,
    stamp,
    crossover_rate,
    seed,
    n_parents,
    n_offspring,
    parent_selection,
    tournament_size,
    search_eval_mode,
    inner_val_size,
    inner_val_seed,
    temperature,
    tabpfn_mode,
    require_tabpfn,
    tabpfn_eval_source,
    eval_timeout_override=None,
    eval_cpus=1,
):
    """
    Run a single OpenML task in its own process.
    """

    import os
    import warnings
    import openml
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    warnings.filterwarnings("ignore")

    # Keep math libs single-threaded in each worker to avoid thread storms.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # Cores that n_jobs=-1 in a generated pipeline may use (0 = no limit).
    if eval_cpus:
        os.environ["LOKY_MAX_CPU_COUNT"] = str(eval_cpus)

    # OpenML cache lives under the shared results root
    openml.config.set_root_cache_directory(
        os.getenv("OPENML_CACHE_DIR", os.path.join(results_root, "openml_cache"))
    )
    openml.config.server = "https://www.openml.org/api/v1"
    openml.config.set_retry_policy("robot", n_retries=20)

    # Decide timeout based on whether this is a heavy task
    if tid in HEAVY_OPENML_TASKS:
        eval_timeout = HEAVY_TIMEOUT_SEC
        size_label = "HEAVY"
    else:
        eval_timeout = NORMAL_TIMEOUT_SEC
        size_label = "NORMAL"

    if eval_timeout_override is not None:
        eval_timeout = int(eval_timeout_override)
        size_label += "/override"

    print(
        f"[TASK {tid}] Category={size_label}, eval_timeout={eval_timeout/3600:.1f}h",
        flush=True,
    )

    task_type = get_openml_task_type(tid)
    tabpfn_info = check_tabpfn_for_run(
        tabpfn_mode=tabpfn_mode,
        require_tabpfn=require_tabpfn,
        task_type=task_type,
    )
    tabpfn_signature = tabpfn_signature_for_task(tabpfn_info, task_type)
    tabpfn_allowed_kwargs = tabpfn_kwargs_for_task(tabpfn_info, task_type)
    if tabpfn_mode != "off" or require_tabpfn:
        print(
            f"[TASK {tid}] tabpfn_mode={tabpfn_mode}, "
            f"task_type={task_type}, "
            f"tabpfn_available={tabpfn_info.get('available')}, "
            f"tabpfn_version={tabpfn_info.get('version')}, "
            f"required_estimators={tabpfn_info.get('required_estimators')}, "
            f"constructor_signature={tabpfn_signature}, "
            f"error={tabpfn_info.get('error')}",
            flush=True,
        )

    # Problem / logging directory
    prob = AutoML(
        openml_task_id=tid,
        name=f"AutoML-OpenML-{tid}",
        eval_timeout=eval_timeout,
        search_eval_mode=search_eval_mode,
        inner_val_size=inner_val_size,
        inner_val_seed=inner_val_seed,
        tabpfn_mode=tabpfn_mode,
        tabpfn_available=tabpfn_info.get("available"),
        tabpfn_version=tabpfn_info.get("version"),
        tabpfn_eval_source=tabpfn_eval_source,
        tabpfn_signature=tabpfn_signature,
        tabpfn_allowed_kwargs=tabpfn_allowed_kwargs,
    )
    print(
        f"[TASK {tid}] search_eval_mode={search_eval_mode}, "
        f"inner_val_size={inner_val_size}, inner_val_seed={inner_val_seed}",
        flush=True,
    )
    prob_dir = os.path.join(
        results_root, "results", "automl-openml-validation", stamp, prob.name
    )
    os.makedirs(prob_dir, exist_ok=True)
    write_run_config(
        prob_dir,
        {
            "model": model,
            "budget": budget,
            "search_eval_mode": search_eval_mode,
            "task_id": tid,
            "seed": seed,
            "tabpfn_mode": tabpfn_mode,
            "tabpfn_available": tabpfn_info.get("available"),
            "tabpfn_version": tabpfn_info.get("version"),
            "tabpfn_error": tabpfn_info.get("error"),
            "tabpfn_eval_source": tabpfn_eval_source,
            "tabpfn_signature": tabpfn_signature,
            "tabpfn_allowed_kwargs": tabpfn_allowed_kwargs,
            "tabpfn_required_estimators": tabpfn_info.get("required_estimators"),
            "require_tabpfn": require_tabpfn,
            "task_type": task_type,
            "stamp": stamp,
            "temperature": temperature,
            "eval_timeout": eval_timeout,
            "inner_val_size": inner_val_size,
            "inner_val_seed": inner_val_seed,
            "crossover_rate": crossover_rate,
            "n_parents": n_parents,
            "n_offspring": n_offspring,
            "parent_selection": parent_selection,
            "tournament_size": tournament_size,
        },
    )
    logger = ExperimentLogger(prob_dir)

    if model.startswith("deepseek"):
        llm = make_deepseek_llm(model=model, temperature=temperature)
    elif model.startswith("gemini"):
        llm = make_gemini_llm(model=model)
    elif model.startswith("gpt-"):
        llm = make_openai_llm(model=model, temperature=temperature)
    else:
        # Derive port from OLLAMA_URL (host is assumed local)
        port = urlparse(ollama_url).port or 11434
        llm = Ollama_LLM(model=model, port=port)

    mutation_prompts = [
        # change the model family
        "Change the model family used (e.g., linear <-> tree ensemble <-> kernel method), while keeping the same class interface and avoiding internal CV/HPO loops.",
        # refine the preprocessing
        "Refine the preprocessing part of the pipeline: add/remove/adjust scaling, imputation handling, or simple feature selection, but keep the estimator family unchanged.",
        # add or remove one model in a small ensemble
        "Add or remove ONE base model to a small ensemble (e.g., VotingClassifier) while keeping everything simple and within constraints. Do not introduce internal CV/search.",
    ]
    crossover_prompts = [
        "Combine the two parent pipelines by creating a simple ensemble such as VotingClassifier using both parents' estimators. Keep preprocessing shared and minimal. 2 base models is enough. No internal CV/HPO.",
    ]
    # llamea 1.3 expects a list of operators instead of the prompt lists.
    # The weight sets how often an operator is picked: crossover gets crossover_rate,
    # the mutations share the rest. Keyword arguments so prompt and name cannot be swapped.
    cx_rate = float(crossover_rate)
    mut_w = (1.0 - cx_rate) / len(mutation_prompts) if mutation_prompts else 0.0
    operators = [
        Operator(
            prompt=text,
            name=f"mutation_{i + 1}",
            weight=mut_w,
            number_of_parents=1,
        )
        for i, text in enumerate(mutation_prompts)
    ]
    if cx_rate > 0 and crossover_prompts:
        cx_w = cx_rate / len(crossover_prompts)
        operators += [
            Operator(
                prompt=text,
                name=f"crossover_{i + 1}",
                weight=cx_w,
                number_of_parents=2,
            )
            for i, text in enumerate(crossover_prompts)
        ]

    method = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA",
        operators=operators,
        n_parents=n_parents,
        n_offspring=n_offspring,
        elitism=True,
        HPO=True,
        parallel_backend="threading",
        parent_selection=parent_selection,
        tournament_size=tournament_size,
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
        "--crossover-rate",
        "--crossover_rate",
        dest="crossover_rate",
        type=float,
        default=0.0,
        help="Probability of crossover per offspring (0.0-1.0).",
    )
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
    parser.add_argument(
        "--tasks",
        type=str,
        default="",
        help="Comma-separated OpenML task IDs to run instead of loading a suite.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for API-based LLMs such as DeepSeek/OpenAI.",
    )
    parser.add_argument("--n-parents", type=int, default=4)
    parser.add_argument("--n-offspring", type=int, default=4)
    parser.add_argument(
        "--parent-selection",
        type=str,
        default="random",
        choices=["random", "tournament", "roulette"],
    )
    parser.add_argument("--tournament-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--search-eval-mode",
        type=str,
        default="inner_val",
        choices=["official_test", "inner_val"],
        help="Which score to feed back to LLaMEA during search.",
    )
    parser.add_argument(
        "--inner-val-size",
        type=float,
        default=0.2,
        help="Validation fraction taken from the OpenML train fold when search-eval-mode=inner_val.",
    )
    parser.add_argument(
        "--inner-val-seed",
        type=int,
        default=42,
        help="Base seed for inner train/validation splits.",
    )
    parser.add_argument(
        "--tabpfn-mode",
        choices=VALID_TABPFN_MODES,
        default="off",
        help=(
            "TabPFN ablation mode: off keeps the current prompt and "
            "evaluation dependencies; passive makes TabPFN available and "
            "logs detection without mentioning it in the prompt; explicit "
            "also appends TabPFN availability text to the prompt."
        ),
    )
    parser.add_argument(
        "--eval-timeout",
        type=int,
        default=None,
        help="Seconds per candidate evaluation, overrides the task default (1h, 4h for heavy tasks).",
    )
    parser.add_argument(
        "--eval-cpus",
        type=int,
        default=1,
        help="Cores that n_jobs=-1 may use in a candidate evaluation (0 = no limit).",
    )
    parser.add_argument(
        "--require-tabpfn",
        action="store_true",
        help="Fail early if TabPFN cannot be imported for the OpenML task type.",
    )
    parser.add_argument(
        "--tabpfn-eval-source",
        choices=VALID_TABPFN_EVAL_SOURCES,
        default="env",
        help=(
            "How BLADE evaluation subprocesses should access TabPFN. "
            "'env' inherits packages from the runner conda env; 'pip' "
            "installs TabPFN into each temporary evaluation virtualenv."
        ),
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
    openml.config.set_root_cache_directory(
        os.getenv("OPENML_CACHE_DIR", os.path.join(RESULTS_ROOT, "openml_cache"))
    )
    openml.config.server = "https://www.openml.org/api/v1"
    openml.config.set_retry_policy("robot", n_retries=20)

    base_dir = os.path.join(RESULTS_ROOT, "results", "automl-openml-validation", stamp)
    os.makedirs(base_dir, exist_ok=True)

    # Build the task list
    if args.tasks:
        # Use explicit TASKS list
        all_task_ids = [int(t.strip()) for t in args.tasks.split(",") if t.strip()]
    else:
        # Fallback: load tasks from OpenML suite
        clf_suite = openml.study.get_suite(args.suite)
        # Make sure they are ints, not strings
        all_task_ids = [int(t) for t in clf_suite.tasks]

    # Optional: skip some tasks
    if args.skip:
        skip_ids = {int(x) for x in args.skip.split(",") if x.strip()}
        all_task_ids = [t for t in all_task_ids if t not in skip_ids]

    # Optional: limit
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
                args.crossover_rate,
                args.seed,
                args.n_parents,
                args.n_offspring,
                args.parent_selection,
                args.tournament_size,
                args.search_eval_mode,
                args.inner_val_size,
                args.inner_val_seed,
                args.temperature,
                args.tabpfn_mode,
                args.require_tabpfn,
                args.tabpfn_eval_source,
                args.eval_timeout,
                args.eval_cpus,
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
