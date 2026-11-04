"""Step 1: run LLaMEA with a HookedLLM on MA-BBOB and record activations.

Every LLM query stores per-layer residual-stream summaries next to the BLADE
logs (see ``HookedLLM``). Shard the seeds over GPUs by starting one process
per GPU, e.g.::

    for g in 0 1 2 3; do
      uv run python experiments/mi/exp01_probing/scripts/01_generate.py \\
          --config main --device cuda:$g --shard $g --n-shards 4 &
    done

Re-running the same command resumes: finished runs are skipped.
"""

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # experiments/mi
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # exp01_probing

from common.models import MODELS
from common.paths import data_dir
from exp_config import EXPERIMENT, OPERATORS, PRESETS
from llamea import Operator

from iohblade.benchmarks import MA_BBOB
from iohblade.experiment import Experiment
from iohblade.llm import HookedLLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="smoke", choices=sorted(PRESETS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--model", help="override the preset's model key")
    args = ap.parse_args()

    cfg = PRESETS[args.config]
    if args.model:
        cfg.model = args.model
    preset = MODELS[cfg.model]
    seeds = cfg.seeds[args.shard :: args.n_shards]
    out = data_dir(EXPERIMENT, "raw", cfg.tag)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(f"[{cfg.tag}] model={preset.hf_id} seeds={seeds} -> {out}")

    llm = HookedLLM(
        preset.hf_id,
        device_map=args.device,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens,
        chat_template_kwargs=preset.chat_template_kwargs,
    )
    operators = [
        Operator(prompt=OPERATORS[k], name=k, weight=1.0, number_of_parents=1)
        for k in cfg.operators
    ]
    method = LLaMEA(
        llm,
        budget=cfg.budget,
        name="LLaMEA",
        operators=operators,
        n_parents=cfg.n_parents,
        n_offspring=cfg.n_offspring,
        elitism=cfg.elitism,
    )
    problem = MA_BBOB(
        dims=cfg.dims,
        budget_factor=cfg.budget_factor,
        training_instances=cfg.training_instances,
        eval_timeout=cfg.eval_timeout,
        dependencies=cfg.eval_dependencies,
    )
    logger = ExperimentLogger(str(out / f"shard-{args.shard}"))
    Experiment(
        methods=[method],
        problems=[problem],
        seeds=seeds,
        budget=cfg.budget,
        n_jobs=1,  # one model per process; parallelise over GPUs instead
        show_stdout=False,
        log_stdout=True,
        exp_logger=logger,
    )()


if __name__ == "__main__":
    main()
