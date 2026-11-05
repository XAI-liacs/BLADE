"""Step 5: held-out test of the child-failure pre-screen inside LLaMEA.

Runs the same LLaMEA set-up as ``01_generate.py`` on new seeds, with the
probe and threshold frozen from an earlier campaign
(``04_export_probes.py``). Every seed runs under two problems:

* ``MA_BBOB_shadow``: the probe score is logged, LLaMEA sees every result.
  Tests whether the probe generalises to new runs.
* ``MA_BBOB_screened``: algorithms scored above the threshold are shown to
  LLaMEA as failed. Tests what the screen does to the search.

All algorithms are still evaluated, so the true outcome of every flagged
algorithm is in the log (``metadata["true_*"]``). Both conditions of a
shard run in parallel threads that share one model::

    SCRIPT=05_prescreen_generate.py scripts/launch.sh prescreen 12 4
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
from common.paths import data_dir, results_dir
from common.prescreen import ScreenedHookedLLM, load_probe
from common.problems import PreScreenedMABBOB
from exp_config import EXPERIMENT, OPERATORS, PRESETS
from llamea import Operator

from iohblade.experiment import Experiment
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="prescreen", choices=sorted(PRESETS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument(
        "--smoke", action="store_true", help="one tiny run to check the pipeline"
    )
    args = ap.parse_args()

    cfg = PRESETS[args.config]
    if args.smoke:
        cfg.tag, cfg.seeds, cfg.budget = f"{cfg.tag}-smoke", [999], 4
        cfg.n_parents, cfg.n_offspring = 2, 2
        cfg.training_instances = [0]
    if not cfg.prescreen_probe:
        raise SystemExit(f"preset {cfg.tag!r} has no prescreen_probe")
    preset = MODELS[cfg.model]
    probe_dir = results_dir(EXPERIMENT, cfg.prescreen_source) / "probes"
    summary = json.loads((probe_dir / "probes.json").read_text())
    name = cfg.prescreen_probe
    if name not in summary:  # a target name: that target's exported probe
        name = next(n for n, i in summary.items() if i["target"] == name)
    info = summary[name]
    probe = load_probe(probe_dir / f"{name}.npz")
    seeds = cfg.seeds[args.shard :: args.n_shards]
    out = data_dir(EXPERIMENT, "raw", cfg.tag)
    (out / "config.json").write_text(
        json.dumps({**cfg.to_dict(), "probe": info}, indent=2)
    )
    print(f"[{cfg.tag}] probe={name} {info} seeds={seeds}")

    llm = ScreenedHookedLLM(
        preset.hf_id,
        probe=probe,
        position=info["position"],
        layer=info["layer"],
        device_map=args.device,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens,
        chat_template_kwargs=preset.chat_template_kwargs,
        store_dtype=preset.store_dtype,
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
    problems = [
        PreScreenedMABBOB(
            dims=cfg.dims,
            budget_factor=cfg.budget_factor,
            training_instances=cfg.training_instances,
            eval_timeout=cfg.eval_timeout,
            dependencies=cfg.eval_dependencies,
            name=f"MA_BBOB_{condition}",
            threshold=info["threshold"],
            act=condition == "screened",
        )
        for condition in ("shadow", "screened")
    ]
    logger = ExperimentLogger(str(out / f"shard-{args.shard}"))
    Experiment(
        methods=[method],
        problems=problems,
        seeds=seeds,
        budget=cfg.budget,
        n_jobs=len(problems) * len(seeds),  # threads share the model
        show_stdout=False,
        log_stdout=True,
        exp_logger=logger,
    )()


if __name__ == "__main__":
    main()
