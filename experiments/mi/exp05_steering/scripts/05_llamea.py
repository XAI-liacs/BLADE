"""Step 5: LLaMEA with a steered variation operator.

Every arm (``exp_config.ARMS``) runs experiment 1's LLaMEA set-up
(Qwen3-8B, 4 parents, 12 offspring, elitism, budget-enforced MA-BBOB)
with the operators refine, new and simplify plus one extra operator of
equal weight. For ``levers`` and ``random`` the extra operator uses the
refine prompt and adds a steering vector while the child is generated:
one lever of ``levers.json`` (``04_select_levers.py``), drawn at random
per child, or a random unit direction with the same layer and norm.

Every logged solution carries ``metadata["operator"]`` (``init`` for the
initial population), ``metadata["steer"]`` (lever name or ``None``) and
``metadata["steer_norm"]``. Shards split the seeds; all arms and seeds of a shard run as
threads of one ``Experiment`` that share one model (generation is serialised by ``HookedLLM``'s lock;
evaluation, the slow part, overlaps)::

    for k in $(seq 0 7); do CUDA_VISIBLE_DEVICES=$((2 + k % 2)) \\
      uv run python 05_llamea.py --tag llamea --shard $k --n-shards 8 & done

Re-running resumes: finished runs are skipped by BLADE's logger.
"""

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MI_ROOT = HERE.parents[1]
sys.path.insert(0, str(MI_ROOT))
sys.path.insert(0, str(HERE.parent))

from common.models import MODELS
from common.paths import data_dir
from common.problems import BudgetEnforcedMABBOB
from exp_config import EXPERIMENT, LLAMEA_CAMPAIGNS, MODEL, SOURCE_EXPERIMENT
from llamea import LLaMEA as LLaMEAAlgorithm
from llamea import Operator

from iohblade.experiment import Experiment
from iohblade.llm import HookedLLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA


def load_source_config():
    from importlib import util

    path = MI_ROOT / SOURCE_EXPERIMENT / "exp_config.py"
    spec = util.spec_from_file_location("source_exp_config", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OperatorSteeredLLM(HookedLLM):
    """``HookedLLM`` that steers the children of some operators.

    ``plans`` maps an operator name to a list of levers
    ``{"name", "layer", "vector"}``; a child of that operator is generated
    with one of them (uniformly at random) added at its layer. The
    algorithm sets ``next_operator`` before each query (same thread).
    """

    def __init__(self, model: str, plans: dict | None = None, **kwargs):
        super().__init__(model, **kwargs)
        self.plans = plans or {}
        self.next_operator = None

    def sample_solution(self, session_messages: list, parent_ids=[], **kwargs):
        op, self.next_operator = self.next_operator or "init", None
        lever = None
        if self.plans.get(op):
            options = self.plans[op]
            lever = options[np.random.randint(len(options))]
            self.set_steering({lever["layer"]: lever["vector"]})
        try:
            solution = super().sample_solution(session_messages, parent_ids, **kwargs)
        finally:
            if lever is not None:
                self.clear_steering()
        solution.add_metadata("operator", op)
        solution.add_metadata("steer", lever["name"] if lever else None)
        solution.add_metadata(
            "steer_norm", float(np.linalg.norm(lever["vector"])) if lever else 0.0
        )
        return solution


class _Algorithm(LLaMEAAlgorithm):
    def _evolve_solution(self, individual, operator):
        self.llm.next_operator = operator.name
        return super()._evolve_solution(individual, operator)


class SteeredLLaMEA(LLaMEA):
    """BLADE's LLaMEA method, with the operator name passed to the LLM."""

    def __call__(self, problem):
        self.llamea_instance = _Algorithm(
            f=problem,
            llm=self.llm,
            role_prompt="You are an excellent Python programmer.",
            task_prompt=problem.task_prompt,
            example_prompt=problem.example_prompt,
            output_format_prompt=problem.format_prompt,
            log=None,
            budget=self.budget,
            max_workers=1,
            minimization=getattr(problem, "minimisation", False),
            **self.kwargs,
        )
        return self.llamea_instance.run()


def lever_plans(source_tag: str) -> dict[str, list[dict]]:
    """``levers`` and ``random`` lever lists from ``levers.json``."""
    proc = data_dir(EXPERIMENT, "processed", source_tag)
    levers = json.loads((proc / "levers.json").read_text())
    rng = np.random.default_rng(12345)
    plans = {"levers": [], "random": []}
    for lv in levers:
        src = data_dir(EXPERIMENT, "processed", lv.get("source", source_tag))
        dirs = np.load(src / "directions.npz")
        vec = lv["dose"] * lv["scale"] * dirs[lv["direction"]]
        plans["levers"].append(
            {"name": lv["direction"], "layer": lv["layer"], "vector": vec}
        )
        r = rng.standard_normal(vec.shape).astype(np.float32)
        r *= np.linalg.norm(vec) / np.linalg.norm(r)
        plans["random"].append(
            {"name": f"random-{lv['direction']}", "layer": lv["layer"], "vector": r}
        )
    return plans


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="llamea-smoke", choices=list(LLAMEA_CAMPAIGNS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    camp = LLAMEA_CAMPAIGNS[args.tag]
    src_mod = load_source_config()
    src = src_mod.PRESETS["main"]
    preset = MODELS[MODEL]
    plans = lever_plans(camp.lever_source)
    seeds = camp.seeds[args.shard :: args.n_shards]
    out = data_dir(EXPERIMENT, "raw", camp.tag)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "campaign": camp.tag,
        "arms": [a.__dict__ for a in camp.arms],
        "levers": {k: [x["name"] for x in v] for k, v in plans.items()},
        "lever_norms": {
            k: [float(np.linalg.norm(x["vector"])) for x in v] for k, v in plans.items()
        },
    }
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    print(f"[{camp.tag}] shard {args.shard}: seeds {seeds}, {len(camp.arms)} arms")

    base_llm = OperatorSteeredLLM(
        preset.hf_id,
        capture=False,
        device_map=args.device,
        temperature=src.temperature,
        top_p=src.top_p,
        max_new_tokens=camp.max_new_tokens,
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
    ops = src_mod.OPERATORS
    methods = []
    for arm in camp.arms:
        llm = base_llm.__deepcopy__({})
        operators = [
            Operator(prompt=ops[k], name=k, weight=1.0, number_of_parents=1)
            for k in ("refine", "new", "simplify")
        ]
        prompt = ops["new"] if arm.extra == "new" else ops["refine"]
        operators.append(
            Operator(
                prompt=prompt, name=f"x_{arm.extra}", weight=1.0, number_of_parents=1
            )
        )
        if arm.extra in plans:
            llm.plans = {f"x_{arm.extra}": plans[arm.extra]}
        methods.append(
            SteeredLLaMEA(
                llm,
                budget=camp.budget,
                name=f"LLaMEA-{arm.name}",
                operators=operators,
                n_parents=camp.n_parents,
                n_offspring=camp.n_offspring,
                elitism=True,
            )
        )
    # One Experiment for all arms: it deletes the problem's evaluation
    # environment when it finishes, so runs must not span Experiments.
    Experiment(
        methods=methods,
        problems=[problem],
        seeds=seeds,
        budget=camp.budget,
        n_jobs=len(methods) * len(seeds),  # threads share the model
        show_stdout=False,
        log_stdout=True,
        exp_logger=ExperimentLogger(str(out / f"shard-{args.shard}")),
    )()
    print(f"[{camp.tag}] shard {args.shard}: done")


if __name__ == "__main__":
    main()
