"""Compare plain LLaMEA against RLM pre-screening variants.

Each invocation runs one problem suite (separate experiments, separate
result folders):

* ``mabbob``: MA-BBOB, instances 300-349 (not in the RLM training data).
* ``photonics``: the Bragg-mirror photonics problem (a real-world problem
  the RLM has never seen).
* ``bbob``: BBOB functions 1, 3, 6, 8, 10, 13, 15, 17, 21, 23.

Every method makes the same number of real benchmark evaluations
(``--budget``). For ``threshold_reject``, rejected candidates use up a
LLaMEA budget slot without being evaluated, so those variants get a larger
LLaMEA budget (see ``threshold_llamea_budget``). The experiment-level
budget still caps real evaluations per run at ``--budget`` for every method.

The ``rlm_*`` kwargs are forwarded unchanged by ``iohblade.methods.LLaMEA``
to ``llamea.LLaMEA``. Setup:

    uv sync --group rlm-surrogate --group extraproblems

Before a real run, confirm each checkpoint's feature settings (they are
not stored in the checkpoint) and update ``CHECKPOINT_FEATURES``:

    uv run python examples/check-rlm-feature-config.py \\
        --checkpoint-dir /data/neocortex/rlm/checkpoints/production_lhs200/ \\
        --expect-mode lhs --expect-points 200 --expect-seed 0

Usage:
    # Cheap pilot: Dummy LLM, budget 16, 1 run, few instances.
    uv run python examples/run-rlm-screening.py --suite mabbob --pilot \\
        --ckpt-lhs200 /data/neocortex/rlm/checkpoints/production_lhs200/

    # Full matrix with Claude (credentials: ANTHROPIC_API_KEY or an
    # `ant auth login` profile).
    uv run python examples/run-rlm-screening.py --suite photonics \\
        --ckpt-lhs200 /data/neocortex/rlm/checkpoints/production_lhs200/ \\
        --ckpt-lhs20 /data/neocortex/rlm/checkpoints/production_lhs20/ --with-lhs20
"""

import argparse
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import ioh

import iohblade.benchmarks.BBOB.mabbob as blade_mabbob
from iohblade.benchmarks import BBOB_SBOX, MA_BBOB, Photonics
from iohblade.experiment import Experiment
from iohblade.llm import Claude_LLM, Dummy_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA
from iohblade.solution import Solution

# Feature settings each checkpoint was trained with. Confirm with
# examples/check-rlm-feature-config.py before trusting any result: a wrong
# value gives out-of-distribution inputs and meaningless predictions,
# silently.
CHECKPOINT_FEATURES = {
    "lhs200": {
        "rlm_feature_mode": "lhs",
        "rlm_n_lhs_points": 200,
        "rlm_lhs_seed": 0,
    },
    "lhs20": {
        "rlm_feature_mode": "lhs",
        "rlm_n_lhs_points": 20,
        "rlm_lhs_seed": 0,
    },
}

THRESHOLD_QUANTILE = 0.5
BBOB_FIDS = [1, 2, 3]
MA_BBOB_TRAIN = range(300, 350)
MA_BBOB_TEST = range(350, 400)
BLADE_MA_BBOB_DIR = Path(blade_mabbob.__file__).parent / "mabbob"


# --------------------------------------------------------------------------
# Problem suites and the matching rlm_problem_instances
# --------------------------------------------------------------------------


def build_suite(suite: str, pilot: bool) -> tuple[Any, list[dict], dict]:
    """Returns (BLADE problem, rlm_problem_instances, custom instance kinds).

    The instances describe the functions the problem evaluates candidates
    on during evolution (its training instances). Custom kinds map a
    ``kind`` string to an ioh-style problem the RLM can fingerprint; see
    ``install_rlm_instance_hooks``.
    """
    if suite == "bbob":
        fids = [1, 21] if pilot else BBOB_FIDS
        problem = BBOB_SBOX(
            training_instances=[(f, i) for f in fids for i in range(1, 6)],
            test_instances=[(f, i) for f in fids for i in range(5, 16)],
            dims=[5],
            budget_factor=2000,
            eval_timeout=600,
            name="BBOB",
            # The RLM rebuilds kind="bbob" with ioh.get_problem (BBOB class);
            problem_type=ioh.ProblemClass.BBOB,
        )
        instances = [
            {"kind": "bbob", "dim": dim, "fid_or_idx": fid, "iid": iid}
            for dim in problem.dims
            for fid, iid in problem.training_instances
        ]
        return problem, instances, {}

    if suite == "mabbob":
        problem = MA_BBOB(
            training_instances=range(300, 302) if pilot else MA_BBOB_TRAIN,
            test_instances=MA_BBOB_TEST,
            dims=[5],
            budget_factor=2000,
            eval_timeout=600,
            name="MA_BBOB",
        )
        instances = [
            {"kind": "ma_bbob", "dim": dim, "fid_or_idx": idx}
            for dim in problem.dims
            for idx in problem.training_instances
        ]
        return problem, instances, {}

    if suite == "photonics":
        problem = Photonics(
            None,
            problem_type="bragg",
            name="Bragg",
            budget_factor=100 if pilot else 500,
            seeds=1 if pilot else 5,
            eval_timeout=3600,
        )
        # A separate copy for fingerprinting, so the LHS probes do not
        # touch the instance BLADE evaluates on.
        fingerprint_problem = Photonics(None, problem_type="bragg").problem
        dim = fingerprint_problem.meta_data.n_variables
        kind = "photonics:bragg"
        instances = [{"kind": kind, "dim": dim, "fid_or_idx": 0}]
        return problem, instances, {kind: fingerprint_problem}

    raise ValueError(f"Unknown suite: {suite!r}")


def install_rlm_instance_hooks(custom_kinds: dict[str, Any]) -> None:
    """Lets LLaMEA's RLM screener describe the problems used here.

    LLaMEA's ``ProblemInstance`` only knows ``bbob`` and ``ma_bbob``, so:

    * MA-BBOB: LLaMEA looks for its MA-BBOB tables outside the installed
      package. BLADE ships byte-identical copies, so point it at those.
    * Other kinds (e.g. photonics): build the problem text with LLaMEA's
      own ``lhs_fingerprint``/``lhs_summary_stats`` on the given problem,
      in the same format as the BBOB/MA-BBOB training text.

    Experiment runs use threads, so patching once here covers every run.
    """
    import llamea.rlm_surrogate.problem_instances as pi
    import llamea.rlm_surrogate.screening as screening

    if not (pi.MA_BBOB_DATA_DIR / "weights.csv").exists():
        pi.MA_BBOB_DATA_DIR = BLADE_MA_BBOB_DIR

    if not custom_kinds:
        return
    original = pi.compute_problem_feature_text

    def feature_text(instance, *, n_points=20, seed=0, mode="lhs"):
        problem = custom_kinds.get(instance.kind)
        if problem is None:
            return original(instance, n_points=n_points, seed=seed, mode=mode)
        if mode == "lhs":
            return pi.lhs_fingerprint(problem, instance.dim, n_points, seed)
        if mode == "lhs_stats":
            return pi.lhs_summary_stats(problem, instance.dim, n_points, seed)
        raise ValueError(
            f"feature_mode={mode!r} needs static BBOB/MA-BBOB metadata, "
            f"which {instance.kind!r} does not have."
        )

    screening.compute_problem_feature_text = feature_text


def check_rlm_screeners(
    checkpoints: dict[str, str], instances: list[dict], rlm_extra: dict
) -> None:
    """Loads each RLM screener once and scores one test candidate.

    LLaMEA builds its screener inside every run, so a screener that cannot
    load only shows up as failed runs, after LLM budget has been spent.
    This runs the same setup once, before any run starts, and exits with
    the error if it fails. Call it after ``install_rlm_instance_hooks``.
    """
    import numpy as np
    from llamea.rlm_surrogate.problem_instances import ProblemInstance
    from llamea.rlm_surrogate.screening import RLMScreener, RLMScreeningConfig

    test_candidate = Solution(
        code=(
            "import numpy as np\n\n"
            "class RandomSearch:\n"
            "    def __init__(self, budget=10000, dim=10):\n"
            "        self.budget = budget\n"
        ),
        name="RandomSearch",
        description="Random search, used to test the RLM screener.",
    )
    problem_instances = [ProblemInstance(**inst) for inst in instances]
    for tag, ckpt in checkpoints.items():
        features = CHECKPOINT_FEATURES[tag]
        config = RLMScreeningConfig(
            checkpoint_dir=ckpt,
            strategy="log_only",
            feature_mode=features["rlm_feature_mode"],
            n_lhs_points=features["rlm_n_lhs_points"],
            lhs_seed=features["rlm_lhs_seed"],
            device=rlm_extra.get("rlm_device"),
            num_samples_point_pred=rlm_extra.get("rlm_num_samples_point_pred"),
        )
        start = time.time()
        try:
            screener = RLMScreener(config, problem_instances)
            (score,) = screener.score([test_candidate])
            if not np.isfinite(score):
                raise ValueError(f"test candidate scored {score}")
        except Exception as e:
            traceback.print_exc()
            sys.exit(
                f"RLM screener check failed for {tag} ({ckpt}): "
                f"{type(e).__name__}: {e}. No runs were started."
            )
        print(
            f"RLM screener {tag}: OK ({len(problem_instances)} instances, "
            f"test score {score:.4g}, {time.time() - start:.0f}s)",
            flush=True,
        )
        del screener
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# --------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------


def threshold_llamea_budget(
    real_evals: int, n_parents: int, n_offspring: int, quantile: float
) -> int:
    """LLaMEA budget giving ``threshold_reject`` ``real_evals`` real evals.

    The initial population is always evaluated. Each later generation uses
    ``n_offspring`` budget slots but only evaluates the candidates at or
    above the ``quantile`` of the batch's predictions:
    ``n_offspring - ceil(quantile * (n_offspring - 1))`` of them (with
    ``np.quantile``'s linear interpolation and no tied predictions).
    """
    kept = n_offspring - math.ceil(quantile * (n_offspring - 1))
    generations = math.ceil(max(real_evals - n_parents, 0) / kept)
    return n_parents + generations * n_offspring


def build_methods(
    llm,
    budget: int,
    n_parents: int,
    n_offspring: int,
    checkpoints: dict[str, str],
    instances: list[dict],
    rlm_extra: dict,
) -> list[LLaMEA]:
    """Baseline plus, per checkpoint, the log-only/top-k/threshold variants."""

    def variant(name, llamea_budget=budget, **kwargs):
        return LLaMEA(
            llm,
            budget=llamea_budget,
            name=name,
            n_parents=n_parents,
            n_offspring=n_offspring,
            elitism=False,
            **kwargs,
        )

    methods = [variant("baseline-no-screening")]
    for tag, ckpt in checkpoints.items():
        rlm = {
            "rlm_checkpoint_dir": ckpt,
            "rlm_problem_instances": instances,
            **CHECKPOINT_FEATURES[tag],
            **rlm_extra,
        }
        q = int(THRESHOLD_QUANTILE * 100)
        methods += [
            # Control: scores every offspring, never filters.
            variant(f"log-only-{tag}", rlm_screening="log_only", **rlm),
            # Keep the predicted-best 50% / 25% of an over-generated pool.
            # Discarded candidates use no budget, so budgets stay equal.
            variant(
                f"topk-{tag}-k50",
                rlm_screening="topk_filter",
                rlm_overgenerate_factor=2.0,
                **rlm,
            ),
            variant(
                f"topk-{tag}-k25",
                rlm_screening="topk_filter",
                rlm_overgenerate_factor=4.0,
                **rlm,
            ),
            # Reject the predicted-worse part of each offspring batch. The
            # larger LLaMEA budget keeps real evaluations equal to the rest.
            variant(
                f"threshold-{tag}-q{q}",
                llamea_budget=threshold_llamea_budget(
                    budget, n_parents, n_offspring, THRESHOLD_QUANTILE
                ),
                rlm_screening="threshold_reject",
                rlm_threshold_quantile=THRESHOLD_QUANTILE,
                **rlm,
            ),
        ]
    return methods


def main():
    p = argparse.ArgumentParser(description="LLaMEA vs. RLM pre-screening.")
    p.add_argument("--suite", required=True, choices=["mabbob", "photonics", "bbob"])
    p.add_argument("--pilot", action="store_true", help="Dummy LLM, tiny run.")
    p.add_argument("--ckpt-lhs200", default=os.getenv("RLM_CKPT_LHS200"))
    p.add_argument("--ckpt-lhs20", default=os.getenv("RLM_CKPT_LHS20"))
    p.add_argument(
        "--with-lhs20",
        action="store_true",
        help="Also run the lhs20-checkpoint mirrors of every RLM variant.",
    )
    p.add_argument("--model", default="claude-haiku-5-5", help="Claude model ID.")
    p.add_argument(
        "--effort",
        default="medium",
        choices=["low", "medium", "high", "xhigh", "max"],
        help="Claude effort level (thinking depth and token spend).",
    )
    p.add_argument("--budget", type=int, default=None, help="Real evals/run.")
    p.add_argument("--runs", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--rlm-device", default=None, help="cuda / cpu / auto")
    p.add_argument(
        "--rlm-num-samples",
        type=int,
        default=None,
        help="Override the checkpoint's num_samples_point_pred (speed).",
    )
    args = p.parse_args()

    if not args.ckpt_lhs200:
        p.error("--ckpt-lhs200 (or RLM_CKPT_LHS200) is required")
    checkpoints = {"lhs200": args.ckpt_lhs200}
    if args.with_lhs20:
        if not args.ckpt_lhs20:
            p.error("--with-lhs20 needs --ckpt-lhs20 (or RLM_CKPT_LHS20)")
        checkpoints["lhs20"] = args.ckpt_lhs20
    for ckpt in checkpoints.values():
        for f in ("model.pt", "config.yaml"):
            if not os.path.exists(os.path.join(ckpt, f)):
                p.error(f"{ckpt} has no {f}")

    if args.pilot:
        llm = Dummy_LLM("dummy-model")
        budget, runs = args.budget or 16, args.runs or 1
        # Small population so the pilot spans several generations.
        n_parents, n_offspring = 2, 4
        results_dir = f"results/rlm-screening-{args.suite}-pilot"
    else:
        llm = Claude_LLM(
            os.getenv("CLAUDE_API_KEY"),  # None: SDK resolves credentials
            args.model,
            max_tokens=16000,  # thinking + the generated algorithm
            effort=args.effort,
            # On a safety refusal, re-run the request on Anthropic's
            # recommended fallback model instead of losing the offspring.
            fallbacks="default",
        )
        budget, runs = args.budget or 100, args.runs or 5
        n_parents, n_offspring = 4, 12
        results_dir = f"results/rlm-screening-{args.suite}"
    results_dir = args.results_dir or results_dir

    problem, instances, custom_kinds = build_suite(args.suite, args.pilot)
    install_rlm_instance_hooks(custom_kinds)

    rlm_extra = {}
    if args.rlm_device:
        rlm_extra["rlm_device"] = args.rlm_device
    if args.rlm_num_samples:
        rlm_extra["rlm_num_samples_point_pred"] = args.rlm_num_samples

    # Fail now, with the real error, rather than in every screening run.
    check_rlm_screeners(checkpoints, instances, rlm_extra)

    # One instance list, shared by every RLM variant on this suite.
    methods = build_methods(
        llm, budget, n_parents, n_offspring, checkpoints, instances, rlm_extra
    )
    os.makedirs(os.path.dirname(results_dir) or ".", exist_ok=True)
    experiment = Experiment(
        methods=methods,
        problems=[problem],
        runs=runs,
        # Caps real evaluations per run, also for the threshold variants.
        budget=budget,
        exp_logger=ExperimentLogger(results_dir),
        n_jobs=min(len(methods), args.n_jobs),
        show_stdout=args.pilot,
    )
    experiment()


if __name__ == "__main__":  # needed for the spawn start method
    main()
