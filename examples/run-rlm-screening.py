"""Compare plain LLaMEA against RLM pre-screening variants on BBOB.

Every method gets the same LLaMEA ``budget``. The RLM settings
(``rlm_*`` kwargs) are forwarded unchanged by ``iohblade.methods.LLaMEA``
to ``llamea.LLaMEA``; this needs the ``llamea`` feature branch pinned in
``pyproject.toml`` plus the ``rlm-surrogate`` dependency group:

    uv sync --group rlm-surrogate

Before a real run, confirm each checkpoint's feature settings against its
training data (they are not stored in the checkpoint):

    uv run python examples/check-rlm-feature-config.py \\
        --data-dir <data_pipeline output dir> --expect-mode lhs \\
        --expect-points 200 --expect-seed 0

Usage:
    # Cheap pilot: Dummy LLM, budget 16, 1 run, 2 BBOB functions.
    uv run python examples/run-rlm-screening.py --pilot \\
        --ckpt-lhs200 /path/to/checkpoints/production_lhs200

    # Full matrix (needs OPENAI_API_KEY).
    uv run python examples/run-rlm-screening.py \\
        --ckpt-lhs200 /path/to/production_lhs200 \\
        --ckpt-lhs20 /path/to/production_lhs --with-lhs20

Budget accounting (see ``llamea.rlm_surrogate.screening.RLMScreeningConfig``):

* ``log_only``: no change, every offspring is really evaluated.
* ``topk_filter``: discarded candidates use no budget; every budget slot is
  a real evaluation. LLM calls per generation grow by
  ``overgenerate_factor``.
* ``threshold_reject``: rejected candidates DO use a budget slot but are
  never evaluated (and so never appear in BLADE's ``log.jsonl``). With
  ``threshold_quantile=0.5`` a run makes roughly half as many real
  evaluations as the baseline. Compare it at equal LLM calls, or account
  for this when comparing at equal real evaluations.
"""

import argparse
import os

import ioh

from iohblade.benchmarks import BBOB_SBOX
from iohblade.experiment import Experiment
from iohblade.llm import Dummy_LLM, OpenAI_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA

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

# BBOB functions/instances, same set as examples/run-bbob-logging.py.
FIDS = [1, 3, 6, 8, 10, 13, 15, 17, 21, 23]
PILOT_FIDS = [1, 21]


def problem_instances_for_bbob(problem: BBOB_SBOX) -> list[dict]:
    """Describes the functions ``problem`` evaluates candidates on.

    Uses the training instances only: LLaMEA never calls ``f`` on the
    test instances during evolution.

    Raises:
        ValueError: if ``problem`` is not on the BBOB problem class. The
            RLM rebuilds ``kind="bbob"`` instances with
            ``ioh.get_problem(fid, iid, dim)`` (BBOB), so an SBOX-COST
            problem would be fingerprinted as the wrong functions.
    """
    if problem.problem_type != ioh.ProblemClass.BBOB:
        raise ValueError(
            "RLM screening instances are BBOB functions; build BBOB_SBOX with "
            "problem_type=ioh.ProblemClass.BBOB (the default is SBOX)."
        )
    return [
        {"kind": "bbob", "dim": dim, "fid_or_idx": fid, "iid": iid}
        for dim in problem.dims
        for fid, iid in problem.training_instances
    ]


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

    def variant(name, **kwargs):
        return LLaMEA(
            llm,
            budget=budget,
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
        methods += [
            # Control: scores every offspring, never filters.
            variant(f"log-only-{tag}", rlm_screening="log_only", **rlm),
            # Keep predicted-best 50% / 25% of an over-generated pool.
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
            # Reject the predicted-worse half of each offspring batch.
            variant(
                f"threshold-{tag}-q50",
                rlm_screening="threshold_reject",
                rlm_threshold_quantile=0.5,
                **rlm,
            ),
        ]
    return methods


def main():
    p = argparse.ArgumentParser(description="LLaMEA vs. RLM pre-screening.")
    p.add_argument("--pilot", action="store_true", help="Dummy LLM, tiny run.")
    p.add_argument("--ckpt-lhs200", default=os.getenv("RLM_CKPT_LHS200"))
    p.add_argument("--ckpt-lhs20", default=os.getenv("RLM_CKPT_LHS20"))
    p.add_argument(
        "--with-lhs20",
        action="store_true",
        help="Also run the lhs20-checkpoint mirrors of every RLM variant.",
    )
    p.add_argument("--model", default="gpt-4.1-2025-04-14")
    p.add_argument("--budget", type=int, default=None)
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
        fids, results_dir = PILOT_FIDS, "results/rlm-screening-pilot"
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            p.error("OPENAI_API_KEY is not set")
        llm = OpenAI_LLM(api_key, args.model)
        budget, runs = args.budget or 100, args.runs or 5
        n_parents, n_offspring = 4, 12
        fids, results_dir = FIDS, "results/rlm-screening"
    results_dir = args.results_dir or results_dir

    os.makedirs(os.path.dirname(results_dir) or ".", exist_ok=True)
    logger = ExperimentLogger(results_dir)
    problem = BBOB_SBOX(
        training_instances=[(f, i) for f in fids for i in range(1, 6)],
        test_instances=[(f, i) for f in fids for i in range(5, 16)],
        dims=[5],
        budget_factor=2000,
        eval_timeout=600,
        name="BBOB",
        problem_type=ioh.ProblemClass.BBOB,
    )
    # One list, shared by every RLM variant, so they all describe the same
    # benchmark functions.
    instances = problem_instances_for_bbob(problem)

    rlm_extra = {}
    if args.rlm_device:
        rlm_extra["rlm_device"] = args.rlm_device
    if args.rlm_num_samples:
        rlm_extra["rlm_num_samples_point_pred"] = args.rlm_num_samples

    methods = build_methods(
        llm, budget, n_parents, n_offspring, checkpoints, instances, rlm_extra
    )
    experiment = Experiment(
        methods=methods,
        problems=[problem],
        runs=runs,
        budget=budget,
        exp_logger=logger,
        n_jobs=min(len(methods), args.n_jobs),
        show_stdout=args.pilot,
    )
    experiment()


if __name__ == "__main__":  # needed for the spawn start method
    main()
