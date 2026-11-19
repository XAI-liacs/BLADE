"""Review follow-up (#5): the failure pre-screen inside LLaMEA on Bragg.

The frozen MA-BBOB child-failure probe of experiment 1
(``child_failed_gen_mean_L21``, no retraining) screens LLaMEA's children on
mini-Bragg. Its MA-BBOB threshold does not transfer (the transfer screens
fail 70% of the time), so the threshold is a running quantile that needs
no calibration on the problem: a child is flagged if its score is in the
top 21% of the last 24 scores of its run (the share skipped on MA-BBOB);
the initial population is never flagged (``PreScreenedPhotonics``).

Every seed runs twice, as in experiment 1's pre-registered test:
``shadow`` (LLaMEA sees every result; the flag is logged) and ``screened``
(flagged children are shown as failed). Every child is still evaluated, so
the true outcome and evaluation time of each flagged child are logged.
LLaMEA set-up as the Bragg campaigns of step 5d (``05_llamea.py``):
4 parents, 12 offspring, elitist, refine/new/simplify, budget 100.

    python scripts/17_prescreen_loop.py --shard 0 --n-shards 3
    python scripts/17_prescreen_loop.py --analyze
"""

import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MI = HERE.parents[1]
sys.path[:0] = [str(MI), str(HERE.parent), str(HERE)]

from common.models import MODELS
from common.paths import data_dir, results_dir
from common.prescreen import ScreenedHookedLLM, load_probe
from exp_config import EXPERIMENT, EXTRA_EVAL_DEPENDENCIES, MODEL
from llamea import Operator

from iohblade.experiment import Experiment
from iohblade.loggers import ExperimentLogger
from iohblade.methods import LLaMEA

TAG = "prescreen-bragg"
SEEDS = list(range(440, 452))  # new seeds
PROBE = "child_failed_gen_mean_L21"
SHARE, WINDOW, WARMUP = 0.21, 24, 12


def exp01():
    spec = importlib.util.spec_from_file_location(
        "exp01_config", MI / "exp01_probing/exp_config.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(args):
    from common.transfer_problems import PreScreenedPhotonics

    src = exp01()
    cfg = src.PRESETS["prescreen"]
    preset = MODELS[MODEL]
    probe_dir = results_dir("exp01_probing", cfg.prescreen_source) / "probes"
    info = json.loads((probe_dir / "probes.json").read_text())[PROBE]
    probe = load_probe(probe_dir / f"{PROBE}.npz")
    seeds = (SEEDS if not args.smoke else [998])[args.shard :: args.n_shards]
    tag = TAG + ("-smoke" if args.smoke else "")
    out = data_dir(EXPERIMENT, "raw", tag)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            {"probe": info, "share": SHARE, "window": WINDOW, "warmup": WARMUP},
            indent=1,
        )
    )
    print(f"[{tag}] shard {args.shard}: seeds {seeds}", flush=True)
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
        Operator(prompt=src.OPERATORS[k], name=k, weight=1.0, number_of_parents=1)
        for k in ("refine", "new", "simplify")
    ]
    budget = 12 if args.smoke else 100
    method = LLaMEA(
        llm,
        budget=budget,
        name="LLaMEA",
        operators=operators,
        n_parents=2 if args.smoke else 4,
        n_offspring=2 if args.smoke else 12,
        elitism=True,
    )
    problems = [
        PreScreenedPhotonics(
            "bragg",
            extra_dependencies=EXTRA_EVAL_DEPENDENCIES,
            act=act,
            share=SHARE,
            window=WINDOW,
            warmup=2 if args.smoke else WARMUP,
        )
        for act in (False, True)
    ]
    Experiment(
        methods=[method],
        problems=problems,
        seeds=seeds,
        budget=budget,
        n_jobs=len(problems) * len(seeds),  # threads share the model
        show_stdout=False,
        log_stdout=True,
        exp_logger=ExperimentLogger(str(out / f"shard-{args.shard}")),
    )()


def analyze():
    """Paired shadow vs screened: best (true) fitness, skipped share,
    probe AUROC on the shadow runs, evaluation time saved."""
    from scipy import stats
    from sklearn.metrics import roc_auc_score

    rows = []
    for log in sorted(data_dir(EXPERIMENT, "raw", TAG).glob("shard-*/run-*/log.jsonl")):
        cond = "screened" if "_screened" in log.parent.name else "shadow"
        seed = int(log.parent.name.rsplit("-", 1)[1])
        for i, line in enumerate(log.read_text().splitlines()):
            d = json.loads(line)
            m = d["metadata"]
            rows.append(
                {
                    "cond": cond,
                    "seed": seed,
                    "i": i,
                    "score": m.get("prescreen_score"),
                    "flagged": bool(m.get("prescreen_flagged")),
                    "failed": m.get("true_fitness") is None,
                    "true_fitness": m.get("true_fitness"),
                    "seconds": m.get("eval_seconds"),
                }
            )
    df = pd.DataFrame(rows)
    dest = results_dir(EXPERIMENT, TAG)
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "children.csv", index=False)
    # best true fitness of each run: what the search found, screened or not
    # (a flagged child is still evaluated, but LLaMEA never sees it, so
    # only unflagged children count for the screened runs)
    seen = df[(df.cond == "shadow") | ~df.flagged]
    best = seen.groupby(["cond", "seed"]).true_fitness.max().unstack(0)
    d = (best.screened - best.shadow).dropna()
    rng = np.random.default_rng(0)
    bs = rng.choice(d.to_numpy(), (10000, len(d))).mean(1)
    sh = df[df.cond == "shadow"].dropna(subset=["score"])
    sc = df[df.cond == "screened"]
    lines = [
        f"runs: {best.shape[0]} seeds paired",
        f"best true fitness: shadow {best.shadow.mean():.3f}, screened"
        f" {best.screened.mean():.3f}; diff {d.mean():+.3f}"
        f" [95% CI {np.quantile(bs, 0.025):+.3f}, {np.quantile(bs, 0.975):+.3f}],"
        f" screened wins {(d > 0).sum()}/{len(d)},"
        f" Wilcoxon p = {stats.wilcoxon(d).pvalue:.3f}",
        f"shadow: failure rate {sh.failed.mean():.3f}, probe AUROC"
        f" {roc_auc_score(sh.failed, sh.score):.3f}, flagged {sh.flagged.mean():.3f},"
        f" precision {sh[sh.flagged].failed.mean():.3f}, failures caught"
        f" {sh[sh.failed].flagged.mean():.3f}",
        f"screened: evaluations skipped {sc.flagged.mean():.3f}; evaluation time"
        f" of flagged children {sc[sc.flagged].seconds.sum() / sc.seconds.sum():.3f}"
        " of the total (time saved if not evaluated)",
        f"median evaluation time: failing {sh[sh.failed].seconds.median():.1f} s,"
        f" working {sh[~sh.failed].seconds.median():.1f} s",
    ]
    (dest / "summary.md").write_text(
        "# Pre-screen in the loop on Bragg\n\n"
        + "\n".join(f"* {l}" for l in lines)
        + "\n"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    analyze() if a.analyze else run(a)
