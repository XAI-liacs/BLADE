"""Step 3: build the generation-level counterfactual campaign.

Step 1 measured how likely the *recorded* response is under edited prompts.
This step resamples the response under each edit and evaluates it, so the
effect is measured on what the model writes and how well that code does.

Two strata of recorded mutation queries:

* ``scored``: the parent ran, so its feedback is a score. From experiment 1
  ``main``, ``--n-scored`` per operator. Variants:

  - ``full``: the original prompt;
  - ``score_low`` / ``score_high``: the parent's score set to 0.010 / 0.900
    **both in the feedback and in the population list**, so the true score
    is no longer visible anywhere (step 1 only changed the feedback);
  - ``no_feedback``: the feedback span deleted;
  - ``budget_msg``: only if the parent overspent the evaluation budget on
    some instances. The feedback says so explicitly (MA-BBOB's feedback
    never does);
  - ``steer_up`` / ``steer_down``: ``+/- alpha * v`` added to the residual
    stream at the parent-fitness probe's layer (15 for Qwen3-8B) on every
    position, where ``v`` is the difference of
    the mean ``prompt_mean`` activation between top- and bottom-tercile
    parents (experiment 1's parent-fitness representation; adding ``v``
    moves the exported probe's readout by the tercile gap);
  - ``steer_random``: a random direction of the same norm, orthogonal to
    ``v`` (control for unspecific perturbation).

* ``error``: the parent crashed, so its feedback is an error message. These
  are rare in one campaign (elitism seldom selects a failed parent), so
  the source campaign and the ``shadow`` runs of the pre-screen campaigns
  (``--error-tags``) are pooled (identical prompts). Variants: ``full``,
  ``no_error`` (feedback and error deleted) and ``error_generic`` (replaced
  by "The algorithm crashed during evaluation.").

``--source-tag`` selects the experiment 1 campaign (and so the model), e.g.
``main`` (Qwen3-8B) or ``main-qwen3-14b``.

Writes ``data/processed/<tag>/cf_queries.jsonl`` (one line per query, with
every variant's prompt), ``cf_steering.npz`` (``v``, the random control,
layer and alpha) and ``cf_meta.json`` (source campaign, model, layer).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from common.paths import data_dir, results_dir
from common.runs import iter_run_dirs, load_arrays, load_run, prompt_text
from exp_config import EXPERIMENT, SOURCE_EXPERIMENT, load_source_config
from spans import ablate, feedback_score, segment

OPERATORS = load_source_config().OPERATORS
GENERIC_ERROR = "## Feedback:\n\t The algorithm crashed during evaluation.\n\n"


def set_parent_score(text: str, spans: dict, value: str) -> tuple[str, int]:
    """Replaces the parent's score in the feedback and the population list.

    Returns the new text and the number of population entries changed.
    """
    s, e = spans["feedback"]
    fb = text[s:e]
    m = re.search(r"The algorithm (\S+) scored ([0-9.]+) on AOCC", fb)
    name, shown = m.group(1), float(m.group(2))
    fb = fb[: m.start(2)] + value + fb[m.end(2) :]
    ps, pe = spans["population"]
    pop, n = text[ps:pe], 0

    def repl(mm):
        nonlocal n
        if round(float(mm.group(2)), 3) != round(shown, 3):
            return mm.group(0)
        n += 1
        return f"{mm.group(1)}{value})"

    pop = re.sub(
        rf"(^{re.escape(name)}:.*\(Score: )([-0-9.e]+)\)", repl, pop, flags=re.M
    )
    return text[:ps] + pop + text[pe:s] + fb + text[e:], n


def add_budget_message(text: str, spans: dict, n_over: int, n_inst: int) -> str:
    s, e = spans["feedback"]
    fb = text[s:e]
    body = fb.rstrip()
    msg = (
        f"\n\t The algorithm called func more times than its budget allows on "
        f"{n_over} of {n_inst} instances. Those runs were stopped when the "
        "budget ran out, so it was scored only on the evaluations made within "
        "the budget. Make sure the algorithm never exceeds self.budget calls."
    )
    return text[:s] + body + msg + fb[len(body) :] + text[e:]


def candidate_queries(tag: str, condition: str | None) -> list[dict]:
    meta = pd.read_csv(
        data_dir(SOURCE_EXPERIMENT, "processed", tag) / "meta.csv",
        dtype={"record_id": str, "run": str},
    ).set_index("record_id")
    if condition is not None:
        meta = meta[meta["condition"] == condition]
    out = []
    for run_dir in iter_run_dirs(data_dir(SOURCE_EXPERIMENT, "raw", tag)):
        sols, records = load_run(run_dir)
        for rec in records:
            if rec["record_id"] not in meta.index or len(rec["prompt"]) != 1:
                continue
            m = meta.loc[rec["record_id"]]
            if m["operator"] not in OPERATORS:
                continue
            text = prompt_text(rec)
            spans = segment(text, OPERATORS[m["operator"]])
            parent = sols.get((rec.get("parent_ids") or [None])[0])
            if spans is None or parent is None:
                continue
            md = parent.get("metadata") or {}
            out.append(
                {
                    "record_id": rec["record_id"],
                    "source_tag": tag,
                    "run": f"{tag}/{m['run']}",
                    "operator": m["operator"],
                    "parent_failed": bool(m["parent_failed"]),
                    "parent_fitness": m["parent_fitness"],
                    "parent_over_budget_instances": int(
                        md.get("over_budget_instances", 0) or 0
                    ),
                    "parent_code": parent.get("code", ""),
                    "parent_name": parent.get("name", ""),
                    "orig_child_fitness": m["child_fitness"],
                    "orig_child_failed": m["child_failed"],
                    "text": text,
                    "spans": spans,
                    "run_dir": str(run_dir),
                    "file": rec.get("file"),
                    "layers": rec.get("layers"),
                }
            )
    return out


def fitness_direction(queries: list[dict], layer: int) -> tuple[np.ndarray, dict]:
    """Top-minus-bottom tercile mean of ``prompt_mean`` at ``layer``."""
    X, y = [], []
    for q in queries:
        f = q["parent_fitness"]
        if q["parent_failed"] or not np.isfinite(f) or not q["file"]:
            continue
        a = load_arrays(q["run_dir"], q)
        X.append(a["prompt_mean"][q["layers"].index(layer)].astype(np.float32))
        y.append(f)
    X, y = np.array(X), np.array(y)
    lo, hi = np.quantile(y, [1 / 3, 2 / 3])
    v = X[y >= hi].mean(0) - X[y <= lo].mean(0)
    info = {
        "n": len(y),
        "tercile_low": float(lo),
        "tercile_high": float(hi),
        "fitness_gap": float(y[y >= hi].mean() - y[y <= lo].mean()),
        "norm": float(np.linalg.norm(v)),
        "mean_activation_norm": float(np.linalg.norm(X, axis=1).mean()),
    }
    return v.astype(np.float32), info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="cf")
    ap.add_argument("--source-tag", default="main", help="experiment 1 campaign")
    ap.add_argument(
        "--error-tags",
        default="prescreen,prescreen2",
        help="pre-screen campaigns whose shadow runs add crashed parents",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="steering layer (default: the parent-fitness probe's layer)",
    )
    ap.add_argument("--n-scored", type=int, default=32, help="per operator")
    ap.add_argument("--n-error", type=int, default=96)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = data_dir(EXPERIMENT, "processed", args.tag)

    src = load_source_config().PRESETS[args.source_tag]
    layer = args.layer
    if layer is None:
        summary = json.loads(
            (
                results_dir(SOURCE_EXPERIMENT, args.source_tag, "probes")
                / "probes.json"
            ).read_text()
        )
        layer = next(
            i["layer"] for i in summary.values() if i["target"] == "parent_fitness"
        )
    main_q = candidate_queries(args.source_tag, None)
    v, info = fitness_direction(main_q, layer)
    r = rng.standard_normal(v.shape).astype(np.float32)
    r -= (r @ v) / (v @ v) * v
    r *= np.linalg.norm(v) / np.linalg.norm(r)
    np.savez(
        out / "cf_steering.npz",
        v=v,
        random=r,
        layer=layer,
        alpha=args.alpha,
    )
    (out / "cf_meta.json").write_text(
        json.dumps(
            {
                "source_tag": args.source_tag,
                "model": src.model,
                "layer": int(layer),
                "alpha": args.alpha,
                "error_tags": args.error_tags.split(",") if args.error_tags else [],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"fitness direction at layer {layer}: {info}")

    scored = [
        q
        for q in main_q
        if not q["parent_failed"] and feedback_score(q["text"], q["spans"]) is not None
    ]
    picked = []
    for op in OPERATORS:
        pool = [q for q in scored if q["operator"] == op]
        idx = rng.choice(len(pool), min(args.n_scored, len(pool)), replace=False)
        picked += [dict(pool[i], stratum="scored") for i in sorted(idx)]

    errors = [q for q in main_q if q["parent_failed"]]
    for tag in [t for t in args.error_tags.split(",") if t]:
        errors += [q for q in candidate_queries(tag, "shadow") if q["parent_failed"]]
    idx = rng.choice(len(errors), min(args.n_error, len(errors)), replace=False)
    picked += [dict(errors[i], stratum="error") for i in sorted(idx)]
    print(f"error-parent queries available: {len(errors)}")

    n_pop = []
    with open(out / "cf_queries.jsonl", "w") as f:
        for q in picked:
            text, spans = q["text"], q["spans"]
            variants = {"full": {"text": text}}
            if q["stratum"] == "scored":
                for label, value in (("score_low", "0.010"), ("score_high", "0.900")):
                    t, n = set_parent_score(text, spans, value)
                    variants[label] = {"text": t}
                    n_pop.append(n)
                variants["no_feedback"] = {"text": ablate(text, spans, "feedback")}
                if q["parent_over_budget_instances"] > 0:
                    variants["budget_msg"] = {
                        "text": add_budget_message(
                            text, spans, q["parent_over_budget_instances"], 10
                        )
                    }
                for label, steer in (
                    ("steer_up", "up"),
                    ("steer_down", "down"),
                    ("steer_random", "random"),
                ):
                    variants[label] = {"text": text, "steer": steer}
            else:
                s, e = spans["feedback"]
                variants["no_error"] = {"text": ablate(text, spans, "feedback")}
                variants["error_generic"] = {
                    "text": text[:s] + GENERIC_ERROR + text[e:]
                }
            keep = {
                k: q[k]
                for k in (
                    "record_id",
                    "source_tag",
                    "run",
                    "stratum",
                    "operator",
                    "parent_failed",
                    "parent_fitness",
                    "parent_over_budget_instances",
                    "parent_code",
                    "parent_name",
                    "orig_child_fitness",
                    "orig_child_failed",
                )
            }
            keep = {
                k: (None if isinstance(x, float) and not np.isfinite(x) else x)
                for k, x in keep.items()
            }
            line = {**keep, "variants": variants}
            f.write(json.dumps(line, default=lambda x: x.item()) + "\n")

    counts = pd.Series([q["stratum"] for q in picked]).value_counts().to_dict()
    print(f"queries: {counts}")
    print(
        "population entries changed per score edit: "
        f"{pd.Series(n_pop).value_counts().to_dict()}"
    )
    n_jobs = sum(
        len(json.loads(line)["variants"]) for line in open(out / "cf_queries.jsonl")
    )
    print(f"variants in total: {n_jobs}; wrote {out}")


if __name__ == "__main__":
    main()
