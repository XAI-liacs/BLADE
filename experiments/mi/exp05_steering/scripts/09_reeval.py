"""Step 5b.4: re-evaluate families that failed on missing packages.

Quasi-random initial designs (``scipy.stats.qmc``) and surrogate models
(``sklearn``) failed in experiment 1's sandbox because those packages are
not installed. This script re-evaluates the existing code (no new
generation) with ``exp_config.EXTRA_EVAL_DEPENDENCIES`` added, on the same
traced problem as the original evaluation:

* ``elicit``: the corpus algorithms elicited for ``--targets`` (and the
  plain-prompt ``none`` ones as a reference);
* ``contrast-c``: the steered children of the ``--targets`` directions of
  the ``contrast-c`` screen and their unsteered siblings (``none``).

Writes ``data/raw/reeval/<source>.jsonl`` (resumable) and
``results/reeval/summary.md``: failure rate before/after, AOCC, the rate
of the target concept, and the paired change against the unsteered
children for the steered ones.

    uv run python 09_reeval.py --workers 48
"""

import argparse
import importlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MI_ROOT = HERE.parents[1]
sys.path.insert(0, str(MI_ROOT))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common.paths import data_dir, results_dir
from common.problems import TracedMABBOB
from common.runs import read_jsonl
from exp_config import CORPUS, EXPERIMENT, EXTRA_EVAL_DEPENDENCIES

prep = importlib.import_module("01_prepare")
gen = importlib.import_module("02_generate")
corpus = importlib.import_module("07_corpus")
labels_mod = prep.exp03_module("03_concept_labels")

TARGETS = ["quasi_random_init", "surrogate", "archive"]


def elicit_items(targets) -> list[dict]:
    raw = data_dir(EXPERIMENT, "raw", CORPUS["tag"])
    items = []
    for path in sorted(raw.glob("elicit-shard-*.jsonl")):
        for r in read_jsonl(path):
            if r["target"] in (*targets, "none") and r.get("code"):
                items.append(
                    {
                        "id": r["id"],
                        "group": r["target"],
                        "concept": r["target"] if r["target"] != "none" else None,
                        "code": r["code"],
                        "failed_before": bool(r.get("failed")),
                        "fitness_before": r.get("fitness"),
                    }
                )
    return items


def contrast_items(targets) -> list[dict]:
    raw = data_dir(EXPERIMENT, "raw", "contrast-c")
    items = []
    for path in sorted(raw.glob("shard-*.jsonl")):
        for r in read_jsonl(path):
            d = r.get("direction") or ""
            concept = d[2:].split("@")[0] if d.startswith("c_") else None
            if r["condition"] != "none" and concept not in targets:
                continue
            items.append(
                {
                    "id": f"{r['condition']}/{r['record_id']}/{r['k']}",
                    "group": r["condition"],
                    "record_id": r["record_id"],
                    "k": r["k"],
                    "concept": concept,
                    "code": r.get("code") or "",
                    "failed_before": bool(r.get("failed")),
                    "fitness_before": r.get("fitness"),
                }
            )
    return items


def make_problem(src, instances):
    p = TracedMABBOB(
        dims=src.dims,
        budget_factor=src.budget_factor,
        training_instances=instances,
        eval_timeout=src.eval_timeout,
        dependencies=src.eval_dependencies + EXTRA_EVAL_DEPENDENCIES,
        name="MA_BBOB_traced",
    )
    p.shared_namespace = True
    p._ensure_env()
    return p


def run(items, problem, path: Path, workers: int):
    done = set()
    if path.exists():
        done = {r["id"] for r in read_jsonl(path)}
    todo = [it for it in items if it["id"] not in done]
    print(f"{path.name}: {len(items)} items, {len(todo)} to do", flush=True)
    lock = threading.Lock()
    out = open(path, "a")

    def work(it):
        row = dict(it)
        if it["code"].strip():
            row.update(
                corpus.evaluate(problem, it["code"], corpus.class_name(it["code"]))
            )
        else:
            row.update(fitness=None, failed=True, error="no code")
        with lock:
            out.write(json.dumps(row) + "\n")
            out.flush()

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(work, todo))
    out.close()


def concept_present(code: str, concept: str) -> bool:
    return bool(code) and bool(labels_mod.labels(code).get(concept))


def summarise(df: pd.DataFrame, concept_of) -> pd.DataFrame:
    rows = []
    for g, x in df.groupby("group", sort=False):
        c = concept_of(g, x)
        ok = ~x.failed
        rows.append(
            {
                "group": g,
                "n": len(x),
                "failed_before": x.failed_before.mean(),
                "failed_after": x.failed.mean(),
                "aocc": np.where(ok, x.fitness.fillna(0), 0).mean(),
                "aocc_working": x.fitness[ok].mean(),
                "concept": c or "",
                "concept_rate": (
                    np.mean([concept_present(k, c) for k in x.code]) if c else np.nan
                ),
                "concept_working": (
                    np.mean([concept_present(k, c) for k in x.code[ok]])
                    if c and ok.any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_failures(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df.group == "none"].set_index(["record_id", "k"])
    rows = []
    for g, x in df[df.group != "none"].groupby("group"):
        x = x.set_index(["record_id", "k"])
        b = base.reindex(x.index)
        rows.append(
            {
                "group": g,
                "d_failed_before": (
                    x.failed_before.astype(float) - b.failed_before.astype(float)
                ).mean(),
                "d_failed_after": (
                    x.failed.astype(float) - b.failed.astype(float)
                ).mean(),
                "d_aocc_after": (
                    np.where(x.failed, 0, x.fitness.fillna(0))
                    - np.where(b.failed, 0, b.fitness.fillna(0))
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def md(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    src = gen.load_source_config().PRESETS["main"]
    raw = data_dir(EXPERIMENT, "raw", "reeval")
    raw.mkdir(parents=True, exist_ok=True)
    out = results_dir(EXPERIMENT, "reeval")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    run(
        elicit_items(args.targets),
        make_problem(src, CORPUS["trace_instances"]),
        raw / "elicit.jsonl",
        args.workers,
    )
    run(
        contrast_items(args.targets),
        make_problem(src, src.training_instances),
        raw / "contrast-c.jsonl",
        args.workers,
    )
    print(f"evaluated in {(time.time() - t0) / 60:.1f} min", flush=True)

    el = pd.DataFrame(read_jsonl(raw / "elicit.jsonl"))
    el["failed"] = el.failed.astype(bool)
    se = summarise(el, lambda g, x: x.concept.iloc[0])
    cc = pd.DataFrame(read_jsonl(raw / "contrast-c.jsonl"))
    cc["failed"] = cc.failed.astype(bool)
    sc = summarise(cc, lambda g, x: x.concept.dropna().iloc[0] if g != "none" else None)
    sc = sc.merge(paired_failures(cc), on="group", how="left")
    errs = (
        el[el.failed & el.group.ne("none")]
        .assign(err=el.error.fillna("").str.strip().str.split("\n").str[-1].str[:90])
        .groupby(["group", "err"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values("n", ascending=False)
        .groupby("group")
        .head(4)
    )
    lines = [
        "# Re-evaluation with scipy and scikit-learn",
        "",
        "Existing code re-evaluated with "
        + ", ".join(f"`{d}`" for d in EXTRA_EVAL_DEPENDENCIES)
        + " added to the sandbox (same traced problem as the original"
        " evaluation). `aocc` counts failures as 0; `concept_working` is the"
        " rate of the target concept among the working algorithms.",
        "",
        "## Elicited algorithms (corpus, 24 per target)",
        "",
        md(se),
        "",
        "Most frequent remaining errors:",
        "",
        md(errs),
        "",
        "## Steered children (`contrast-c`, 16 per condition)",
        "",
        "`d_*`: paired change against the unsteered child of the same prompt.",
        "",
        md(sc),
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines))
    se.to_csv(out / "elicit.csv", index=False)
    sc.to_csv(out / "contrast-c.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
