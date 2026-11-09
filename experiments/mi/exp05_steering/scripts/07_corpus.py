"""Step 5b.1: a labelled code corpus for contrastive levers.

Two parts, both written to ``data/raw/corpus/<part>-shard-<k>.jsonl`` (one
line per algorithm; rerun to resume):

* ``natural`` (CPU): every algorithm of experiment 1's ``main`` campaign
  and of step 5's ``llamea`` campaign, re-evaluated on the traced problem
  (algorithms that failed originally are kept, without behaviour).
* ``elicit`` (GPU): for each target of ``exp_config.ELICIT``,
  ``n_per_target`` initial queries of experiment 1 with one extra sentence
  ("The algorithm must use <description>."), plus as many plain initial
  queries (target ``none``). Unsteered; the sampling seed depends on the
  target and the index.

Every working algorithm is evaluated on experiment 1's budget-enforced
MA-BBOB with ``TracedMABBOB``, which adds ``behaviour``: exploration %,
distance to the best so far, intensification, stagnation and success rate
(means over the training instances). Concept labels are added in step
5b.2.

    uv run python 07_corpus.py --part natural --workers 48
    CUDA_VISIBLE_DEVICES=2 uv run python 07_corpus.py --part elicit \\
        --shard 0 --n-shards 2
"""

import argparse
import copy
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MI_ROOT = HERE.parents[1]
sys.path.insert(0, str(MI_ROOT))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import importlib

from common.models import MODELS
from common.paths import data_dir
from common.problems import TracedMABBOB
from common.runs import read_jsonl
from exp_config import CORPUS, ELICIT, ELICIT_TEMPLATE, EXPERIMENT, MODEL

from iohblade.solution import Solution
from iohblade.utils import NoCodeException

prep = importlib.import_module("01_prepare")
gen = importlib.import_module("02_generate")


def class_name(code: str) -> str:
    m = re.search(r"^class\s+(\w+)", code, re.M)
    return m.group(1) if m else "Algorithm"


def evaluate(problem, code: str, name: str) -> dict:
    t0 = time.time()
    try:
        s = problem(Solution(code=code, name=name))
        fit = float(s.fitness)
        err = s.error or ""
    except Exception as e:
        return {"fitness": None, "failed": True, "error": f"INFRA: {e!r}"[:300]}
    ok = np.isfinite(fit) and not err
    return {
        "fitness": fit if np.isfinite(fit) else None,
        "failed": bool(not ok),
        "error": err[:300],
        "behaviour": s.metadata.get("behaviour") if ok else None,
        "eval_s": time.time() - t0,
    }


def natural_items() -> list[dict]:
    items = []
    tag = CORPUS["natural"]["exp01_probing"]
    path = data_dir("exp01_probing", "processed", tag) / "code.jsonl"
    import pandas as pd

    meta = pd.read_csv(
        data_dir("exp01_probing", "processed", tag) / "meta.csv",
        dtype={"record_id": str},
    ).set_index("record_id")
    for r in read_jsonl(path):
        rid = r["record_id"]
        items.append(
            {
                "id": f"exp01/{rid}",
                "source": "exp01",
                "target": None,
                "operator": meta.loc[rid, "operator"],
                "code": r["code"] or "",
                "was_failed": bool(meta.loc[rid, "child_failed"]),
            }
        )
    tag = CORPUS["natural"]["exp05_steering"]
    for log in sorted(data_dir(EXPERIMENT, "raw", tag).glob("*/run-*/log.jsonl")):
        arm = log.parent.name.split("-")[2]
        for r in read_jsonl(log):
            md = r.get("metadata") or {}
            fit = r.get("fitness")
            failed = fit is None or not np.isfinite(float(fit)) or bool(r.get("error"))
            items.append(
                {
                    "id": f"llamea/{log.parent.name}/{r['id']}",
                    "source": "llamea",
                    "target": None,
                    "operator": md.get("operator"),
                    "steer": md.get("steer"),
                    "arm": arm,
                    "code": r.get("code") or "",
                    "was_failed": bool(failed),
                }
            )
    return items


def make_problem(src):
    p = TracedMABBOB(
        dims=src.dims,
        budget_factor=src.budget_factor,
        training_instances=CORPUS["trace_instances"],
        eval_timeout=src.eval_timeout,
        dependencies=src.eval_dependencies,
        name="MA_BBOB_traced",
    )
    p._ensure_env()
    return p


def done_ids(raw: Path, part: str) -> set:
    done = set()
    for path in raw.glob(f"{part}-shard-*.jsonl"):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not str(r.get("error", "")).startswith("INFRA"):
                done.add(r["id"])
    return done


def run_natural(args, src, raw):
    items = natural_items()
    done = done_ids(raw, "natural")
    todo = [it for it in items if it["id"] not in done][args.shard :: args.n_shards]
    print(f"natural: {len(items)} items, {len(todo)} to do", flush=True)
    problem = make_problem(src)
    lock = threading.Lock()
    out = open(raw / f"natural-shard-{args.shard}.jsonl", "a")

    def work(it):
        row = dict(it)
        if it["was_failed"] or not it["code"].strip():
            row.update(fitness=None, failed=True, error="failed originally")
        else:
            row.update(evaluate(problem, it["code"], class_name(it["code"])))
        with lock:
            out.write(json.dumps(row) + "\n")
            out.flush()

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(work, todo)):
            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t0) * 60
                print(f"natural: {i + 1}/{len(todo)}, {rate:.0f}/min", flush=True)
    out.close()


def run_elicit(args, src, raw):
    from iohblade.llm import HookedLLM

    records = prep.exp03_module("records")
    recs = records.load_records(["main"])
    init = recs[recs.operator == "init"].reset_index(drop=True)
    n = CORPUS["n_per_target"]
    tasks = [(t, i) for t in ["none", *ELICIT] for i in range(n)]
    done = done_ids(raw, "elicit")
    tasks = [x for x in tasks if f"elicit/{x[0]}/{x[1]}" not in done]
    tasks = tasks[args.shard :: args.n_shards]
    print(f"elicit shard {args.shard}: {len(tasks)} to do", flush=True)
    if not tasks:
        return
    preset = MODELS[MODEL]
    llm = HookedLLM(
        preset.hf_id,
        capture=False,
        device_map=args.device,
        temperature=src.temperature,
        top_p=src.top_p,
        max_new_tokens=src.max_new_tokens,
        chat_template_kwargs=preset.chat_template_kwargs,
    )
    problem = make_problem(src)
    lock = threading.Lock()
    out = open(raw / f"elicit-shard-{args.shard}.jsonl", "a")

    def finish(row, code, name):
        if code:
            row.update(evaluate(problem, code, name))
        with lock:
            out.write(json.dumps(row) + "\n")
            out.flush()

    pool = ThreadPoolExecutor(args.workers)
    t0 = time.time()
    for k, (target, i) in enumerate(tasks):
        session = copy.deepcopy(init.session.iloc[i % len(init)])
        if target != "none":
            session[-1]["content"] += ELICIT_TEMPLATE.format(ELICIT[target])
        seed = gen.sample_seed(f"elicit-{target}", i)
        np.random.seed(seed)
        row = {
            "id": f"elicit/{target}/{i}",
            "source": "elicit",
            "target": target,
            "operator": "init",
            "seed": seed,
        }
        code = name = None
        try:
            s = llm.sample_solution(session)
            code, name = s.code, s.name
            row.update(code=code, name=name)
        except NoCodeException:
            row.update(code="", fitness=None, failed=True, error="NoCodeException")
        except Exception as e:
            row.update(code="", fitness=None, failed=True, error=f"INFRA: {e!r}"[:300])
        llm.records.clear()
        pool.submit(finish, row, code, name)
        if (k + 1) % 10 == 0:
            rate = (k + 1) / (time.time() - t0) * 60
            print(f"elicit: {k + 1}/{len(tasks)}, {rate:.1f}/min", flush=True)
    pool.shutdown(wait=True)
    out.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--part", choices=["natural", "elicit"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    src = gen.load_source_config().PRESETS["main"]
    raw = data_dir(EXPERIMENT, "raw", CORPUS["tag"])
    raw.mkdir(parents=True, exist_ok=True)
    (run_natural if args.part == "natural" else run_elicit)(args, src, raw)
    print("done", flush=True)


if __name__ == "__main__":
    main()
