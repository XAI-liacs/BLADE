"""Step 1: steering directions, their scales and the prompts of a campaign.

* **Directions.** For every feature of the campaign: the SAE decoder
  column (unit norm) and its scale ``a`` = median activation of its top
  examples in experiment 3. One random unit vector per ``random_layers``
  entry, with the median scale of all ``DIRECTIONS`` features at that
  layer (or of the campaign's, if none).
* **Residual norm.** With ``--device``, the mean per-token residual norm
  of the generated code at each steering layer (20 recorded queries,
  teacher forced), so a dose can be read as a fraction of it.
* **Prompts.** ``main``: ``n_init`` initial and ``n_mut`` mutation
  queries of experiment 1's ``main`` campaign whose parent worked;
  ``undefined``: every recorded query whose response used a name that is
  not defined (experiment 3, step 13 labels).

Writes ``data/processed/<tag>/directions.npz``, ``queries.jsonl`` and
``campaign.json``.
"""

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MI_ROOT = HERE.parents[1]
sys.path.insert(0, str(MI_ROOT))
sys.path.insert(0, str(HERE.parent))

from common.paths import data_dir
from exp_config import (
    CAMPAIGNS,
    CONCEPT_EXPERIMENT,
    CONCEPT_TAG,
    DIRECTIONS,
    EXPERIMENT,
    MODEL,
    SAE,
    SOURCE_TAGS,
)


def exp03_module(name: str):
    """Imports a module of experiment 3 (its ``exp_config`` shadows ours)."""
    root = MI_ROOT / CONCEPT_EXPERIMENT
    saved = sys.modules.pop("exp_config", None)
    sys.path[:0] = [str(root), str(root / "scripts")]
    try:
        path = root / "scripts" / f"{name}.py"
        if not path.exists():
            path = root / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"exp03_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        del sys.path[:2]
        sys.modules.pop("exp_config", None)
        if saved is not None:
            sys.modules["exp_config"] = saved
    return mod


def feature_scales(features: list[tuple[int, int]]) -> dict:
    """(layer, feature) -> median activation of its top examples."""
    step4 = exp03_module("04_concept_features")
    enc = step4.load_encodings(CONCEPT_TAG)
    out = {}
    for layer, f in features:
        val, row, _ = enc[(layer, "top")]
        v = val[f][row[f] >= 0]
        out[(layer, f)] = float(np.median(v))
    return out


def residual_norms(layers, n, device) -> dict:
    import torch
    from common.models import MODELS

    from iohblade.llm import HookedLLM

    records = exp03_module("records")
    recs = records.load_records(["main"]).sample(n, random_state=0)
    preset = MODELS[MODEL]
    llm = HookedLLM(
        preset.hf_id,
        capture=False,
        device_map=device,
        chat_template_kwargs=preset.chat_template_kwargs,
    )
    fwd = records.Forward(llm, list(layers))
    norms = {l: [] for l in layers}
    for r in recs.itertuples():
        ids, n_p, offsets = fwd.encode(r.session, r.response)
        pos = fwd.code_positions(r.response, n_p, offsets)
        states, _ = fwd.run(ids)
        for l in layers:
            h = states[l].float()[torch.as_tensor(pos, device=states[l].device)]
            norms[l].append(h.norm(dim=-1).cpu().numpy())
    return {l: float(np.mean(np.concatenate(v))) for l, v in norms.items()}


def select_queries(camp, rng) -> list[dict]:
    records = exp03_module("records")
    if camp.queries == "main":
        recs = records.load_records(["main"])
        init = recs[recs.operator == "init"]
        mut = recs[(recs.operator != "init") & (recs.parent_failed == False)]  # noqa
        pick = [
            init.iloc[rng.choice(len(init), camp.n_init, replace=False)],
            mut.iloc[rng.choice(len(mut), camp.n_mut, replace=False)],
        ]
        import pandas as pd

        recs = pd.concat(pick)
    elif camp.queries == "undefined":
        step13 = exp03_module("13_undefined_generality")
        recs = records.load_records(list(SOURCE_TAGS))
        keep = []
        for r in recs.itertuples():
            a, b = records.code_span(r.response)
            groups = {x["group"] for x in step13.references(r.response[a:b])}
            keep.append(bool(groups & {"func", "elsewhere", "never"}))
        recs = recs[np.array(keep)]
    else:
        raise ValueError(camp.queries)
    return [
        {
            "record_id": r.record_id,
            "source_tag": r.tag,
            "run": r.run,
            "operator": r.operator,
            "session": r.session,
            "child_failed": int(r.child_failed),
        }
        for r in recs.itertuples()
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="pilot", choices=list(CAMPAIGNS))
    ap.add_argument("--device", default=None, help="measure residual norms")
    ap.add_argument("--n-norm", type=int, default=20)
    args = ap.parse_args()
    camp = CAMPAIGNS[args.tag]
    rng = np.random.default_rng(camp.seed)
    out = data_dir(EXPERIMENT, "processed", camp.tag)
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from common.sae import TopKSAE

    feats = {n: DIRECTIONS[n] for n in camp.directions}
    scales = feature_scales([(l, f) for l, f, _ in DIRECTIONS.values()])
    layers = sorted({l for l, _, _ in feats.values()} | set(camp.random_layers))
    arrays, info = {}, {}
    for layer in layers:
        sae = TopKSAE.from_hub(SAE, layer, "cpu")
        for name, (l, f, concept) in feats.items():
            if l != layer:
                continue
            d = sae.W_dec[:, f].numpy().astype(np.float32)
            arrays[name] = d / np.linalg.norm(d)
            info[name] = {
                "layer": l,
                "feature": f,
                "concept": concept,
                "scale": scales[(l, f)],
                "decoder_norm": float(np.linalg.norm(d)),
            }
        if layer in camp.random_layers:
            at_layer = [scales[(l, f)] for l, f, _ in DIRECTIONS.values() if l == layer]
            r = rng.standard_normal(sae.W_dec.shape[0]).astype(np.float32)
            arrays[f"random{layer}"] = r / np.linalg.norm(r)
            info[f"random{layer}"] = {
                "layer": layer,
                "feature": None,
                "concept": None,
                "scale": float(np.median(at_layer)),
            }
        del sae
        torch.cuda.empty_cache()
    np.savez(out / "directions.npz", **arrays)
    norms = residual_norms(layers, args.n_norm, args.device) if args.device else {}
    for v in info.values():
        if v["layer"] in norms:
            v["residual_norm"] = norms[v["layer"]]
    queries = select_queries(camp, rng)
    with open(out / "queries.jsonl", "w") as fh:
        for q in queries:
            fh.write(json.dumps(q) + "\n")
    meta = {"campaign": dataclasses.asdict(camp), "directions": info}
    (out / "campaign.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(info, indent=1))
    print(f"{len(queries)} queries -> {out}")


if __name__ == "__main__":
    main()
