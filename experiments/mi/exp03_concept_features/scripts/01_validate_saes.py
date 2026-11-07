"""Step 1: do the base-model SAEs transfer to Qwen3-8B (post-trained)?

The Qwen-Scope SAEs were trained on Qwen3-8B-Base. For a sample of
recorded LLaMEA queries this script teacher-forces prompt + response and,
per SAE and layer, measures on the generated code tokens:

* ``fvu``: fraction of variance unexplained by the reconstruction;
* ``cos``: mean cosine between state and reconstruction;
* ``ce_clean`` / ``ce_splice`` / ``ce_ablate``: next-token loss (nats) of
  the code tokens with the block output untouched, replaced by its SAE
  reconstruction, or replaced by its mean over the sequence;
* ``recovered``: ``(ce_ablate - ce_splice) / (ce_ablate - ce_clean)``.

Position 0 (the attention sink, with a very large norm) is never replaced.
Run once with the post-trained model and once with ``--hf-id
Qwen/Qwen3-8B-Base`` for the reference the SAEs were trained on.

Writes ``results/<tag>/sae_validation.csv`` (one row per query, SAE and
layer) and prints the means.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp03_concept_features

from common.models import MODELS
from common.paths import results_dir
from common.sae import TopKSAE
from exp_config import EXPERIMENT, LAYERS, MODEL, SAES
from records import Forward, load_records

from iohblade.llm import HookedLLM


def code_ce(logits, ids, n_prompt, pos):
    """Mean next-token loss over the code positions ``pos``."""
    lp = torch.log_softmax(logits, -1)  # rows: positions n_prompt-1 ...
    rows = torch.as_tensor(pos - n_prompt)  # predicts token at pos
    target = ids[0, pos].to(lp.device)
    return float(-lp[rows, target].mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="qwen3-8b")
    ap.add_argument("--hf-id", default=None, help="default: the experiment model")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-tokens", type=int, default=3000)
    args = ap.parse_args()

    recs = load_records(["main"])
    recs = recs[recs.response.str.contains("```")]
    sample = (
        recs.sample(frac=1.0, random_state=0)
        .groupby("operator")
        .head(args.n // recs.operator.nunique() + 1)
        .head(args.n)
    )
    preset = MODELS[MODEL]
    llm = HookedLLM(
        args.hf_id or preset.hf_id,
        capture=False,
        device_map=args.device,
        chat_template_kwargs=preset.chat_template_kwargs,
    )
    fwd = Forward(llm, LAYERS)
    saes = {
        (name, layer): TopKSAE.from_hub(repo, layer, args.device)
        for name, repo in SAES.items()
        for layer in LAYERS
    }
    rows = []
    t0 = time.time()
    for i, (_, r) in enumerate(sample.iterrows()):
        ids, n_p, offsets = fwd.encode(r.session, r.response)
        if ids.shape[1] > args.max_tokens:
            continue
        pos = fwd.code_positions(r.response, n_p, offsets)
        if len(pos) < 10:
            continue
        states, logits = fwd.run(ids, logits_from=n_p - 1)
        clean = code_ce(logits, ids, n_p, pos)
        for (name, layer), sae in saes.items():
            h = states[layer].float()
            rec = sae.reconstruct(h)
            x, xr = h[pos], rec[pos]
            fvu = float(((x - xr) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())
            cos = float(torch.nn.functional.cosine_similarity(x, xr, dim=-1).mean())
            mean = h[1:].mean(0)

            def splice(hs, sae=sae):
                out = hs.clone()
                out[0, 1:] = sae.reconstruct(hs[0, 1:].float()).to(hs.dtype)
                return out

            def ablate(hs, mean=mean):
                out = hs.clone()
                out[0, 1:] = mean.to(hs.dtype)
                return out

            _, lg = fwd.run(ids, patch={layer: splice}, logits_from=n_p - 1)
            ce_splice = code_ce(lg, ids, n_p, pos)
            _, lg = fwd.run(ids, patch={layer: ablate}, logits_from=n_p - 1)
            ce_ablate = code_ce(lg, ids, n_p, pos)
            rows.append(
                {
                    "record_id": r.record_id,
                    "operator": r.operator,
                    "sae": name,
                    "layer": layer,
                    "n_code_tokens": len(pos),
                    "fvu": fvu,
                    "cos": cos,
                    "ce_clean": clean,
                    "ce_splice": ce_splice,
                    "ce_ablate": ce_ablate,
                    "recovered": (ce_ablate - ce_splice) / (ce_ablate - clean),
                }
            )
        print(f"  {i + 1}/{len(sample)} ({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    out = results_dir(EXPERIMENT, args.tag)
    df.to_csv(out / "sae_validation.csv", index=False)
    summary = df.groupby(["sae", "layer"])[
        ["fvu", "cos", "ce_clean", "ce_splice", "ce_ablate", "recovered"]
    ].mean()
    summary["delta_ce"] = summary.ce_splice - summary.ce_clean
    print(summary.round(4).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
