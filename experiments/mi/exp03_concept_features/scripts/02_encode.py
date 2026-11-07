"""Step 2: SAE features of every recorded algorithm.

Teacher-forces each recorded query (prompt + response) and encodes the
residual stream of the generated code tokens with one SAE per layer. Per
solution and feature it keeps

* ``max``: the largest activation over the code tokens;
* ``mean``: the sum over code tokens divided by their number;
* ``count``: on how many code tokens the feature is active;

as sparse matrices (solutions x features), and the top-k codes of the last
prompt token (what the model represents just before it writes). For each
feature it also keeps the ``--n-top`` largest per-solution activations with
their record and token position, for interpretation.

Writes ``data/processed/<tag>/shard-<k>.npz`` per shard.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp03_concept_features

from common.models import MODELS
from common.paths import data_dir
from common.sae import TopKSAE
from exp_config import EXPERIMENT, LAYERS, MODEL, SAES, SOURCE_TAGS
from records import Forward, load_records

from iohblade.llm import HookedLLM


class Sparse:
    """Row-by-row builder of a CSR matrix."""

    def __init__(self):
        self.indptr, self.indices, self.data = [0], [], []

    def add(self, idx, val):
        self.indices.append(idx.astype(np.int32))
        self.data.append(val)
        self.indptr.append(self.indptr[-1] + len(idx))

    def arrays(self, prefix, dtype):
        return {
            f"{prefix}_indptr": np.array(self.indptr, dtype=np.int64),
            f"{prefix}_indices": np.concatenate(self.indices or [np.zeros(0, int)]),
            f"{prefix}_data": np.concatenate(self.data or [np.zeros(0)]).astype(dtype),
        }


class TopExamples:
    """The ``n`` largest per-solution activations of every feature."""

    def __init__(self, d_sae, n):
        self.val = np.zeros((d_sae, n), np.float32)
        self.row = np.full((d_sae, n), -1, np.int32)
        self.tok = np.full((d_sae, n), -1, np.int32)

    def update(self, row, feats, vals, toks):
        slot = self.val[feats].argmin(1)
        better = vals > self.val[feats, slot]
        f, s = feats[better], slot[better]
        self.val[f, s] = vals[better]
        self.row[f, s] = row
        self.tok[f, s] = toks[better]


def aggregate(vals, idx, pos, d_sae):
    """Per-feature max, sum, count and argmax token over the code tokens."""
    v = vals.flatten()
    i = idx.flatten()
    t = torch.as_tensor(pos, device=v.device).repeat_interleave(vals.shape[1])
    mx = torch.zeros(d_sae, device=v.device).scatter_reduce_(0, i, v, "amax")
    sm = torch.zeros(d_sae, device=v.device).index_add_(0, i, v)
    ct = torch.zeros(d_sae, device=v.device).index_add_(0, i, (v > 0).float())
    feats = torch.nonzero(mx > 0)[:, 0]
    # token of the max: last write wins among equal features, so write in
    # ascending order of value
    order = v.argsort()
    best = torch.full((d_sae,), -1, device=v.device, dtype=torch.long)
    best[i[order]] = t[order]
    return (
        feats.cpu().numpy(),
        mx[feats].cpu().numpy(),
        sm[feats].cpu().numpy(),
        ct[feats].cpu().numpy(),
        best[feats].cpu().numpy(),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="k50")
    ap.add_argument("--sae", default="k50", choices=list(SAES))
    ap.add_argument("--layers", type=int, nargs="*", default=list(LAYERS))
    ap.add_argument("--source-tags", nargs="*", default=list(SOURCE_TAGS))
    ap.add_argument("--n-top", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    recs = load_records(args.source_tags)
    recs = recs.sort_values("record_id").reset_index(drop=True)
    recs = recs.iloc[args.shard :: args.n_shards]
    if args.limit:
        recs = recs.head(args.limit)
    print(f"shard {args.shard}: {len(recs)} records", flush=True)

    preset = MODELS[MODEL]
    llm = HookedLLM(
        preset.hf_id,
        capture=False,
        device_map=args.device,
        chat_template_kwargs=preset.chat_template_kwargs,
    )
    fwd = Forward(llm, args.layers)
    saes = {l: TopKSAE.from_hub(SAES[args.sae], l, args.device) for l in args.layers}
    d_sae = next(iter(saes.values())).d_sae
    stats = {l: {m: Sparse() for m in ("max", "mean", "count")} for l in args.layers}
    last = {l: ([], []) for l in args.layers}
    tops = {l: TopExamples(d_sae, args.n_top) for l in args.layers}
    kept, n_code = [], []
    t0 = time.time()
    for k, (_, r) in enumerate(recs.iterrows()):
        ids, n_p, offsets = fwd.encode(r.session, r.response)
        pos = fwd.code_positions(r.response, n_p, offsets)
        if ids.shape[1] > args.max_tokens or len(pos) == 0:
            continue
        states, _ = fwd.run(ids)
        row = len(kept)
        for layer, sae in saes.items():
            h = states[layer].float()
            vals, idx = sae.encode(h[torch.as_tensor(pos, device=h.device)])
            f, mx, sm, ct, best = aggregate(vals, idx, pos, d_sae)
            st = stats[layer]
            st["max"].add(f, mx)
            st["mean"].add(f, sm / len(pos))
            st["count"].add(f, ct)
            tops[layer].update(row, f, mx, best)
            lv, li = sae.encode(h[n_p - 1])
            last[layer][0].append(li.cpu().numpy().astype(np.int32))
            last[layer][1].append(lv.cpu().numpy().astype(np.float16))
        kept.append(r.record_id)
        n_code.append(len(pos))
        if (k + 1) % 100 == 0:
            rate = (k + 1) / (time.time() - t0) * 60
            print(
                f"shard {args.shard}: {k + 1}/{len(recs)} ({rate:.0f}/min)", flush=True
            )

    arrays = {
        "record_id": np.array(kept),
        "n_code_tokens": np.array(n_code),
        "layers": np.array(args.layers),
        "d_sae": np.array(d_sae),
    }
    for layer in args.layers:
        for m, dtype in (
            ("max", np.float16),
            ("mean", np.float16),
            ("count", np.int16),
        ):
            arrays |= stats[layer][m].arrays(f"L{layer}_{m}", dtype)
        arrays[f"L{layer}_last_idx"] = np.stack(last[layer][0])
        arrays[f"L{layer}_last_val"] = np.stack(last[layer][1])
        arrays[f"L{layer}_top_val"] = tops[layer].val
        arrays[f"L{layer}_top_row"] = tops[layer].row
        arrays[f"L{layer}_top_tok"] = tops[layer].tok
    out = data_dir(EXPERIMENT, "processed", args.tag)
    np.savez_compressed(out / f"shard-{args.shard}.npz", **arrays)
    print(f"shard {args.shard}: wrote {len(kept)} records in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
