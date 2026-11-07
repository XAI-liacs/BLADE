"""Step 5: top-activating contexts of the selected concept features.

For the features in ``concept_features.csv`` (``--ranks`` per concept and
layer) this renders the solutions where each feature is strongest, with
the max token in «», and the most common max tokens over its ``n_top``
examples (a feature whose max token is always the same identifier is a
keyword detector; one with varied tokens is closer to a concept).

Writes ``results/<tag>/feature_contexts.md`` and
``feature_tokens.csv`` (one row per feature: top tokens and their share).
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp03_concept_features

from common.models import MODELS
from common.paths import data_dir, results_dir
from exp_config import EXPERIMENT, MODEL
from records import load_records


def load_tops(tag: str):
    """Top examples per layer, merged over shards (record ids, not rows)."""
    files = sorted(data_dir(EXPERIMENT, "processed", tag).glob("shard-*.npz"))
    zs = [np.load(f) for f in files]
    ids = np.concatenate([z["record_id"] for z in zs])
    offsets = np.cumsum([0] + [len(z["record_id"]) for z in zs])
    tops = {}
    for l in zs[0]["layers"]:
        val = np.concatenate([z[f"L{l}_top_val"] for z in zs], 1)
        row = np.concatenate(
            [
                np.where(z[f"L{l}_top_row"] >= 0, z[f"L{l}_top_row"] + o, -1)
                for z, o in zip(zs, offsets)
            ],
            1,
        )
        tok = np.concatenate([z[f"L{l}_top_tok"] for z in zs], 1)
        order = np.argsort(-val, 1)
        tops[int(l)] = tuple(np.take_along_axis(a, order, 1) for a in (val, row, tok))
    return ids, tops


class Contexts:
    def __init__(self, recs):
        from transformers import AutoTokenizer

        preset = MODELS[MODEL]
        self.tok = AutoTokenizer.from_pretrained(preset.hf_id)
        self.kw = preset.chat_template_kwargs
        self.recs = recs.set_index("record_id")
        self.cache = {}

    def _tokens(self, record_id):
        if record_id not in self.cache:
            r = self.recs.loc[record_id]
            enc = self.tok.apply_chat_template(
                r.session,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **self.kw,
            )
            resp = self.tok(r.response, add_special_tokens=False)["input_ids"]
            self.cache[record_id] = (len(enc["input_ids"]), resp)
        return self.cache[record_id]

    def token(self, record_id, pos) -> str:
        n_p, toks = self._tokens(record_id)
        return self.tok.decode(toks[pos - n_p : pos - n_p + 1])

    def __call__(self, record_id, pos, width=14):
        n_p, toks = self._tokens(record_id)
        i = pos - n_p
        piece = lambda a, b: self.tok.decode(toks[max(a, 0) : b])
        text = piece(i - width, i) + "«" + piece(i, i + 1) + "»" + piece(i + 1, i + 5)
        return " ".join(text.split()).replace("`", "'")[:180]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="k100")
    ap.add_argument("--ranks", type=int, default=3)
    ap.add_argument("--n-contexts", type=int, default=5)
    args = ap.parse_args()

    out = results_dir(EXPERIMENT, args.tag)
    feats = pd.read_csv(out / "concept_features.csv")
    feats = feats[feats["rank"] < args.ranks]
    ids, tops = load_tops(args.tag)
    labels = pd.read_csv(data_dir(EXPERIMENT, "processed") / "labels.csv")
    ctx = Contexts(load_records(sorted(labels.tag.unique())))

    md = ["# Top features per concept", ""]
    tok_rows = []
    for (layer, concept), block in feats.groupby(["layer", "concept"], sort=False):
        md += [f"## L{layer} · {concept}", ""]
        val, row, tok = tops[layer]
        for _, f in block.iterrows():
            fid = int(f.feature)
            ex = [
                (v, ids[r], int(t))
                for v, r, t in zip(val[fid], row[fid], tok[fid])
                if r >= 0
            ]
            counts = Counter(ctx.token(rid, t).strip() for _, rid, t in ex)
            common = counts.most_common(3)
            share = common[0][1] / max(len(ex), 1) if common else np.nan
            tok_rows.append(
                {
                    "layer": layer,
                    "concept": concept,
                    "rank": int(f["rank"]),
                    "feature": fid,
                    "auroc_val": f.auroc_val,
                    "top_tokens": " | ".join(f"{t!r}:{n}" for t, n in common),
                    "top_token_share": share,
                }
            )
            md.append(
                f"- **f{fid}** AUROC {f.auroc_val:.3f} (val), active "
                f"{f.p_active_pos:.2f} / {f.p_active_neg:.2f}; max tokens "
                + ", ".join(f"`{t!r}` ×{n}" for t, n in common)
            )
            for v, rid, t in ex[: args.n_contexts]:
                md.append(f"  - {v:.1f}: `{ctx(rid, t)}`")
        md.append("")
    (out / "feature_contexts.md").write_text("\n".join(md))
    pd.DataFrame(tok_rows).to_csv(out / "feature_tokens.csv", index=False)
    print(pd.DataFrame(tok_rows).query("rank == 0").to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
