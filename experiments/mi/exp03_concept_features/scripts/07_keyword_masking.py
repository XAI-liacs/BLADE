"""Step 7: keyword detector or concept feature? Re-score with the keyword masked.

The concept labels of step 3 are regex matches, so a feature that fires
only on the matched identifier (``mutant``, ``restart``) scores a perfect
AUROC without representing anything about the algorithm. Here each
selected feature of ``concept_features.csv`` is re-scored with its
per-solution max taken over the code tokens *except*

* ``keyword``: the tokens overlapping a match of the concept's pattern
  (on the raw code, so matches in comments and strings count too),
  widened to the whole identifier;
* ``line``: every line that contains a match (also hides the tokens next
  to the keyword, e.g. ``mutant = x[a] + F * (...)``);
* ``all_keywords``: the keyword tokens of *every* concept (a feature
  that survives ``keyword`` masking may still read a co-occurring
  concept's keyword, e.g. ``mutant`` for crossover);
* ``random``: as many random code tokens as ``keyword`` masks (control:
  the drop from simply seeing fewer tokens).

A feature that keeps its AUROC under ``keyword``/``line`` masking detects
the concept from the code around the keyword; one that falls to ~0.5 is
a keyword detector.

Usage: run the shards (GPU), then ``--analyze`` (CPU). Shards write
``data/processed/<tag>/masked-<k>.npz``; the analysis writes
``results/<tag>/keyword_masking.csv`` and ``keyword_masking.md``.
"""

import argparse
import importlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp03_concept_features
sys.path.insert(0, str(HERE))

from common.paths import data_dir, results_dir
from exp_config import EXPERIMENT, MODEL, SAES, SOURCE_TAGS
from records import code_span, load_records

concept_labels = importlib.import_module("03_concept_labels")

MASKS = ("none", "keyword", "line", "random", "all_keywords")


def keyword_spans(code: str, pattern: str) -> list[tuple[int, int]]:
    """Character spans of the pattern's matches, widened to whole words."""
    spans = []
    for m in re.finditer(pattern, code):
        a, b = m.span()
        while a > 0 and (code[a - 1].isalnum() or code[a - 1] == "_"):
            a -= 1
        while b < len(code) and (code[b].isalnum() or code[b] == "_"):
            b += 1
        spans.append((a, max(b, a + 1)))
    return spans


def line_spans(code: str, spans) -> list[tuple[int, int]]:
    out = []
    for a, b in spans:
        s = code.rfind("\n", 0, a) + 1
        e = code.find("\n", b)
        out.append((s, len(code) if e < 0 else e))
    return out


def token_mask(tok_spans: np.ndarray, spans) -> np.ndarray:
    """Which tokens (char spans ``tok_spans``) overlap any of ``spans``."""
    hit = np.zeros(len(tok_spans), bool)
    for a, b in spans:
        hit |= (tok_spans[:, 0] < b) & (tok_spans[:, 1] > a)
    return hit


def selected_features(tag: str, ranks: int) -> pd.DataFrame:
    feats = pd.read_csv(results_dir(EXPERIMENT, tag) / "concept_features.csv")
    return feats[feats["rank"] < ranks].reset_index(drop=True)


def encode_shard(args):
    import torch
    from common.models import MODELS
    from common.sae import TopKSAE
    from records import Forward

    from iohblade.llm import HookedLLM

    feats = selected_features(args.tag, args.ranks)
    layers = sorted(feats.layer.unique())
    concepts = sorted(feats.concept.unique())
    patterns = {c: concept_labels.CONCEPTS[c] for c in concepts}
    # per layer: the distinct feature ids and, per selected row, its column
    fid = {l: np.unique(feats.feature[feats.layer == l]) for l in layers}
    col = np.array(
        [np.searchsorted(fid[r.layer], r.feature) for r in feats.itertuples()]
    )

    recs = load_records(args.source_tags)
    recs = recs.sort_values("record_id").reset_index(drop=True)
    recs = recs.iloc[args.shard :: args.n_shards]
    if args.limit:
        recs = recs.head(args.limit)
    print(f"shard {args.shard}: {len(recs)} records, {len(feats)} features", flush=True)

    preset = MODELS[MODEL]
    llm = HookedLLM(
        preset.hf_id,
        capture=False,
        device_map=args.device,
        chat_template_kwargs=preset.chat_template_kwargs,
    )
    fwd = Forward(llm, layers)
    saes = {l: TopKSAE.from_hub(SAES[args.sae], l, args.device) for l in layers}
    fid_t = {l: torch.as_tensor(fid[l], device=args.device) for l in layers}

    kept, out, n_masked = [], [], []
    t0 = time.time()
    for k, (_, r) in enumerate(recs.iterrows()):
        ids, n_p, offsets = fwd.encode(r.session, r.response)
        pos = fwd.code_positions(r.response, n_p, offsets)
        if ids.shape[1] > args.max_tokens or len(pos) == 0:
            continue
        a, b = code_span(r.response)
        code = r.response[a:b]
        tok_spans = offsets[pos - n_p] - a
        rng = np.random.default_rng([args.shard, k])
        masks = {}  # concept -> (n_masks, n_tok) keep-masks
        nm = []
        m_all = token_mask(
            tok_spans,
            [
                s
                for p in concept_labels.CONCEPTS.values()
                for s in keyword_spans(code, p)
            ],
        )
        for c in concepts:
            kw = keyword_spans(code, patterns[c])
            m_kw = token_mask(tok_spans, kw)
            m_ln = token_mask(tok_spans, line_spans(code, kw))
            m_rd = np.zeros(len(pos), bool)
            m_rd[rng.choice(len(pos), min(m_kw.sum(), len(pos)), replace=False)] = True
            masks[c] = ~np.stack([np.zeros(len(pos), bool), m_kw, m_ln, m_rd, m_all])
            nm.append((m_kw.sum(), m_ln.sum()))
        states, _ = fwd.run(ids)
        res = np.zeros((len(feats), len(MASKS)), np.float32)
        acts = {}
        for l in layers:
            h = states[l].float()[torch.as_tensor(pos, device=args.device)]
            vals, idx = saes[l].encode(h)
            hit = idx.unsqueeze(-1) == fid_t[l]  # (tok, k, n_feat)
            acts[l] = (vals.unsqueeze(-1) * hit).sum(1).cpu().numpy()  # (tok, n_feat)
        for i, row in enumerate(feats.itertuples()):
            A = acts[row.layer][:, col[i]]
            keep = masks[row.concept]
            res[i] = np.where(keep, A[None], 0).max(1)
        kept.append(r.record_id)
        out.append(res)
        n_masked.append(nm)
        if (k + 1) % 100 == 0:
            rate = (k + 1) / (time.time() - t0) * 60
            print(
                f"shard {args.shard}: {k + 1}/{len(recs)} ({rate:.0f}/min)", flush=True
            )

    path = data_dir(EXPERIMENT, "processed", args.tag) / f"masked-{args.shard}.npz"
    np.savez_compressed(
        path,
        record_id=np.array(kept),
        act=np.stack(out).astype(np.float16),  # (rec, feat, mask)
        n_masked=np.array(n_masked, np.int32),  # (rec, concept, kw/line)
        concepts=np.array(concepts),
        feat_layer=feats.layer.to_numpy(),
        feat_concept=feats.concept.to_numpy().astype(str),
        feat_id=feats.feature.to_numpy(),
    )
    print(f"shard {args.shard}: wrote {path} in {time.time() - t0:.0f} s")


def analyze(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    files = sorted(data_dir(EXPERIMENT, "processed", args.tag).glob("masked-*.npz"))
    zs = [np.load(f, allow_pickle=True) for f in files]  # own files
    ids = np.concatenate([z["record_id"] for z in zs])
    act = np.concatenate([z["act"] for z in zs]).astype(np.float32)
    feats = pd.DataFrame(
        {
            "layer": zs[0]["feat_layer"],
            "concept": zs[0]["feat_concept"],
            "feature": zs[0]["feat_id"],
        }
    )
    sel = selected_features(args.tag, args.ranks)
    feats = feats.merge(sel, on=["layer", "concept", "feature"], how="left")
    labels = pd.read_csv(data_dir(EXPERIMENT, "processed") / "labels.csv")
    labels = labels.set_index("record_id").loc[ids]
    recs = load_records(sorted(labels.tag.unique())).set_index("record_id").loc[ids]
    # same discovery/validation split as step 4
    run = recs.run.to_numpy()
    rng = np.random.default_rng(0)
    runs = np.unique(run)
    disc_runs = set(rng.permutation(runs)[: len(runs) // 2])
    val = np.array([r not in disc_runs for r in run])
    print(f"{len(ids)} solutions ({val.sum()} validation), {len(feats)} features")

    # baselines without the SAE: code length, and the other concepts' labels
    # (+ length) in a logistic regression fitted on the discovery half
    n_tok = {}
    for z in map(
        np.load, data_dir(EXPERIMENT, "processed", args.tag).glob("shard-*.npz")
    ):
        n_tok.update(zip(z["record_id"], z["n_code_tokens"]))
    length = np.log(np.array([n_tok[r] for r in ids], float))
    concepts = list(concept_labels.CONCEPTS)
    base = {}
    for c in feats.concept.unique():
        y = labels[c].to_numpy()
        X = np.column_stack([labels[[o for o in concepts if o != c]], length])
        lr = LogisticRegression(max_iter=2000).fit(X[~val], y[~val])
        base[c] = {
            "auroc_length": roc_auc_score(y[val], length[val]),
            "auroc_other_labels": roc_auc_score(y[val], lr.decision_function(X[val])),
        }

    rows = []
    for i, f in feats.iterrows():
        y = labels[f.concept].to_numpy()[val]
        row = f.to_dict() | base[f.concept]
        for j, m in enumerate(MASKS):
            x = act[val, i, j]
            row[f"auroc_{m}"] = roc_auc_score(y, x)
            row[f"active_pos_{m}"] = float((x[y == 1] > 0).mean())
            row[f"active_neg_{m}"] = float((x[y == 0] > 0).mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    tok = results_dir(EXPERIMENT, args.tag) / "feature_tokens.csv"
    if tok.exists():
        t = pd.read_csv(tok)[["layer", "concept", "feature", "top_tokens"]]
        df = df.merge(t, on=["layer", "concept", "feature"], how="left")
    out = results_dir(EXPERIMENT, args.tag)
    df.to_csv(out / "keyword_masking.csv", index=False)

    # most robust of the top-n per concept and layer once no keyword is visible
    best = df.loc[df.groupby(["concept", "layer"]).auroc_all_keywords.idxmax()]
    cols = [
        "concept",
        "layer",
        "rank",
        "feature",
        "auroc_none",
        "auroc_random",
        "auroc_keyword",
        "auroc_line",
        "auroc_all_keywords",
        "auroc_other_labels",
        "auroc_length",
    ]
    md = [
        "# Keyword masking",
        "",
        "Validation AUROC of each concept feature with the concept's keyword",
        "tokens (`keyword`), their whole lines (`line`), as many random code",
        "tokens (`random`) or the keyword tokens of every concept",
        "(`all_keywords`) excluded from the per-solution max. Baselines",
        "without the SAE: the other concepts' labels + log code length",
        "(`other_labels`, logistic regression on the discovery half) and the",
        "log code length alone (`length`).",
        "",
        "## Rank-0 features (chosen on unmasked AUROC)",
        "",
        to_markdown(df[df["rank"] == 0][cols]),
        "",
        f"## Most robust of the top-{args.ranks} per concept and layer "
        "(max `all_keywords` AUROC)",
        "",
        to_markdown(best[cols]),
        "",
    ]
    (out / "keyword_masking.md").write_text("\n".join(md))
    summary = df[df["rank"] == 0].groupby("layer")[[f"auroc_{m}" for m in MASKS]]
    print(summary.median().round(3).to_string())
    print(best[cols].round(3).to_string(index=False))
    print(f"wrote {out}")


def to_markdown(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="k100")
    ap.add_argument("--sae", default="k100", choices=list(SAES))
    ap.add_argument("--ranks", type=int, default=5)
    ap.add_argument("--source-tags", nargs="*", default=list(SOURCE_TAGS))
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    analyze(args) if args.analyze else encode_shard(args)


if __name__ == "__main__":
    main()
