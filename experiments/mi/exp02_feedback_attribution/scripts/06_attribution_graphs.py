"""Step 6: attribution graphs for error-driven and score-driven edits.

Runs in a separate environment with ``circuit-tracer`` (it pins
``transformers<=4.57.3``; the project uses 5.x), see the README.

For recorded mutation queries from ``cf_queries.jsonl`` whose recorded
child worked, the target is the first token where the child's code departs
from the parent's code (the *edit token*): in the hunk that replaces the
line named in the error message if there is one, else in the first changed
hunk. The control target is the first token of a line copied unchanged from
the parent, the nearest one before the edit (the *copy token*). For each
target, the context is the chat prompt plus the response up to that token.

``circuit-tracer`` builds the attribution graph of the target logit
through the model's per-layer transcoders (``ModelPreset.transcoders``, e.g.
``mwhanna/qwen3-8b-transcoders``; the model is the source campaign's, from
``cf_meta.json``). From
the graph's node influence on the target logit, every node (feature,
error, token embedding) is assigned to its token position, and positions
to prompt spans (``spans.SPAN_NAMES``), the chat template or the response
prefix. Reported per target: influence share per span (all nodes, and
token embeddings only), the graph's replacement and completeness scores,
and the top feature nodes on the feedback span.

Writes ``data/processed/<tag>/attribution_graphs-<shard>.jsonl``.
"""

import argparse
import difflib
import json
import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
MI_ROOT = HERE.parents[1]
sys.path.insert(0, str(MI_ROOT))
sys.path.insert(0, str(HERE.parent))

from common.models import MODELS
from common.paths import data_dir
from common.runs import iter_run_dirs, read_jsonl
from exp_config import EXPERIMENT, SOURCE_EXPERIMENT, cf_source, load_source_config
from spans import segment

OPERATORS = load_source_config().OPERATORS
TRIVIAL = re.compile(r"^\s*(#.*|\"\"\".*|'''.*|class \w+.*:|)\s*$")


def recorded_responses(tags: set[str], wanted: set[str]) -> dict[str, str]:
    out = {}
    for tag in tags:
        for run_dir in iter_run_dirs(data_dir(SOURCE_EXPERIMENT, "raw", tag)):
            for rec in read_jsonl(run_dir / "mi_activations" / "index.jsonl"):
                if rec["record_id"] in wanted:
                    out[rec["record_id"]] = rec["response"]
    return out


def code_block(response: str) -> tuple[int, str] | None:
    m = re.search(r"```(?:python)?\n(.*?)```", response, re.S)
    return (m.start(1), m.group(1)) if m else None


def error_line(text: str, spans: dict) -> str | None:
    s, e = spans["feedback"]
    m = re.search(r"On line: (.+)", text[s:e])
    return m.group(1).strip() if m else None


def pick_targets(parent: str, response: str, err: str | None):
    """Char offsets in ``response`` of the edit token and the copy token."""
    cb = code_block(response)
    if cb is None:
        return None
    base, child = cb
    p_lines, c_lines = parent.splitlines(), child.splitlines()
    c_starts = np.cumsum([0] + [len(l) + 1 for l in c_lines])
    sm = difflib.SequenceMatcher(None, p_lines, c_lines, autojunk=False)
    ops = sm.get_opcodes()
    edit, edit_on_error = None, False
    for tag, i1, i2, j1, j2 in ops:
        if tag in ("replace", "insert") and j2 > j1:
            touches = err is not None and any(err in p_lines[i] for i in range(i1, i2))
            for j in range(j1, j2):
                if TRIVIAL.match(c_lines[j]):
                    continue
                line = c_lines[j]
                if tag == "replace" and j - j1 < i2 - i1:
                    old = p_lines[i1 + j - j1]
                    n = len(re.match(r"\s*", line).group(0))
                    while n < min(len(line), len(old)) and line[n] == old[n]:
                        n += 1
                    if n >= len(line):
                        continue
                else:
                    n = len(re.match(r"\s*", line).group(0))
                cand = (int(c_starts[j]) + n, j)
                if touches and not edit_on_error:
                    edit, edit_on_error = cand, True
                elif edit is None:
                    edit = cand
                break
        if edit_on_error:
            break
    if edit is None:
        return None
    copy = None
    for tag, i1, i2, j1, j2 in ops:
        if tag != "equal":
            continue
        for j in range(j1, j2):
            if (
                j < edit[1]
                and not TRIVIAL.match(c_lines[j])
                and len(c_lines[j].strip()) > 8
            ):
                n = len(re.match(r"\s*", c_lines[j]).group(0))
                copy = int(c_starts[j]) + n
    return {
        "edit": base + edit[0],
        "edit_on_error_line": edit_on_error,
        "copy": None if copy is None else base + copy,
    }


class Context:
    """Token ids of chat prompt + response prefix, with a span per position."""

    def __init__(self, tok, chat_kwargs, user_text, spans, response):
        session = [{"role": "user", "content": user_text}]
        rendered = tok.apply_chat_template(
            session, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
        # Some templates (Gemma) trim the message: locate the trimmed text
        # and shift by the leading whitespace so span offsets still apply.
        lead = len(user_text) - len(user_text.lstrip())
        off = rendered.index(user_text.strip()) - lead
        enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
        renc = tok(response, add_special_tokens=False, return_offsets_mapping=True)
        self.prompt_ids = enc["input_ids"]
        self.resp_ids = renc["input_ids"]
        self.resp_starts = [s for s, _ in renc["offset_mapping"]]
        lo, hi = 0, len(user_text)
        if user_text not in rendered:  # the template trimmed it
            lo, hi = lead, len(user_text.rstrip())
        labels = []
        for s, _ in enc["offset_mapping"]:
            c = s - off
            name = "template"
            if lo <= c < hi:
                name = next(k for k, (a, b) in spans.items() if a <= c < b)
            labels.append(name)
        self.prompt_labels = labels

    def upto(self, char: int):
        """(ids, labels, target id, target text) for the token covering ``char``."""
        t = max(i for i, s in enumerate(self.resp_starts) if s <= char)
        ids = self.prompt_ids + self.resp_ids[:t]
        labels = self.prompt_labels + ["response"] * t
        return ids, labels, self.resp_ids[t]


def node_positions(graph) -> np.ndarray:
    n_pos = len(graph.input_tokens)
    feats = graph.active_features[graph.selected_features][:, 1].cpu().numpy()
    n_layers = graph.cfg.n_layers
    err = np.tile(np.arange(n_pos), n_layers)
    tok = np.arange(n_pos)
    logit = np.full(len(graph.logit_targets), -1)
    return feats, np.concatenate([feats, err, tok, logit])


def analyse(graph, labels, top_k=15):
    from circuit_tracer.graph import compute_graph_scores, compute_node_influence

    adj = graph.adjacency_matrix
    n_logits = len(graph.logit_targets)
    w = torch.zeros(adj.shape[0])
    w[-n_logits:] = graph.logit_probabilities.cpu()
    infl = compute_node_influence(adj.cpu(), w).numpy()
    feats_pos, pos = node_positions(graph)
    n_feat, n_pos = len(feats_pos), len(graph.input_tokens)
    tok_slice = slice(
        n_feat + graph.cfg.n_layers * n_pos, n_feat + (graph.cfg.n_layers + 1) * n_pos
    )
    lab = np.array(labels + ["logit"])
    names = sorted(set(labels))
    keep = pos >= 0
    by_pos = np.bincount(pos[keep], weights=infl[keep], minlength=n_pos)
    tok_infl = infl[tok_slice]
    share = {k: float(by_pos[lab[:n_pos] == k].sum() / by_pos.sum()) for k in names}
    tok_share = {
        k: float(tok_infl[lab[:n_pos] == k].sum() / tok_infl.sum()) for k in names
    }
    n_tok = {k: int((lab[:n_pos] == k).sum()) for k in names}
    replacement, completeness = compute_graph_scores(graph)
    fb = np.where(lab[feats_pos] == "feedback")[0]
    top = fb[np.argsort(-infl[fb])[:top_k]]
    af = graph.active_features[graph.selected_features].cpu().numpy()
    top_features = [
        {
            "layer": int(af[i, 0]),
            "pos": int(af[i, 1]),
            "feature": int(af[i, 2]),
            "influence": float(infl[i]),
        }
        for i in top
    ]
    return {
        "share": share,
        "token_share": tok_share,
        "n_tokens": n_tok,
        "replacement_score": replacement,
        "completeness_score": completeness,
        "n_feature_nodes": int(n_feat),
        "top_feedback_features": top_features,
    }


def attribute_fit(model, ids, tid, args):
    """Attribute with a batch size that fits in memory, or return None.

    The forward pass keeps the attention patterns of every batch copy for the
    backward pass, so memory grows with ``batch_size * n**2``: scale the
    batch down beyond ``--ref-tokens`` and halve it on out-of-memory.
    """
    from circuit_tracer import attribute

    scale = (args.ref_tokens / len(ids)) ** 2
    batch = max(1, min(args.batch_size, int(args.batch_size * scale)))
    while batch >= 1:
        try:
            return attribute(
                torch.tensor(ids),
                model,
                attribution_targets=torch.tensor([tid]),
                max_feature_nodes=args.max_feature_nodes,
                batch_size=batch,
                offload="cpu",
            )
        except torch.OutOfMemoryError as e:
            where = traceback.extract_tb(e.__traceback__)[-1]
            torch.cuda.empty_cache()
            print(
                f"out of memory at n={len(ids)}, batch {batch} "
                f"({where.name}:{where.lineno}, {str(e).split('.')[0]})",
                flush=True,
            )
            batch //= 2
    return None


def load_replacement_model(hf_id: str, transcoders: str, args):
    """Load the model with its transcoders.

    ``transcoders`` is a Hugging Face reference (``org/repo[/subfolder]``)
    or a local directory with a ``config.yaml`` listing local files, as
    written by ``fix_transcoders.py`` for sets that circuit-tracer cannot
    read directly (``mwhanna/qwen3-14b-transcoders``).
    """
    from circuit_tracer import ReplacementModel
    from circuit_tracer.utils.hf_utils import load_transcoders

    kw = dict(device=torch.device(args.device), dtype=torch.bfloat16)
    lazy = dict(lazy_encoder=not args.eager_encoder, lazy_decoder=True)
    local = Path(os.path.expandvars(transcoders))
    if not local.is_dir():
        return ReplacementModel.from_pretrained(hf_id, transcoders, **kw, **lazy)
    import yaml

    config = yaml.safe_load((local / "config.yaml").read_text())
    tc = load_transcoders(config, **kw, **lazy)
    return ReplacementModel.from_pretrained_and_transcoders(hf_id, tc, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="cf")
    ap.add_argument("--n-error", type=int, default=24)
    ap.add_argument("--n-scored", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=2200)
    ap.add_argument("--max-feature-nodes", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument(
        "--ref-tokens",
        type=int,
        default=1750,
        help="context length up to which --batch-size fits in 96 GB",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--transcoders", default=None, help="default: the model preset's transcoders"
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument(
        "--eager-encoder",
        action="store_true",
        help="keep all encoders on the GPU (~48 GB; faster on an idle GPU)",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from transformers import AutoTokenizer

    proc = data_dir(EXPERIMENT, "processed", args.tag)
    queries = [json.loads(l) for l in open(proc / "cf_queries.jsonl")]
    queries = [q for q in queries if q["orig_child_failed"] == 0]
    responses = recorded_responses(
        {q["source_tag"] for q in queries}, {q["record_id"] for q in queries}
    )
    preset = MODELS[cf_source(args.tag)["model"]]
    transcoders = args.transcoders or preset.transcoders
    tok = AutoTokenizer.from_pretrained(preset.hf_id)
    chat_kwargs = preset.chat_template_kwargs

    jobs = []
    for q in queries:
        text = q["variants"]["full"]["text"]
        spans = segment(text, OPERATORS[q["operator"]])
        resp = responses.get(q["record_id"])
        if spans is None or resp is None:
            continue
        t = pick_targets(q["parent_code"], resp, error_line(text, spans))
        if t is None or t["copy"] is None:
            continue
        ctx = Context(tok, chat_kwargs, text, spans, resp)
        ids, _, _ = ctx.upto(t["edit"])
        if len(ids) > args.max_tokens:
            continue
        jobs.append((q, ctx, t, len(ids)))
    picked = []
    for stratum, n in (("error", args.n_error), ("scored", args.n_scored)):
        # Edits on the line named in the error first, then shortest contexts
        # (graph size grows with tokens^2).
        pool = sorted(
            [j for j in jobs if j[0]["stratum"] == stratum],
            key=lambda j: (not j[2]["edit_on_error_line"], j[3]),
        )
        picked += pool[:n]
    picked = picked[args.shard :: args.n_shards][: args.limit]
    print(f"{len(jobs)} usable queries; attributing {len(picked)} x 2 targets")

    model = load_replacement_model(preset.hf_id, transcoders, args)
    out_path = proc / f"attribution_graphs-{args.shard}.jsonl"
    done = set()
    if out_path.exists():
        done = {(r["record_id"], r["target"]) for r in read_jsonl(out_path)}
    with open(out_path, "a") as f:
        for q, ctx, t, _ in picked:
            for target in ("edit", "copy"):
                if (q["record_id"], target) in done:
                    continue
                ids, labels, tid = ctx.upto(t[target])
                t0 = time.time()
                graph = attribute_fit(model, ids, tid, args)
                if graph is None:
                    print(f"skip {q['record_id']} {target}: out of memory")
                    continue
                row = {
                    "record_id": q["record_id"],
                    "stratum": q["stratum"],
                    "operator": q["operator"],
                    "target": target,
                    "edit_on_error_line": t["edit_on_error_line"],
                    "target_token": tok.decode([tid]),
                    "context_tail": tok.decode(ids[-12:]),
                    "n_ctx": len(ids),
                    "target_prob": float(graph.logit_probabilities[0]),
                    **analyse(graph, labels),
                    "seconds": time.time() - t0,
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(
                    f"{q['stratum']:6s} {target:4s} n={len(ids)} "
                    f"fb={row['share'].get('feedback', 0):.3f} "
                    f"code={row['share'].get('parent_code', 0):.3f} "
                    f"repl={row['replacement_score']:.2f} {row['seconds']:.0f}s",
                    flush=True,
                )
                del graph
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
