"""Step 1: how much does each prompt span drive the generated response?

For a stratified sample of mutation queries recorded in experiment 1, the
model re-reads the prompt and its own response (teacher forcing) under
several prompt variants:

* ``full``: the original prompt;
* ``-<span>``: the prompt with one span deleted (``spans.SPAN_NAMES``);
* ``score_low`` / ``score_high``: the parent's score in the feedback replaced
  by 0.010 / 0.900 (only when the parent has a score).

Per variant it records the mean log-probability of the response tokens,
over the whole response and over its code part. ``drop = full - variant``
(nats per token) is the causal dependence of the response on that span.

It also computes gradient x input on the input embeddings for the summed
response log-probability (one backward pass), and reports per span the
share of the total attribution and the share per token (density).

Writes ``data/processed/<tag>/span_attribution.csv`` (one row per query and
variant) and ``span_gradients.csv`` (one row per query and span).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MI_ROOT = HERE.parents[1]
sys.path.insert(0, str(MI_ROOT))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp02_feedback_attribution

from common.models import MODELS
from common.paths import data_dir
from common.runs import iter_run_dirs, load_run, prompt_text
from exp_config import EXPERIMENT, SOURCE_EXPERIMENT, load_source_config
from spans import SPAN_NAMES, ablate, feedback_score, segment

OPERATORS = load_source_config().OPERATORS

import torch

from iohblade.llm import HookedLLM


def load_queries(source_tag: str) -> pd.DataFrame:
    meta = pd.read_csv(
        data_dir(SOURCE_EXPERIMENT, "processed", source_tag) / "meta.csv",
        dtype={"record_id": str, "run": str},
    ).set_index("record_id")
    rows = []
    raw = data_dir(SOURCE_EXPERIMENT, "raw", source_tag)
    for run_dir in iter_run_dirs(raw):
        _, records = load_run(run_dir)
        for rec in records:
            if rec["record_id"] not in meta.index:
                continue
            m = meta.loc[rec["record_id"]]
            if m["operator"] not in OPERATORS:
                continue
            text = prompt_text(rec)
            spans = segment(text, OPERATORS[m["operator"]])
            if spans is None or len(rec["prompt"]) != 1:
                continue
            rows.append(
                {
                    "record_id": rec["record_id"],
                    "run": m["run"],
                    "operator": m["operator"],
                    "parent_failed": bool(m["parent_failed"]),
                    "parent_fitness": m["parent_fitness"],
                    "child_failed": m["child_failed"],
                    "improved": m["improved"],
                    "text": text,
                    "spans": spans,
                    "response": rec["response"],
                }
            )
    return pd.DataFrame(rows)


class Scorer:
    """Teacher-forced response log-probs and gradient x input."""

    def __init__(self, llm: HookedLLM):
        import torch

        self.torch = torch
        self.llm = llm
        self.model = llm.hf_model
        self.tok = llm.tokenizer
        self.device = self.model.get_input_embeddings().weight.device
        # Recompute activations in the backward pass: the gradient pass
        # must fit next to generation jobs on the same GPU. Training mode
        # only enables checkpointing (Qwen3 has no dropout).
        self.model.gradient_checkpointing_enable()
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _encode(self, user_text: str, response: str):
        session = [{"role": "user", "content": user_text}]
        prompt_ids = self.llm._encode_chat(session)["input_ids"]
        resp_ids = self.tok(response, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ]
        return prompt_ids, resp_ids

    def _code_start(self, response: str, resp_ids) -> int:
        """Index of the first response token of the code block."""
        cut = response.find("```")
        if cut < 0:
            return 0
        n = len(self.tok(response[:cut], add_special_tokens=False)["input_ids"])
        return min(n, resp_ids.shape[1] - 1)

    def logprobs(self, user_text: str, response: str) -> tuple[float, float]:
        torch = self.torch
        prompt_ids, resp_ids = self._encode(user_text, response)
        ids = torch.cat([prompt_ids, resp_ids], dim=1).to(self.device)
        n_p = prompt_ids.shape[1]
        n_r = resp_ids.shape[1]
        with torch.no_grad():
            logits = self.model(input_ids=ids, logits_to_keep=n_r + 1).logits
        logits = logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, ids[0, n_p:, None])[:, 0]
        c = self._code_start(response, resp_ids)
        return float(lp.mean()), float(lp[c:].mean())

    def gradient_x_input(self, user_text: str, response: str, spans: dict):
        """Per-span share of |grad . embedding| over prompt tokens."""
        torch = self.torch
        session = [{"role": "user", "content": user_text}]
        rendered = self.tok.apply_chat_template(
            session,
            tokenize=False,
            add_generation_prompt=True,
            **(self.llm.chat_template_kwargs or {}),
        )
        offset = rendered.index(user_text)
        enc = self.tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
        prompt_ids, resp_ids = self._encode(user_text, response)
        if enc["input_ids"] != prompt_ids[0].tolist():
            raise ValueError("re-tokenised prompt differs from the chat encoding")
        ids = torch.cat([prompt_ids, resp_ids], dim=1).to(self.device)
        n_p = prompt_ids.shape[1]
        n_r = resp_ids.shape[1]
        emb = self.model.get_input_embeddings()(ids).detach().requires_grad_(True)
        logits = self.model(inputs_embeds=emb, logits_to_keep=n_r + 1).logits
        logits = logits[0, :-1].float()
        lp = torch.log_softmax(logits, -1).gather(1, ids[0, n_p:, None]).sum()
        lp.backward()
        attr = (emb.grad[0, :n_p] * emb[0, :n_p].detach()).float().sum(-1).abs()
        attr = attr.cpu().numpy()
        self.model.zero_grad(set_to_none=True)
        total = attr.sum()
        starts = np.array([s for s, _ in enc["offset_mapping"]]) - offset
        out = {}
        for name, (a, b) in spans.items():
            mask = (starts >= a) & (starts < b)
            out[name] = (float(attr[mask].sum() / total), int(mask.sum()))
        out["template"] = (
            float(attr[starts < 0].sum() / total),
            int((starts < 0).sum()),
        )
        return out


def variants(row) -> dict[str, str]:
    text, spans = row["text"], row["spans"]
    out = {"full": text}
    for name in SPAN_NAMES:
        out[f"-{name}"] = ablate(text, spans, name)
    if feedback_score(text, spans) is not None:
        s, e = spans["feedback"]
        fb = text[s:e]
        for label, value in (("score_low", "0.010"), ("score_high", "0.900")):
            new_fb = fb.replace(
                fb[fb.index("scored ") + 7 : fb.index(" on AOCC")], value, 1
            )
            out[label] = text[:s] + new_fb + text[e:]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-tag", default="main")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--n-per-operator", type=int, default=60)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model", default=None, help="default: the source campaign's")
    ap.add_argument("--no-gradients", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    queries = load_queries(args.source_tag)
    sample = (
        queries.sample(frac=1.0, random_state=args.seed)
        .groupby("operator")
        .head(args.n_per_operator)
        .reset_index(drop=True)
    )
    print(f"{len(queries)} segmentable mutation queries; sampled {len(sample)}")

    preset = MODELS[args.model or load_source_config().PRESETS[args.source_tag].model]
    llm = HookedLLM(
        preset.hf_id,
        capture=False,
        device_map=args.device,
        chat_template_kwargs=preset.chat_template_kwargs,
        store_dtype=preset.store_dtype,
    )
    scorer = Scorer(llm)
    out = data_dir(EXPERIMENT, "processed", args.tag)
    lp_rows, grad_rows = [], []
    t0 = time.time()
    for i, row in sample.iterrows():
        info = {
            k: row[k]
            for k in (
                "record_id",
                "run",
                "operator",
                "parent_failed",
                "parent_fitness",
                "child_failed",
                "improved",
            )
        }
        for name, text in variants(row).items():
            lp_all, lp_code = scorer.logprobs(text, row["response"])
            lp_rows.append(
                {**info, "variant": name, "logprob": lp_all, "logprob_code": lp_code}
            )
        if not args.no_gradients:
            try:
                grads = scorer.gradient_x_input(
                    row["text"], row["response"], row["spans"]
                )
            except torch.OutOfMemoryError:
                print(f"  OOM in gradient pass, skipped {row['record_id']}")
                torch.cuda.empty_cache()
                grads = {}
            for span, (share, n_tok) in grads.items():
                grad_rows.append(
                    {**info, "span": span, "grad_share": share, "n_tokens": n_tok}
                )
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(sample)} ({time.time() - t0:.0f} s)", flush=True)
            pd.DataFrame(lp_rows).to_csv(out / "span_attribution.csv", index=False)
            pd.DataFrame(grad_rows).to_csv(out / "span_gradients.csv", index=False)
    pd.DataFrame(lp_rows).to_csv(out / "span_attribution.csv", index=False)
    pd.DataFrame(grad_rows).to_csv(out / "span_gradients.csv", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
