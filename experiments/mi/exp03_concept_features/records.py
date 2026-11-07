"""Recorded queries of experiment 1 and teacher-forced residual states."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # experiments/mi

from common.paths import data_dir
from common.runs import iter_run_dirs, load_run
from exp_config import SOURCE_EXPERIMENT


def load_records(tags) -> pd.DataFrame:
    """One row per recorded query with its prompt, response and labels.

    ``parent_code`` is the code of the first parent ("" for initial
    queries).

    Labels (operator, fitness, failure, ...) come from experiment 1's
    ``meta.csv``; only queries listed there are kept.
    """
    rows = []
    for tag in tags:
        meta = pd.read_csv(
            data_dir(SOURCE_EXPERIMENT, "processed", tag) / "meta.csv",
            dtype={"record_id": str, "run": str},
        ).set_index("record_id")
        for run_dir in iter_run_dirs(data_dir(SOURCE_EXPERIMENT, "raw", tag)):
            solutions, records = load_run(run_dir)
            for rec in records:
                rid = rec["record_id"]
                if rid not in meta.index:
                    continue
                parents = [solutions.get(p) for p in rec.get("parent_ids", [])]
                parent = parents[0] if parents and parents[0] else None
                rows.append(
                    {
                        "record_id": rid,
                        "tag": tag,
                        "session": rec["prompt"],
                        "response": rec["response"],
                        "parent_code": parent["code"] if parent else "",
                        **meta.loc[rid].to_dict(),
                    }
                )
    df = pd.DataFrame(rows)
    df["run"] = df["tag"] + "/" + df["run"].astype(str)
    return df


def code_span(response: str) -> tuple[int, int]:
    """Character span of the first fenced code block (whole text if none)."""
    start = response.find("```")
    if start < 0:
        return 0, len(response)
    start = response.find("\n", start) + 1
    end = response.find("```", start)
    return start, end if end >= 0 else len(response)


class Forward:
    """Teacher-forced forward pass returning residual states at ``layers``.

    States are the outputs of the decoder blocks (``resid_post``), the hook
    point of the Qwen-Scope SAEs and of ``HookedLLM``'s records. Optional
    ``patch`` callables replace a block's output (for spliced-in SAE
    reconstructions).
    """

    def __init__(self, llm, layers):
        self.llm = llm
        self.tok = llm.tokenizer
        self.layers = list(layers)
        self.blocks = llm.hf_model.model.layers
        self.device = llm.hf_model.get_input_embeddings().weight.device

    def encode(self, session, response):
        import torch

        prompt_ids = self.llm._encode_chat(session)["input_ids"]
        enc = self.tok(
            response,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        ids = torch.cat([prompt_ids, enc["input_ids"]], dim=1)
        offsets = enc["offset_mapping"][0].numpy()
        return ids, int(prompt_ids.shape[1]), offsets

    def code_positions(self, response, n_prompt, offsets) -> np.ndarray:
        """Token positions (in the full sequence) inside the code block."""
        a, b = code_span(response)
        inside = (offsets[:, 0] >= a) & (offsets[:, 1] <= b)
        return n_prompt + np.flatnonzero(inside)

    def run(self, ids, patch: dict | None = None, logits_from: int | None = None):
        """Returns ``(states by layer, logits or None)``."""
        import torch

        states, handles = {}, []

        def make(layer):
            def hook(module, inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                if patch and layer in patch:
                    hs = patch[layer](hs)
                    output = (
                        (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
                    )
                if layer in self.layers:
                    states[layer] = hs[0].detach()
                return output if patch and layer in patch else None

            return hook

        for layer in set(self.layers) | set(patch or {}):
            handles.append(self.blocks[layer].register_forward_hook(make(layer)))
        try:
            with torch.no_grad():
                keep = 1 if logits_from is None else ids.shape[1] - logits_from
                out = self.llm.hf_model(
                    input_ids=ids.to(self.device), logits_to_keep=keep
                )
        finally:
            for h in handles:
                h.remove()
        logits = out.logits[0].float() if logits_from is not None else None
        return states, logits
