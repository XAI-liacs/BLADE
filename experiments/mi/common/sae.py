"""TopK sparse autoencoders on the residual stream (Qwen-Scope format).

``Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_<k>`` stores one ``layer<n>.sae.pt``
dict per layer with ``W_enc`` (d_sae, d_model), ``b_enc`` (d_sae,),
``W_dec`` (d_model, d_sae) and ``b_dec`` (d_model,). The hook point is the
output of block ``n`` (``resid_post``), the same tensor ``HookedLLM``
captures for layer ``n``. Encoding keeps the ``k`` largest pre-activations
(as in the model card; there is no ``b_dec`` subtraction before encoding).
"""

import json

import torch


class TopKSAE:
    def __init__(self, W_enc, b_enc, W_dec, b_dec, k: int):
        self.W_enc, self.b_enc = W_enc, b_enc
        self.W_dec, self.b_dec = W_dec, b_dec
        self.k = k
        self.d_sae = W_enc.shape[0]

    @classmethod
    def from_hub(cls, repo: str, layer: int, device="cuda:0", dtype=torch.float32):
        from huggingface_hub import hf_hub_download

        config = json.load(open(hf_hub_download(repo, "config.json")))
        path = hf_hub_download(repo, f"layer{layer}.sae.pt")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        t = {k: v.to(device=device, dtype=dtype) for k, v in sd.items()}
        return cls(t["W_enc"], t["b_enc"], t["W_dec"], t["b_dec"], config["k"])

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``x`` (..., d_model) -> top-k ``(values, indices)``, each (..., k)."""
        pre = x.to(self.W_enc.dtype) @ self.W_enc.T + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        return vals.clamp_min(0), idx

    def decode(self, vals: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Reconstruction from top-k codes: sum of decoder columns + bias."""
        cols = self.W_dec.T[idx]  # (..., k, d_model)
        return (vals.unsqueeze(-1) * cols).sum(-2) + self.b_dec

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(*self.encode(x)).to(x.dtype)
