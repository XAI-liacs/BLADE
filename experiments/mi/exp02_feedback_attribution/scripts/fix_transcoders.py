"""Write a circuit-tracer-ready local copy of a transcoder set.

``mwhanna/qwen3-14b-transcoders`` differs from the Qwen3-8B set that
circuit-tracer 0.5 reads: it has no ``config.yaml`` and stores ``W_dec``
as ``(d_model, d_transcoder)`` instead of ``(d_transcoder, d_model)``.
circuit-tracer's lazy decoder reads rows of ``W_dec`` by feature index, so
the transposed layout fails (or would silently read the wrong vectors).

This writes ``layer_<i>.safetensors`` with ``W_dec`` transposed where
needed, plus a ``config.yaml`` listing the local files. Point the model
preset (or ``06_attribution_graphs.py --transcoders``) at the directory::

    python fix_transcoders.py mwhanna/qwen3-14b-transcoders Qwen/Qwen3-14B \\
        $MI_CACHE/transcoders/qwen3-14b
"""

import argparse
import glob
import os
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo_id")
    ap.add_argument("model_name")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--input-hook", default="mlp.hook_in")
    ap.add_argument("--output-hook", default="mlp.hook_out")
    args = ap.parse_args()

    src = Path(snapshot_download(args.repo_id, allow_patterns=["layer_*"]))
    files = glob.glob(str(src / "layer_*.safetensors"))
    n_layers = len(files)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_layers):
        out = args.out_dir / f"layer_{i}.safetensors"
        paths.append(str(out))
        if out.exists():
            continue
        with safe_open(str(src / f"layer_{i}.safetensors"), "pt") as f:
            t = {k: f.get_tensor(k) for k in f.keys()}
        d_tc = t["b_enc"].shape[0]
        if t["W_dec"].shape[0] != d_tc:  # (d_model, d_tc) -> (d_tc, d_model)
            t["W_dec"] = t["W_dec"].T.contiguous()
        assert t["W_dec"].shape[0] == d_tc == t["W_enc"].shape[0]
        tmp = out.with_suffix(".tmp")
        save_file(t, str(tmp))
        os.replace(tmp, out)
        print(f"layer {i}: W_dec {tuple(t['W_dec'].shape)}", flush=True)

    config = {
        "model_name": args.model_name,
        "model_kind": "transcoder_set",
        "feature_input_hook": args.input_hook,
        "feature_output_hook": args.output_hook,
        "transcoders": paths,
        "scan": f"{args.repo_id} (W_dec transposed)",
    }
    (args.out_dir / "config.yaml").write_text(yaml.safe_dump(config))
    print(f"wrote {n_layers} layers and config.yaml to {args.out_dir}")


if __name__ == "__main__":
    main()
