"""Step 4: fit the final probes on a whole campaign and save them.

The saved probes are used in two places:

* the in-loop pre-screen (``05_prescreen_generate.py``), which scores every
  generated algorithm with the ``child_failed`` probe on new seeds;
* experiments 2 and 5, which use the probe directions as attribution and
  steering targets.

Each probe is the same pipeline as in ``03_train_probes.py``
(``StandardScaler`` + ``RidgeCV``), refit on all rows of the campaign at a
fixed (position, layer). ``results/<tag>/probes/<name>.npz`` stores:

* ``mean``, ``scale``, ``coef``, ``intercept``, ``alpha``: the probe, with
  ``score(x) = ((x - mean) / scale) @ coef + intercept`` for a residual-stream
  vector ``x`` (float32, ``d_model``);
* ``direction``: ``coef / scale``, normalised to unit length. This is the
  residual-stream direction along which the score grows (for steering);
* ``threshold``: for binary probes, the out-of-fold score above which a
  fraction ``skip_rate`` of the campaign's queries would be flagged.

Each probe sits at the best layer of its campaign for that target and
position, read from ``oof_<target>_<position>.csv`` (``03_train_probes.py``),
so every model gets its own layers. The name records it, e.g.
``child_failed_gen_mean_L21`` for Qwen3-8B. The layers were picked after
seeing the scores, so they are only validated on new seeds (the pre-screen
campaign). ``probes.json`` lists the probes with target, position and layer;
use :func:`find_probe` to look one up by target.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common.paths import data_dir, results_dir
from exp_config import EXPERIMENT

probes = importlib.import_module("03_train_probes")

# (target, position); the layer is the campaign's best one
EXPORT = [
    ("child_failed", "gen_mean"),
    ("over_budget", "gen_mean"),
    ("parent_fitness", "prompt_mean"),
]


def find_probe(probe_dir: Path, target: str) -> tuple[str, dict]:
    """Name and info of the exported probe for ``target`` in ``probe_dir``."""
    summary = json.loads((Path(probe_dir) / "probes.json").read_text())
    hits = [(n, i) for n, i in summary.items() if i["target"] == target]
    if len(hits) != 1:
        raise KeyError(f"{len(hits)} probes for {target} in {probe_dir}")
    return hits[0]


def fit_probe(X: np.ndarray, y: np.ndarray, kind: str) -> dict:
    scaler = StandardScaler().fit(X)
    target = 2.0 * y - 1.0 if kind == "binary" else y
    ridge = RidgeCV(alphas=probes.ALPHAS).fit(scaler.transform(X), target)
    scale = scaler.scale_.astype(np.float32)
    direction = ridge.coef_ / scale
    return {
        "mean": scaler.mean_.astype(np.float32),
        "scale": scale,
        "coef": ridge.coef_.astype(np.float32),
        "intercept": np.float32(ridge.intercept_),
        "alpha": np.float32(ridge.alpha_),
        "direction": (direction / np.linalg.norm(direction)).astype(np.float32),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="main")
    ap.add_argument("--skip-rate", type=float, default=0.2)
    args = ap.parse_args()

    proc = data_dir(EXPERIMENT, "processed", args.tag)
    meta = pd.read_csv(proc / "meta.csv", dtype={"record_id": str, "run": str})
    acts = np.load(proc / "activations.npz")
    layers = list(acts["layers"])
    out = results_dir(EXPERIMENT, args.tag) / "probes"
    out.mkdir(exist_ok=True)

    summary = {}
    for target, pos in EXPORT:
        oof = pd.read_csv(results_dir(EXPERIMENT, args.tag) / f"oof_{target}_{pos}.csv")
        layer = int(oof["layer"].iloc[0])
        name = f"{target}_{pos}_L{layer}"
        kind, row_filter, _ = probes.TARGETS[target]
        mask = row_filter(meta).to_numpy()
        X = acts[pos][mask, layers.index(layer)].astype(np.float32)
        y = meta.loc[mask, target].to_numpy(dtype=float)
        p = fit_probe(X, y, kind)
        info = {"target": target, "position": pos, "layer": layer, "kind": kind}
        info.update({"n": int(mask.sum()), "alpha": float(p["alpha"])})
        if kind == "binary":
            p["threshold"] = np.float32(np.quantile(oof["pred"], 1 - args.skip_rate))
            info.update(
                {"skip_rate": args.skip_rate, "threshold": float(p["threshold"])}
            )
        np.savez(out / f"{name}.npz", **p)
        summary[name] = info
        print(name, info)
    (out / "probes.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
