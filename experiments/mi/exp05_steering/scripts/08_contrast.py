"""Step 5b.2: contrastive directions from the labelled corpus.

``--stage acts`` (GPU): every algorithm of the corpus (``07_corpus.py``) is
fed to the model as the answer to one fixed initial prompt of experiment 1
(teacher forced), and the residual state is averaged over the code tokens
at ``CONTRAST_LAYERS``. With one prompt for all, activation differences
come from the code, not from the instruction that elicited it. Writes
``data/processed/corpus/acts.npz`` and ``rows.csv`` (concept labels of
experiment 3, behaviour, fitness).

``--stage directions`` (CPU), per layer:

* **concept** ``c`` (at least ``--min-pos`` algorithms with it): mean state
  of the code with ``c`` minus the mean of the code without;
* **behaviour** ``m`` (``BEHAVIOUR_METRICS``, working algorithms): mean of
  the top third by ``m`` minus the bottom third;
* ``<m>_resid``: the same after regressing the states on the concept
  labels, so the direction is not just "more PSO" for exploration.

Each direction is validated out of sample (5 folds, direction from the
other folds): AUROC of the projection for concepts, Spearman correlation
with the metric for behaviours. The unit directions go to
``data/processed/corpus/directions.npz``; ``info.json`` has the layer, the
norm of the mean difference (the natural dose unit), the mean per-token
residual norm at the layer and the validation score. ``results/corpus/``
gets ``directions.md`` with the table, the cosine between the behaviour
and concept directions, and corpus statistics.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MI_ROOT = HERE.parents[1]
sys.path.insert(0, str(MI_ROOT))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import importlib

from common.paths import data_dir, results_dir
from common.runs import read_jsonl
from exp_config import BEHAVIOUR_METRICS, CONTRAST_LAYERS, CORPUS, EXPERIMENT, MODEL

prep = importlib.import_module("01_prepare")
labels_mod = prep.exp03_module("03_concept_labels")
CONCEPTS = list(labels_mod.CONCEPTS)
MAX_CODE_TOKENS = 3072


def load_corpus() -> pd.DataFrame:
    raw = data_dir(EXPERIMENT, "raw", CORPUS["tag"])
    rows = {}
    for path in sorted(raw.glob("*-shard-*.jsonl")):
        for r in read_jsonl(path):
            if str(r.get("error", "")).startswith("INFRA"):
                continue
            rows[r["id"]] = r  # last line wins
    df = pd.DataFrame(list(rows.values()))
    df["code"] = df.code.fillna("")
    df = df[df.code.str.strip() != ""].reset_index(drop=True)
    for m in BEHAVIOUR_METRICS:
        df[m] = [b.get(m) if isinstance(b, dict) else np.nan for b in df.behaviour]
    lab = [labels_mod.labels(c) for c in df.code]
    for c in CONCEPTS:
        df[c] = [int(bool(l.get(c))) for l in lab]
    return df


def stage_acts(args):
    import torch
    from common.models import MODELS

    from iohblade.llm import HookedLLM

    df = load_corpus()
    records = prep.exp03_module("records")
    recs = records.load_records(["main"])
    session = recs[recs.operator == "init"].session.iloc[0]
    preset = MODELS[MODEL]
    llm = HookedLLM(
        preset.hf_id,
        capture=False,
        device_map=args.device,
        chat_template_kwargs=preset.chat_template_kwargs,
    )
    fwd = records.Forward(llm, CONTRAST_LAYERS)
    L = len(CONTRAST_LAYERS)
    acts = np.zeros((len(df), L, llm.hf_model.config.hidden_size), np.float16)
    tok_norm = np.zeros((len(df), L), np.float32)
    for i, code in enumerate(df.code):
        response = f"# Description: -\n# Code:\n```python\n{code}\n```"
        ids, n_p, offsets = fwd.encode(session, response)
        pos = fwd.code_positions(response, n_p, offsets)[:MAX_CODE_TOKENS]
        ids = ids[:, : pos[-1] + 1]
        states, _ = fwd.run(ids)
        p = torch.as_tensor(pos, device=states[CONTRAST_LAYERS[0]].device)
        for j, l in enumerate(CONTRAST_LAYERS):
            h = states[l].float()[p]
            acts[i, j] = h.mean(0).cpu().numpy()
            tok_norm[i, j] = h.norm(dim=-1).mean().item()
        if (i + 1) % 200 == 0:
            print(f"acts: {i + 1}/{len(df)}", flush=True)
    out = data_dir(EXPERIMENT, "processed", CORPUS["tag"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "acts.npz", acts=acts, tok_norm=tok_norm, layers=CONTRAST_LAYERS)
    df.drop(columns=["code", "behaviour"]).to_csv(out / "rows.csv", index=False)
    print(f"wrote {len(df)} rows", flush=True)


def md(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table (pandas' ``to_markdown`` needs ``tabulate``)."""
    if index:
        df = df.reset_index()
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    head = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def auroc(score, y):
    from scipy.stats import rankdata

    y = np.asarray(y, bool)
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def folds(n, k, rng):
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


def concept_dir(X, y):
    return X[y == 1].mean(0) - X[y == 0].mean(0)


def behaviour_dir(X, v):
    lo, hi = np.nanpercentile(v, [100 / 3, 200 / 3])
    return X[v >= hi].mean(0) - X[v <= lo].mean(0)


def residualise(X_fit, C_fit, X_apply, C_apply):
    """Removes the part of X explained by concept labels (OLS fit on X_fit)."""
    A = np.c_[np.ones(len(C_fit)), C_fit]
    B, *_ = np.linalg.lstsq(A, X_fit, rcond=None)
    return X_apply - np.c_[np.ones(len(C_apply)), C_apply] @ B


def stage_directions(args):
    from scipy.stats import spearmanr

    proc = data_dir(EXPERIMENT, "processed", CORPUS["tag"])
    z = np.load(proc / "acts.npz")
    acts, tok_norm, layers = z["acts"].astype(np.float32), z["tok_norm"], z["layers"]
    df = pd.read_csv(proc / "rows.csv")
    rng = np.random.default_rng(0)
    C = df[CONCEPTS].to_numpy(float)
    work = (~df.failed.astype(bool)) & df[BEHAVIOUR_METRICS[0]].notna()
    vecs, info = {}, {}
    for j, layer in enumerate(layers):
        X = acts[:, j]
        res_norm = float(tok_norm[:, j].mean())
        for c in CONCEPTS:
            y = df[c].to_numpy()
            if y.sum() < args.min_pos or (1 - y).sum() < args.min_pos:
                continue
            d = concept_dir(X, y)
            score = np.zeros(len(y))
            for f in folds(len(y), 5, rng):
                tr = np.setdiff1d(np.arange(len(y)), f)
                score[f] = X[f] @ concept_dir(X[tr], y[tr])
            name = f"c_{c}@L{layer}"
            vecs[name] = d / np.linalg.norm(d)
            info[name] = {
                "kind": "concept",
                "target": c,
                "layer": int(layer),
                "n_pos": int(y.sum()),
                "diff_norm": float(np.linalg.norm(d)),
                "residual_norm": res_norm,
                "auroc": float(auroc(score, y)),
            }
        Xw, Cw = X[work.to_numpy()], C[work.to_numpy()]
        for m in BEHAVIOUR_METRICS:
            v = df.loc[work, m].to_numpy(float)
            for resid in (False, True):
                Xm = residualise(Xw, Cw, Xw, Cw) if resid else Xw
                d = behaviour_dir(Xm, v)
                score = np.zeros(len(v))
                for f in folds(len(v), 5, rng):
                    tr = np.setdiff1d(np.arange(len(v)), f)
                    if resid:
                        Xtr = residualise(Xw[tr], Cw[tr], Xw[tr], Cw[tr])
                        Xf = residualise(Xw[tr], Cw[tr], Xw[f], Cw[f])
                    else:
                        Xtr, Xf = Xw[tr], Xw[f]
                    score[f] = Xf @ behaviour_dir(Xtr, v[tr])
                name = f"b_{m}{'_resid' if resid else ''}@L{layer}"
                vecs[name] = d / np.linalg.norm(d)
                info[name] = {
                    "kind": "behaviour",
                    "target": m,
                    "resid": resid,
                    "layer": int(layer),
                    "n_pos": int(len(v)),
                    "diff_norm": float(np.linalg.norm(d)),
                    "residual_norm": res_norm,
                    "spearman": float(spearmanr(score, v).statistic),
                }
        print(f"L{layer}: done", flush=True)
    np.savez(proc / "directions.npz", **vecs)
    (proc / "info.json").write_text(json.dumps(info, indent=1))
    report(df, vecs, info, layers)


def report(df, vecs, info, layers):
    out = results_dir(EXPERIMENT, CORPUS["tag"])
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# Contrastive directions (`{CORPUS['tag']}`)", ""]
    # corpus: concept rate per elicited target, behaviour per source
    el = df[df.source == "elicit"]
    if len(el):
        rows = []
        for t, g in el.groupby("target"):
            row = {"target": t, "n": len(g), "failed": g.failed.astype(bool).mean()}
            if t in CONCEPTS:
                row["target_rate"] = g[t].mean()
                row["target_rate_none"] = el[el.target == "none"][t].mean()
            for m in ("exploration_pct", "intensification"):
                row[m] = g[m].median()
            rows.append(row)
        lines += [
            "## Elicitation (median behaviour of working code)",
            "",
            md(pd.DataFrame(rows), index=False),
            "",
        ]
    lines += [
        "## Concept prevalence in the corpus",
        "",
        md(df.groupby("source")[CONCEPTS].mean().T),
        "",
    ]
    t = pd.DataFrame(info).T.reset_index(names="direction")
    cols = ["direction", "layer", "n_pos", "diff_norm", "residual_norm"]
    for kind, score in (("concept", "auroc"), ("behaviour", "spearman")):
        s = t[t.kind == kind]
        piv = s.pivot_table(
            index="target" if kind == "concept" else s.direction.str.split("@").str[0],
            columns="layer",
            values=score,
        )
        lines += [
            f"## {kind} directions: held-out {score} by layer",
            "",
            md(piv.astype(float)),
            "",
        ]
        rel = s.assign(rel=s.diff_norm / s.residual_norm).pivot_table(
            index="target" if kind == "concept" else s.direction.str.split("@").str[0],
            columns="layer",
            values="rel",
        )
        lines += [
            f"## {kind} directions: |mean difference| / residual norm",
            "",
            md(rel.astype(float)),
            "",
        ]
    # cosine of behaviour directions with concept directions
    for layer in layers:
        names = [n for n in vecs if n.endswith(f"@L{layer}")]
        cn = [n for n in names if n.startswith("c_")]
        bn = [n for n in names if n.startswith("b_")]
        if not cn or not bn:
            continue
        M = pd.DataFrame(
            [[float(vecs[b] @ vecs[c]) for c in cn] for b in bn],
            index=[b.split("@")[0] for b in bn],
            columns=[c.split("@")[0][2:] for c in cn],
        )
        lines += [f"## Cosine, behaviour x concept directions, L{layer}", "", md(M), ""]
    (out / "directions.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["acts", "directions"], required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min-pos", type=int, default=20)
    args = ap.parse_args()
    (stage_acts if args.stage == "acts" else stage_directions)(args)


if __name__ == "__main__":
    main()
