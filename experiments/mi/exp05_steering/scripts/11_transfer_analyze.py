"""Step 5d: analyse the transfer screens of ``10_transfer.py``.

``--stage acts`` (GPU): every generated algorithm is fed to the model as
the answer to experiment 1's fixed initial MA-BBOB prompt (teacher forced,
as the corpus in ``08_contrast.py``) and the residual state is averaged
over the code tokens at ``READOUT_LAYERS``. Writes
``data/processed/<tag>/acts.npz``.

``--stage report`` (CPU): a ridge probe trained on the corpus (working
algorithms; activations -> measured ``exploration_pct``) reads out the
exploration of each generated algorithm (``probe_expl``), a measure that
needs no trace and so also works for circle packing. On Bragg it is
checked against the measured exploration. Writes
``results/<tag>/summary.md``, ``conditions.csv``, ``slopes.csv`` and
``dose.png``: per condition, failure rate, score, behaviour (Bragg),
probe readout and concept rates; the slope of each on the signed dial
dose (paired: fixed effects per prompt and replicate, 95% CI from a
bootstrap over them); and the paired change of the random direction.
"""

import argparse
import importlib
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

from common.paths import data_dir, results_dir
from common.runs import read_jsonl
from exp_config import CORPUS, EXPERIMENT, MODEL, TRANSFER_CAMPAIGNS

prep = importlib.import_module("01_prepare")
labels_mod = prep.exp03_module("03_concept_labels")

READOUT_LAYERS = [15, 21]
PROBE_LAYER = 15
MAX_CODE_TOKENS = 3072
BEHAVIOUR = ["exploration_pct", "intensification", "last_improvement"]
CONCEPTS = [
    "population",
    "differential_evolution",
    "particle_swarm",
    "restart",
    "simulated_annealing",
    "local_search",
    "gradient",
]
SCORE = {"bragg": "AOCC", "packing26": "sum of radii (repaired)"}


def load(tag: str) -> pd.DataFrame:
    raw = data_dir(EXPERIMENT, "raw", tag)
    rows = []
    for phase in ("init", "refine"):
        path = raw / f"{phase}.jsonl"
        if path.exists():
            rows += [r for r in read_jsonl(path) if not r.get("infra")]
    df = pd.DataFrame(rows)
    df["failed"] = df.failed.fillna(True).astype(bool)
    df["code"] = df.code.fillna("")
    df["unit"] = df.prompt + "/" + df.k.astype(str)
    for m in BEHAVIOUR:
        df[m] = pd.to_numeric(
            df.behaviour.map(
                lambda b: (b or {}).get(m) if isinstance(b, dict) else None
            ),
            errors="coerce",
        )
    for c in CONCEPTS:
        df[c] = [
            float(bool(labels_mod.labels(k).get(c))) if k else np.nan for k in df.code
        ]
    return df


def stage_acts(args):
    import torch
    from common.models import MODELS

    from iohblade.llm import HookedLLM

    df = load(args.tag)
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
    fwd = records.Forward(llm, READOUT_LAYERS)
    acts = np.full(
        (len(df), len(READOUT_LAYERS), llm.hf_model.config.hidden_size),
        np.nan,
        np.float32,
    )
    for i, code in enumerate(df.code):
        if not code.strip():
            continue
        response = f"# Description: -\n# Code:\n```python\n{code}\n```"
        ids, n_p, offsets = fwd.encode(session, response)
        pos = fwd.code_positions(response, n_p, offsets)[:MAX_CODE_TOKENS]
        if not len(pos):
            continue
        states, _ = fwd.run(ids[:, : pos[-1] + 1])
        p = torch.as_tensor(pos, device=states[READOUT_LAYERS[0]].device)
        for j, layer in enumerate(READOUT_LAYERS):
            acts[i, j] = states[layer].float()[p].mean(0).cpu().numpy()
        if (i + 1) % 100 == 0:
            print(f"acts: {i + 1}/{len(df)}", flush=True)
    out = data_dir(EXPERIMENT, "processed", args.tag)
    np.savez(out / "acts.npz", acts=acts, layers=READOUT_LAYERS)
    df[["prompt", "k", "condition"]].to_csv(out / "acts_rows.csv", index=False)
    print(f"wrote {len(df)} rows", flush=True)


def corpus_probe(layer: int):
    """Ridge probe (standardised activations -> exploration_pct) on the
    working corpus algorithms, and its 5-fold CV Spearman correlation."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    proc = data_dir(EXPERIMENT, "processed", CORPUS["tag"])
    a = np.load(proc / "acts.npz")
    rows = pd.read_csv(proc / "rows.csv")
    keep = (rows.exploration_pct.notna() & ~rows.failed.astype(bool)).to_numpy()
    X = a["acts"][keep, list(a["layers"]).index(layer)].astype(np.float32)
    y = rows.exploration_pct.to_numpy()[keep]
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(1, 5, 9)))
    cv = cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0))
    return model.fit(X, y), float(spearmanr(cv, y)[0])


def slopes(df: pd.DataFrame, cols, n_boot: int, rng) -> pd.DataFrame:
    """Slope of each column on the signed dial dose (``none`` = 0), with
    fixed effects per unit (prompt x replicate) and a bootstrap over units."""
    x0 = df[(df.kind == "dial") | (df.kind == "none")]
    rows = []
    for col in cols:
        x = x0[x0[col].notna()]
        xc = x.dose - x.groupby("unit").dose.transform("mean")
        yc = x[col] - x.groupby("unit")[col].transform("mean")
        sxy = (xc * yc).groupby(x.unit).sum().to_numpy()
        sxx = (xc * xc).groupby(x.unit).sum().to_numpy()
        if sxx.sum() == 0:
            continue
        i = rng.integers(0, len(sxy), (n_boot, len(sxy)))
        boots = sxy[i].sum(1) / np.maximum(sxx[i].sum(1), 1e-12)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(
            {
                "metric": col,
                "n": len(x),
                "slope": sxy.sum() / sxx.sum(),
                "lo": lo,
                "hi": hi,
            }
        )
    return pd.DataFrame(rows)


def paired_delta(df, cond, col, n_boot, rng):
    """Mean paired change of ``col`` from ``none`` to ``cond`` (same unit)."""
    a = df[df.condition == cond].set_index("unit")[col]
    b = df[df.condition == "none"].set_index("unit")[col]
    d = (a - b.reindex(a.index)).dropna().to_numpy()
    if len(d) < 3:
        return np.nan, np.nan, np.nan
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return d.mean(), *np.percentile(boots, [2.5, 97.5])


def md(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)


def cond_order(df):
    order = ["none"] + sorted(
        [c for c in df.condition.unique() if c.startswith("dial@")],
        key=lambda c: float(c.split("@")[1]),
    )
    return order + sorted(c for c in df.condition.unique() if c.startswith("random@"))


def plot(df, cols, labels, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cols), figsize=(3.3 * len(cols), 3.0))
    colors = {"init": "#2a6fbb", "refine": "#c4572a"}
    for ax, col, lab in zip(np.atleast_1d(axes), cols, labels):
        for phase, g in df.groupby("phase"):
            d = g[(g.kind == "dial") | (g.kind == "none")]
            m = d.groupby("dose")[col].agg(["mean", "sem"])
            ax.errorbar(
                m.index,
                m["mean"],
                yerr=1.96 * m["sem"],
                marker="o",
                ms=4,
                lw=1.5,
                capsize=2,
                color=colors.get(phase, "k"),
                label=phase,
            )
            r = g[g.kind == "random"]
            if len(r):
                ax.errorbar(
                    r.dose.iloc[0] + 0.03,
                    r[col].mean(),
                    yerr=1.96 * r[col].sem(),
                    marker="x",
                    ms=6,
                    lw=0,
                    elinewidth=1,
                    capsize=2,
                    color=colors.get(phase, "k"),
                )
        ax.axvline(0, color="#bbb", lw=0.8, zorder=0)
        ax.set_xlabel("dial dose (x residual norm)")
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "o: dial (dose < 0: exploit, > 0: explore)   x: random direction", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stage_report(args):
    from scipy.stats import spearmanr

    camp = TRANSFER_CAMPAIGNS[args.tag]
    rng = np.random.default_rng(0)
    df = load(args.tag)
    df["score"] = np.where(
        df.failed, np.nan, pd.to_numeric(df.fitness, errors="coerce")
    )
    df["failed_f"] = df.failed.astype(float)
    df["probe_expl"] = np.nan
    acts_path = data_dir(EXPERIMENT, "processed", args.tag) / "acts.npz"
    probe_cv = None
    if acts_path.exists():
        a = np.load(acts_path)
        keys = pd.read_csv(acts_path.with_name("acts_rows.csv"))
        if (
            len(keys) == len(df)
            and (keys.condition.values == df.condition.values).all()
        ):
            model, probe_cv = corpus_probe(PROBE_LAYER)
            X = a["acts"][:, list(a["layers"]).index(PROBE_LAYER)]
            ok = np.isfinite(X).all(1)
            df.loc[ok, "probe_expl"] = model.predict(X[ok])
        else:
            print("acts.npz is stale; rerun --stage acts")
    traced = camp.problem == "bragg"
    # packing: strictly feasible rate (all samples) next to the repaired score
    beh = BEHAVIOUR if traced else ["feasible_f"]
    if not traced:
        feas = df["feasible"] if "feasible" in df else pd.Series(False, df.index)
        df["feasible_f"] = (feas == True).astype(float)  # noqa: E712
    # probe readout of working algorithms only (as the corpus probe)
    df["probe_expl_w"] = np.where(df.failed, np.nan, df.probe_expl)

    rows = []
    for (phase, cond), g in df.groupby(["phase", "condition"]):
        row = {
            "phase": phase,
            "condition": cond,
            "n": len(g),
            "failed": g.failed.mean(),
            "score": g.score.mean(),
            **{m: g[m].mean() for m in beh},
            "probe_expl": g.probe_expl_w.mean(),
            **{c: g.loc[~g.failed, c].mean() for c in CONCEPTS},
            "gen_tokens": g.n_gen_tokens.mean(),
        }
        rows.append(row)
    summ = pd.DataFrame(rows)
    order = {c: i for i, c in enumerate(cond_order(df))}
    summ = summ.sort_values(
        ["phase", "condition"],
        key=lambda s: s.map(order) if s.name == "condition" else s,
    )

    cols = beh + ["probe_expl_w", "score", "failed_f"] + CONCEPTS
    sl = []
    for phase, g in [("pooled", df), *df.groupby("phase")]:
        s = slopes(g, cols, args.n_boot, rng)
        s.insert(0, "phase", phase)
        sl.append(s)
    sl = pd.concat(sl)

    rand = []
    for c in [c for c in order if c != "none"]:
        for col in beh + ["probe_expl_w", "score", "failed_f"]:
            m, lo, hi = paired_delta(df, c, col, args.n_boot, rng)
            rand.append({"condition": c, "metric": col, "delta": m, "lo": lo, "hi": hi})
    rand = pd.DataFrame(rand)

    out = results_dir(EXPERIMENT, args.tag)
    summ.to_csv(out / "conditions.csv", index=False)
    sl.to_csv(out / "slopes.csv", index=False)
    rand.to_csv(out / "paired.csv", index=False)
    pcols = (["exploration_pct", "intensification"] if traced else ["feasible_f"]) + [
        "probe_expl_w",
        "score",
        "failed_f",
    ]
    labels = {
        "exploration_pct": "exploration % (measured)",
        "intensification": "intensification (measured)",
        "probe_expl_w": "exploration % (probe readout)",
        "score": SCORE[camp.problem] + " (working)",
        "failed_f": "failure rate",
        "feasible_f": "strictly feasible rate",
    }
    plot(df, pcols, [labels[c] for c in pcols], out / "dose.png")

    lines = [
        f"# Transfer screen: {args.tag} ({camp.problem})",
        "",
        f"Dial `{camp.direction}`, doses {camp.doses} x residual norm; random "
        f"direction at {camp.random_dose}. Score: {SCORE[camp.problem]} of the "
        "working algorithms. `probe_expl`: exploration % read out by a ridge "
        f"probe on the L{PROBE_LAYER} activations, trained on the MA-BBOB corpus"
        + (f" (CV Spearman there {probe_cv:.2f})" if probe_cv is not None else "")
        + "; concept rates among working algorithms.",
        "",
        "## Conditions",
        "",
        md(summ),
        "",
        "## Slopes on the signed dial dose",
        "",
        "Fixed effects per prompt x replicate; 95% CI from a bootstrap over them."
        " `failed_f`: failure rate (all samples); other metrics: working only.",
        "",
        md(sl),
        "",
        "## Paired change against the unsteered sample",
        "",
        md(rand),
        "",
    ]
    if traced and df.probe_expl_w.notna().sum() > 10:
        ok = df.probe_expl_w.notna() & df.exploration_pct.notna()
        rho = spearmanr(df.probe_expl_w[ok], df.exploration_pct[ok])[0]
        lines += [
            "## Probe readout vs measured exploration",
            "",
            f"Spearman {rho:.2f} over {ok.sum()} working algorithms.",
            "",
        ]
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", required=True, choices=list(TRANSFER_CAMPAIGNS))
    ap.add_argument("--stage", choices=["acts", "report"], default="report")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    (stage_acts if args.stage == "acts" else stage_report)(args)


if __name__ == "__main__":
    main()
