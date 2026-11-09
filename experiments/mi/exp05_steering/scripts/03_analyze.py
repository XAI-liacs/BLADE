"""Step 3: what does each steering condition do to the generated algorithms?

Per sample (``data/raw/<tag>/shard-*.jsonl``):

* concept labels of the code (experiment 3's regex labels) and undefined
  references (experiment 3, step 13: ``func`` out of scope, a name bound
  in another function, a name bound nowhere, ``self.<attr>`` never
  assigned);
* similarity of the code to the unsteered sample of the same prompt and
  replicate (``difflib`` ratio; with common random numbers an
  ineffective steer reproduces it exactly).

Per condition: failure rate (``NoCodeException`` counts as a failure),
fitness with failures as 0 (``aocc``) and of working code, the rate of the
direction's target concept, the rate of undefined references, the
similarity to ``none``, and paired differences to ``none`` with 95%
bootstrap CIs over prompts.

Writes ``results/<tag>/samples.csv``, ``summary.csv`` and ``summary.md``
(and ``dose_response.png`` if there is more than one positive dose).
"""

import argparse
import difflib
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
from exp_config import BEHAVIOUR_METRICS, CAMPAIGNS, EXPERIMENT

# dist_to_best is unbounded (algorithms that leave the box); not shown.
BEHAVIOUR_SHOWN = ["exploration_pct", "intensification", "last_improvement"]

prep = importlib.import_module("01_prepare")
labels_mod = prep.exp03_module("03_concept_labels")
step13 = prep.exp03_module("13_undefined_generality")

UNDEF = ("func", "elsewhere", "never", "self_attr")


def sample_features(code: str) -> dict:
    code = code or ""
    lab = labels_mod.labels(code) if code else {}
    refs = step13.references(code) if code else []
    groups = {x["group"] for x in refs}
    out = {f"c_{k}": int(v) for k, v in lab.items()}
    for g in UNDEF:
        out[f"u_{g}"] = int(g in groups)
    out["u_name"] = int(bool(groups & {"func", "elsewhere", "never"}))
    out["code_chars"] = len(code)
    return out


def load_samples(tag: str) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir(EXPERIMENT, "raw", tag).glob("shard-*.jsonl")):
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("infra"):
                continue
            rows.append(r)
    df = pd.DataFrame(rows)
    df["code"] = df.get("code", pd.Series([""] * len(df))).fillna("")
    df["failed"] = df.failed.fillna(True).astype(bool)
    df["aocc"] = np.where(df.failed, 0.0, df.fitness.fillna(0.0))
    for m in BEHAVIOUR_METRICS:
        beh = df.get("behaviour", pd.Series([None] * len(df)))
        df[f"b_{m}"] = [b.get(m) if isinstance(b, dict) else np.nan for b in beh]
    feats = pd.DataFrame([sample_features(c) for c in df.code], index=df.index)
    df = pd.concat([df, feats], axis=1)
    base = df[df.condition == "none"].set_index(["record_id", "k"]).code
    sim = []
    for r in df.itertuples():
        b = base.get((r.record_id, r.k))
        if b is None or not r.code or not b:
            sim.append(np.nan)
        else:
            sim.append(difflib.SequenceMatcher(None, r.code, b, autojunk=False).ratio())
    df["sim_none"] = sim
    return df


def paired_delta(df, cond, col, n_boot, rng):
    """Mean of (cond - none) over prompts x replicates, CI over prompts."""
    a = df[df.condition == cond].set_index(["record_id", "k"])[col].astype(float)
    b = df[df.condition == "none"].set_index(["record_id", "k"])[col].astype(float)
    d = (a - b.reindex(a.index)).dropna().astype(float)
    if d.empty:
        return np.nan, np.nan, np.nan
    by = d.groupby(level=0).mean()
    w = d.groupby(level=0).size()
    vals, ws = by.to_numpy(), w.to_numpy()
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(vals), len(vals))
        boots.append(np.average(vals[i], weights=ws[i]))
    return float(d.mean()), *np.percentile(boots, [2.5, 97.5])


def summarise(df, info, n_boot, rng) -> pd.DataFrame:
    rows = []
    for cond, g in df.groupby("condition", sort=False):
        direction = g.direction.iloc[0]
        concept = info.get(direction, {}).get("concept") if direction else None
        metric = info.get(direction, {}).get("behaviour") if direction else None
        row = {
            "condition": cond,
            "direction": direction or cond,
            "dose": float(g.dose.iloc[0]),
            "n": len(g),
            "steer_norm": float(g.steer_norm.iloc[0]),
            "nocode": float((g.error == "NoCodeException").mean()),
            "failed": float(g.failed.mean()),
            "aocc": float(g.aocc.mean()),
            "fitness_working": float(g.fitness[~g.failed].mean()),
            "n_gen_tokens": float(g.n_gen_tokens.mean()),
            "sim_none": float(g.sim_none.mean()),
            "undefined_name": float(g.u_name.mean()),
            "undefined_attr": float(g.u_self_attr.mean()),
            "concept": concept or "",
            "concept_rate": float(g[f"c_{concept}"].mean()) if concept else np.nan,
            "behaviour": metric or "",
        }
        for m in BEHAVIOUR_SHOWN:
            row[f"{m}"] = float(g[f"b_{m}"].mean())
        if cond != "none":
            for col, key in (
                ("failed", "d_failed"),
                ("aocc", "d_aocc"),
                ("u_name", "d_undefined_name"),
            ):
                d, lo, hi = paired_delta(df, cond, col, n_boot, rng)
                row.update({key: d, f"{key}_lo": lo, f"{key}_hi": hi})
            if concept:
                d, lo, hi = paired_delta(df, cond, f"c_{concept}", n_boot, rng)
                row.update(d_concept=d, d_concept_lo=lo, d_concept_hi=hi)
            for m in BEHAVIOUR_SHOWN:  # pairs where both children work
                d, lo, hi = paired_delta(df, cond, f"b_{m}", n_boot, rng)
                row.update({f"d_{m}": d, f"d_{m}_lo": lo, f"d_{m}_hi": hi})
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_o"] = out.condition.ne("none")
    return out.sort_values(["_o", "direction", "dose"]).drop(columns="_o")


def dose_slopes(df, n_boot, rng) -> pd.DataFrame:
    """Slope of each behaviour metric on signed dose, per direction.

    All doses of a direction plus the unsteered children, working children
    only, with prompt fixed effects (both sides centred per prompt);
    95% CI from a bootstrap over prompts.
    """
    base = df[df.condition == "none"].assign(dose=0.0)
    steered = df[df.direction.notna() & (df.condition != "none")]
    steered = steered[steered.direction != "resample"]
    rows = []
    for d, g in steered.groupby("direction", sort=False):
        row = {"direction": d, "doses": len(g.dose.unique())}
        for m in BEHAVIOUR_SHOWN:
            x = pd.concat([g, base])
            x = x[x[f"b_{m}"].notna()]
            xc = x.dose - x.groupby("record_id").dose.transform("mean")
            yc = x[f"b_{m}"] - x.groupby("record_id")[f"b_{m}"].transform("mean")
            sxy = (xc * yc).groupby(x.record_id).sum().to_numpy()
            sxx = (xc * xc).groupby(x.record_id).sum().to_numpy()
            if sxx.sum() == 0:
                continue
            i = rng.integers(0, len(sxy), (n_boot, len(sxy)))
            boots = sxy[i].sum(1) / np.maximum(sxx[i].sum(1), 1e-12)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            row.update(
                {
                    "n": len(x),
                    f"slope_{m}": sxy.sum() / sxx.sum(),
                    f"slope_{m}_lo": lo,
                    f"slope_{m}_hi": hi,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def concept_matrix(df, info) -> pd.DataFrame:
    """Rate of every direction's concept under every condition."""
    concepts = sorted({v["concept"] for v in info.values() if v.get("concept")})
    cols = [f"c_{c}" for c in concepts]
    m = df.groupby("condition")[cols].mean()
    m.columns = concepts
    return m.reset_index()


def plot(summary, path, unit="x feature peak activation"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.plotting import INK, NEUTRAL, SERIES, setup

    setup()
    s = summary[~summary.direction.isin(["none", "resample"])]
    base = summary[summary.direction == "none"].iloc[0]
    resample = summary[summary.direction == "resample"]
    dirs = list(dict.fromkeys(s.direction))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    panels = (
        ("concept_rate", "target concept in the code"),
        ("failed", "failure rate"),
        ("sim_none", "similarity to unsteered code"),
    )
    colors, styles = {}, {}
    i, j = 0, 0
    for d in dirs:
        if d.startswith("random"):
            colors[d], styles[d] = NEUTRAL, ("--", ":", "-.")[j % 3]
            j += 1
        else:
            colors[d], styles[d] = SERIES[i % len(SERIES)], "-"
            i += 1
    for ax, (col, title) in zip(axes, panels):
        for d in dirs:
            g = s[s.direction == d]
            if g[col].isna().all():
                continue
            if col == "concept_rate":
                b = g.concept_rate.iloc[0] - g.d_concept.iloc[0]
            elif col == "sim_none":
                b = 1.0
            else:
                b = base[col]
            xy = sorted([(0.0, b)] + list(zip(g.dose, g[col])))
            ax.plot(*zip(*xy), styles[d], marker="o", ms=4, color=colors[d], label=d)
        if col != "concept_rate" and len(resample):
            ax.axhline(
                resample[col].iloc[0], color=INK, lw=0.8, ls=":", label="resample"
            )
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel(f"dose ({unit})")
    fig.legend(
        *axes[1].get_legend_handles_labels(),
        loc="lower center",
        ncol=len(dirs) + 1,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_behaviour(summary, path, unit):
    """Paired change of the behaviour metrics against signed dose."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.plotting import INK, NEUTRAL, SERIES, setup

    setup()
    s = summary[~summary.direction.isin(["none", "resample"])]
    dirs = list(dict.fromkeys(s.direction))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    i = j = 0
    for d in dirs:
        g = s[s.direction == d].sort_values("dose")
        if d.startswith("random"):
            c, ls = NEUTRAL, ("--", ":", "-.")[j % 3]
            j += 1
        else:
            c, ls = SERIES[i % len(SERIES)], "-"
            i += 1
        for ax, m in zip(axes, BEHAVIOUR_SHOWN):
            x = np.r_[0.0, g.dose]
            y = np.r_[0.0, g[f"d_{m}"]]
            o = np.argsort(x)
            ax.plot(x[o], y[o], ls, marker="o", ms=3.5, color=c, label=d, lw=1.4)
            ax.vlines(g.dose, g[f"d_{m}_lo"], g[f"d_{m}_hi"], color=c, lw=0.8)
    for ax, m in zip(axes, BEHAVIOUR_SHOWN):
        ax.axhline(0, color=INK, lw=0.6)
        ax.set_title(f"change in {m}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel(f"signed dose ({unit})")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        ncol=min(len(dirs), 4),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def to_markdown(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="pilot", choices=list(CAMPAIGNS))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    proc = data_dir(EXPERIMENT, "processed", args.tag)
    info = json.loads((proc / "campaign.json").read_text())["directions"]
    out = results_dir(EXPERIMENT, args.tag)
    out.mkdir(parents=True, exist_ok=True)

    camp = CAMPAIGNS[args.tag]
    unit = {
        "peak": "x feature peak activation",
        "residual": "x residual norm",
        "diff": "x mean-difference norm",
    }.get(getattr(camp, "dose_unit", "peak"), "")
    df = load_samples(args.tag)
    keep = [c for c in df.columns if c not in ("code", "response", "error")]
    df[keep + ["error"]].assign(error=df.error.fillna("").str[-200:]).to_csv(
        out / "samples.csv", index=False
    )
    summary = summarise(df, info, args.n_boot, rng)
    summary.to_csv(out / "summary.csv", index=False)
    cm = concept_matrix(df, info)
    show = [
        "condition",
        "n",
        "steer_norm",
        "nocode",
        "failed",
        "d_failed",
        "d_failed_lo",
        "d_failed_hi",
        "aocc",
        "d_aocc",
        "fitness_working",
        "sim_none",
        "n_gen_tokens",
        "concept",
        "concept_rate",
        "d_concept",
        "d_concept_lo",
        "d_concept_hi",
        "undefined_name",
        "d_undefined_name",
        "d_undefined_name_lo",
        "d_undefined_name_hi",
        "undefined_attr",
        *[x for m in BEHAVIOUR_SHOWN for x in (m, f"d_{m}", f"d_{m}_lo", f"d_{m}_hi")],
    ]
    show = [c for c in show if c in summary]
    lines = [
        f"# Steering campaign `{args.tag}`",
        "",
        f"{df.record_id.nunique()} prompts, {len(df)} samples. Doses {unit};",
        "`steer_norm` is the norm of the added",
        "vector (residual norms: "
        + ", ".join(
            f"L{v['layer']} {v['residual_norm']:.0f}"
            for v in {
                v["layer"]: v for v in info.values() if "residual_norm" in v
            }.values()
        )
        + ").",
        "",
        to_markdown(summary[show]),
        "",
        "## Concept rates by condition",
        "",
        to_markdown(cm),
        "",
    ]
    if (summary.dose < 0).any():
        sl = dose_slopes(df, args.n_boot, rng)
        sl.to_csv(out / "slopes.csv", index=False)
        lines += [
            "## Behaviour slopes on signed dose",
            "",
            "Change per unit dose (working children, prompt fixed effects,"
            " CI over prompts).",
            "",
            to_markdown(sl),
            "",
        ]
    (out / "summary.md").write_text("\n".join(lines))
    if (summary.dose > 0).groupby(summary.direction).sum().max() > 1:
        plot(summary, out / "dose_response.png", unit)
    if (summary.behaviour != "").any():
        plot_behaviour(summary, out / "behaviour_dose.png", unit)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
