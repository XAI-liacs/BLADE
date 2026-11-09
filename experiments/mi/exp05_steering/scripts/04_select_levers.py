"""Step 4: pick the levers for step 5 from a screen campaign.

A direction is a *lever* at a dose if steering it

* adds its concept: the paired increase over the unsteered child has a
  95% bootstrap CI above 0 and is at least ``--min-gain``;
* without breaking the code: the failure rate rises by at most
  ``--max-fail`` over the unsteered children.

Per direction the qualifying dose with the largest concept gain is kept,
and per concept the direction with the largest gain (two levers for the
same concept would make the operator less diverse, not more).

Several screens (``--tags``) are pooled: every (direction, dose) of every
screen is a candidate, compared with the unsteered children of its own
screen. Reads ``results/<tag>/summary.csv`` (``03_analyze.py``); writes
``data/processed/<first tag>/levers.json`` (each lever names the screen
whose ``directions.npz`` holds it) and ``results/<first tag>/levers.md``.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from common.paths import data_dir, results_dir
from exp_config import CAMPAIGNS, EXPERIMENT


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tags", nargs="+", default=["screen"], choices=list(CAMPAIGNS))
    ap.add_argument("--min-gain", type=float, default=0.2)
    ap.add_argument("--max-fail", type=float, default=0.15)
    args = ap.parse_args()
    tag = args.tags[0]
    proc = data_dir(EXPERIMENT, "processed", tag)
    info, frames, base_fail = {}, [], {}
    for t in args.tags:
        p = data_dir(EXPERIMENT, "processed", t) / "campaign.json"
        for name, v in json.loads(p.read_text())["directions"].items():
            info[(t, name)] = v
        f = pd.read_csv(results_dir(EXPERIMENT, t) / "summary.csv").assign(source=t)
        base_fail[t] = float(f[f.condition == "none"].failed.iloc[0])
        frames.append(f)
    s = pd.concat(frames, ignore_index=True)
    s = s[s.concept.notna() & (s.concept != "")].copy()
    s["lever"] = (
        (s.d_concept_lo > 0)
        & (s.d_concept >= args.min_gain)
        & (s.d_failed <= args.max_fail)
    )
    table = s[
        [
            "source",
            "direction",
            "dose",
            "concept",
            "concept_rate",
            "d_concept",
            "d_concept_lo",
            "d_concept_hi",
            "failed",
            "d_failed",
            "sim_none",
            "lever",
        ]
    ].sort_values(["concept", "direction", "dose"])
    picks = (
        s[s.lever]
        .sort_values("d_concept", ascending=False)
        .drop_duplicates(["source", "direction"])
        .drop_duplicates("concept")
    )
    levers = [
        {
            "direction": r.direction,
            "source": r.source,
            "concept": r.concept,
            "layer": info[(r.source, r.direction)]["layer"],
            "feature": info[(r.source, r.direction)]["feature"],
            "dose": float(r.dose),
            "scale": info[(r.source, r.direction)]["scale"],
            "d_concept": float(r.d_concept),
            "failed": float(r.failed),
        }
        for r in picks.itertuples()
    ]
    (proc / "levers.json").write_text(json.dumps(levers, indent=1))
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = [
        f"# Lever screen `{' + '.join(args.tags)}`",
        "",
        f"Lever: concept gain CI above 0 and >= {args.min_gain}, failure rate"
        f" up by at most {args.max_fail}. Unsteered failure rate: "
        + ", ".join(f"{t} {v:.3f}" for t, v in base_fail.items())
        + ".",
        "",
        "| " + " | ".join(table.columns) + " |",
        "|" + "---|" * len(table.columns),
    ]
    for _, r in table.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    lines += [
        "",
        f"## Selected ({len(levers)})",
        "",
        *[
            f"- `{l['direction']}` (L{l['layer']} f{l['feature']}) at dose "
            f"{l['dose']}: {l['concept']} +{l['d_concept']:.2f}, "
            f"failures {l['failed']:.2f}"
            for l in levers
        ],
        "",
    ]
    (results_dir(EXPERIMENT, tag) / "levers.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
