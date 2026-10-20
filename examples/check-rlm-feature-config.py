"""Recover the RLM feature settings a checkpoint was trained with.

LLaMEA's RLM screener needs ``rlm_feature_mode``, ``rlm_n_lhs_points``,
``rlm_lhs_seed``, ``rlm_include_description`` and ``rlm_include_configspace``
to match what ``llamea.rlm_surrogate.data_pipeline`` used when it built the
checkpoint's training examples. None of these are stored in the checkpoint's
``config.yaml``, and a mismatch gives meaningless predictions, not an error.

This script reads the training examples themselves (``train.jsonl`` written
by ``data_pipeline.py``) and recovers the settings from them:

* feature mode and LHS point count come from the problem-text header
  (``LHS[200pts,dim=5]: ...``, ``STATS[...]``, ``family: ...``);
* the LHS seed is found by regenerating the problem text for the example's
  instance with candidate seeds and checking for an exact string match;
* the description/configspace flags come from ``stats.json`` when present,
  otherwise from the section headers in the examples.

Usage (``--checkpoint-dir`` finds the data via the checkpoint's
``train_manifest.json``; use ``--data-dir`` if that path has moved):
    uv run python examples/check-rlm-feature-config.py \\
        --checkpoint-dir /path/to/checkpoints/production_lhs200 \\
        --expect-mode lhs --expect-points 200 --expect-seed 0

Exits with status 1 if any ``--expect-*`` value does not match.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import llamea.rlm_surrogate.problem_instances as pi
from llamea.rlm_surrogate.problem_instances import (
    ProblemInstance,
    compute_problem_feature_text,
)

import iohblade.benchmarks.BBOB.mabbob as blade_mabbob

PROBLEM_SEP = "\n\n# Problem\n"

# The installed llamea package lacks its MA-BBOB tables; BLADE ships
# byte-identical copies.
if not (pi.MA_BBOB_DATA_DIR / "weights.csv").exists():
    pi.MA_BBOB_DATA_DIR = Path(blade_mabbob.__file__).parent / "mabbob"
HEADER_RE = re.compile(r"(LHS|STATS)\[(\d+)pts,dim=(\d+)\]")


def detect_mode(feature_text: str) -> tuple[str, int | None, int | None]:
    """Returns (feature_mode, n_points, dim) parsed from the problem text."""
    has_meta = feature_text.startswith("family:")
    m = HEADER_RE.search(feature_text)
    if m is None:
        if has_meta:
            dim = re.search(r"dim: (\d+)", feature_text)
            return "meta", None, int(dim.group(1)) if dim else None
        raise ValueError(f"Unrecognised problem text: {feature_text[:80]!r}")
    kind = "lhs" if m.group(1) == "LHS" else "lhs_stats"
    mode = f"meta+{kind}" if has_meta else kind
    return mode, int(m.group(2)), int(m.group(3))


def find_seed(
    example: dict,
    feature_text: str,
    mode: str,
    n_points: int,
    dim: int,
    seeds: list[int],
    max_iid: int,
) -> tuple[int, int | None] | None:
    """Brute-forces (seed, iid) so the regenerated text equals the logged one.

    ``iid`` is not stored in the examples, so BBOB rows try iids 1..max_iid.
    """
    kind = example.get("instance_kind")
    fid_or_idx = example.get("instance_fid_or_idx")
    if kind not in ("bbob", "ma_bbob") or fid_or_idx is None or fid_or_idx < 0:
        return None
    iids = range(1, max_iid + 1) if kind == "bbob" else [None]
    for seed in seeds:
        for iid in iids:
            inst = ProblemInstance(kind=kind, dim=dim, fid_or_idx=fid_or_idx, iid=iid)
            text = compute_problem_feature_text(
                inst, n_points=n_points, seed=seed, mode=mode
            )
            if text == feature_text:
                return seed, iid
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--data-dir",
        help="data_pipeline.py --output-dir (holds train.jsonl and stats.json).",
    )
    src.add_argument(
        "--checkpoint-dir",
        help="train.py --output-dir; its train_manifest.json names train.jsonl.",
    )
    p.add_argument("--max-examples", type=int, default=2000)
    p.add_argument("--seed-checks", type=int, default=5)
    p.add_argument("--max-iid", type=int, default=15)
    p.add_argument("--candidate-seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--expect-mode", default=None)
    p.add_argument("--expect-points", type=int, default=None)
    p.add_argument("--expect-seed", type=int, default=None)
    args = p.parse_args()

    if args.checkpoint_dir:
        manifest = Path(args.checkpoint_dir) / "train_manifest.json"
        if not manifest.exists():
            print(
                f"No train_manifest.json in {args.checkpoint_dir}; "
                "pass --data-dir instead.",
                file=sys.stderr,
            )
            return 2
        data_dir = Path(json.loads(manifest.read_text())["train_path"]).parent
    else:
        data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        print(f"No train.jsonl in {data_dir}", file=sys.stderr)
        return 2

    examples = []
    with open(train_path) as fh:
        for line in fh:
            examples.append(json.loads(line))
            if len(examples) >= args.max_examples:
                break

    modes, points, has_desc, has_cs = set(), set(), False, False
    parsed = []
    for ex in examples:
        base, sep, feature_text = ex["x"].rpartition(PROBLEM_SEP)
        if not sep:
            continue
        mode, n_pts, dim = detect_mode(feature_text)
        modes.add(mode)
        points.add(n_pts)
        has_desc |= base.startswith("# Description\n")
        has_cs |= "# Config space\n" in base
        parsed.append((ex, feature_text, mode, n_pts, dim))

    if not parsed:
        print(
            "No examples with a '# Problem' section: was the data built with "
            "--target aucs_per_instance?",
            file=sys.stderr,
        )
        return 2

    seeds = list(
        dict.fromkeys(
            ([args.expect_seed] if args.expect_seed is not None else [])
            + args.candidate_seeds
        )
    )
    found_seeds, checked = set(), 0
    if all(m != "meta" for m in modes):
        for ex, text, mode, n_pts, dim in parsed:
            if checked >= args.seed_checks:
                break
            hit = find_seed(ex, text, mode, n_pts, dim, seeds, args.max_iid)
            checked += 1
            found_seeds.add(hit[0] if hit else None)

    include_desc, include_cs = has_desc, has_cs
    stats_path = data_dir / "stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        include_desc = stats.get("include_description", include_desc)
        include_cs = stats.get("include_configspace", include_cs)

    print(f"examples read          : {len(parsed)} ({train_path})")
    print(f"feature_mode           : {sorted(modes)}")
    print(f"n_lhs_points           : {sorted(p for p in points if p is not None)}")
    print(f"lhs_seed (checked {checked}) : {sorted(found_seeds, key=str)}")
    print(f"include_description    : {include_desc}")
    print(f"include_configspace    : {include_cs}")

    ok = len(modes) == 1 and len(points) == 1
    if None in found_seeds or len(found_seeds) > 1:
        print(
            "WARNING: could not pin down a single LHS seed "
            "(try --candidate-seeds / --max-iid)."
        )
        ok = False
    if args.expect_mode is not None and modes != {args.expect_mode}:
        print(f"MISMATCH: expected feature_mode={args.expect_mode}")
        ok = False
    if args.expect_points is not None and points != {args.expect_points}:
        print(f"MISMATCH: expected n_lhs_points={args.expect_points}")
        ok = False
    if args.expect_seed is not None and found_seeds != {args.expect_seed}:
        print(f"MISMATCH: expected lhs_seed={args.expect_seed}")
        ok = False
    if ok and len(found_seeds) == 1:
        (seed,) = found_seeds
        print("\nLLaMEA kwargs for this checkpoint:")
        print(f"    rlm_feature_mode={next(iter(modes))!r},")
        print(f"    rlm_n_lhs_points={next(iter(points))},")
        print(f"    rlm_lhs_seed={seed},")
        print(f"    rlm_include_description={include_desc},")
        print(f"    rlm_include_configspace={include_cs},")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
