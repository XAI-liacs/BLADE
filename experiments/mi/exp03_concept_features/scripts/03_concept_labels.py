"""Step 3: label every algorithm with the metaheuristic concepts it uses.

Labels come from patterns on the code with comments and strings removed
(Python ``tokenize``; the raw code if it does not tokenise), so a concept
counts only when the code names or implements it. ``CONCEPTS`` maps each
concept to its regular expression; several are deliberately structural
(``F * (x_b - x_c)`` for DE mutation, ``sigma *=`` for step-size
adaptation) rather than lexical.

Writes ``data/processed/labels.csv`` (one row per record, one 0/1 column
per concept) and prints the prevalence of each concept per campaign.
"""

import argparse
import io
import json
import re
import sys
import tokenize
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp03_concept_features

from common.paths import data_dir
from exp_config import EXPERIMENT, SOURCE_EXPERIMENT, SOURCE_TAGS

CONCEPTS = {
    "population": r"\bpop(ulation)?(_size)?\b|\bpopulation\w*|\bswarm\w*",
    "differential_evolution": (
        r"\bmutant\w*|differential|"
        r"\bF\w*\s*\*\s*\(\s*\w+(\[[^\]]*\])?\s*-\s*\w+"  # F * (x_b - x_c)
    ),
    "crossover": r"crossover|recombin|\bCR\b",
    "particle_swarm": r"velocit|inertia|\bp_?best|\bg_?best|personal_best|social|cognitive",
    "covariance": r"covarian|\bnp\.cov\(|linalg\.(eigh?|cholesky)\(|\bC\s*=\s*np\.(eye|identity)",
    "step_size_adaptation": (
        r"(sigma|step_size|step|mutation_strength)\w*\s*(\*=|/=)|"
        r"success_(rate|ratio|count)|one_fifth|1\s*/\s*5"
    ),
    "restart": r"restart",
    "simulated_annealing": r"temperature|\btemp\b|anneal|cooling",
    "local_search": r"local_search|local_refine|nelder|hill_climb|pattern_search|"
    r"coordinate_(search|descent)|scipy\.optimize|\bminimize\(",
    "gradient": r"gradient|\bgrad\b|finite_diff|\bnp\.gradient\(",
    "levy_flight": r"levy|mantegna",
    "elitism": r"elit",
    "archive": r"archive",
    "opposition": r"opposit",
    "surrogate": r"surrogate|gaussian_process|GaussianProcess|kriging|\brbf\b",
    "quasi_random_init": r"latin|lhs\b|sobol|halton|\bqmc\b",
    "tournament_selection": r"tournament",
    "adaptive_parameters": r"self_adapt|adaptive_\w+|jade|shade|\bmu_?[FC]R?\b",
}


def strip_comments_and_strings(code: str) -> str:
    """Blanks comments and string literals, keeping everything else."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code
    lines = code.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    chars = list(code)
    drop = {tokenize.COMMENT, tokenize.STRING}
    drop |= {getattr(tokenize, n) for n in ("FSTRING_MIDDLE",) if hasattr(tokenize, n)}
    for t in toks:
        if t.type in drop:
            a = starts[t.start[0] - 1] + t.start[1]
            b = starts[t.end[0] - 1] + t.end[1]
            for i in range(a, min(b, len(chars))):
                if chars[i] != "\n":
                    chars[i] = " "
    return "".join(chars)


def labels(code: str) -> dict[str, int]:
    text = strip_comments_and_strings(code)
    return {c: int(re.search(p, text) is not None) for c, p in CONCEPTS.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-tags", nargs="*", default=list(SOURCE_TAGS))
    args = ap.parse_args()
    rows = []
    for tag in args.source_tags:
        proc = data_dir(SOURCE_EXPERIMENT, "processed", tag)
        with open(proc / "code.jsonl") as f:
            for line in f:
                c = json.loads(line)
                rows.append(
                    {"record_id": c["record_id"], "tag": tag, **labels(c["code"])}
                )
    df = pd.DataFrame(rows)
    out = data_dir(EXPERIMENT, "processed")
    df.to_csv(out / "labels.csv", index=False)
    print(df.groupby("tag")[list(CONCEPTS)].mean().T.round(3).to_string())
    print(f"{len(df)} records; wrote {out / 'labels.csv'}")


if __name__ == "__main__":
    main()
