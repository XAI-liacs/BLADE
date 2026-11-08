"""Step 12: is there a feature for a reference to an undefined name?

Step 11 found that the SAE code predicts ``NameError`` far better than
the code text (AUROC 0.92 vs 0.72), mostly through one L21 feature. The
most common NameError (39 of 92) is ``name 'func' is not defined``: the
algorithm uses ``func`` (the objective, a parameter of ``__call__``) in a
helper method where it is not in scope. That gives a controlled test:
the same token, in scope or out of scope.

Every load of the name ``func`` in the code is labelled from the AST with
Python's scoping rules (bound if a parameter or assignment of an
enclosing function, or bound at module level; class bodies are skipped,
as in Python): ``valid`` or ``undefined``. The residual stream at those
tokens is encoded with the SAE at ``--layer``. Reports

* every feature's token-level AUROC for undefined vs valid references
  (is there a feature that tells them apart, and which?);
* the activation of ``--feature`` (step 11's NameError feature) at
  undefined / valid ``func`` tokens and elsewhere in the code;
* whether solutions with an undefined reference fail, and how.

Writes ``results/<tag>/undefined_names.csv`` (top features) and
``undefined_names.md``.
"""

import argparse
import ast
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp03_concept_features
sys.path.insert(0, str(HERE))

from common.paths import results_dir
from exp_config import EXPERIMENT, MODEL, SAES, SOURCE_TAGS
from records import code_span, load_records

step4 = importlib.import_module("04_concept_features")
step10 = importlib.import_module("10_sparse_failure_probe")
step11 = importlib.import_module("11_failure_types")


def _bound(fn) -> set[str]:
    """Names bound in a function's own scope (params, assignments, defs)."""
    a = fn.args
    names = {x.arg for x in a.args + a.kwonlyargs + a.posonlyargs}
    names |= {x.arg for x in (a.vararg, a.kwarg) if x}
    body = fn.body if isinstance(fn.body, list) else [fn.body]
    stack = list(body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
            continue  # own scope
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names |= {(x.asname or x.name).split(".")[0] for x in n.names}
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            names |= set(n.names)
        stack.extend(ast.iter_child_nodes(n))
    return names


def func_references(code: str, name: str = "func"):
    """``(start, end, valid, inside __call__)`` for every load of ``name``."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    module = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module.add(n.name)
        for m in (
            ast.walk(n) if not isinstance(n, (ast.FunctionDef, ast.ClassDef)) else []
        ):
            if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Store):
                module.add(m.id)
            elif isinstance(m, (ast.Import, ast.ImportFrom)):
                module |= {(x.asname or x.name).split(".")[0] for x in m.names}
    starts = np.cumsum([0] + [len(l) for l in code.splitlines(keepends=True)])
    out = []

    def visit(node, scopes, chain=()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for d in getattr(node, "decorator_list", []) + node.args.defaults:
                visit(d, scopes, chain)
            scopes = scopes + [_bound(node)]
            chain = chain + (getattr(node, "name", "<lambda>"),)
            for c in node.body if isinstance(node.body, list) else [node.body]:
                visit(c, scopes, chain)
            return
        if isinstance(node, ast.Name) and node.id == name:
            if isinstance(node.ctx, ast.Load):
                ok = any(name in s for s in scopes) or name in module
                a = int(starts[node.lineno - 1] + node.col_offset)
                out.append((a, a + len(name), ok, "__call__" in chain))
        for c in ast.iter_child_nodes(node):
            visit(c, scopes, chain)

    visit(tree, [])
    return out


def groups_table(meta, Z, feats):
    """Activity of ``feats`` at undefined refs, valid refs inside
    ``__call__`` and valid refs elsewhere (helpers that receive ``func`` as a
    parameter, closures)."""
    groups = {
        "undefined": ~meta.valid,
        "valid, in __call__": meta.valid & meta.in_call,
        "valid, elsewhere": meta.valid & ~meta.in_call,
    }
    rows = []
    for name, m in groups.items():
        m = m.to_numpy()
        row = {"group": name, "n": int(m.sum())}
        for f in feats:
            row[f"f{f}_active"] = float((Z[m, f] > 0).mean()) if m.any() else np.nan
            row[f"f{f}_mean"] = float(Z[m, f].mean()) if m.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="k100")
    ap.add_argument("--sae", default="k100", choices=list(SAES))
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--feature", type=int, default=7944)
    ap.add_argument("--n-valid", type=int, default=400)
    ap.add_argument("--n-helper", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    recs = load_records(list(SOURCE_TAGS)).sort_values("record_id")
    errors = step11.load_errors(list(SOURCE_TAGS))
    refs = {}
    for r in recs.itertuples():
        a, b = code_span(r.response)
        found = func_references(r.response[a:b])
        if found:
            refs[r.record_id] = [(s + a, e + a, ok, c) for s, e, ok, c in found]
    has_bad = [k for k, v in refs.items() if not all(ok for _, _, ok, _ in v)]
    only_ok = [k for k, v in refs.items() if all(ok for _, _, ok, _ in v)]
    # valid references outside __call__ (helpers taking func, closures) are
    # rare in a random sample, so solutions with them are added on purpose
    helper = [k for k in only_ok if any(not c for *_, c in refs[k])]
    rest = [k for k in only_ok if k not in set(helper)]
    pick = (
        has_bad
        + list(rng.choice(helper, min(args.n_helper, len(helper)), False))
        + list(rng.choice(rest, min(args.n_valid, len(rest)), False))
    )
    sub = recs.set_index("record_id").loc[pick]
    etype = {
        r: (step11.error_type(errors.get(r, "")) if f else "works")
        for r, f in zip(sub.index, sub.child_failed)
    }
    print(
        f"{len(refs)} solutions reference func; {len(has_bad)} have an undefined "
        f"reference; outcomes: {pd.Series([etype[k] for k in has_bad]).value_counts().to_dict()}",
        flush=True,
    )

    import torch
    from common.models import MODELS
    from common.sae import TopKSAE
    from records import Forward

    from iohblade.llm import HookedLLM

    preset = MODELS[MODEL]
    llm = HookedLLM(
        preset.hf_id,
        capture=False,
        device_map=args.device,
        chat_template_kwargs=preset.chat_template_kwargs,
    )
    fwd = Forward(llm, [args.layer])
    sae = TopKSAE.from_hub(SAES[args.sae], args.layer, args.device)
    tok_codes, tok_rows, other = [], [], []
    t0 = time.time()
    for k, (rid, r) in enumerate(sub.iterrows()):
        ids, n_p, offsets = fwd.encode(r.session, r.response)
        if ids.shape[1] > args.max_tokens:
            continue
        pos = fwd.code_positions(r.response, n_p, offsets)
        states, _ = fwd.run(ids)
        h = states[args.layer].float()
        vals, idx = sae.encode(h[torch.as_tensor(pos, device=h.device)])
        dense = torch.zeros(len(pos), sae.d_sae, device=h.device)
        dense.scatter_(1, idx, vals)
        tok_spans = offsets[pos - n_p]
        is_ref = np.zeros(len(pos), bool)
        for s, e, ok, in_call in refs[rid]:
            hit = np.flatnonzero((tok_spans[:, 0] < e) & (tok_spans[:, 1] > s))
            if len(hit) == 0:
                continue
            j = hit[0]
            is_ref[hit] = True
            tok_codes.append(dense[j].cpu().numpy().astype(np.float16))
            tok_rows.append(
                {
                    "record_id": rid,
                    "valid": ok,
                    "in_call": in_call,
                    "solution_has_undefined": rid in has_bad,
                    "outcome": etype[rid],
                }
            )
        f_other = dense[torch.as_tensor(~is_ref, device=h.device), args.feature]
        other.append(f_other.cpu().numpy())
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(sub)} ({time.time() - t0:.0f} s)", flush=True)

    Z = np.stack(tok_codes).astype(np.float32)
    meta = pd.DataFrame(tok_rows)
    y = (~meta.valid).to_numpy().astype(np.float32)
    # token-level AUROC of every feature, undefined vs valid references
    R = rankdata(Z, axis=0)
    auc = step4.auroc_all(R.astype(np.float32), y)
    order = np.argsort(-np.abs(auc - 0.5))[:20]
    top = pd.DataFrame(
        {
            "feature": order,
            "auroc_undefined_vs_valid": auc[order],
            "mean_undefined": Z[y == 1][:, order].mean(0),
            "mean_valid": Z[y == 0][:, order].mean(0),
            "p_active_undefined": (Z[y == 1][:, order] > 0).mean(0),
            "p_active_valid": (Z[y == 0][:, order] > 0).mean(0),
        }
    )
    out = results_dir(EXPERIMENT, args.tag)
    top.to_csv(out / "undefined_names.csv", index=False)
    meta.assign(target_feature=Z[:, args.feature]).to_csv(
        out / "undefined_names_tokens.csv", index=False
    )
    f = Z[:, args.feature]
    oth = np.concatenate(other)
    # within the solutions that have an undefined reference, valid vs undefined
    within = meta.solution_has_undefined.to_numpy()
    auc_f = step10.auroc(y, f)
    auc_within = step10.auroc(y[within], f[within])
    lines = [
        "# Undefined-name references",
        "",
        f"{len(refs)} solutions reference `func`; {len(has_bad)} have at least",
        "one reference where it is not in scope. Outcome of those:",
        "",
        step10.to_markdown(
            pd.Series([etype[k] for k in has_bad])
            .value_counts()
            .rename_axis("outcome")
            .reset_index()
        ),
        "",
        f"Tokens: {int(y.sum())} undefined, {int((1 - y).sum())} valid "
        f"`func` references (from {len(sub)} solutions).",
        "",
        f"## Step 11's NameError feature L{args.layer} f{args.feature}",
        "",
        f"- AUROC undefined vs valid `func` tokens: {auc_f:.3f}"
        f" (within solutions with an undefined reference: {auc_within:.3f})",
        f"- active at undefined refs {np.mean(f[y == 1] > 0):.2f}, at valid refs "
        f"{np.mean(f[y == 0] > 0):.2f}, at all other code tokens "
        f"{np.mean(oth > 0):.3f}",
        f"- mean activation: undefined {f[y == 1].mean():.2f}, valid "
        f"{f[y == 0].mean():.2f}, other tokens {oth.mean():.3f}",
        "",
        "## By reference group (activation > 0 / mean)",
        "",
        step10.to_markdown(groups_table(meta, Z, [args.feature, int(order[0])])),
        "",
        "## Features that best separate undefined from valid references",
        "",
        step10.to_markdown(top),
        "",
    ]
    (out / "undefined_names.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
