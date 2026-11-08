"""Step 13: do the out-of-scope features fire on every undefined reference?

Step 12 found two L21 features (f7944, f9410) that fire on ``func`` where
it is not in scope and stay silent on in-scope ``func``. That is one
name. This step labels *every* name load in the code with Python's
scoping rules (function scopes, class bodies, module, builtins) and every
``self.<attr>`` load against the attributes the class assigns or defines:

* ``func``: ``func`` out of scope (step 12's case);
* ``elsewhere``: another name that is bound in some other function of the
  code (a local of another method, e.g. ``best_x``);
* ``never``: a name bound nowhere in the code (a missing import such as
  ``norm`` / ``minimize``, or a typo);
* ``self_attr``: ``self.<attr>`` that the class never assigns and does not
  define (classes with base classes are skipped);
* ``valid``: every other name / ``self`` attribute load.

The labels are checked against the logged errors (``name 'x' is not
defined`` / ``has no attribute 'x'``). The residual stream at the
reference tokens is SAE-encoded at ``--layer``; per group, the activity of
the step 12 features, and per group the features that best separate it
from valid references of the same kind (names / attributes).

``--scan`` only labels the code (CPU) and prints the label check.

Writes ``results/<tag>/undefined_generality.csv`` and
``undefined_generality.md``.
"""

import argparse
import ast
import builtins
import importlib
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # experiments/mi
sys.path.insert(0, str(HERE.parent))  # exp03_concept_features
sys.path.insert(0, str(HERE))

from common.paths import results_dir
from exp_config import EXPERIMENT, MODEL, SAES, SOURCE_TAGS
from records import code_span, load_records

step10 = importlib.import_module("10_sparse_failure_probe")
step11 = importlib.import_module("11_failure_types")

BUILTINS = set(dir(builtins)) | {"__name__", "__file__", "__class__"}
GROUPS = ("func", "elsewhere", "never", "self_attr")


def _targets(node) -> set[str]:
    """Names a single statement / node binds in the current scope."""
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return {node.id}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(x.asname or x.name).split(".")[0] for x in node.names}
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return set(node.names)
    if isinstance(node, ast.ExceptHandler) and node.name:
        return {node.name}
    if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
        return {node.name}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    return set()


def scope_bound(body) -> set[str]:
    """Names bound in a scope whose statements are ``body`` (nested
    functions / classes / lambdas are their own scopes)."""
    names, stack = set(), list(body)
    while stack:
        n = stack.pop()
        names |= _targets(n)
        if isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            continue
        stack.extend(ast.iter_child_nodes(n))
    return names


def fn_bound(fn) -> set[str]:
    a = fn.args
    names = {x.arg for x in a.args + a.kwonlyargs + a.posonlyargs}
    names |= {x.arg for x in (a.vararg, a.kwarg) if x}
    return names | scope_bound(fn.body if isinstance(fn.body, list) else [fn.body])


def references(code: str):
    """Every name load and ``self.<attr>`` load in ``code``.

    Returns dicts with ``start``, ``end`` (character span of the token),
    ``name``, ``kind`` ("name" / "attr"), ``group`` (one of ``GROUPS`` or
    "valid") and ``in_call`` (inside ``__call__``).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    module = scope_bound(tree.body)
    every_fn = set()  # names bound in any function scope
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            every_fn |= fn_bound(n)
    starts = np.cumsum([0] + [len(l) for l in code.splitlines(keepends=True)])
    out = []

    def span_end(node, text):
        b = int(starts[node.end_lineno - 1] + node.end_col_offset)
        return b - len(text), b

    def visit(node, scopes, chain, cls):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for d in getattr(node, "decorator_list", []) + node.args.defaults:
                visit(d, scopes, chain, cls)
            scopes = scopes + [fn_bound(node)]
            chain = chain + (getattr(node, "name", "<lambda>"),)
            for c in node.body if isinstance(node.body, list) else [node.body]:
                visit(c, scopes, chain, cls)
            return
        if isinstance(node, ast.ClassDef):
            for d in node.bases + node.decorator_list:
                visit(d, scopes, chain, cls)
            info = None
            if not node.bases:  # inherited attributes cannot be checked
                attrs = scope_bound(node.body)
                for m in ast.walk(node):
                    if (
                        isinstance(m, ast.Attribute)
                        and isinstance(m.value, ast.Name)
                        and m.value.id == "self"
                        and isinstance(m.ctx, (ast.Store, ast.Del))
                    ):
                        attrs.add(m.attr)
                    elif (
                        isinstance(m, ast.Call)
                        and isinstance(m.func, ast.Name)
                        and m.func.id == "setattr"
                    ):
                        info = None
                        break
                else:
                    info = attrs
            # class body names are visible in the class body only
            body_scope = scope_bound(node.body)
            for c in node.body:
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit(c, scopes, chain, info)
                else:
                    visit(c, scopes + [body_scope], chain, info)
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            ok = (
                any(node.id in s for s in scopes)
                or node.id in module
                or node.id in BUILTINS
            )
            if ok:
                group = "valid"
            elif node.id == "func":
                group = "func"
            elif node.id in every_fn:
                group = "elsewhere"
            else:
                group = "never"
            a = int(starts[node.lineno - 1] + node.col_offset)
            out.append(
                {
                    "start": a,
                    "end": a + len(node.id),
                    "name": node.id,
                    "kind": "name",
                    "group": group,
                    "in_call": "__call__" in chain,
                }
            )
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and cls is not None
            and not node.attr.startswith("__")
        ):
            a, b = span_end(node, node.attr)
            out.append(
                {
                    "start": a,
                    "end": b,
                    "name": node.attr,
                    "kind": "attr",
                    "group": "valid" if node.attr in cls else "self_attr",
                    "in_call": "__call__" in chain,
                }
            )
        for c in ast.iter_child_nodes(node):
            visit(c, scopes, chain, cls)

    visit(tree, [], (), None)
    return out


def logged_missing(message: str) -> tuple[str, str] | None:
    """(kind, name) of a logged NameError / AttributeError, else None."""
    m = re.findall(r"name '(\w+)' is not defined", message)
    if m:
        return "name", m[-1]
    m = re.findall(r"object has no attribute '(\w+)'", message)
    if m:
        return "attr", m[-1]
    return None


def label_check(recs, refs, errors):
    """How well do the labels match the logged errors?"""
    rows = []
    for r in recs.itertuples():
        found = refs.get(r.record_id, [])
        bad = {(x["kind"], x["name"]) for x in found if x["group"] != "valid"}
        logged = logged_missing(errors.get(r.record_id, "")) if r.child_failed else None
        rows.append(
            {
                "record_id": r.record_id,
                "flagged": bool(bad),
                "flag_name": any(k == "name" for k, _ in bad),
                "flag_attr": any(k == "attr" for k, _ in bad),
                "logged": logged is not None,
                "logged_kind": logged[0] if logged else "",
                "logged_hit": logged in bad if logged else False,
                "failed": bool(r.child_failed),
            }
        )
    df = pd.DataFrame(rows)
    out = []
    for kind in ("name", "attr"):
        f = df[f"flag_{kind}"]
        lg = df.logged_kind == kind
        out.append(
            {
                "kind": kind,
                "n_flagged": int(f.sum()),
                "flagged_fail": float(df.failed[f].mean()),
                "flagged_logged_same_kind": float(lg[f].mean()),
                "n_logged": int(lg.sum()),
                "logged_found": float(df.logged_hit[lg].mean()),
            }
        )
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="k100")
    ap.add_argument("--sae", default="k100", choices=list(SAES))
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--features", default="7944,9410,57487,16349")
    ap.add_argument("--n-valid", type=int, default=400)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--min-active", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()
    feats = [int(f) for f in args.features.split(",")]
    rng = np.random.default_rng(0)
    out = results_dir(EXPERIMENT, args.tag)

    recs = load_records(list(SOURCE_TAGS)).sort_values("record_id")
    errors = step11.load_errors(list(SOURCE_TAGS))
    refs = {}
    for r in recs.itertuples():
        a, b = code_span(r.response)
        found = references(r.response[a:b])
        for x in found:
            x["start"] += a
            x["end"] += a
        refs[r.record_id] = found
    check = label_check(recs, refs, errors)
    allrefs = pd.DataFrame(
        [dict(x, record_id=k) for k, v in refs.items() for x in v if x]
    )
    bad = allrefs[allrefs.group != "valid"]
    per_sol = bad.groupby("group").record_id.nunique()
    print(step10.to_markdown(check))
    print("references by group:", allrefs.group.value_counts().to_dict())
    print("solutions by group:", per_sol.to_dict())
    for g in GROUPS:
        print(g, Counter(bad[bad.group == g].name).most_common(12))
    if args.scan:
        return

    has_bad = sorted(bad.record_id.unique())
    clean = sorted(set(recs.record_id) - set(has_bad))
    pick = has_bad + list(rng.choice(clean, min(args.n_valid, len(clean)), False))
    sub = recs.set_index("record_id").loc[pick]
    fail = {
        r: (step11.error_type(errors.get(r, "")) if f else "works")
        for r, f in zip(sub.index, sub.child_failed)
    }

    import torch
    from common.models import MODELS
    from common.sae import TopKSAE
    from records import Forward
    from scipy import sparse

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
    rows, vals_l, idx_l, other = [], [], [], []
    t0 = time.time()
    for k, (rid, r) in enumerate(sub.iterrows()):
        ids, n_p, offsets = fwd.encode(r.session, r.response)
        if ids.shape[1] > args.max_tokens:
            continue
        pos = fwd.code_positions(r.response, n_p, offsets)
        states, _ = fwd.run(ids)
        h = states[args.layer].float()
        vals, idx = sae.encode(h[torch.as_tensor(pos, device=h.device)])
        vals, idx = vals.cpu().numpy(), idx.cpu().numpy()
        spans = offsets[pos - n_p]
        is_ref = np.zeros(len(pos), bool)
        for x in refs[rid]:
            hit = np.flatnonzero((spans[:, 0] < x["end"]) & (spans[:, 1] > x["start"]))
            if len(hit) == 0:
                continue
            is_ref[hit] = True
            j = hit[0]
            vals_l.append(vals[j].astype(np.float32))
            idx_l.append(idx[j])
            rows.append(
                dict(
                    {c: x[c] for c in ("name", "kind", "group", "in_call")},
                    record_id=rid,
                    outcome=fail[rid],
                )
            )
        # step 12 features on all non-reference code tokens
        dense = np.zeros((len(pos), len(feats)), np.float32)
        for c, f in enumerate(feats):
            m = idx == f
            dense[:, c] = (vals * m).sum(1)
        other.append(dense[~is_ref])
        if (k + 1) % 100 == 0:
            print(f"{k + 1}/{len(sub)} ({time.time() - t0:.0f} s)", flush=True)

    meta = pd.DataFrame(rows)
    n = len(meta)
    Z = sparse.csr_matrix(
        (
            np.concatenate(vals_l),
            np.concatenate(idx_l),
            np.arange(0, n * len(idx_l[0]) + 1, len(idx_l[0])),
        ),
        shape=(n, sae.d_sae),
    )
    oth = np.concatenate(other)
    gmask = {g: (meta.group == g).to_numpy() for g in GROUPS}
    valid = {
        "name": ((meta.group == "valid") & (meta.kind == "name")).to_numpy(),
        "attr": ((meta.group == "valid") & (meta.kind == "attr")).to_numpy(),
    }
    tab = []
    for g, m in list(gmask.items()) + [
        ("valid name", valid["name"]),
        ("valid self attr", valid["attr"]),
    ]:
        row = {"group": g, "n": int(m.sum())}
        for f in feats:
            x = Z[m][:, f].toarray().ravel()
            row[f"f{f}"] = float((x > 0).mean()) if m.any() else np.nan
        tab.append(row)
    tab.append(
        dict(
            {"group": "other code tokens", "n": len(oth)},
            **{f"f{f}": float((oth[:, c] > 0).mean()) for c, f in enumerate(feats)},
        )
    )
    tab = pd.DataFrame(tab)

    # per group: AUROC of the step 12 features and the best features
    best_rows, auc_rows = [], []
    for g, m in gmask.items():
        if m.sum() < 10:
            continue
        ctrl = valid["attr" if g == "self_attr" else "name"]
        sel = m | ctrl
        y = m[sel].astype(int)
        Zs = Z[sel]
        act = np.asarray((Zs[y == 1] > 0).mean(0)).ravel()
        cand = np.flatnonzero(act >= args.min_active)
        cand = np.union1d(cand, feats)
        D = Zs[:, cand].toarray()
        auc = np.array([roc_auc_score(y, D[:, c]) for c in range(len(cand))])
        for f in feats:
            c = int(np.flatnonzero(cand == f)[0])
            auc_rows.append({"group": g, "feature": f, "auroc": auc[c]})
        for rank, c in enumerate(np.argsort(-auc)[:5]):
            best_rows.append(
                {
                    "group": g,
                    "rank": rank,
                    "feature": int(cand[c]),
                    "auroc": auc[c],
                    "p_active_group": float((D[y == 1, c] > 0).mean()),
                    "p_active_valid": float((D[y == 0, c] > 0).mean()),
                }
            )
    aucs = pd.DataFrame(auc_rows).pivot(index="group", columns="feature")
    aucs.columns = [f"f{f}" for _, f in aucs.columns]
    aucs = aucs.reset_index()
    best = pd.DataFrame(best_rows)
    # per undefined name: activity of the first two features
    names = (
        meta[meta.group != "valid"]
        .assign(
            **{
                f"f{f}": Z[:, f].toarray().ravel()[meta.group != "valid"] > 0
                for f in feats[:2]
            }
        )
        .groupby(["group", "name"])
        .agg(n=("name", "size"), **{f"f{f}": (f"f{f}", "mean") for f in feats[:2]})
        .reset_index()
        .sort_values("n", ascending=False)
        .groupby("group")
        .head(6)
    )
    outcomes = (
        meta[meta.group != "valid"]
        .drop_duplicates(["record_id", "group"])
        .groupby("group")
        .outcome.value_counts()
        .unstack(fill_value=0)
    )
    keep = [c for c in ("works", "NameError", "AttributeError") if c in outcomes]
    outcomes = outcomes.assign(other=outcomes.drop(columns=keep).sum(1))[
        keep + ["other"]
    ].reset_index()

    pd.concat([tab.assign(table="activity"), aucs.assign(table="auroc")]).to_csv(
        out / "undefined_generality.csv", index=False
    )
    best.to_csv(out / "undefined_generality_best.csv", index=False)
    lines = [
        "# Undefined references beyond `func`",
        "",
        "## Label check against the logged errors (all 8400 solutions)",
        "",
        step10.to_markdown(check),
        "",
        f"Solutions sampled: {len(sub)}"
        f" ({len(has_bad)} with an undefined reference + {args.n_valid} clean);"
        f" reference tokens: {n}.",
        "",
        "## Outcome of solutions with each kind of undefined reference",
        "",
        step10.to_markdown(outcomes),
        "",
        f"## Activity (fraction > 0) of the step 12 features, L{args.layer}",
        "",
        step10.to_markdown(tab),
        "",
        "## AUROC against valid references of the same kind",
        "",
        step10.to_markdown(aucs),
        "",
        "## Best features per group",
        "",
        step10.to_markdown(best),
        "",
        "## Most common undefined names (fraction active)",
        "",
        step10.to_markdown(names),
        "",
    ]
    (out / "undefined_generality.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
