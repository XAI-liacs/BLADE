import json
import os
import subprocess
import time
import urllib
from pathlib import Path

import jsonlines
import matplotlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from iohblade.assets import LOGO_DARK_B64, LOGO_LIGHT_B64
from iohblade.loggers import ExperimentLogger
from iohblade.plots import (
    CEG_FEATURE_LABELS,
    CEG_FEATURES,
    _resolve_minimisation,
    _solution_fitness,
    add_method_labels,
    aggregate_best_so_far,
    best_so_far,
    final_best_per_run,
    plotly_code_evolution,
    run_columns,
)

LOGO_LIGHT = f"data:image/png;base64,{LOGO_LIGHT_B64}"
LOGO_DARK = f"data:image/png;base64,{LOGO_DARK_B64}"
AUTO_REFRESH_SECONDS = 60


def convergence_dataframe(
    logger: ExperimentLogger, include_running: bool = False
) -> pd.DataFrame:
    """Best-so-far curves of every run of every problem of an experiment.

    Uses the scalar fitness (multi-objective: mean of the objectives) and
    the logged direction of each problem; failed evaluations never improve
    the curve.

    Args:
        logger (ExperimentLogger): The experiment to read.
        include_running (bool): Also include runs that are not finished.

    Returns:
        pd.DataFrame: The evaluations with ``best_so_far``, ``eval``
        (1-based), ``minimisation``, ``method_label`` and ``run_label``
        columns (empty if there is no data).
    """
    _, problems = logger.get_methods_problems()
    if include_running:
        problems = sorted(set(problems) | set(_running_problems(logger)))
    frames = []
    for p in problems:
        df = logger.get_problem_data(problem_name=p, include_running=include_running)
        if df.empty:
            continue
        df = df.drop(columns=["code"], errors="ignore")
        df["problem_name"] = p
        minimise = _resolve_minimisation(logger, p)
        curves = best_so_far(df, minimisation=minimise)
        curves["minimisation"] = minimise
        frames.append(curves)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    add_method_labels(df)
    df["run_label"] = _run_labels(df)
    return df


def _running_problems(logger: ExperimentLogger) -> list[str]:
    progress = getattr(logger, "progress", None) or {}
    return [
        r["problem_name"]
        for r in progress.get("runs", []) or []
        if r.get("problem_name") and r.get("start_time")
    ]


def _run_labels(df: pd.DataFrame) -> pd.Series:
    """A readable, unique label per run: method (LLM), problem and seed."""
    label = df["method_label"].astype(str) + "-" + df["problem_name"].astype(str)
    label = label + "-seed" + df["seed"].astype(str)
    if "run" in df.columns:
        # the same method, LLM and seed in two read directories
        per_label = df.groupby(label)["run"].transform("nunique")
        label = label.where(per_label <= 1, label + " [" + df["run"].astype(str) + "]")
    return label


def final_best_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """The best fitness of each run (``NaN`` for a run with only failures).

    Args:
        df (pd.DataFrame): Output of :func:`convergence_dataframe`.

    Returns:
        pd.DataFrame: One row per run with ``best_so_far``.
    """
    cols = run_columns(df)
    if "problem_name" not in cols:
        cols = cols + ["problem_name"]
    return final_best_per_run(df, cols)


def top_solutions(
    runs: pd.DataFrame, method: str, minimisation_of=None, n: int = 3
) -> pd.DataFrame:
    """The best final solutions of a method, per problem.

    Args:
        runs (pd.DataFrame): ``ExperimentLogger.get_data()``.
        method (str): The method name.
        minimisation_of (callable, optional): Maps a problem name to whether
            it is minimised (default: maximised).
        n (int): Number of solutions per problem.

    Returns:
        pd.DataFrame: Rows of ``runs`` with a ``fitness_value`` column (float,
        ``NaN`` for failures), best first per problem; failures come last.
    """
    if runs.empty or "method_name" not in runs.columns:
        return pd.DataFrame()
    m_runs = runs[runs["method_name"] == method].copy()
    if m_runs.empty:
        return m_runs
    m_runs["fitness_value"] = _solution_fitness(m_runs)
    frames = []
    for problem, group in m_runs.groupby("problem_name", sort=True):
        minimise = bool(minimisation_of(problem)) if minimisation_of else False
        group = group.sort_values(
            by="fitness_value", ascending=minimise, na_position="last"
        )
        frames.append(group.head(n))
    return pd.concat(frames) if frames else m_runs.head(0)


# helper: convert '#RRGGBB' or 'rgb(r,g,b)' to rgba with alpha
def _rgba(color: str, alpha: float) -> str:
    # handles '#rrggbb'
    if color.startswith("#") and len(color) == 7:
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    # handles 'rgb(r,g,b)'
    if color.startswith("rgb(") and color.endswith(")"):
        nums = color[4:-1]
        return f"rgba({nums},{alpha})"
    # fallback: let Plotly parse; many qualitative colors are hex so this usually won’t hit
    return color


def plotly_convergence(df: pd.DataFrame, aggregate: bool = False) -> go.Figure:
    """Convergence (best-so-far) figure of :func:`convergence_dataframe`.

    Args:
        df (pd.DataFrame): Output of :func:`convergence_dataframe`.
        aggregate (bool): Mean ± std over the runs of each method (and LLM)
            per problem (shorter runs carried forward), instead of one line
            per run.
    """
    fig = go.Figure()
    palette = px.colors.qualitative.Plotly  # or px.colors.qualitative.D3
    if df.empty:
        return fig
    if "method_label" not in df.columns:
        df = add_method_labels(df.copy())
    run_cols = run_columns(df)

    if aggregate:
        groups = df.groupby(["method_label", "problem_name"], sort=True)
        for i, ((m, p), g) in enumerate(groups):
            summary = aggregate_best_so_far(g, "mean", "std", run_cols)
            summary["std"] = summary["std"].fillna(0.0)  # a single run
            color = palette[i % len(palette)]
            group = f"{m}-{p}"

            # Draw LOWER bound first
            fig.add_trace(
                go.Scatter(
                    x=summary["eval"],
                    y=summary["mean"] - summary["std"],
                    mode="lines",
                    line=dict(width=0, color=color),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=group,
                    name=group,  # not shown because showlegend=False
                )
            )
            # Then UPPER bound, fill to previous (lower)
            fig.add_trace(
                go.Scatter(
                    x=summary["eval"],
                    y=summary["mean"] + summary["std"],
                    mode="lines",
                    line=dict(width=0, color=color),
                    fill="tonexty",
                    fillcolor=_rgba(color, 0.18),  # match line color, add alpha
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=group,
                )
            )
            # Finally the MEAN line on top
            fig.add_trace(
                go.Scatter(
                    x=summary["eval"],
                    y=summary["mean"],
                    mode="lines",
                    line=dict(color=color),
                    name=group,
                    legendgroup=group,
                )
            )
    else:
        # per-run lines (no bands)
        labels = df["run_label"] if "run_label" in df.columns else None
        for key, g in df.groupby(run_cols, sort=False, dropna=False):
            name = labels.loc[g.index[0]] if labels is not None else str(key)
            fig.add_trace(
                go.Scatter(
                    x=g["eval"],
                    y=g["best_so_far"],
                    mode="lines",
                    name=name,
                )
            )

    fig.update_layout(
        xaxis_title="Evaluations",
        yaxis_title="Best fitness",
        legend=dict(
            groupclick="togglegroup"
        ),  # click one legend item toggles its whole group
    )
    return fig


def run_evaluations(
    logger: ExperimentLogger,
    df: pd.DataFrame,
    problem: str,
    run_label: str,
    include_running: bool = False,
) -> pd.DataFrame:
    """All evaluations (with code) of the run named ``run_label``.

    Args:
        logger (ExperimentLogger): The experiment.
        df (pd.DataFrame): Output of :func:`convergence_dataframe`.
        problem (str): The problem of the run.
        run_label (str): The ``run_label`` of the run.
        include_running (bool): Also read runs that are not finished.
    """
    rows = df[df["run_label"] == run_label]
    if rows.empty:
        return pd.DataFrame()
    data = logger.get_problem_data(problem, include_running=include_running)
    if data.empty:
        return data
    mask = pd.Series(True, index=data.index)
    for col in run_columns(rows):
        if col in data.columns:
            value = rows[col].iloc[0]
            if pd.isna(value):
                mask &= data[col].isna()
            else:
                mask &= data[col] == value
    return data[mask].sort_values("_id", kind="stable")


RESULTS_DIR = "results"


def discover_experiments(root=RESULTS_DIR):
    """Return list of experiment directories containing an experimentlog.jsonl."""
    exps = []
    if os.path.isdir(root):
        for entry in os.listdir(root):
            path = os.path.join(root, entry)
            if os.path.isdir(path) and os.path.exists(
                os.path.join(path, "progress.json")
            ):
                exps.append(entry)
    return sorted(exps)


def run_status_rows(progress: dict | None) -> list[dict]:
    """One row per run of ``progress.json``: progress, best fitness, errors."""
    rows = []
    for r in (progress or {}).get("runs", []):
        if r.get("end_time"):
            status = "done"
        elif r.get("error"):
            status = "failed"
        elif r.get("start_time"):
            status = "running"
        else:
            status = "pending"
        counts = r.get("error_counts") or {}
        rows.append(
            {
                "Method": r.get("method_name"),
                "LLM": r.get("llm_name") or "",
                "Problem": r.get("problem_name"),
                "Seed": r.get("seed"),
                "Evaluations": f"{r.get('evaluations', 0)}/{r.get('budget', '?')}",
                "Best fitness": r.get("best_fitness"),
                "Errors in code": counts.get("candidate", 0),
                "Harness errors": counts.get("harness", 0),
                "Timeouts": counts.get("timeout", 0),
                "Status": status,
                "Last error": r.get("error") or r.get("last_error") or "",
            }
        )
    return rows


def read_record(exp_dir: str) -> dict:
    """The recorded configuration of an experiment (see iohblade.config.record).

    Returns:
        dict: ``resolved`` (YAML text or None), ``sources`` (module ->
        {source, sha256}) and ``archived`` (older records, newest first).
    """
    record = os.path.join(exp_dir, "blade_config")
    resolved_path = os.path.join(record, "resolved.yml")
    sources_path = os.path.join(record, "sources.json")
    resolved = None
    if os.path.exists(resolved_path):
        with open(resolved_path) as f:
            resolved = f.read()
    sources = {}
    if os.path.exists(sources_path):
        try:
            with open(sources_path) as f:
                sources = json.load(f)
        except ValueError:
            sources = {}
    archived = (
        sorted(
            (d for d in os.listdir(exp_dir) if d.startswith("blade_config-")),
            reverse=True,
        )
        if os.path.isdir(exp_dir)
        else []
    )
    return {"resolved": resolved, "sources": sources, "archived": archived}


def tail_file(path: str, lines: int = 200) -> str | None:
    """The last ``lines`` lines of a text file (None if it does not exist)."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-lines:])


def read_progress(exp_dir):
    """Read the progress of an experiment from the directory"""
    path = os.path.join(exp_dir, "progress.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _refresh_due(now: float | None = None) -> bool:
    """Whether the page was last refreshed more than a refresh period ago."""
    now = time.time() if now is None else now
    last = st.session_state.get("last_refresh")
    if last is None:
        st.session_state["last_refresh"] = now
        return False
    if now - last >= AUTO_REFRESH_SECONDS:
        st.session_state["last_refresh"] = now
        return True
    return False


def _refresh_timer() -> None:
    """Rerun the whole app when a refresh is due (see :func:`_auto_refresh`)."""
    if _refresh_due():
        st.rerun(scope="app")


def _make_refresh_fragment():
    """``_refresh_timer`` as a fragment rerun on a timer, or None if the
    installed Streamlit has no ``st.fragment(run_every=...)`` (< 1.37)."""
    fragment = getattr(st, "fragment", None)
    if fragment is None:
        return None
    try:
        return fragment(run_every=AUTO_REFRESH_SECONDS)(_refresh_timer)
    except TypeError:
        return None


_REFRESH_FRAGMENT = _make_refresh_fragment()


def _auto_refresh() -> None:
    """Rerun the whole app every ``AUTO_REFRESH_SECONDS`` seconds.

    Uses a fragment that Streamlit reruns on a timer (``st.fragment`` with
    ``run_every``, Streamlit >= 1.37), so a running experiment's page
    refreshes without any interaction. With an older Streamlit the page is
    only refreshed on the first interaction after the period.
    """
    if _REFRESH_FRAGMENT is None:
        if _refresh_due():
            st.rerun()
        return
    st.session_state["last_refresh"] = time.time()  # this (full) run
    _REFRESH_FRAGMENT()


def run() -> None:
    st.set_page_config(
        page_title="BLADE Experiment Browser",
        layout="wide",
        menu_items={"Get Help": "https://xai-liacs.github.io/BLADE/"},
    )

    params = st.query_params
    conv = params.get("conv")
    exp = params.get("exp")
    if conv and exp:
        log_path = os.path.join(RESULTS_DIR, exp, conv, "conversationlog.jsonl")
        st.title("Conversation Log")
        if os.path.exists(log_path):
            with jsonlines.open(log_path) as f:
                for msg in f:
                    role = msg.get("role", "assistant")
                    content = msg.get("content", "")
                    chat_role = (
                        "user" if role.lower() in {"client", "user"} else "assistant"
                    )
                    with st.chat_message(chat_role):
                        st.markdown(content)
        else:
            st.write(f"No conversation log found in {log_path}.")
        return

    _auto_refresh()

    with st.sidebar:
        st.image(
            LOGO_LIGHT if st.context.theme.type == "light" else LOGO_DARK,
            width=150,
        )
    st.sidebar.markdown("### Experiments")

    experiments = discover_experiments()
    selected = st.session_state.get("selected_exp")

    if experiments:
        for exp in experiments:
            prog = read_progress(os.path.join(RESULTS_DIR, exp))
            icon = "✅" if prog and prog.get("end_time") else "⏳"
            if st.sidebar.button(exp, key=f"btn_{exp}", type="tertiary", icon=icon):
                st.session_state["selected_exp"] = exp
                selected = exp
            if prog and not prog.get("end_time"):
                total = prog.get("total", 1)
                pct = prog.get("current", 0) / total if total else 0
                st.sidebar.progress(pct)
    else:
        st.sidebar.write("No experiments found in 'results'.")

    if selected is None and experiments:
        selected = experiments[0]

    if selected:
        exp_dir = os.path.join(RESULTS_DIR, selected)
        logger = ExperimentLogger(exp_dir, read=True)

        prog = read_progress(exp_dir)
        finished = prog and prog.get("end_time")
        if prog:
            if prog.get("end_time"):
                st.success("Finished")
            else:
                total = prog.get("total", 1)
                pct = prog.get("current", 0) / total if total else 0
                st.progress(pct)
            rows = run_status_rows(prog)
            if rows:
                st.markdown("#### Runs")
                st.dataframe(
                    pd.DataFrame(rows), hide_index=True, use_container_width=True
                )
            if prog.get("runs") and not finished:
                st.markdown("#### Run Progress")
                for r in prog["runs"]:
                    llm = f" | {r['llm_name']}" if r.get("llm_name") else ""
                    label = (
                        f"{r['method_name']}{llm} | {r['problem_name']} "
                        f"| seed {r['seed']}"
                    )
                    budget = r.get("budget", 1)
                    evaluations = r.get("evaluations", 0)
                    pct_run = evaluations / budget if budget else 0
                    st.progress(pct_run, text=label)

        record = read_record(exp_dir)
        if record["resolved"] or record["sources"]:
            with st.expander("Configuration"):
                if record["resolved"]:
                    st.code(record["resolved"], language="yaml")
                if record["sources"]:
                    st.markdown("**Custom code**")
                    for module, info in record["sources"].items():
                        st.markdown(
                            f"- `{module}`: {info.get('source')} "
                            f"(sha256 `{str(info.get('sha256', ''))[:12]}`)"
                        )
                if record["archived"]:
                    st.markdown(
                        "Earlier versions (the experiment was resumed with a "
                        "changed configuration or code): "
                        + ", ".join(f"`{a}`" for a in record["archived"])
                    )
        log_text = tail_file(os.path.join(exp_dir, "experiment.log"))
        if log_text:
            with st.expander("Experiment log (last 200 lines)"):
                st.code(log_text, language="text")

        st.header(f"Convergence - {selected}")
        df = convergence_dataframe(logger, include_running=not finished)
        if not df.empty:
            methods = sorted(df["method_name"].unique())
            problems = sorted(df["problem_name"].unique())
            method_sel = st.multiselect("Methods", methods, default=methods)
            problem_sel = st.multiselect("Problems", problems, default=problems)
            aggregate = st.checkbox("Aggregate runs", value=True)
            df_filt = df[
                df["method_name"].isin(method_sel)
                & df["problem_name"].isin(problem_sel)
            ]
            fig = plotly_convergence(df_filt, aggregate=aggregate)
            st.plotly_chart(fig, use_container_width=True)

            final_df = final_best_dataframe(df_filt)
            box_fig = go.Figure()
            for prob in problem_sel:
                subset = final_df[final_df["problem_name"] == prob]
                box_fig.add_trace(
                    go.Box(
                        y=subset["best_so_far"],
                        x=subset["method_label"],
                        name=prob,
                    )
                )
            box_fig.update_layout(yaxis_title="Best fitness", xaxis_title="Method")
            st.plotly_chart(box_fig, use_container_width=True)

            st.markdown("#### Code Evolution Graph")
            ceg_method = st.selectbox("Method", methods, key="ceg_method")
            ceg_problem = st.selectbox("Problem", problems, key="ceg_problem")
            run_options = df[
                (df["method_name"] == ceg_method) & (df["problem_name"] == ceg_problem)
            ].drop_duplicates("run_label")
            run_options = run_options.sort_values(["seed", "run_label"])
            ceg_run = st.selectbox(
                "Run", run_options["run_label"].tolist(), key="ceg_run"
            )
            feature_labels = [CEG_FEATURE_LABELS.get(f, f) for f in CEG_FEATURES]
            label_to_feature = {CEG_FEATURE_LABELS.get(f, f): f for f in CEG_FEATURES}
            selected_label = st.selectbox("Feature", feature_labels, key="ceg_feature")
            feature = label_to_feature[selected_label]
            if st.button("Plot Evolution Graph", key="plot_ceg"):
                run_df = run_evaluations(
                    logger, df, ceg_problem, ceg_run, include_running=not finished
                )
                if not run_df.empty:
                    ceg_fig = plotly_code_evolution(
                        run_df,
                        feature=feature,
                        minimisation=_resolve_minimisation(logger, ceg_problem),
                    )
                    st.plotly_chart(ceg_fig, use_container_width=True)
                else:
                    st.write("No data for selected run.")

            st.markdown("#### Top Solutions")
            runs = logger.get_data()
            for m in method_sel:
                top = top_solutions(
                    runs,
                    m,
                    minimisation_of=lambda p: _resolve_minimisation(logger, p),
                )
                with st.expander(m):
                    for i, (_, row) in enumerate(top.iterrows()):
                        sol = row["solution"]
                        fname = f"{sol.get('name','sol')}.py"
                        st.write(
                            f"{row['problem_name']}: {sol.get('name')} - "
                            f"{sol.get('fitness')}"
                        )
                        st.download_button(
                            label=f"Download {fname}",
                            data=sol.get("code", ""),
                            file_name=fname,
                            mime="text/x-python",
                            key=f"download_{m}_{i}_{sol.get('id', 'id')}",
                        )

            st.markdown("#### Run Logs")
            for m in method_sel if "method_name" in runs.columns else []:
                m_runs = runs[runs["method_name"] == m].sort_values(by=["seed"])
                with st.expander(m):
                    for _, r in m_runs.iterrows():
                        llm = r.get("llm_name")
                        llm = f" | {llm}" if isinstance(llm, str) and llm else ""
                        label = (
                            f"{r['method_name']}{llm} | {r['problem_name']} "
                            f"| seed {r['seed']}"
                        )
                        log_dir = r.get("log_dir")
                        if log_dir:
                            conv_path = os.path.join(
                                exp_dir, log_dir, "conversationlog.jsonl"
                            )
                            if os.path.exists(conv_path):
                                selected_url = urllib.parse.quote_plus(selected)
                                log_dir_url = urllib.parse.quote_plus(log_dir)
                                link = f"?exp={selected_url}&conv={log_dir_url}"
                                st.markdown(
                                    f"- {label} - <a href='{link}' target='_blank'>Conversation Log</a>",
                                    unsafe_allow_html=True,
                                )
        else:
            st.write("No data")

    else:
        st.write("Select an experiment from the sidebar to view results.")


def main() -> None:
    subprocess.run(["streamlit", "run", str(Path(__file__))], check=True)


if __name__ == "__main__":
    run()
