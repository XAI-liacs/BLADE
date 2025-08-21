import json
import os
import subprocess
import time
from pathlib import Path

import matplotlib
import pandas as pd
import plotly.graph_objects as go
import jsonlines
import streamlit as st

from iohblade.plots import CEG_FEATURES, CEG_FEATURE_LABELS, plotly_code_evolution

from iohblade.assets import LOGO_DARK_B64, LOGO_LIGHT_B64
from iohblade.loggers import ExperimentLogger

LOGO_LIGHT = f"data:image/png;base64,{LOGO_LIGHT_B64}"
LOGO_DARK = f"data:image/png;base64,{LOGO_DARK_B64}"


def convergence_dataframe(logger: ExperimentLogger) -> pd.DataFrame:
    methods, problems = logger.get_methods_problems()
    frames = []
    for p in problems:
        df = logger.get_problem_data(problem_name=p).drop(columns=["code"])
        df.replace([-pd.NA, float("inf"), float("-inf")], 0, inplace=True)
        df["problem_name"] = p
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["cummax_fitness"] = df.groupby(["method_name", "problem_name", "seed"])[
        "fitness"
    ].cummax()
    df["eval"] = df["_id"] + 1
    return df


def plotly_convergence(df: pd.DataFrame, aggregate: bool = False) -> go.Figure:
    fig = go.Figure()
    if aggregate:
        for (m, p), g in df.groupby(["method_name", "problem_name"]):
            summary = (
                g.groupby("eval")["cummax_fitness"].agg(["mean", "std"]).reset_index()
            )
            fig.add_trace(
                go.Scatter(
                    x=summary["eval"],
                    y=summary["mean"],
                    mode="lines",
                    name=f"{m}-{p}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=summary["eval"],
                    y=summary["mean"] + summary["std"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=summary["eval"],
                    y=summary["mean"] - summary["std"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    showlegend=False,
                )
            )
    else:
        for (m, p, s), g in df.groupby(["method_name", "problem_name", "seed"]):
            fig.add_trace(
                go.Scatter(
                    x=g["eval"],
                    y=g["cummax_fitness"],
                    mode="lines",
                    name=f"{m}-{p}-seed{s}",
                )
            )
    fig.update_layout(xaxis_title="Evaluations", yaxis_title="Best fitness")
    return fig


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


def read_progress(exp_dir):
    path = os.path.join(exp_dir, "progress.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


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
                    chat_role = "user" if role.lower() in {"client", "user"} else "assistant"
                    with st.chat_message(chat_role):
                        st.markdown(content)
        else:
            st.write("No conversation log found.")
        return

    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    elif time.time() - st.session_state["last_refresh"] > 60:
        st.session_state["last_refresh"] = time.time()
        st.rerun()

    logo_html = f"""
    <picture>
      <source media='(prefers-color-scheme: dark)' srcset='{LOGO_DARK}'>
      <source media='(prefers-color-scheme: light)' srcset='{LOGO_LIGHT}'>
      <img src='{LOGO_LIGHT}' style='width: 150px;'>
    </picture>
    """
    st.sidebar.markdown(logo_html, unsafe_allow_html=True)
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
            if prog.get("runs") and not finished:
                st.markdown("#### Run Progress")
                for r in prog["runs"]:
                    label = (
                        f"{r['method_name']} | {r['problem_name']} | seed {r['seed']}"
                    )
                    budget = r.get("budget", 1)
                    evaluations = r.get("evaluations", 0)
                    pct_run = evaluations / budget if budget else 0
                    st.progress(pct_run, text=label)

        if prog and prog.get("runs"):
            st.markdown("#### Run Logs")
            for r in prog["runs"]:
                label = f"{r['method_name']} | {r['problem_name']} | seed {r['seed']}"
                log_dir = r.get("log_dir")
                if log_dir:
                    conv_path = os.path.join(exp_dir, log_dir, "conversationlog.jsonl")
                    if os.path.exists(conv_path):
                        link = f"?exp={selected}&conv={log_dir}"
                        st.markdown(
                            f"- {label} - <a href='{link}' target='_blank'>Conversation Log</a>",
                            unsafe_allow_html=True,
                        )

        st.header(f"Convergence - {selected}")
        df = convergence_dataframe(logger)
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

            final_df = (
                df_filt.groupby(["method_name", "problem_name", "seed"])
                .last()
                .reset_index()
            )
            box_fig = go.Figure()
            for prob in problem_sel:
                subset = final_df[final_df["problem_name"] == prob]
                box_fig.add_trace(
                    go.Box(
                        y=subset["fitness"],
                        x=subset["method_name"],
                        name=prob,
                    )
                )
            box_fig.update_layout(yaxis_title="Fitness", xaxis_title="Method")
            st.plotly_chart(box_fig, use_container_width=True)

            st.markdown("#### Code Evolution Graph")
            ceg_method = st.selectbox("Method", methods, key="ceg_method")
            ceg_problem = st.selectbox("Problem", problems, key="ceg_problem")
            seed_options = df[
                (df["method_name"] == ceg_method) & (df["problem_name"] == ceg_problem)
            ]["seed"].unique()
            ceg_seed = st.selectbox(
                "Seed", sorted(seed_options.tolist()), key="ceg_seed"
            )
            feature_labels = [CEG_FEATURE_LABELS.get(f, f) for f in CEG_FEATURES]
            label_to_feature = {CEG_FEATURE_LABELS.get(f, f): f for f in CEG_FEATURES}
            selected_label = st.selectbox("Feature", feature_labels, key="ceg_feature")
            feature = label_to_feature[selected_label]
            run_df = logger.get_problem_data(ceg_problem)
            run_df = run_df[
                (run_df["method_name"] == ceg_method) & (run_df["seed"] == ceg_seed)
            ]
            if st.button("Plot Evolution Graph", key="plot_ceg"):
                if not run_df.empty:
                    ceg_fig = plotly_code_evolution(run_df, feature=feature)
                    st.plotly_chart(ceg_fig, use_container_width=True)
                else:
                    st.write("No data for selected run.")

            st.markdown("#### Top Solutions")
            runs = logger.get_data()
            for m in method_sel:
                m_runs = runs[runs["method_name"] == m]
                m_runs = m_runs.sort_values(
                    by=["solution"],
                    key=lambda col: col.apply(
                        lambda s: s.get("fitness", -float("inf"))
                    ),
                    ascending=False,
                )
                top = m_runs.head(3)
                with st.expander(m):
                    for _, row in top.iterrows():
                        sol = row["solution"]
                        fname = f"{sol.get('name','sol')}.py"
                        st.write(f"{sol.get('name')} - {sol.get('fitness')}")
                        st.download_button(
                            label=f"Download {fname}",
                            data=sol.get("code", ""),
                            file_name=fname,
                            mime="text/x-python",
                            key=f"download_{m}_{sol.get('id', 'id')}",
                        )
        else:
            st.write("No data")
    else:
        st.write("Select an experiment from the sidebar to view results.")


def main() -> None:
    subprocess.run(["streamlit", "run", str(Path(__file__))], check=True)


if __name__ == "__main__":
    run()
