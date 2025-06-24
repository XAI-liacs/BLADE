import os
import json
import streamlit as st
import matplotlib
from iohblade.loggers import ExperimentLogger
from iohblade.plots import plot_convergence

RESULTS_DIR = "results"


def discover_experiments(root=RESULTS_DIR):
    """Return list of experiment directories containing an experimentlog.jsonl."""
    exps = []
    if os.path.isdir(root):
        for entry in os.listdir(root):
            path = os.path.join(root, entry)
            if os.path.isdir(path) and os.path.exists(
                os.path.join(path, "progress.jsonl")
            ):
                exps.append(entry)
    return sorted(exps)


def read_progress(exp_dir):
    path = os.path.join(exp_dir, "progress.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


st.set_page_config(page_title="BLADE Experiment Browser", layout="wide")

st.sidebar.title("Experiments")
experiments = discover_experiments()
refresh = st.sidebar.button("Refresh")
if refresh:
    st.experimental_rerun()

if experiments:
    selected = st.sidebar.selectbox("Select an experiment", experiments)
    st.sidebar.markdown("### Progress")
    for exp in experiments:
        prog = read_progress(os.path.join(RESULTS_DIR, exp))
        if prog is None:
            st.sidebar.write(f"{exp}: ?")
        elif prog.get("end_time"):
            st.sidebar.success(f"{exp} - finished")
        else:
            total = prog.get("total", 1)
            pct = prog.get("current", 0) / total if total else 0
            st.sidebar.write(exp)
            st.sidebar.progress(pct)
else:
    st.sidebar.write("No experiments found in 'results'.")
    selected = None

if selected:
    exp_dir = os.path.join(RESULTS_DIR, selected)
    logger = ExperimentLogger(exp_dir, read=True)

    prog = read_progress(exp_dir)
    if prog:
        if prog.get("end_time"):
            st.success("Finished")
        else:
            total = prog.get("total", 1)
            pct = prog.get("current", 0) / total if total else 0
            st.progress(pct, text=f"{prog.get('current',0)} / {total}")

    matplotlib.use("Agg")
    fig = plot_convergence(logger, metric="fitness", save=False, return_fig=True, separate_lines=True)
    st.header(f"Convergence - {selected}")
    st.pyplot(fig)
    st.caption("Plot refreshes when the page reruns")
else:
    st.write("Select an experiment from the sidebar to view results.")
