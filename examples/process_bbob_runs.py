import iohinspector
import polars as pl
from iohblade.behaviour_metrics import compute_behavior_metrics
from tqdm import tqdm
import pandas as pd
import numpy as np
from iohblade.loggers import ExperimentLogger
from iohblade import plot_convergence, plot_experiment_CEG, plot_boxplot_fitness_hue, plot_boxplot_fitness, fitness_table
import os

data_dir = "/data/neocortex/BBOB-2"

logger = ExperimentLogger(data_dir, True)

data = logger.get_data()


tqdm.pandas(desc="Processing runs")

func_ids        = [1, 3, 6, 8, 10, 13, 15, 17, 21, 23]
runs_per_func   = 5          # we know each function has 5 runs

def avg_auc_for_fid(aucs: list[float], fid: int, runs_per_func=5, func_ids=[1, 3, 6, 8, 10, 13, 15, 17, 21, 23]) -> float:
    """
    aucs = 50-long list in metadata.
    Take the five entries belonging to the requested fid and average them.
    """
    if len(aucs) < 50:
        return 0
    block = func_ids.index(fid) * runs_per_func
    slice_ = aucs[block : block + runs_per_func]
    return float(np.mean(slice_))

def process_run(row, func_ids= [1, 3, 6, 8, 10, 13, 15, 17, 21, 23], runs_per_func=5, root=f"{data_dir}/ioh/"):
    rows = []
    # each algorithm has 50 different runs = 50 different directories we need to process
    algid = row['id']
    fitness = row['fitness']
    aucs = row['aucs_list']
    method_name = row['method_name']
    problem_name = row['problem_name']

    counter = 0
    for fid in func_ids:
        for run in range(runs_per_func):
            
            path = f"{root}{algid}"
            if counter > 0:
                path = f"{root}{algid}-{counter}"
            # each algorithm has 50 different runs = 50 different directories we need to process
            if not os.path.exists(path):
                continue
            manager = iohinspector.DataManager()
            manager.add_folder(path)
            # load run data for one folder
            df = manager.load(monotonic=False, include_meta_data=True)
            metrics = compute_behavior_metrics(df)
            counter += 1

            f_fid = avg_auc_for_fid(aucs, fid, runs_per_func=runs_per_func, func_ids=func_ids)
            metrics.update(
                {
                    "id":           algid,
                    "fid":          fid,
                    "fitness_fid":  f_fid,
                    "method_name":  method_name,
                    "problem_name": problem_name,
                    "seed" :        row['seed'], 
                    "_id" :         row['_id'],
                }
            )
            rows.append(metrics)
    return pd.DataFrame(rows)

def get_aucs(d):
    """Return the list under key 'aucs' (or [] if anything is wrong)."""
    try:
        return d.get("aucs", [])
    except (ValueError, SyntaxError):
        return []


methods, problems = logger.get_methods_problems()
print(methods)
print(problems)

for problem in problems:
    print(f"Problem: {problem}")
    df = logger.get_problem_data(problem_name=problem)
    print(df.columns)

    df["aucs_list"] = df["metadata"].apply(get_aucs)


    frames = []

    for _, algo_row in tqdm(df.iterrows(),  # <‑‑ your dataframe of algorithms
                            total=len(df),
                            desc="algorithms"):
        try:
            df_runs = process_run(algo_row)      # your function above
            frames.append(df_runs)
        except Exception as e:
            print(f"⚠︎  {algo_row['id']} skipped ({e})")

    all_runs = pd.concat(frames, ignore_index=True)

    print(f"✅  built dataframe with {len(all_runs)} runs")

    print(all_runs.columns)
    all_runs.to_pickle(f"./{problem}2.pkl")