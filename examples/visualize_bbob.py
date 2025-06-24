import iohinspector
import polars as pl
from iohblade.behaviour_metrics import compute_behavior_metrics
from tqdm import tqdm
import pandas as pd
import numpy as np
from iohblade.loggers import ExperimentLogger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
from pandas.plotting import parallel_coordinates
from iohblade import plot_convergence, plot_experiment_CEG, plot_boxplot_fitness_hue, plot_boxplot_fitness, fitness_table
import os
import tqdm
from sklearn.preprocessing import RobustScaler, MinMaxScaler


# ── 2. HELPER FUNCTION (drop it right after `behaviour_feats` is first defined) ─
def robust_minmax(df, cols, q=(0.01, 0.99)):
    """
    Robust [0-1] scaling that flattens outliers.

    • Clips each feature to the q-th and (1-q)-th quantile.  
    • Then rescales the clipped values to the 0-1 interval.
    """
    lo = df[cols].quantile(q[0])
    hi = df[cols].quantile(q[1])
    df[cols] = df[cols].clip(lo, hi, axis=1)
    return (df[cols] - lo) / (hi - lo)

"""
    mutation_prompts1 = [
        "Refine and simplify the selected algorithm to improve it.",  # simplify mutation
    ]
    mutation_prompts2 = [
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    ]
    mutation_prompts3 = [
        "Refine and simplify the selected solution to improve it.",  # simplify mutation
        "Generate a new algorithm that is different from the algorithms you have tried before.", #new random solution
    ]
    LLaMEA_method1 = LLaMEA(llm, budget=budget, name=f"LLaMEA-1", mutation_prompts=mutation_prompts1, n_parents=4, n_offspring=12, elitism=False) 
    LLaMEA_method2 = LLaMEA(llm, budget=budget, name=f"LLaMEA-2", mutation_prompts=mutation_prompts2, n_parents=4, n_offspring=12, elitism=False) 
    LLaMEA_method3 = LLaMEA(llm, budget=budget, name=f"LLaMEA-3", mutation_prompts=mutation_prompts3, n_parents=4, n_offspring=12, elitism=False) 
    LLaMEA_method4 = LLaMEA(llm, budget=budget, name=f"LLaMEA-4", mutation_prompts=mutation_prompts3, n_parents=1, n_offspring=1, elitism=True) 
    LLaMEA_method5 = LLaMEA(llm, budget=budget, name=f"LLaMEA-5", mutation_prompts=None, adaptive_mutation=True, n_parents=4, n_offspring=12, elitism=False) 
    LLaMEA_method6 = LLaMEA(llm, budget=budget, name=f"LLaMEA-6", mutation_prompts=None, adaptive_mutation=True, n_parents=1, n_offspring=1, elitism=True) 
# """

# | **Metric**                              | **What it Measures**                                    | **Reference**                          |
# | --------------------------------------- | ------------------------------------------------------- | -------------------------------------- |
# | **Nearest-Neighbor Distance** (avg NND) | Novelty of new samples (exploration)                    | Distance-based exploration     https://arxiv.org/html/2410.14573#:~:text=removing%20points%20from%20the%20Pareto,various%20optimization%20scenarios%20and%20objectives        |
# | **Coverage / Dispersion**               | Fraction of space covered (exploration)                 | Space-filling designs in DOE   https://www.nature.com/articles/s41598-024-68436-1#:~:text=To%20perform%20population%20diversity%20analysis%2C,used%20for%20population%20diversity%20estimation        |
# | **Spatial Entropy**                     | Spread/uncertainty of sample distribution               | Entropy for batch diversity    https://www.nature.com/articles/s41598-024-68436-1#:~:text=To%20perform%20population%20diversity%20analysis%2C,used%20for%20population%20diversity%20estimation        |
# | **Proximity to Best**                   | Focus on known optima (exploitation)                    | Exploitation = staying in neighborhood https://romisatriawahono.net/lecture/rm/survey/softcomputing/Crepinsek%20-%20Exploration%20and%20Exploitation%20in%20Evolutionary%20Algorithms%20-%202013.pdf#:~:text=space,regions%20of%20a%20search%20space  |
# | **Exploration–Exploitation %**          | Trade-off via diversity (time-varying)                  | Pop. diversity ratio                 https://www.nature.com/articles/s41598-024-68436-1#:~:text=%24%24exploration%5C%3B%5Cleft%28iteration%5Cright%29%3D%5Cleft%28%5Cfrac |
# | **Success/Improvement Rate**            | Frequency of improving moves (convergence/exploitation) | 1/5th success rule concept           https://www.nature.com/articles/s41598-024-68436-1#:~:text=%24%24exploration%5C%3B%5Cleft%28iteration%5Cright%29%3D%5Cleft%28%5Cfrac  |
# | **Area Under Conv. Curve**              | Overall speed of convergence                            | – (common in benchmarking)             |
# | **Average Convergence Rate** (ACR)      | Geometric mean error reduction (convergence speed)      | Convergence rate analysis             https://www.nature.com/articles/s41598-024-68436-1#:~:text=%24%24exploration%5C%3B%5Cleft%28iteration%5Cright%29%3D%5Cleft%28%5Cfrac |
# | **No-Improvement Streak**               | Longest stagnation period                               | Used as stopping criterion            https://www.sciencedirect.com/science/article/pii/S037722172200159X#:~:text=The%20second%20parameter%20to%20be,In%20this%20case%2C%20we |
# | **Last Improvement Fraction**           | Portion of run spent stagnating                         | – (derived from convergence curve)     |
# | **Pop. Diversity (to centroid)**        | Dispersion of points in search space                    | “Distance to average” measure         https://www.sciencedirect.com/science/article/pii/S037722172200159X#:~:text=The%20second%20parameter%20to%20be,In%20this%20case%2C%20we |
# | **Mean Pairwise Distance**              | Overall pairwise diversity                              | Diversity measures in EAs             https://romisatriawahono.net/lecture/rm/survey/softcomputing/Crepinsek%20-%20Exploration%20and%20Exploitation%20in%20Evolutionary%20Algorithms%20-%202013.pdf#:~:text=match%20at%20L885%20%E2%80%94Distance,used%20type%20of%20diversity%20measure |
# | **Coverage Volume**                     | Volume spanned by samples (diversity)                   | DPP-based diversity                   https://romisatriawahono.net/lecture/rm/survey/softcomputing/Crepinsek%20-%20Exploration%20and%20Exploitation%20in%20Evolutionary%20Algorithms%20-%202013.pdf#:~:text=match%20at%20L885%20%E2%80%94Distance,used%20type%20of%20diversity%20measure |
# | **Average Step Length**                 | Typical move size (exploration intensity)               | Step size reduction in search         https://stats.stackexchange.com/questions/304813/stopping-criterion-for-nelder-mead#:~:text=Stopping%20criterion%20for%20Nelder%20Mead,than%20some%20tolerance%20TOL |
# | **Total Path Length**                   | Total distance traveled                                 | – (trajectory length analysis)         |
# | **Path Efficiency**                     | Directness of search path                               | –                                      |
# | **Step Length Trend**                   | Change of step size over time (convergence indicator)   | – (implied by 1/5th rule)              |


#df.to_pickle("./BBOB0.pkl")

llm_per_folder = {"BBOB0": "gpt-4.1",
                  "BBOB1": "gemini-2.0-flash",
                  "BBOB2": "o4-mini",
                  "BBOB3": "gemma3-27b"}

for folder in tqdm.tqdm(["BBOB0", "BBOB1", "BBOB2", "BBOB3"]): # 
    #print(f"Processing {folder}...")

    llm_name = llm_per_folder[folder]

    df = pd.read_pickle(f"{folder}.pkl")  
    img_dir = f"examples/img/"
    #df.describe()

    # Some columns you’ll want handy later
    behaviour_feats = [
        "avg_nearest_neighbor_distance", "dispersion", "avg_exploration_pct",
        "avg_distance_to_best", 
        "average_convergence_rate", "avg_improvement", "success_rate",
        "longest_no_improvement_streak", 
    ]
    all_behaviour_feats = [
        "avg_nearest_neighbor_distance", "dispersion", "avg_exploration_pct",
        "avg_distance_to_best",  "intensification_ratio", "avg_exploitation_pct", "last_improvement_fraction",
        "average_convergence_rate", "avg_improvement", "success_rate",
        "longest_no_improvement_streak", 
    ]
    #leave out: 
    agg_map  = {c: 'mean' for c in all_behaviour_feats} | {           # mean for numbers
                'id': 'first',                              # keep id constant
                'fitness_fid': 'mean',                     # average fitness per id
                'fid': 'first',                             
                '_id': 'first',
                'seed': 'first',
                'method_name': 'first',                    # keep one value
                'problem_name': 'first'}                   #  – they should be constant per id
    print(df['_id'].describe())

    df_all = (df
            .groupby('id', as_index=False)
            .agg(agg_map))

    print(df_all['_id'].head())



    # ------------------------------------------------------------------
    # 0  House‑keeping
    # ------------------------------------------------------------------
    # Make a copy we can mangle
    data = df_all.copy()

    # Replace ±∞ with NaN, then drop rows that have no finite fitness at all
    data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    #data = data.dropna(subset=["fitness"])

    # Normalise fitness to [0, 1] per algorithm so colour scales are comparable
    data["fitness_norm"] = (
        data.groupby("method_name")["fitness_fid"]
            .transform(lambda s: (s - s.min()) / (s.max() - s.min()))
    )

    
    nice_names = {
        "avg_nearest_neighbor_distance":  "NN-dist",
        "dispersion":                     "Disp",
        "avg_exploration_pct":            "Expl %",
        "avg_distance_to_best":           "Dist→best",
        "intensification_ratio":          "Inten-ratio",
        "avg_exploitation_pct":           "Explt %",
        "average_convergence_rate":       "Conv-rate",
        "avg_improvement":                "Δ fitness",
        "success_rate":                   "Success %",
        "longest_no_improvement_streak":  "No-imp streak",
        "last_improvement_fraction":      "Last-imp frac",
    }

    log_folder = {"BBOB0": "BBOB",
                "BBOB1": "BBOB-1",
                "BBOB2": "BBOB-2",
                "BBOB3": "BBOB-3"}
    loc = log_folder[folder]
    logger = ExperimentLogger(f'/data/neocortex/{loc}', True)
    df_log = logger.get_problem_data(problem_name='BBOB')
    

    data = data.merge(df_log[['id', 'parent_ids']], on='id', how='left') #add parent_ids


    if True:
        # ── NEW BLOCK — STN-style export (drop this *after* `behaviour_feats` is defined
        #    and `data` has been assembled for the current folder) ──────────────────────

        stn_dir = "stn/"
        os.makedirs(stn_dir, exist_ok=True)

        feature_cols = [
            "avg_exploration_pct",
            "average_convergence_rate", "avg_improvement",
            "success_rate",
            "longest_no_improvement_streak",
        ]  # the columns that go into the STN feature list (subset)
        #         "avg_nearest_neighbor_distance", "dispersion", "avg_exploration_pct",
        # "avg_distance_to_best",  "intensification_ratio", "avg_exploitation_pct", "last_improvement_fraction",
        # "average_convergence_rate", "avg_improvement", "success_rate",
        # "longest_no_improvement_streak", 

        data_stn = data.copy()
        data_stn[behaviour_feats] = robust_minmax(data_stn, behaviour_feats)
        for method, g in data_stn.groupby("method_name", sort=False):


            lines = []
            lines_best = []
            # one file line per evaluation, ordered by run (seed) then _id
            for seed, sg in g.groupby("seed", sort=False):
                # Quick lookup table: id  ➜ row (Series)
                #id_lookup = sg.set_index("id", drop=False)
                id_lookup = (
                    sg.sort_values("_id")           # ensures _id ascending
                    .drop_duplicates("id", keep="first")
                    .set_index("id", drop=False)
                )

                current_best = -np.inf
                for _, child in sg.sort_values("_id").iterrows():
                    # print(child)
                    # a =aaaa

                    # ── get parent row, if any ───────────────────────────────
                    parent = None
                    parent_list = child["parent_ids"]
                    if isinstance(parent_list, (list, tuple)) and parent_list:
                        pid = parent_list[0]
                        if pid in id_lookup.index:
                            parent = id_lookup.loc[pid]
                    
                    if parent is None:
                        # No parent found, use the current child as parent
                        parent = child
                            
                    # ── stringify parent & child info ───────────────────────
                    # parent block
                    if parent is not None:
                        try:
                            feat_str_parent = ",".join(f"{parent[c]:.12g}" for c in feature_cols)
                            parent_block = f"{parent['fitness_fid']:.12g}\t{feat_str_parent}"
                            # child block
                            feat_str_child = ",".join(f"{child[c]:.12g}" for c in feature_cols)
                            child_block = f"{child['fitness_fid']:.12g}\t{feat_str_child}"
                            # complete line: seed ⟶ parent ⟶ child
                            lines.append(f"{seed}\t{parent_block}\t{child_block}")
                        except Exception as e:
                            print(f"Error processing seed {seed} with parent {pid}: {e}")
                            print(parent)
                            continue

                    # feat_str = ",".join(f"{row[c]:.12g}" for c in feature_cols)
                    # lines.append(f"{seed}\t{row['fitness_fid']:.12g}\t{feat_str}")
                    # if row["fitness_fid"] > current_best:
                    #     current_best = row["fitness_fid"]
                    #     lines_best.append(f"{seed}\t{row['fitness_fid']:.12g}\t{feat_str}")

            fname = f"{llm_name}_{method}_all.csv".replace(" ", "_")
            with open(os.path.join(stn_dir, fname), "w") as fh:
                fh.write("\n".join(lines))

        #continue #only do STN for now

            # fname_best = f"{llm_name}_{method}_best.csv".replace(" ", "_")
            # with open(os.path.join(stn_dir, fname_best), "w") as fh:
            #     fh.write("\n".join(lines_best))



        # ------------------------------------------------------------------
        # 0  Pre-filter: keep only rows where fitness_fid strictly improves
        #    relative to the previous evaluation of the *same seed*.
        # ------------------------------------------------------------------
        improving = (
            data.sort_values(["seed", "_id"])          # just to be safe
                .groupby("seed")["fitness_fid"]
                .apply(lambda s: s > s.shift())        # True where improvement
                .reset_index(level=0, drop=True)       # align with original index
        )

        data_imp = data[improving].copy()              # <- only improving evals

            
        # ------------------------------------------------------------------
        # 1  Visualise behaviour features *along those improving steps only*
        # ------------------------------------------------------------------
        for algo, g in data_imp.groupby("method_name", sort=False):
            g = g.sort_values(["seed", "_id"])          # nicer lines

            for feat in behaviour_feats:
                plt.figure(figsize=(8, 4))

                # individual improving stretches
                for _, seed_group in g.groupby("seed"):
                    plt.plot(
                        seed_group["_id"],
                        seed_group[feat],
                        linewidth=0.6,
                        alpha=0.35,
                    )

                # mean trend of improving steps
                mean_curve = (
                    g.groupby("_id")[feat]
                    .mean()
                    .reset_index()
                    .sort_values("_id")
                )
                plt.plot(
                    mean_curve["_id"],
                    mean_curve[feat],
                    linewidth=2,
                    label="mean on improving evals",
                )

                plt.title(f"{algo}")
                plt.xlabel("Evaluations")
                plt.ylabel(f"{feat} - of best so far")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{img_dir}{llm_name}_{algo}_{feat}_improving.png")
                plt.clf()
                plt.close()
        
        # ------------------------------------------------------------------
        # 1  Behaviour features over evaluations
        #     (_id is assumed to be monotonically increasing per seed)
        # ------------------------------------------------------------------
        for algo, g in data.groupby("method_name", sort=False):
            # Pre‑sort once for nicer lines
            g = g.sort_values(["seed", "_id"])

            for feat in behaviour_feats:
                plt.figure(figsize=(8, 4))

                # All individual runs in the background
                for _, seed_group in g.groupby("seed"):
                    plt.plot(
                        seed_group["_id"],
                        seed_group[feat],
                        linewidth=0.6,
                        alpha=0.3,
                    )

                # Mean trend over runs on top
                mean_curve = (
                    g.groupby("_id")[feat]
                    .mean()
                    .reset_index()
                    .sort_values("_id")
                )
                plt.plot(
                    mean_curve["_id"],
                    mean_curve[feat],
                    linewidth=2,
                    label="per‑eval mean",
                )

                plt.title(f"{algo}")
                plt.xlabel("Evaluations")
                plt.ylabel(feat)
                plt.legend(loc="best")
                plt.tight_layout()
                plt.savefig(f"{img_dir}{llm_name}_{algo}_{feat}.png")
                plt.clf()
                plt.close()

        # ------------------------------------------------------------------
        # 2  All methods in one figure per behaviour feature
        # ------------------------------------------------------------------
        palette = sns.color_palette("tab10", n_colors=data["method_name"].nunique())

        for feat in behaviour_feats:
            plt.figure(figsize=(10, 6))

            # Average each method over seeds at every evaluation step
            mean_per_method = (
                data.groupby(["method_name", "_id"])[feat]
                    .median()
                    .reset_index()
                    .sort_values("_id")
            )

            # One line per algorithm
            for (algo, g), colour in zip(mean_per_method.groupby("method_name", sort=False),
                                        palette):
                plt.plot(
                    g["_id"],
                    g[feat],
                    label=algo,
                    linewidth=2,
                    alpha=0.8,
                    color=colour,
                )

            plt.title(f"Median {nice_names.get(feat, feat)}")
            plt.xlabel("Evaluations")
            plt.ylabel(nice_names.get(feat, feat))
            plt.legend(loc="best", frameon=False, fontsize="small")
            plt.tight_layout()
            plt.savefig(f"{img_dir}{llm_name}_ALL_{feat}.png")
            plt.clf()
            plt.close()

    if False:
        # ------------------------------------------------------------------
        # 1  Correlation heat‑map 
        # ------------------------------------------------------------------
        #for algo, g in data.groupby("method_name", sort=False):

        nice_names_corr = {
            "avg_nearest_neighbor_distance":  "NN-dist",
            "dispersion":                     "Disp",
            "avg_exploration_pct":            "Expl %",
            "avg_distance_to_best":           "Dist→best",
            "intensification_ratio":          "Inten-ratio",
            "avg_exploitation_pct":           "Explt %",
            "average_convergence_rate":       "Conv-rate",
            "avg_improvement":                "Δ fitness",
            "success_rate":                   "Success %",
            "longest_no_improvement_streak":  "No-imp streak",
            "last_improvement_fraction":      "Last-imp frac",
            "fitness_norm":                   "Normalized Fitness",
        }

        
        corr_data = data.copy()
        corr_data[all_behaviour_feats] = robust_minmax(corr_data, all_behaviour_feats)
        corr_data.rename(columns=nice_names_corr, inplace=True)
        
        corr_data.drop(columns=["id", "fitness_fid", "fid", "_id", "seed", "method_name", "problem_name", 'parent_ids'], inplace=True)

        corr_data.dropna(inplace=True)  # drop rows with NaN values
        plot_df = corr_data.corr()

        plt.figure(figsize=(16, 14))
        sns.heatmap(
            plot_df,
            cmap="coolwarm",
            annot=True,
            fmt=".1f",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws=dict(label="Pearson r"),
        )
        plt.title(f"Behaviour metrics\ncorrelation matrix")
        plt.tight_layout()
        plt.savefig(f"{img_dir}{llm_name}_behaviour-metrics-correlation.png")
        #plt.show()
        plt.clf()
    
    if False:
        # ------------------------------


        # Parallel coordinates for all!
        pc = data[behaviour_feats + ["fitness_norm"]].copy()

        # 2‑a.  Min‑max scale features (parallel plots hate disparate ranges)
        # pc[behaviour_feats] = (
        #     pc[behaviour_feats] - pc[behaviour_feats].min()
        # ) / (pc[behaviour_feats].max() - pc[behaviour_feats].min())
        pc[behaviour_feats] = robust_minmax(pc, behaviour_feats)

        quartile_labels = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
        quart_cat = pd.CategoricalDtype(categories=quartile_labels, ordered=True)

        # 2‑b.  Quartile‑bin fitness so we can colour by performance
        pc["fitness_group"] = pd.qcut(
            pc["fitness_norm"], 4, labels=quartile_labels, duplicates="drop"
        )
        pc["fitness_group"] = pc["fitness_group"].astype(quart_cat)
        pc = pc.sort_values("fitness_group", key=lambda s: s.cat.codes)

        # ── optional: subsample to keep the plot readable ──────────────
        #sample = pc.sample(n=min(len(pc), 600), random_state=42)

        fig, ax = plt.subplots(figsize=(16, 12))

        plot_df = pc.rename(columns=nice_names)
        parallel_coordinates(
            plot_df, "fitness_group", alpha=0.3, linewidth=1.0,
            colormap="seismic", ax=ax,
        )

        handles, _ = ax.get_legend_handles_labels()
        # ax.legend(handles, quartile_labels, title="fitness quartile",
        #     loc="upper left", frameon=True)
        plt.title(f"Behaviour profile")
        plt.ylabel("scaled feature value")
        plt.xticks(rotation=90, ha="right")
        plt.tight_layout()
        plt.savefig(f"{img_dir}{llm_name}_behaviour-pc.png")
        plt.clf()
        plt.close()

        # ------------------------------------------------------------------
        # 2  Parallel‑coordinate plot per algorithm
        # ------------------------------------------------------------------
        for algo, g in data.groupby("method_name", sort=False):

            pc = g[behaviour_feats + ["fitness_norm"]].copy()

            # 2‑a.  Min‑max scale features (parallel plots hate disparate ranges)
            # pc[behaviour_feats] = (
            #     pc[behaviour_feats] - pc[behaviour_feats].min()
            # ) / (pc[behaviour_feats].max() - pc[behaviour_feats].min())
            pc[behaviour_feats] = robust_minmax(pc, behaviour_feats)

            # 2‑b.  Quartile‑bin fitness so we can colour by performance
            pc["fitness_group"] = pd.qcut(
                pc["fitness_norm"], 4, labels=quartile_labels, duplicates="drop" #["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
            )
            pc["fitness_group"] = pc["fitness_group"].astype(quart_cat)

            pc = pc.sort_values("fitness_group", key=lambda s: s.cat.codes)
            # ── optional: subsample to keep the plot readable ──────────────
            #sample = pc.sample(n=min(len(pc), 600), random_state=42)

            plt.figure(figsize=(16, 12))
            parallel_coordinates(
                pc, "fitness_group", alpha=0.4, linewidth=1.0,
                colormap="seismic",
            )
            plt.title(f"{algo} — parallel‑coordinate behaviour profile")
            plt.ylabel("scaled feature value")
            plt.xticks(rotation=90, ha="right")
            plt.tight_layout()
            plt.savefig(f"{img_dir}{llm_name}_{algo}_pc.png")
            plt.clf()
            plt.close()

        # ------------------------------------------------------------------
        # 0  Aggregate the 50 raw auc‑columns into 10 function‑level scores
        # ------------------------------------------------------------------



        func_ids        = [1, 3, 6, 8, 10, 13, 15, 17, 21, 23]
        runs_per_func   = 5          # we know each function has 5 runs
        quart_labels    = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
        quart_cat       = pd.CategoricalDtype(quart_labels, ordered=True)



        # ------------------------------------------------------------------
        # 1  Build a helper with min‑max‑scaled behaviour features
        # ------------------------------------------------------------------
        behaviour_feats = [
            "avg_nearest_neighbor_distance", "dispersion", "avg_exploration_pct",
            "avg_distance_to_best", "intensification_ratio", "avg_exploitation_pct",
            "average_convergence_rate", "avg_improvement", "success_rate",
            "longest_no_improvement_streak", "last_improvement_fraction",
        ]

        nice_names = {
            "avg_nearest_neighbor_distance":  "NN-dist",
            "dispersion":                     "Disp",
            "avg_exploration_pct":            "Expl %",
            "avg_distance_to_best":           "Dist→best",
            "intensification_ratio":          "Inten-ratio",
            "avg_exploitation_pct":           "Explt %",
            "average_convergence_rate":       "Conv-rate",
            "avg_improvement":                "Δ fitness",
            "success_rate":                   "Success %",
            "longest_no_improvement_streak":  "No-imp streak",
            "last_improvement_fraction":      "Last-imp frac",
        }

        data = df.copy()
        # data[behaviour_feats] = (
        #     data[behaviour_feats] - data[behaviour_feats].min()
        # ) / (data[behaviour_feats].max() - data[behaviour_feats].min())
        data[behaviour_feats] = robust_minmax(data, behaviour_feats)

        # ------------------------------------------------------------------
        # 2  Parallel‑coordinate plot for each function‑level AUC
        # ------------------------------------------------------------------
        from pandas.plotting import parallel_coordinates
        import matplotlib.pyplot as plt

        col = "fitness_fid"
        for fid in func_ids:
            num_cols = data.select_dtypes('number').columns          # all numeric columns
            agg_map  = {c: 'mean' for c in num_cols} | {           # mean for numbers
                        'method_name': 'first',                    # keep one value
                        'problem_name': 'first'}                   #  – they should be constant per id

            df_fid = data[data["fid"] == fid].copy()
            df_f = (df_fid
                    .groupby('id', as_index=False)
                    .agg(agg_map))


            if df_f.empty:       # nothing to draw for this function
                continue



            # quartile‑bin the aggregated fitness score
            q = pd.qcut(df_f[col], 4, labels=quart_labels).astype(quart_cat)

            # frame to plot: scaled behaviour + quartile label
            plot_df = df_f[behaviour_feats + ["fitness_fid"]].copy()
            plot_df["fitness_quartile"] = q

            # (optional) thin out lines for readability
            #plot_df = plot_df.sample(n=min(len(plot_df), 2000), random_state=42)
            plot_df = plot_df.sort_values("fitness_quartile", key=lambda s: s.cat.codes)

            fig, ax = plt.subplots(figsize=(16, 12))
            
            plot_df = plot_df.rename(columns=nice_names)
            parallel_coordinates(
                plot_df, "fitness_quartile", ax=ax,
                alpha=0.35, linewidth=1.0, colormap="seismic",
            )

            # handles, _ = ax.get_legend_handles_labels()
            # ax.legend(handles, quart_labels, title="fitness quartile",
            #           loc="upper left", frameon=True)

            ax.set_title(f"Behaviour profile for $f_{{{fid}}}$")
            ax.set_ylabel("scaled feature value")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
            plt.tight_layout()
            plt.savefig(f"{img_dir}{llm_name}_f{fid}_pc.png")
            plt.clf()
            plt.close()

        TOP_K     = 100            # ← change this to whatever "top‑k" you want

        # ---------------------------------------------------------------
        # 1  Gather top‑k rows per function (by aggregated auc_f*)
        # ---------------------------------------------------------------
        pieces = []

        behaviour_feats = [
            "avg_nearest_neighbor_distance", "dispersion", "avg_exploration_pct",
            "avg_distance_to_best", "intensification_ratio", "avg_exploitation_pct",
            #"average_convergence_rate", "avg_improvement", "success_rate",
            "longest_no_improvement_streak", "last_improvement_fraction",
        ]
        nice_names = {
            "avg_nearest_neighbor_distance":  "NN-dist",
            "dispersion":                     "Disp",
            "avg_exploration_pct":            "Expl %",
            "avg_distance_to_best":           "Dist→best",
            "intensification_ratio":          "Inten-ratio",
            "avg_exploitation_pct":           "Explt %",
        #    "average_convergence_rate":       "Conv-rate",
        #    "avg_improvement":                "Δ fitness",
        #    "success_rate":                   "Success %",
            "longest_no_improvement_streak":  "No-imp streak",
            "last_improvement_fraction":      "Last-imp frac",
        }

        col = "fitness_fid"
        for fid in func_ids:

            df_fid = data[data["fid"] == fid].copy()
            df_f = (df_fid
                    .groupby('id', as_index=False)
                    .agg(agg_map))

            # take the k rows with highest score for this function
            topk = df_f.nlargest(TOP_K, columns=col)

            # scaled behaviour + fid label
            
            frame = df_f[behaviour_feats + ["fitness_fid"]].loc[topk.index].copy()
            frame["fid"] = f"f{fid}"
            pieces.append(frame)

        # Concatenate every function’s Q4 slice
        plot_df = pd.concat(pieces, ignore_index=True)

        # Rename cols for prettier x‑ticks
        plot_df = plot_df.rename(columns=nice_names)

        # ------------------------------------------------------------------
        # 3  Parallel‑coordinate plot
        # ------------------------------------------------------------------
        from pandas.plotting import parallel_coordinates
        import matplotlib.pyplot as plt

        # Build a colour list (one per fid) – tab10 has 10 distinct hues
        cmap   = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(func_ids))]

        fig, ax = plt.subplots(figsize=(16, 12))
        parallel_coordinates(
            plot_df, "fid",
            ax=ax,
            linewidth=1.1,
            alpha=0.9,
            color=colors,
        )

        ax.set_title("Top-100 behaviour profiles per function id")
        ax.set_ylabel("scaled feature value")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
        ax.legend(title="function id", frameon=True, loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{img_dir}{llm_name}_fids_pc.png")
        plt.clf()
        plt.close()

        # BLADE loggers
        log_folder = {"BBOB0": "BBOB",
                    "BBOB1": "BBOB-1",
                    "BBOB2": "BBOB-2",
                    "BBOB3": "BBOB-3"}
        loc = log_folder[folder]
        logger = ExperimentLogger(f'/data/neocortex/{loc}', True)
        plot_convergence(logger, metric="AOCC", save=True, budget=100)
        plot_experiment_CEG(logger, save=True, budget=100, max_seeds=5)

