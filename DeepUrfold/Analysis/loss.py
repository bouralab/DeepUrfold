from itertools import groupby

import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def draw_loss_plots(superfamily):
    print("Running", superfamily)
    api = wandb.Api()
    projects = {project.name:project for project in api.projects() if superfamily in project.name}

    all_metrics = pd.DataFrame()
    stop = False
    for project_name, project in projects.items():
        for run in api.runs(project.name):
            history = run.history()

            if len(history) == 0 or "epoch" not in history:
                continue

            all_metrics = pd.concat((history, all_metrics))
            if history.epoch.min()==0:
                stop = True
                break
        if stop:
            break

    def get_latest(epoch_group, metric):
        if metric not in epoch_group:
            return np.nan
        epoch_group = epoch_group.dropna(subset=[metric])
        epoch_group = epoch_group[epoch_group["_timestamp"]==epoch_group["_timestamp"].max()]
        if len(epoch_group) > 0:
            return epoch_group[metric].iloc[0]
        return np.nan

    use_metrics = all_metrics.groupby("epoch").apply(lambda g: pd.Series({"Training":get_latest(g, "train_loss"), "Validation":get_latest(g, "val_loss")}, name=g.iloc[0].epoch))
    try:
        use_metrics = use_metrics.interpolate(method='polynomial', order=2, limit_direction="both")
    except ValueError:
        pass
    use_metrics = use_metrics.interpolate(limit_direction="both")
    print(use_metrics)
    use_metrics = use_metrics.reset_index().melt(id_vars="epoch", var_name="Stage", value_name="Evidence Lower Bound (ELBO)")

    sns.set()
    plt.figure(figsize=(8,6))
    sns.lineplot(x="epoch", y="Evidence Lower Bound (ELBO)", hue="Stage", data=use_metrics)
    plt.title(f"{superfamily} Loss Values")

    fig_path = f"{superfamily}_trian_valid_loss.png"
    plt.savefig(fig_path)

    return fig_path

def create_plots(superfamilies, latex=True):
    supp_mat_file = "supp_mat_losses.tex"
    if latex:
        #Rewrite
        with open(supp_mat_file, "w") as f:
            pass
    for superfamily in superfamilies:
        fig_path = draw_loss_plots(superfamily)
        if latex:
            with open(supp_mat_file, "a") as f:
                print(f"\subsubsection{{{superfamily}}}", file=f)
                print(f"""\\begin{{figure}}[h]
    \centering
    \includegraphics[width=0.6\\textwidth]{{Chapter3/images/losses/{fig_path}}}
    \caption{{The {superfamily} model was trained for 30 epochs. Horizontal stretches during training indicate the values were not saved and interpolated from known positions.}}
    \label{{fig:{superfamily}-loss}}
\end{{figure}}

""", file=f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    parser.add_argument("superfamily", nargs="+")
    args = parser.parse_args()
    print(args.superfamily)
    create_plots(args.superfamily)
