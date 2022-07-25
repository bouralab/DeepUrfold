import os
import re
import itertools as it
from pathlib import Path

import umap
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

import h5pyd


from DeepUrfold.Analysis.AllVsAll import get_representative_domains

import torch

class ExploreLatent():
    @staticmethod
    def reduce(*superfamilies, result_dir=None, data_dir=None, plot=True, method="umap", prefix="UMAP"):
        assert result_dir is not None
        assert data_dir is not None
        assert method in ["umap", "pca", "t-sne"]

        if method.lower() == "umap":
            reducer = umap.UMAP(n_components=2)
        elif method.lower() == "pca":
            reducer = PCA(n_components=2)
        elif method.lower() == "t-sne":
            reducer = TSNE(n_components=2)

        result_dir = Path(result_dir)
        assert result_dir.is_dir()

        embeddings_file = result_dir / f"{prefix}-{method}-all_latent.h5"

        if embeddings_file.exists():
            data = pd.read_hdf(embeddings_file, "table")
        elif len(superfamilies) == 1 and Path(superfamilies[0]).exists():
            data = pd.read_hdf(superfamilies[0], "table")
        else:
            all_latent = None
            reps_domain_and_superfamilies = None

            for superfamily in superfamilies:
                if superfamily in ["3.40.50.300", "1.10.510.10"]: continue
                result_file = result_dir / superfamily / "latent.csv"
                if not result_file.exists():
                    assert 0

                results = pd.read_csv(result_file)

                if "Unnamed: 0" in results:
                    results = results.drop(columns=["Unnamed: 0"])

                if "index" in results and "cathDomain" not in results:
                    if "', '" in results["index"].iloc[0]:
                        results = pd.concat((results, results["index"].str[2:-2].str.split("', '", expand=True).rename(
                            columns={0:"cathDomain", 1:"superfamily"})), axis=1)
                        results = results.drop(columns=["index"])
                        print(results.head())
                    else:
                        results = results.rename(columns={"index":"cathDomain"})
                        if reps_domain_and_superfamilies is None:
                            reps_domain_and_superfamilies = get_representative_domains(superfamilies, data_dir)
                        results = pd.merge(results, reps_domain_and_superfamilies, how="left", on="cathDomain")

                elif "cathDomain" not in results:
                    raise KeyError

                #Get only latent space from superfamilye
                sfam_latent = results[results["superfamily"] == superfamily].set_index(["cathDomain", "superfamily"])

                if all_latent is None:
                    all_latent = sfam_latent
                else:
                    all_latent = pd.concat((all_latent, sfam_latent))

            all_latent.to_hdf("all.hdf", "table")

            embedding = reducer.fit_transform(all_latent.values)

            data = pd.DataFrame(embedding, columns=[f"{method.upper()} Dimension 1", f"{method.upper()} Dimension 2"], index=all_latent.index)
            data = data.assign(ss_score=np.nan, electrostatics=np.nan)

            #Load features from previous run
            for other_method in ["umap", "pca", "t-sne"]:
                other_embedding_file = embeddings_file.with_name(f"{prefix}-{other_method}-all_latent.h5")
                print("Searching for previos file", other_embedding_file)
                if other_embedding_file.exists():
                    other_data = pd.read_hdf(other_embedding_file, "table")
                    data = data.assign(ss_score=other_data.ss_score, electrostatics=other_data.electrostatics)
                    break
            else:
                #Does not exist, calculate features
                with h5pyd.File(data_dir, use_cache=False) as f:
                    for _, (cathDomain, superfamily) in data.index.to_frame().iterrows():
                        try:
                            atoms = f[f"{superfamily.replace('.', '/')}/domains/{cathDomain}/atom"][:]
                            alpha = atoms["is_helix"].sum()
                            beta = atoms["is_sheet"].sum()
                            ss_score = (beta-alpha)/(2*(beta+alpha))+0.5
                            elec = 1-atoms["is_electronegative"].sum()/atoms.shape[0]
                            data.loc[(cathDomain, superfamily), "ss_score"] = ss_score
                            data.loc[(cathDomain, superfamily), "electrostatics"] = elec
                        except KeyError:
                            pass

            data.to_hdf(f"{prefix}-{method}-all_latent.h5", "table")

        if plot:
            ExploreLatent.plot(data, prefix, method)
            ExploreLatent.plot(data, prefix, method, "superfamily")
            ExploreLatent.plot(data, prefix, method, "electrostatics")

    @staticmethod
    def plot(data, prefix, method, feature="ss", old=False):
        assert feature in ["ss", "electrostatics", "superfamily"]
        import matplotlib as mpl
        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams.update({
            "text.usetex": True,
            'text.latex.preamble' : [r'\usepackage{amsmath}']
        })
        figure = plt.figure(figsize=(8,8))

        data = data.reset_index()

        viridis = sns.color_palette("viridis", as_cmap=True)
        if feature == "ss":
            cmap = viridis
            if old:
                data = data.assign(scaled_ss_score2=((data["ss_score2"]-data["ss_score2"].min())/(data["ss_score2"].max()-data["ss_score2"].min())))
                data = data.assign(color=data["scaled_ss_score2"].apply(lambda x: cmap(x)))
            else:
                data = data.assign(color=data["ss_score"].apply(lambda x: cmap(x)))
            colorbar_label = "Secondary Structure score"
        elif feature == "electrostatics":
            cmap = sns.color_palette("vlag", as_cmap=True)
            data = data.assign(color=data["electrostatics"].apply(lambda x: cmap(x)))
            colorbar_label = "Average Electrostatics"
        else:
            colors = {
                '1.10.10.10': '#e6194B',
                '1.10.238.10': '#3cb44b',
                '1.10.490.10': '#ffe119',
                '1.10.510.10': '#4363d8',
                '1.20.1260.10': '#f58231',
                '2.30.30.100': '#911eb4',
                '2.40.50.140': '#42d4f4',
                '2.60.40.10': '#f032e6',
                '3.10.20.30': '#bfef45',
                '3.30.230.10': '#fabed4',
                '3.30.300.20': '#469990',
                '3.30.310.60': '#dcbeff',
                '3.30.1360.40': '#9A6324',
                '3.30.1370.10': '#fffac8',
                '3.30.1380.10': '#800000',
                '3.40.50.300': '#aaffc3',
                '3.40.50.720': '#808000',
                '3.80.10.10': '#ffd8b1',
                '3.90.79.10': '#000075',
                '3.90.420.10': '#a9a9a9'}
            data = data.assign(color=data["superfamily"].apply(lambda x: colors.get(x, '#ffffff')))

        if old:
            mapper = {
                "Mostly Alpha (1)":r"Mostly $\boldsymbol{\alpha}$",
                "Mostly Beta (2)": r"Mostly $\boldsymbol{\beta}$",
                "Alpha/Beta (3)":r"$\boldsymbol{\alpha}$/$\boldsymbol{\beta}$"
            }
            data = data.assign(**{"CATH Class":data["cath_class"].apply(lambda s: mapper[s])})
        else:
            mapper = {
                "1":r"Mostly $\boldsymbol{\alpha}$",
                "2": r"Mostly $\boldsymbol{\beta}$",
                "3":r"$\boldsymbol{\alpha}$/$\boldsymbol{\beta}$"
            }
            data = data.assign(**{"CATH Class":data["superfamily"].map(lambda s: mapper[s.split(".")[0]])})

        d1_name = f"{method.upper()} Dimension 1"
        d2_name = f"{method.upper()} Dimension 2"
        data = data.reset_index()
        with sns.color_palette(viridis([0, 1, 0.3])) as cmap2:
            sns.kdeplot(data=data, x=d1_name, y=d2_name, hue="CATH Class", thresh=.1, alpha=0.3, common_norm=False)

        sns.scatterplot(data=data, x=d1_name, y=d2_name, s=10, c=data["color"]) #, ax=g.ax_joint

        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        ax = plt.gca()
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20

        ax.set_yticklabels(ax.get_yticks(), weight='bold')
        ax.set_xticklabels(ax.get_xticks(), rotation=0, weight='bold')

        if feature in ["ss", "electrostatics"]:

            ax2 = figure.add_axes([0.925, 0.21, .02, 0.6]) #([0.21, -0.05, 0.6, 0.03])

            cb = mpl.colorbar.ColorbarBase(ax2, orientation='vertical',
                                           cmap=cmap, label=colorbar_label)

        #plt.savefig('just_colorbar', bbox_inches='tight')

        #plt.legend(bbox_to_anchor=(-.1, 1.1))

        #plt.legend(loc='lower right')

        plt.savefig(f"{prefix}-{method}-{feature}-latent-full.png", dpi=600, bbox_inches="tight")
        plt.savefig(f"{prefix}-{method}-{feature}-latent-full.pdf", dpi=600, bbox_inches="tight")


        mean, std = data[d1_name].mean(), data[d1_name].std()

        ax.set_xlim([mean-3*std, mean+3*std])

        plt.savefig(f"{prefix}-{method}-{feature}-latent-no-outliers.png", dpi=600, bbox_inches="tight")
        plt.savefig(f"{prefix}-{method}-{feature}-latent-no-outliers.pdf", dpi=600, bbox_inches="tight")

    @staticmethod
    def reduce_old(*files, group_name="superfamily", plot=True, method="umap", prefix="UMAP", **labelled_files):
        assert method in ["umap", "pca", "t-sne"]

        if method.lower() == "umap":
            reducer = umap.UMAP(n_components=2)
        elif method.lower() == "pca":
            reducer = PCA(n_components=2)
        elif method.lower() == "t-sne":
            reducer = TSNE(n_components=2)

        try:
            data = pd.read_hdf(files[0], "table")
        except (IndexError, IOError):
            data = None

        if data is None:
            import torch
            if len(files) > 0:
                file_re = re.compile("model=(?P<model>[A-Za-z0-9]+)_input=(?P<input>[A-Za-z0-9]+)")
                for file in files:
                    m = file_re.match(file)
                    if m:
                        values = m.groupdict()
                        if values["model"] == values["input"]:
                            label = values["model"]
                        else:
                            label = (values["model"], values["input"])
                    elif "model=" in file:
                        label = file.split("=")[1].split("_")[0] #sfam
                    else:
                        label = os.path.splitext(os.path.basename(file))[0]

                    if label in labelled_files:
                        raise RuntimeError(f"Combining labelled and unlabelled files where created labels are the same is not permitted: {label}")

                    labelled_files[label] = file

            data = None

            for label, file in labelled_files.items():
                print("Running", label)
                latent = torch.load(file).numpy()
                latent_df = pd.DataFrame(latent, columns=list(range(latent.shape[-1]))).assign(
                    **{group_name:label}) #Use dict notation to add name from variable

                latent_df = latent_df.assign(ss_score1=np.nan, ss_score2=np.nan, cath_domain=np.nan)
                data_dir = "/home/bournelab/data-eppic-cath-features"
                train_file = f"{data_dir}/train_files/{label.replace('.', '/')}/DomainStructureDataset-representatives.h5"

                if os.path.isfile(train_file):
                    representatives = pd.read_hdf(train_file, "table").reset_index()
                    if not len(representatives)==len(latent_df):
                        print("    Data sets not equal for", label, len(representatives), len(latent_df))
                        continue
                    for i, cathDomain in enumerate(representatives["cathDomain"]):
                        feat_file = f"{data_dir}/cath_features/{label.replace('.', '/')}/{cathDomain}_atom.h5"
                        feats = pd.read_hdf(feat_file, "table")
                        alpha = feats["is_helix"].sum()
                        beta = feats["is_sheet"].sum()
                        unk = feats["Unk_SS"].sum()
                        print(label, cathDomain, f"alpha={alpha}, beta={beta}, unk={unk}")
                        ss_score1 = beta/alpha if alpha !=0 else np.inf

                        total_ss = len(feats)-unk
                        a = alpha/total_ss
                        b = beta/total_ss
                        ss_score2 = np.abs(a-b)
                        ss_score2 *= (-1)**int(a>b)

                        latent_df.loc[i, "ss_score1"] = ss_score1
                        latent_df.loc[i, "ss_score2"] = ss_score2
                        latent_df.loc[i, "cath_domain"] = os.path.basename(feat_file).split("_")[0]
                        del feats

                        if ss_score1 < 1:
                            print("More alpha??", cathDomain)
                        if ss_score2 < 0:
                            print("    More alpha??", cathDomain)
                    del representatives
                else:
                    assert 0, f"{train_file} does not exist"


                if data is None:
                    data = latent_df
                else:
                    data = pd.concat((data, latent_df), axis=0)


            #data.loc[data["ss_score"]==np.inf, "ss_score"] = data["ss_score"][data["ss_score"]!=np.inf].max()
            #data.loc[data["ss_score"]==np.nan, "ss_score"] = data["ss_score"].min()
            ss_mask = data["ss_score1"] != np.inf
            print("Number infs:", len(ss_mask)-ss_mask.astype(int).sum())
            data.loc[~ss_mask, "ss_score1"] = data.loc[ss_mask, "ss_score1"].max()

            ss_nan_mask = data["ss_score1"] != np.nan
            print("Number nans:", len(ss_nan_mask)-ss_nan_mask.astype(int).sum())
            data.loc[~ss_nan_mask, "ss_score1"] = data.loc[ss_nan_mask, "ss_score1"].max()

            print("Min:", data["ss_score1"].min(), data["ss_score2"].min())
            print("Max:", data["ss_score1"].max(), data["ss_score2"].max())

            data = data.assign(scaled_ss_score=((data["ss_score1"]-data["ss_score1"].min())/(data["ss_score1"].max()-data["ss_score1"].min())))

            embedding = reducer.fit_transform(data.drop(columns=[group_name, "ss_score1", "ss_score2", "cath_domain"]).values)
            data = data.assign(**{"Dimension 1": embedding[:, 0], "Dimension 2": embedding[:, 1]})
            data = data.sort_values("superfamily")
            data = data.rename(columns={"superfamily": "Superfamily"})

            data.to_hdf(f"{prefix}-{method}-all_latent.h5", "table")

        if not plot:
            return data

        superfamily_colors = {
          #https://sashamaps.net/docs/resources/20-colors/
          "1.10.10.10":   "red", #"#800000",
          "1.10.238.10":  "red", #"#e6194B",
          "1.10.490.10":  "red", #"#f58231",
          "1.10.510.10":  "red", #"#9A6324",
          "1.20.1260.10": "red", #"#ffe119",
          "2.30.30.100":  "green", #"#808000",
          "2.40.50.140":  "green", #"#ffe119",
          "2.60.40.10":   "green", #"#bfef45",
          "3.10.20.30":   "blue",  #"#aaffc3",
          "3.30.300.20":  "blue",  #"#469990",
          "3.30.230.10":  "blue",  #"#000075",
          "3.30.1360.40": "blue", #"#4363d8",
          "3.30.1370.10": "blue", #"#42d4f4",
          "3.40.50.300":  "blue", #"#911eb4",
          "3.40.50.720":  "blue", #"#f032e6",
          "3.90.79.10":   "blue", #"#fabed4",
          "3.90.420.10":  "blue", #"#dcbeff"
        }

        # superfamily_styles = {
        #   #https://sashamaps.net/docs/resources/20-colors/
        #   "1.10.10.10":   ".", #"#800000",
        #   "1.10.238.10":  "^", #"#e6194B",
        #   "1.10.490.10":  "s", #"#f58231",
        #   "1.10.510.10":  "p", #"#9A6324",
        #   "1.20.1260.10": "P", #"#ffe119",
        #   "2.30.30.100":  ".", #"#808000",
        #   "2.40.50.140":  "^", #"#ffe119",
        #   "2.60.40.10":   "s", #"#bfef45",
        #   "3.10.20.30":   ".",  #"#aaffc3",
        #   "3.30.300.20":  "^",  #"#469990",
        #   "3.30.230.10":  "s",  #"#000075",
        #   "3.30.1360.40": "p", #"#4363d8",
        #   "3.30.1370.10": "P", #"#42d4f4",
        #   "3.40.50.300":  "*", #"#911eb4",
        #   "3.40.50.720":  "h", #"#f032e6",
        #   "3.90.79.10":   "+", #"#fabed4",
        #   "3.90.420.10":  "X", #"#dcbeff"
        # }

        #superfamily_colors = data["Superfamily"]

        # superfamily_styles = {sfam: i for c, ath in \
        #     it.groupby(sorted(data["Superfamily"]), key=lambda s:s[0]) \
        #     for i, (_, sfams) in enumerate(it.groupby(ath)) for sfam in sfams}

        markers = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d"]
        sfam_markers = {sfam: marker for (sfam, _), marker in zip(it.groupby(sorted(data["Superfamily"])), markers)}
        print(sfam_markers)


        #data = data.assign(marker=data["Superfamily"].apply(lambda s: markers.get(s, -1)))

        sizes = {"1":10, "2":40, "3":160}
        cath_classes = {"1":"Mostly Alpha (1)", "2":"Mostly Beta (2)", "3":"Alpha/Beta (3)"}

        data = data.assign(marker=data["Superfamily"].apply(lambda s: sfam_markers.get(s, "$?$")))
        data = data.assign(cath_class=data["Superfamily"].apply(lambda s: cath_classes.get(s.split(".")[0], "Unknown Class")))
        data = data.assign(size=data["Superfamily"].apply(lambda s: sizes.get(s.split(".")[0], 100)))

        #sns.scatterplot(x="Dimension 1", y="Dimension 2", hue="ss_score", #hue="Superfamily", palette=superfamily_colors,
        #    style="marker", data=data, cmap=sns.color_palette("icefire", as_cmap=True))

        def mscatter(x,y,ax=None, m=None, **kw):
            import matplotlib.markers as mmarkers
            if not ax: ax=plt.gca()
            sc = ax.scatter(x,y,**kw)
            if (m is not None) and (len(m)==len(x)):
                paths = []
                for marker in m:
                    if isinstance(marker, mmarkers.MarkerStyle):
                        marker_obj = marker
                    else:
                        marker_obj = mmarkers.MarkerStyle(marker)
                    path = marker_obj.get_path().transformed(
                                marker_obj.get_transform())
                    paths.append(path)
                sc.set_paths(paths)
            return sc

        #cmap = sns.color_palette("icefire", as_cmap=True)
        cmap = sns.color_palette("viridis", as_cmap=True)

        data = data.assign(color=data["ss_score2"].apply(lambda x: cmap(x)))

        fig, ax = plt.subplots(figsize=(8,8))
        points = mscatter(data["Dimension 1"], data["Dimension 2"], m=data["marker"], s=10, c=data["ss_score2"], cmap=cmap, vmin=data["ss_score2"].min(), vmax=data["ss_score2"].max())

        title_kwds = dict(
            visible=False, color="w", s=0, linewidth=0
        )

        legend_data = []
        legend_titles = []

        import matplotlib.markers as mmarkers
        import matplotlib.lines as mlines

        class_title = plt.scatter([], [], label="CATH Class", **title_kwds)
        legend_data.append(class_title)
        legend_titles.append("")
        for cath_class, size in data[["cath_class", "size"]].drop_duplicates().values:
            marker = plt.scatter([], [], label=cath_class, s=size)
            legend_data.append(marker)
            legend_titles.append(cath_class)

        sfam_title = plt.scatter([], [], label="CATH Superfamilies", **title_kwds)
        legend_data.append(sfam_title)
        legend_titles.append("")
        for cath_sfam, marker_style in data[["Superfamily", "marker"]].drop_duplicates().values:
            # marker_obj = mmarkers.MarkerStyle(marker_style)
            # path = marker_obj.get_path().transformed(
            #     marker_obj.get_transform())
            marker = mlines.Line2D([], [], label=cath_sfam, markersize=10, marker=marker_style, linestyle='None')
            legend_data.append(marker)
            legend_titles.append(cath_sfam)

        legend = plt.legend(handles=legend_data, loc="upper left", bbox_to_anchor=(1.04,1))

        cbar = fig.colorbar(points, orientation='horizontal')

        # change colobar ticks labels and locators
        cbar.set_ticks(list(cbar.ax.get_xticks()))
        tick_labels = list(cbar.ax.xaxis.get_ticklabels())
        tick_labels[0] = "Mostly Alpha"
        tick_labels[-1] = "Mostly Beta"
        cbar.set_ticklabels(tick_labels)
        cbar.ax.axes.tick_params(length=0)
        for label in cbar.ax.xaxis.get_ticklabels()[1:-1]:
            label.set_visible(False)

        #sns.utils.adjust_legend_subtitles(legend)

        """
        SS_score: color gradient
        SUperfamily: 17 categories
        Class: 3 categories


        """

        #handles, labels = plt.gca().get_legend_handles_labels()

        #plt.gca().get_legend().remove()

        #plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 4})





        #plt.gca().set_aspect('equal', 'datalim')
        plt.title(f'{method} projection of the Urfold Space: {prefix}') #, fontsize=24)
        print(data["Dimension 1"].min(), data["Dimension 1"].max())
        #plt.xlim(data["Dimension 1"].min()-3, data["Dimension 1"].max()+3)
        #plt.ylim(data["Dimension 2"].min()-3, data["Dimension 2"].max()+3)

        if prefix is None:
            prefix = ""

        plt.savefig(f'{prefix}-{method}.pdf', bbox_extra_artists=(legend,), bbox_inches='tight')

        ExploreLatent.plot(data, prefix, method, old=True)
        ExploreLatent.plot(data, prefix, method, "superfamily", old=True)
        ExploreLatent.plot(data, prefix, method, "electrostatics", old=True)



if __name__ == "__main__":
    import argparse
    sfams = "1.10.10.10 1.10.238.10 1.10.490.10 1.10.510.10 1.20.1260.10 2.30.30.100 2.40.50.140 2.60.40.10 3.10.20.30 3.30.230.10 3.30.300.20 3.30.310.60 3.30.1360.40 3.30.1370.10 3.30.1380.10 3.40.50.300 3.40.50.720 3.80.10.10 3.90.79.10 3.90.420.10".split()
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prefix", default="latent-space")
    parser.add_argument("-d", "--data-dir", required=True)
    parser.add_argument("-r", "--result-dir", required=True)
    parser.add_argument("-m", "--method", nargs="+", default=["umap", "t-sne", "pca"], choices=["umap", "t-sne", "pca"])
    parser.add_argument("superfamilies", nargs="?", default=sfams, help="superfamilies to inlcude", )
    #parser.add_argument("files", nargs="+", default=None, help="Path to pytorch tensof files")
    args = parser.parse_args()

    for method in args.method:
        ExploreLatent.reduce(*args.superfamilies, result_dir=args.result_dir, data_dir=args.data_dir, prefix=args.prefix, method=method)
