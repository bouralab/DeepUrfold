import os
import re
import itertools as it
from pathlib import Path
import pickle

import umap
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, explained_variance_score
import seaborn as sns
import matplotlib.pyplot as plt

import h5pyd
from tqdm import tqdm
from joblib import Parallel, delayed

from DeepUrfold.Analysis.AllVsAll import get_representative_domains

import torch

try:
    import ot
except (ImportError, ModuleNotFoundError):
    print("Unable to run optimal transport without POT package")
    ot = None

class ExploreLatent():
    @staticmethod
    def reduce(*superfamilies, result_dir=None, data_dir=None, plot=True, method="umap", prefix="ExploreLatent", extra_features=None, seperate_sfams=False, optimal_transport=None, ot_combine_mode=None, domain_superfamily_mapping=None, use_pickle=False, pickle_file=None, cosine=False, mahalanobis=False, umap_nneighbors=15, umap_min_dist=0.1, combined_model=False):
        assert result_dir is not None
        assert data_dir is not None
        assert method in ["umap", "pca", "pca-all", "t-sne", "umap-all"]

        if umap_nneighbors != 15 and "umap" in method.lower():
            prefix += f"-umapneigh{umap_nneighbors}"

        spread = 1.
        if umap_min_dist != 0.1 and "umap" in method.lower():
            prefix += f"-umapdist{umap_min_dist}"
            if umap_min_dist>=1:
                spread = umap_min_dist+1

        if cosine:
            metric = "cosine"
            prefix += "-cosine"
            metric_kwds = None
        elif mahalanobis:
            metric = "mahalanobis"
            prefix += "-mahalanobis"
            metric_kwds = {"n_jobs":20}
            metric_kwds = None
        else:
            metric = "euclidean"

        if method.lower() == "umap":
            reducer = umap.UMAP(n_components=2, metric=metric, n_neighbors=umap_nneighbors, min_dist=umap_min_dist, metric_kwds=metric_kwds)
        elif method.lower() == "umap-all":
            reducer = umap.UMAP(n_components=10, metric=metric, n_neighbors=umap_nneighbors, min_dist=umap_min_dist, metric_kwds=metric_kwds)
        elif method.lower() == "pca":
            reducer = PCA(n_components=2)
        elif method.lower() == "pca-all":
            reducer = PCA(n_components=1024)
        elif method.lower() == "t-sne":
            reducer = TSNE(n_components=2, metric=metric)

        if extra_features is not None and not isinstance(extra_features, (list,tuple)):
            extra_features = [extra_features]

        result_dir = Path(result_dir)
        assert result_dir.is_dir()

        embeddings_file = result_dir / f"{prefix}-{method}-all_latent.h5"

        if embeddings_file.exists():
            data = pd.read_hdf(embeddings_file, "table")
        elif len(superfamilies) == 1 and Path(superfamilies[0]).exists():
            data = pd.read_hdf(superfamilies[0], "table")
        else:
            all_latent = None
            if domain_superfamily_mapping is not None:
                reps_domain_and_superfamilies = domain_superfamily_mapping
            else:
                reps_domain_and_superfamilies = None

            if not combined_model:
                latent_dirs = [result_dir / superfamily for superfamily in superfamilies]
            else:
                latent_dirs = [result_dir]

            print("latent_dirs", latent_dirs)

            for latent_dir in latent_dirs: #superfamily in superfamilies:
                #if superfamily in ["3.40.50.300", "1.10.510.10"]: continue
                result_file = latent_dir / "latent.csv" #result_dir / superfamily / "latent.csv"

                superfamily = latent_dir.name
                print("SFAM", superfamily)

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

                if not seperate_sfams and not combined_model:
                    #Get only latent space from superfamily
                    sfam_latent = results[results["superfamily"] == superfamily].set_index(["cathDomain", "superfamily"])
                else:
                    sfam_latent = results.set_index(["cathDomain", "superfamily"])

                if len(superfamilies) == 1:
                    #Add new superfamilies results as new rows
                    if all_latent is None:
                        all_latent = sfam_latent
                    else:
                        all_latent = pd.concat((all_latent, sfam_latent))
                elif len(superfamilies) > 1 and (optimal_transport is None or not optimal_transport):
                    #Add new superfamilies results as new cols
                    if not combined_model:
                        sfam_latent = sfam_latent.rename(columns=
                            {str(idx):f"{idx}_{superfamily}" for idx in range(1024)})
                    if all_latent is None:
                        all_latent = sfam_latent
                    else:
                        print("combining", superfamily)
                        all_latent = pd.concat((all_latent, sfam_latent), axis=1)
                else:
                    if ot_combine_mode != "haddamard":
                        if not combined_model:
                            sfam_latent = sfam_latent.rename(columns=
                                {str(idx):f"{idx}_{superfamily}" for idx in range(1024)})


                    if all_latent is None:
                        #First sfam is set as target
                        target_sfam = sfam_latent
                        all_latent = sfam_latent
                        
                    else:
                        if ot_combine_mode in ["haddamard", "contract"]:
                            # Sinkhorn Transport with Group lasso regularization l1l2
                            ot_l1l2 = ot.da.SinkhornL1l2Transport(reg_e=1e-1, reg_cl=2e0, max_iter=20, verbose=True)
                            ot_l1l2.fit(Xs=sfam_latent.values, ys=np.ones(sfam_latent.shape[0]), Xt=target_sfam.values)
                            transp_sfam_latent = ot_l1l2.transform(Xs=sfam_latent)
                            
                            if ot_combine_mode == "haddamard":
                                all_latent = all_latent * transp_sfam_latent
                            elif ot_combine_mode == "contract":
                                new_mat = []
                                for (_, emb1), emb2 in zip(all_latent.iterrows(), transp_sfam_latent):
                                    logits = torch.einsum('i c, j c -> i j', torch.from_numpy(emb1.values), torch.from_numpy(emb2))
                                    new_mat.append(logits.flatten().numpy())
                                all_latent = pd.DataFrame(new_mat, index=all_latent.index, columns=list(range(new_mat[0].shape[1])))
                                    
                        else:
                            print("combining", superfamily)
                            transp_sfam_latent = pd.DataFrame(transp_sfam_latent, index=sfam_latent.index, columns=sfam_latent.columns)
                            all_latent = pd.concat((all_latent, transp_sfam_latent), axis=1)

            #all_latent.to_hdf(f"{prefix}-{method}-all.h5", "table")

            # if method.lower() == "umap-all":
            #     reducer = umap.UMAP(n_components=len(all_latent.values)-1)

            if isinstance(pickle_file, bool):
                pickle_file = Path(f"{prefix}-{method}-reducer-model.pickle")
            elif isinstance(pickle_file, str):
                pickle_file = Path(pickle_file)

            if use_pickle and isinstance(pickle_file, Path) and pickle_file.is_file():
                with pickle_file.open('rb') as f:
                    reducer = pickle.load(f)
                embedding = reducer.transform(all_latent.values)
            else:
                if False and len(superfamilies)==1 and len(domain_superfamily_mapping.superfamily.drop_duplicates()) > 1:
                    small_latent = all_latent.reset_index()
                    small_latent = small_latent[small_latent.superfamily==superfamilies[0]].set_index(["cathDomain", "superfamily"])
                    print(small_latent)
                    embedding = reducer.fit_transform(small_latent.values)
                    index = small_latent.index
                else:
                    embedding = reducer.fit_transform(all_latent.values)
                    index=all_latent.index
                pickle_file = Path(f"{prefix}-{method}-reducer-model.pickle")
                with pickle_file.open('wb') as f:
                    pickle.dump(reducer, f)

            embedding_df = pd.DataFrame(embedding, columns=range(embedding.shape[1]), index=index)
            embedding_df.to_csv(f"{prefix}-{method}-embedding-values.csv")

            if method.lower() == "pca-all":
                scree = pd.DataFrame(
                    {
                        "Pricipal Component":np.arange(reducer.n_components_)+1,
                        "Proportion of Variance Explained":reducer.explained_variance_ratio_
                    })

                sns.lineplot(data=scree, x="Pricipal Component", y="Proportion of Variance Explained",
                             estimator=None, markers=True, dashes=False)
                plt.title(f"Scree plot of Explained Varaince\n({reducer.n_components_} components)")
                plt.savefig(f"{prefix}-{method}-pca-scree.png", dpi=600, bbox_inches="tight")
                plt.savefig(f"{prefix}-{method}-pca-scree.pdf", dpi=600, bbox_inches="tight")
                plt.clf()

                sns.lineplot(data=scree[scree["Pricipal Component"]<11], x="Pricipal Component", y="Proportion of Variance Explained",
                             estimator=None, markers=True, dashes=False)
                plt.title(f"Scree plot of Explained Varaince\n({reducer.n_components_} components)")
                plt.savefig(f"{prefix}-{method}-pca-scree-small.png", dpi=600, bbox_inches="tight")
                plt.savefig(f"{prefix}-{method}-pca-scree-small.pdf", dpi=600, bbox_inches="tight")

                #scree.to_csv(f"{prefix}-{method}-pca-scree.csv")
                test_indices = np.random.choice(embedding.shape[0]-1, size=(10,), replace=False)
                test_true = all_latent.values[test_indices]

                def reduce_pca_comp(num_components):
                    reducer = PCA(n_components=num_components)
                    embedding = reducer.fit_transform(all_latent.values)
                    #test_embeddings = embedding[:10]
                    inv_transformed_points = reducer.inverse_transform(embedding[test_indices])
                    print(inv_transformed_points)
                    print(inv_transformed_points.shape)
                    mse = mean_squared_error(test_true, inv_transformed_points, multioutput="raw_values")
                    #dim_mse.append(mse.mean())
                    print(num_components, mse.mean())
                    #evs = explained_variance_score(test_true, inv_transformed_points, multioutput="raw_values")
                    #dim_evs.append(evs.mean())

                    return mse.mean()#, evs.mean()
                

                mse_ = Parallel(n_jobs=3)(delayed(reduce_pca_comp)(nc) for nc in tqdm(range(1,10)))
                mse = np.zeros(len(scree))
                mse[:len(mse_)]+=mse_
                
                scree = scree.assign(**{"Mean Square Error": mse})

                plt.clf()

                sns.lineplot(data=scree.iloc[:len(mse_)], x="Pricipal Component", y="Mean Square Error",
                             estimator=None, markers=True, dashes=False)
                plt.title(f"Scree plot of Mean Square Error\n({len(mse_)} dimensions)")
                plt.savefig(f"{prefix}-{method}-pca-mse.png", dpi=600, bbox_inches="tight")
                plt.savefig(f"{prefix}-{method}-pca-mse.pdf", dpi=600, bbox_inches="tight")
                plt.clf()

                return
            
            if method.lower() == "umap-all":
                test_indices = np.random.choice(embedding.shape[0]-1, size=(10,), replace=False)
                test_embeddings = embedding[test_indices]

                test_true = all_latent.values[test_indices]
                dim_mse = []
                dim_evs = []
                
                def reduce_umap_comp(num_components):
                    reducer = umap.UMAP(n_components=num_components, metric=metric, n_neighbors=umap_nneighbors, min_dist=umap_min_dist, metric_kwds=metric_kwds)
                    embedding = reducer.fit_transform(all_latent.values)
                    #test_embeddings = embedding[:10]
                    inv_transformed_points = reducer.inverse_transform(embedding[test_indices])
                    print(inv_transformed_points)
                    print(inv_transformed_points.shape)
                    mse = mean_squared_error(test_true, inv_transformed_points, multioutput="raw_values")
                    #dim_mse.append(mse.mean())
                    print(num_components, mse.mean())
                    evs = explained_variance_score(test_true, inv_transformed_points, multioutput="raw_values")
                    #dim_evs.append(evs.mean())

                    return mse.mean(), evs.mean()
                

                results = Parallel(n_jobs=3)(delayed(reduce_umap_comp)(nc) for nc in tqdm(range(2,7)))
                dim_mse, dim_evs = zip(*results)


                scree = pd.DataFrame(
                    {
                        "UMAP Dimension":np.arange(len(dim_mse))+2,
                        "Mean Square Error":dim_mse,
                        "Explained Variance":dim_evs
                    })

                sns.lineplot(data=scree, x="UMAP Dimension", y="Mean Square Error",
                             estimator=None, markers=True, dashes=False)
                plt.title(f"Scree plot of Mean Square Error\n({len(dim_mse)} dimensions)")
                plt.savefig(f"{prefix}-{method}-umap-mse.png", dpi=600, bbox_inches="tight")
                plt.savefig(f"{prefix}-{method}-umap-mse.pdf", dpi=600, bbox_inches="tight")
                plt.clf()

                sns.lineplot(data=scree, x="UMAP Dimension", y="Explained Variance",
                             estimator=None, markers=True, dashes=False)
                plt.title(f"Scree plot of Explained Variance\n({len(dim_mse)} dimensions)")
                plt.savefig(f"{prefix}-{method}-umap-var.png", dpi=600, bbox_inches="tight")
                plt.savefig(f"{prefix}-{method}-umap-var.pdf", dpi=600, bbox_inches="tight")
                plt.clf()

                scree.to_csv(f"{prefix}-{method}-umap-scree.csv")

                return

            data = pd.DataFrame(embedding, columns=[f"{method.upper()} Dimension 1", f"{method.upper()} Dimension 2"], index=index) #all_latent.index)
            data = data.assign(ss_score=np.nan, electrostatics=np.nan)

            #Load features from previous run
            #for other_method in ["umap", "pca", "t-sne"]:
            for other_embedding_file in Path.cwd().glob("*-all_latent.h5"):
                other_data = pd.read_hdf(other_embedding_file, "table")
                if other_data.index.equals(data.index) and "ss_score" in other_data.columns and "electrostatics" in other_data.columns:
                    data = data.assign(ss_score=other_data.ss_score, electrostatics=other_data.electrostatics)
                    break
        
                # other_embedding_file = embeddings_file.with_name(f"{prefix}-{other_method}-all_latent.h5")
                # print("Searching for previos file", other_embedding_file)
                # if other_embedding_file.exists():
                #     other_data = pd.read_hdf(other_embedding_file, "table")
                #     data = data.assign(ss_score=other_data.ss_score, electrostatics=other_data.electrostatics)
                #     break
            else:
                #Does not exist, calculate features
                print("calculating features")
                with h5pyd.File(data_dir, use_cache=False) as f:
                    d = data.index.to_frame()
                    for _, (cathDomain, superfamily) in tqdm(d.iterrows()):
                        try:
                            atoms = f[f"{superfamily.replace('.', '/')}/domains/{cathDomain}/atom"][:]
                            alpha = atoms["is_helix"].sum()
                            beta = atoms["is_sheet"].sum()
                            ss_score = (beta-alpha)/(2*(beta+alpha))+0.5
                            elec = 1-atoms["is_electronegative"].sum()/atoms.shape[0]
                            data.loc[(cathDomain, superfamily), "ss_score"] = ss_score
                            data.loc[(cathDomain, superfamily), "electrostatics"] = elec
                            if extra_features is not None:
                                for extra_feature in extra_features:
                                    data.loc[(cathDomain, superfamily), extra_feature] = atoms[extra_feature].sum()/atoms.shape[0]
                        except KeyError:
                            pass

            data.to_hdf(f"{prefix}-{method}-all_latent.h5", "table")

        if len(superfamilies)==1 and len(domain_superfamily_mapping.superfamily.drop_duplicates()) > 1:
            highlight_sfam = superfamilies[0]
        else:
            highlight_sfam = None

        if plot:
            ExploreLatent.plot(data, prefix, method, feature="ss", highlight_sfam=highlight_sfam)
            ExploreLatent.plot(data, prefix, method, feature="superfamily", highlight_sfam=highlight_sfam)
            ExploreLatent.plot(data, prefix, method, feature="electrostatics", highlight_sfam=highlight_sfam)
            if extra_features is not None:
                if not isinstance(extra_features, (list, tuple)):
                    extra_features = [extra_features]
                for extra_feature in extra_features:
                    ExploreLatent.plot(data, prefix, method, extra_feature)

    @staticmethod
    def plot(data, prefix, method, feature="ss", highlight_sfam=None, old=False):
        assert feature in ["ss", "electrostatics", "superfamily"]
        import matplotlib as mpl
        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.linewidth'] = 2
        # plt.rcParams.update({
        #     "text.usetex": True,
        #     'text.latex.preamble' : [r'\usepackage{amsmath}']
        # })
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
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
        elif feature == "superfamily":
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
        elif feature in data.columns:
            cmap = sns.color_palette("vlag", as_cmap=True)
            data = data.assign(color=data[feature].apply(lambda x: cmap(x)))
            colorbar_label = f"Average {feature}"
        else:
            raise RuntimeError(f"Invalid feature: {feature}")

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

        if highlight_sfam is not None:
            # data = data.assign(**{"Superfamily": data["superfamily"].apply(
            #     lambda x: f"Is {highlight_sfam}" if x==highlight_sfam else "Other Superfamily"})
            
            sns.scatterplot(data=data[data.superfamily!=highlight_sfam], x=d1_name, y=d2_name, s=10, c=data[data.superfamily!=highlight_sfam]["color"])
            sns.scatterplot(data=data[data.superfamily==highlight_sfam], x=d1_name, y=d2_name, s=10, c=data[data.superfamily==highlight_sfam]["color"], marker="*")
            #sns.scatterplot(data=data, x=d1_name, y=d2_name, s=10, c=data["color"], style="Superfamily")
        else:
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

def run(args, umap_nneighbors=15, min_dist=0.1):
    if not args.seperate_sfams:
        for method in args.method:
            ExploreLatent.reduce(
                *args.superfamilies, 
                result_dir=args.result_dir, 
                data_dir=args.data_dir, 
                prefix=args.prefix, 
                method=method, 
                extra_features=args.extra_features,
                seperate_sfams=args.seperate_sfams,
                use_pickle=args.use_pickle,
                pickle_file=args.pickle_file,
                cosine=args.cosine,
                mahalanobis=args.mahalanobis,
                combined_model=args.combined_model,
                umap_nneighbors=umap_nneighbors,
                umap_min_dist=min_dist
                )
    elif args.seperate_sfams_axis == "col":
        all_reps = get_representative_domains(args.superfamilies, args.data_dir)
        for method in args.method:
            ExploreLatent.reduce(
                *args.superfamilies, 
                result_dir=args.result_dir, 
                data_dir=args.data_dir, 
                prefix=args.prefix, 
                method=method, 
                extra_features=args.extra_features,
                seperate_sfams=True,
                optimal_transport=args.optimal_transport,
                ot_combine_mode=args.ot_combine_mode,
                use_pickle=args.use_pickle,
                pickle_file=args.pickle_file,
                cosine=args.cosine,
                mahalanobis=args.mahalanobis,
                umap_nneighbors=umap_nneighbors,
                umap_min_dist=min_dist
                )
    else:
        all_reps = get_representative_domains(args.superfamilies, args.data_dir)
        for sfam in args.superfamilies:
            for method in args.method:
                try:
                    ExploreLatent.reduce(
                        sfam, 
                        result_dir=args.result_dir, 
                        data_dir=args.data_dir, 
                        prefix=args.prefix+f"-{sfam}", 
                        method=method, 
                        extra_features=args.extra_features,
                        seperate_sfams=True,
                        domain_superfamily_mapping=all_reps,
                        use_pickle=args.use_pickle,
                        pickle_file=args.pickle_file,
                        cosine=args.cosine,
                        mahalanobis=args.mahalanobis,
                        umap_nneighbors=umap_nneighbors,
                        umap_min_dist=min_dist
                        )
                except Exception:
                    raise

if __name__ == "__main__":
    import argparse
    sfams = "1.10.10.10 1.10.238.10 1.10.490.10 1.10.510.10 1.20.1260.10 2.30.30.100 2.40.50.140 2.60.40.10 3.10.20.30 3.30.230.10 3.30.300.20 3.30.310.60 3.30.1360.40 3.30.1370.10 3.30.1380.10 3.40.50.300 3.40.50.720 3.80.10.10 3.90.79.10 3.90.420.10".split()
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prefix", default="latent-space")
    parser.add_argument("-d", "--data-dir", required=True)
    parser.add_argument("-r", "--result-dir", required=True)
    parser.add_argument("-m", "--method", nargs="+", default=["umap", "t-sne", "pca"], choices=["umap", "t-sne", "pca", "pca-all", "umap-all"])
    parser.add_argument("--extra_features", nargs="+", default=None)
    parser.add_argument("--seperate_sfams", action="store_true", default=False)
    parser.add_argument("--seperate_sfams_axis", default="row", choices=["row", "col"])
    parser.add_argument("--use_pickle", default=False, action="store_true")
    parser.add_argument("--pickle_file", default=None)
    parser.add_argument("--cosine", default=False, action="store_true")
    parser.add_argument("--mahalanobis", default=False, action="store_true")
    parser.add_argument("--combined_model", action="store_true", default=False)
    parser.add_argument("--vary_umap_neighbors", action="store_true", default=False)
    parser.add_argument("--vary_umap_min_dist", action="store_true", default=False)
    parser.add_argument("--optimal_transport", action="store_true", default=False)
    parser.add_argument("--ot_combine_mode", default=None, choices=["col", "haddamard", "contract"])
    parser.add_argument("superfamilies", nargs="+", default=sfams, help="superfamilies to include", )
    
    #parser.add_argument("files", nargs="+", default=None, help="Path to pytorch tensof files")
    args = parser.parse_args()

    if args.vary_umap_neighbors:
        for n_neighbors in (2, 5, 10, 20, 50, 100, 200):
            run(args, n_neighbors=n_neighbors)
    elif args.vary_umap_min_dist:
        for min_dist in (0.8, 1.0, 2.0, 10.): #(0., 0.2, 0.5, 1, 10):
            run(args, min_dist=min_dist)
    else:
        run(args)