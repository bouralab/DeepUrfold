import os
import sys
import copy
import glob
import argparse
import subprocess
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd

from DeepUrfold.Analysis.AllVsAll.StructureBased  import StructureBasedAllVsAll
from DeepUrfold.DataModules.DistributedDomainStructureDataModule import DistributedDomainStructureDataModule
from DeepUrfold.util import str2boolorval

#Must be after AVA
import torch


class DeepUrfoldOneModelAllVsAll(StructureBasedAllVsAll):
    NJOBS = 16
    METHOD = "DeepUrfold"
    DATA_INPUT = "CATH S35"
    SCORE_INCREASING = False
    MODEL_TYPE = "3D-CNN VAE"
    SCORE_METRIC = "ELBO"
    SCORE_FN = "negative_log"
    GPU = True

    def __init__(self, superfamilies, data_dir, hparams, latent=False, all=False, classification=None, lrp=False, permutation_dir=None, work_dir=None, force=False, result_dir="", result_dir2=None):
        self.raw_hparams = copy.deepcopy(hparams)
        super().__init__(superfamilies, data_dir, hparams, permutation_dir=permutation_dir, work_dir=work_dir, force=force, downsample_sbm=False, cluster=False)
        self.latent = latent
        self.all = all
        self.classification = classification
        self.lrp = lrp
        self.result_dir = result_dir
        self.result_dir2 = result_dir2

        if result_dir2 is not None:
            flare = os.path.join(result_dir2, "flare.csv")
            if os.path.isfile(flare):
                self.old_flare = flare

        if latent:
            self.SCORE_METRIC = "Latent"
            self.SCORE_INCREASING = True
            self.SCORE_FN = None
            if all:
                self.METHOD += "LatentAll"
            else:
                self.METHOD += "LatentSfam"
        elif self.classification is not None or (isinstance(self.classification, bool) and self.classification):
            self.SCORE_INCREASING = True
            self.SCORE_FN = None
            self.METHOD = "Clasification"
            if self.classification == "auroc" or (isinstance(self.classification, bool) and self.classification):
                self.classification = "auroc"
                self.SCORE_METRIC = "AUROC"
            elif self.classification == "auprc":
                self.SCORE_METRIC = "AUPRC"
            elif self.classification == "validation":
                self.cluster = False
            else:
                raise RuntimeError("classification must be [True, 'auroc', 'auprc']")

    def train_all(self, *args, **kwds):
        """Not implemented
        
        
        """
        completed_model = Path(self.result_dir) / "last.ckpt"
        if completed_model.is_file() and not self.force:
            return completed_model

        subprocess.call([
            "python", "-m", "DeepUrfold.Trainers.DistributedTrainSuperfamilyVAE",
            "--superfamily", *self.superfamily_datasets.keys(),
            "--data_dir", "/home/ed4bu/deepurfold-paper-2.h5",
            "--under_sample",
            "--gpu", "4",
            "--batch_size", "96",
            "--num_workers", "8",
            "--no_early_stopping"
        ])

        if not completed_model.is_file():
            raise RuntimeError("Unable to train model")

        return completed_model

    def completed_inference(self, model_name, result_dir=None):
        results_file = os.path.join(self.result_dir, model_name) if result_dir is None else os.path.join(result_dir, model_name)
        if self.latent:
            if self.all:
                results_file = os.path.join(results_file, "latent.csv")
            else:
                results_file = os.path.join(results_file, "latent_individual.csv")
        elif self.classification in ["auroc", "auprc"]:
            raise NotImplementedError("AUROC and AUPRC scores as similairty scores has been implemented yet")
            results_file = os.path.join(results_file, "classification")
        elif self.classification == "validation":
            results_file = os.path.join(results_file, "validation", "classification  All Combined-ROC_curve.pdf")
            if os.path.isfile(results_file):
                return pd.Series([results_file], name=model_name, index=pd.MultiIndex.from_arrays([["1"], ["1"]], names=("cathDomain", "true_sfam")))
            return None
        elif self.lrp:
            if True:
                suffix = "f_agg=npsum-total_relevance.h5"
            else:
                suffix = "v_agg=arithmetic_mean__f_agg=npsum-total_relevance.75pctquntile.pdb"
            lrp_files = [Path(results_file) / "lrp" / row.cathDomain / f"{row.superfamily.replace('.', '_')}-{row.cathDomain}-{suffix}" \
                    for row in self.combined_dataset.itertuples()]
            missing = [f for f in lrp_files if not f.is_file()]
            print(model_name, len(missing), missing)
            if all([f.is_file() for f in lrp_files]):
                return pd.Series([None], name=model_name, index=pd.MultiIndex.from_arrays([["1"], ["1"]], names=("cathDomain", "true_sfam")))
            return None
        else:
            results_file = os.path.join(results_file, "elbo.csv")

        results = None
        if os.path.isfile(results_file):
            if not self.latent or not self.all:
                results =  pd.read_csv(results_file).set_index("cathDomain")[model_name]
            else:
                results =  pd.read_csv(results_file)
                if "Unnamed: 0" in results:
                    results = results.drop(columns=["Unnamed: 0"])

                if "index" in results and "cathDomain" not in results:
                    if "', '" in results["index"].iloc[0]:
                        #Has superfamlily info
                        results.index = pd.MultiIndex.from_frame(
                            results["index"].str[2:-2].str.split("', '", expand=True).rename(
                                columns={0:"cathDomain", 1:"superfamily"}))
                        results = results.drop(columns=["index"])
                        if "superfamily" in results:
                            results = results.drop(columns=["superfamily"])
                    else:
                        results = results.rename(columns={"index":"cathDomain"})

                results = results.set_index("cathDomain")
                results = pd.Series(list(results.values), name=model_name, index=results.index)
            if len(results.dropna()) != len(results) and len(results) != 3674:
                results = None

        if results is None:
            if result_dir is None and self.result_dir2 is not None:
                #Use other model, not recommended
                return self.completed_inference(model_name, result_dir=self.result_dir2)

        return results

    def infer_all(self, model_path, combined_dataset, gpu=0):
        print(combined_dataset)

        distances = None
        combined_dataset_ = combined_dataset

        #out_dir = "/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/superfamilies_for_paper/"
        out_dir = Path(self.result_dir) / "results"
        out_dir.mkdir()

        print("ou_dir", out_dir)
        print("model_path", model_path)
        print("domains", len(combined_dataset.cathDomain))

        os.makedirs(out_dir, exist_ok=True)

        if self.latent:
            result_path = "latent-mean-mean.pt"
            labels_path = "latent-labels-mean-labels-mean.npy"
            label = "latent"
            command_tail = [
                "--return_latent",
                "--raw", "mean", "log_var",
                "--prefix", "latent"
            ]
        elif self.classification is not None or (isinstance(self.classification, bool) and self.classification):
            #Use AUCROC or AUPRC
            command_tail = [
                "--return_reconstruction",
                "--classification",
                "--prefix", "classification"
            ]

            if self.classification == "auroc" or (isinstance(self.classification, bool) and self.classification):
                result_path = "classification-auroc-auroc.pt"
                labels_path = "classification-labels-auroc-labels-auroc.npy"
                label = "auroc"
            elif self.classification == "auprc":
                result_path = "classification-auprc-auprc.pt"
                labels_path = "classification-labels-auprc-labels-auprc.npy"
                label = "auprc"
            elif self.classification == "validation":
                result_path = "classification  All Combined-ROC_curve.pdf"
                labels_path = "DoesNoTExist"
                out_dir /= "validation"
                out_dir.mkdir(exist_ok=True)
            else:
                raise RuntimeError("classification must be [True, 'auroc', 'auprc', 'validation']")
        elif self.lrp:
            command_tail = [
                "--prefix", "lrp",
                "--return_latent",
                "--no_compute",
                "--lrp"
            ]
            if True:
                command_tail += ["--atomic_relevance", "None", "npsum"]
            result_path = "LRP?"
            labels_path = "DoesNoTExist"
            out_dir /= "lrp"
            out_dir.mkdir(exist_ok=True)
            if True:
                done_domains = [f.parent.stem for f in out_dir.glob("*/*-f_agg=npsum-total_relevance.h5")]
            else:
                done_domains = [f.parent.stem for f in out_dir.glob("*/*-v_agg=arithmetic_mean__f_agg=npsum-total_relevance.75pctquntile.pdb")]
            print(len(done_domains))
            combined_dataset_ = combined_dataset_[~combined_dataset_.cathDomain.isin(done_domains)]
            print(model_name, combined_dataset_)
        else:
            #Use ELBO as default
            result_path = "elbo-elbo-elbo.pt"
            labels_path = "elbo-labels-elbo-labels-elbo.npy"
            label = "elbo"
            command_tail = [
                "--raw", "elbo",
                "--prefix", "elbo"
            ]

        if self.classification == "validation":
            #Only perform on the same sfam
            command = [
                sys.executable,
                "-m", "DeepUrfold.Evaluators.EvaluateDistrubutedDomainStructureVAE",
                "--superfamily", model_name,
            ]
        else:
            command = [
                sys.executable,
                "-m", "DeepUrfold.Evaluators.EvaluateDistrubutedDomainStructureVAE",
                "--superfamily", *combined_dataset_.superfamily.drop_duplicates().tolist(),
                "--domains", *combined_dataset_.cathDomain.tolist(),
            ]

        command += [
            # sys.executable,
            # "-m", "DeepUrfold.Evaluators.EvaluateDistrubutedDomainStructureVAE",
            # "--superfamily", *combined_dataset_.superfamily.drop_duplicates().tolist(),
            # "--domains", *combined_dataset_.cathDomain.tolist(),
            "--data_dir", self.raw_hparams.data_dir, #/home/ed4bu/deepurfold-paper-2.h5",
            "--checkpoint", os.path.abspath(model_path),
            "--features", "H;HD;HS;C;A;N;NA;NS;OA;OS;SA;S;Unk_atom__is_helix;is_sheet;Unk_SS__residue_buried__is_hydrophobic__pos_charge__is_electronegative",
            "--feature_groups", "Atom Type;Secondary Structure;Solvent Accessibility;Hydrophobicity;Charge;Electrostatics",
            "--accelerator", 'gpu',
            "--gpu", f"{gpu},",
            "--batch_size", "128",
            "--num_workers", "60",
            "--batchwise_loss", "False",
        ] + command_tail

        print(" ".join(command))

        results_file = out_dir / result_path
        labels_file = out_dir / labels_path

        if not os.path.exists(results_file):
            p = subprocess.Popen(command, cwd=str(out_dir), stderr=sys.stderr, stdout=sys.stdout)
            p.communicate()
            p.wait()

        if self.classification == "validation":
            distances = pd.Series([results_file], name=model_name, index=pd.MultiIndex.from_arrays([["1"], ["1"]], names=("cathDomain", "true_sfam")))
            return model_name, distances
        elif self.lrp:
            #lrp_files = pd.out_dir / f"{row.cathDomain}-v_agg=arithmetic_mean__f_agg=npsum.h5'" for row in combined_dataset.itertuples()]
            return model_name, pd.Series(["LRP"], name=model_name, index=pd.MultiIndex.from_arrays([["1"], ["1"]], names=("cathDomain", "true_sfam")))
        try:
            result = torch.load(results_file).numpy()
            labels = np.load(labels_file)
            if self.latent and not self.all:
                import umap
                result = umap.UMAP(n_components=1).fit_transform(result).flatten()
                label += "_individual"
        except FileNotFoundError:
            raise
            result = np.array([np.nan]*self.representative_domains)

        if self.latent and self.all:
            distances = pd.DataFrame(result, columns=range(result.shape[1]), index=labels) #self.representative_domains)

            #Reorder array based on CATH Reps
            distances = distances.loc[self.representative_domains["cathDomain"]]
            distances.reset_index().to_csv(os.path.join(out_dir, f"{label}.csv"))
            distances = pd.Series(list(distances.values), name=model_name, index=self.representative_domains["cathDomain"])
        else:
            distances = pd.Series(result, name=model_name, index=self.representative_domains["cathDomain"])
            distances.to_csv(os.path.join(out_dir, f"{label}.csv"))

        # if self.latent and self.all:
        #     sh = distances.shape
        #     all_values = distances.values.resize((sh[0]*sh[1],))
        #     import umap
        #     result = umap.UMAP(n_components=1).fit_transform(all_values).flatten()
        #     distances.values = result.reshape(sh[0], sh[1])

        # for test_family, cathDomains in self.representative_domains.groupby("superfamily"):
        #     #results_file_ = os.path.join(out_dir, model_name, test_family, "elbo", "*.pt")
        #     results_file = result_path(test_family) #list(glob.glob(results_file_))
        #     if len(results_file) == 0 or (len(results_file)>0 and not os.path.isfile(results_file[0])):
        #         #raise RuntimeError(f"No results file found at {results_file_}")
        #         df = cathDomains.copy(deep=True).reset_index(drop=True).rename(
        #             columns={"superfamily":"true_sfam"}).assign(model=model_name)
        #         if self.latent:
        #             df = df.assign(latent=np.nan)
        #         else:
        #             df = df.assign(ELBO=np.nan)
        #     else:
        #         results_file = results_file[0]
        #         df = cathDomains.copy(deep=True).reset_index(drop=True)
        #         df = df.rename(columns={"superfamily":"true_sfam"})
        #         df = df.assign(model=model_name)
        #         if self.latent:
        #             import umap
        #             df = pd.read_hdf(results_file, "table")
        #             s = umap.UMAP(n_components=1).fit_transform(df[list(range(1024))]).flatten()
        #             print(s)
        #             scores = pd.Series(s, name="latent", index=df.index)
        #             print("UMAP")
        #         else:
        #             scores = pd.Series(torch.load(results_file).numpy(), name="ELBO", index=df.index)
        #         df = pd.concat((df, scores), axis=1)
        #     if distances is not None: #sources["model"] in models:
        #         distances = pd.concat((distances, df))
        #     else:
        #         distances = df
        #
        # if distances is not None:
        #     distances = distances.pivot_table(index=['cathDomain', 'true_sfam'], columns='model')
        #     distances.columns = [c[1] for c in distances.columns]
        # else:
        #     distances = pd.DataFrame([], index=pd.MultiIndex(names=['cathDomain', 'true_sfam']), columns=[model_name])

        print(distances)
        return model_name, distances

    def combine_validation_latex(self):
        from zipfile import ZipFile
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set()

        supp_mat_file = Path(self.result_dir) / "supp_mat.tex"
        #Reset figure
        with supp_mat_file.open("w") as f:
            pass

        image_zip = ZipFile(Path(self.result_dir) / "supp_images.zip", 'w')

        for superfamily in self.superfamilies:
            validation_dir = Path(self.result_dir) / superfamily / "validation"

            all_metrics = None
            for metric_file in sorted(validation_dir.glob("*-metrics.csv")):
                if "Separated" in metric_file.name:
                    continue
                metrics = pd.read_csv(str(metric_file))
                metrics["Feature"] = metrics["Feature"].str.replace("classification ", "")
                if all_metrics is None:
                    all_metrics = metrics
                else:
                    all_metrics = pd.concat((all_metrics, metrics))

            plt.figure(figsize=(8,6))
            sns.barplot(x="Feature", y="Value", hue="Metric", data=all_metrics)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(str(validation_dir/"all_metrics.png"))
            image_zip.write(str(validation_dir/"all_metrics.png"), f"classification/{superfamily}/all_metrics.png")

            with supp_mat_file.open("a") as f:
                print(f"\subsubsection{{{superfamily}}}", file=f)
                print(f"""\\begin{{figure}}[h!]
    \centering
    \includegraphics[width=0.3\\textwidth]{{Chapter3/images/classification/{superfamily}/all_metrics.png}}
    \caption{{The {superfamily} model was trained with all features, but individual feature groups were separated to perform micro-averaging ROC, PRC, and F1 scores}}
    \label{{fig:{superfamily}-all}}
\end{{figure}}""", file=f)

            for metric_file in validation_dir.glob("* Separated-metrics.csv"):
                feature_group = metric_file.name.replace("classification", "").split("Separated")[0].strip()
                metrics = pd.read_csv(str(metric_file))
                metrics = metrics[metrics.Metric=="F1"]
                plt.figure(figsize=(8,6))
                sns.barplot(x="Feature", y="Value", data=metrics)
                plt.savefig(str(metric_file.with_name(f"{metric_file.name.split('-')[0]}-F1.png")))

                image_zip.write(
                    str(metric_file.with_name(f"{metric_file.name.split('-')[0]}-roc.pdf")),
                    f"classification/{superfamily}/{metric_file.name.split('-')[0]}-roc.pdf")

                image_zip.write(
                    str(metric_file.with_name(f"{metric_file.name.split('-')[0]}-prc.pdf")),
                    f"classification/{superfamily}/{metric_file.name.split('-')[0]}-prc.pdf")

                image_zip.write(
                    str(metric_file.with_name(f"{metric_file.name.split('-')[0]}-F1.png")),
                    f"classification/{superfamily}/{metric_file.name.split('-')[0]}-F1.png")


                with supp_mat_file.open("a") as f:
                    print(f"""\\begin{{figure}}[h!]
    \\begin{{subfigure}}[h]{{0.3\linewidth}}
        \includegraphics[width=\linewidth]{{Chapter3/images/classification/{superfamily}/{metric_file.name.split('-')[0]}-roc.pdf}}
        \caption{{Micro-averaged ROC Curve for separated features in the {feature_group} feature group.}}
    \end{{subfigure}}
    \\begin{{subfigure}}[h]{{0.3\linewidth}}
        \includegraphics[width=\linewidth]{{Chapter3/images/classification/{superfamily}/{metric_file.name.split('-')[0]}-prc.pdf}}
        \caption{{Micro-averaged PRC Curve for separated features in the {feature_group} feature group.}}
    \end{{subfigure}}
    \\bigskip
    \\begin{{subfigure}}[h]{{0.3\linewidth}}
        \includegraphics[width=\linewidth]{{Chapter3/images/classification/{superfamily}/{metric_file.name.split('-')[0]}-F1.png}}
        \caption{{Micro-averaged F1 values for separated features in the {feature_group} feature group.}}
    \end{{subfigure}}%
    \caption{{Classification metrics for separated features in the {feature_group} feature group.}}
\end{{figure}}

""", file=f)
                    print("\\newpage", file=f)

        image_zip.close()


if __name__ == "__main__":
    sfams = "1.10.10.10 1.10.238.10 1.10.490.10 1.10.510.10 1.20.1260.10 2.30.30.100 2.40.50.140 2.60.40.10 3.10.20.30 3.30.230.10 3.30.300.20 3.30.310.60 3.30.1360.40 3.30.1370.10 3.30.1380.10 3.40.50.300 3.40.50.720 3.80.10.10 3.90.79.10 3.90.420.10".split()
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    parser.add_argument("-w", "--work_dir", default=os.getcwd(), required=False)
    parser.add_argument("--result-dir", default=os.getcwd())
    parser.add_argument("--result-dir2", default=None)
    parser.add_argument("-p", "--permutation_dir", default="/home/bournelab/urfold_runs/multiple_loop_permutations/sh3_3", required=False)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-l", "--latent", action="store_true")
    parser.add_argument("--single_model_umap", action="store_true")
    parser.add_argument("-c", "--classification", type=partial(str2boolorval, choices=["auroc", "auprc", "validation"]), nargs='?',
                       const=True, default=None)
    parser.add_argument("--lrp", action="store_true")
    parser.add_argument("ava_superfamily", nargs="+")
    parser = DistributedDomainStructureDataModule.add_data_specific_args(parser, eval=True)
    parser.set_defaults(
        data_dir="/home/ed4bu/deepurfold-paper-2.h5",
        all_domains=True)
    args = parser.parse_args()
    print(args)
    runner = DeepUrfoldOneModelAllVsAll(
        args.ava_superfamily,
        args.data_dir,
        args,
        latent=args.latent,
        all=not args.single_model_umap,
        classification=args.classification,
        lrp=args.lrp,
        permutation_dir=args.permutation_dir,
        work_dir=args.work_dir,
        force=args.force,
        result_dir=args.result_dir,
        result_dir2=args.result_dir2
    )
    runner.run()
