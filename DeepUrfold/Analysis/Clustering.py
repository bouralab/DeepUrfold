import os
import re
import sys
import glob
import json
import subprocess
import itertools

import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.optimize import brentq
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed

import matplotlib
import seaborn as sns
import graph_tool as gt
import matplotlib.pyplot as plt

from molmimic.util.pdb import get_b

class Clustering(object):
    def __init__(self, distances, tool_name, model, model_input, score_type, increasing=True, sample=False):
        self.distances = distances
        self.tool_name = tool_name
        self.model = model
        self.model_input = model_input
        self.score_type = score_type
        self.increasing = increasing
        self.domain_group_membership = None
        self.sfam_group_membership = None

        self.path_prefix = f"{self.tool_name}-{self.model_input.replace(' ', '_')}_plots"
        os.makedirs(self.path_prefix, exist_ok=True)

        if distances is not None:
            self.sfams = self.distances.index.get_level_values('true_sfam').drop_duplicates()
        else:
            self.sfams = []
        print(self.sfams)

        self.create_discriminators()

        if sample:
            if isinstance(sample, int):
                self.sample(n=sample)
            else:
                self.sample()

    @classmethod
    def from_hdf(cls, distance_file, tool_name, model, model_input, score_type, increasing=True, sample=False, hdf_key="table"):
        distances = pd.read_hdf(distance_file, hdf_key)
        return cls(distances, tool_name, model, model_input, score_type, increasing=increasing, sample=sample)

    @classmethod
    def from_csv(cls, distance_file, tool_name, model, model_input, score_type, increasing=True, sample=False, **csv_params):
        distances = pd.read_csv(distance_file, **csv_params)
        return cls(distances, tool_name, model, model_input, score_type, increasing=increasing, sample=sample)

    @classmethod
    def from_clusters(cls, domain_clusters, superfamily_clusters=None):
        if not isinsance(domain_clusters, pd.DataFrame):
            if os.path.isfile(domain_clusters):
                domain_clusters = pd.read_hdf(domain_clusters, "table")
            else:
                raise RuntimeError("domain_clusters must be a DataFrame or path to hdf file")

        if not isinsance(superfamily_clusters, pd.DataFrame):
            if os.path.isfile(superfamily_clusters):
                superfamily_clusters = pd.read_hdf(superfamily_clusters, "table")
            else:
                raise RuntimeError("superfamily_clusters must be a DataFrame or path to hdf file")


    def sample(self, n=50):
        def _sample(sfam):
            domain_scores = self.distances.loc[(slice(None), sfam), :]
            try:
                domain_scores = domain_scores.sample(n=n)
            except ValueError:
                pass
            return domain_scores
        self.original_distances = self.distances.copy()
        self.distances = pd.concat(_sample(sfam) for sfam in self.sfams)
        print(f"Size {len(self.original_distances)} -> {len(self.distances)}")

    def create_discriminators(self):
        self.kde_descriminators = {}
        self.log_odds_descriminator = {}
        self.log_odds = {}
        for sfam in self.sfams:
            try:
                self.get_descriminator(sfam)
            except KeyError as e:
                print(f"Skipping key {str(e)}")
                #Skip key
                continue

    # Fit KDE
    def _kde_sklearn(self, x, x_grid, **kwargs):
        """Kernel Density Estimation with Scikit-learn"""
        from seaborn.external.kde import gaussian_kde
        try:
            bw = gaussian_kde(x).scotts_factor() * x.std(ddof=1)
        except np.linalg.LinAlgError:
            bw = 0.1
        kde_skl = KernelDensity(bandwidth=bw, kernel="gaussian", **kwargs)
        kde_skl.fit(x[:, np.newaxis])
        # score_samples() returns the log-likelihood of the samples
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
        return kde_skl, np.exp(log_pdf)

    # Find intersection
    def _findIntersection(self, fun1, fun2, lower, upper):
        return brentq(lambda x : fun1(x) - fun2(x), lower, upper)

    def get_descriminator(self, sfam_name, plot=True):
        self.get_descriminator_logodds(sfam_name)
        return self.get_descriminator_kde(sfam_name, plot=plot)

    def _plot(self, sfam, other, sfam_name, prefix="", vertical_lines={}): #descriminator=None, max_sfam=None, max_other=None):
        fig = plt.figure(figsize=(10, 6), dpi=300)
        ax = fig.subplots(1, 1)
        if sfam is not None:
            sns.kdeplot(sfam, label=f"True {sfam_name}", ax=ax)
        sns.kdeplot(other, label="Other Superfamilies", ax=ax)

        # if sklearn_kde is not None:
        #     sns.scatterplot(x_axis, pdfSfam, label=f"True {sfam_name} (sklearn)", ax=ax)
        #     sns.scatterplot(x_axis, pdfOther, label=f"Other Superfamilies (sklearn)", ax=ax)
        # if peaks is not None:
        #     plt.plot(peaks, sfam.values[peaks])
        colors = sns.color_palette("hls", 8)
        for i, (name, value) in enumerate(vertical_lines.items()):
            if value is None: continue
            ax.axvline(value, label="{} ({:.4f})".format(name, value), color=colors[i])

        plt.legend()
        plt.savefig(os.path.join(self.path_prefix, f"{prefix}_kde.png"))

    def find_descriminator_(self, data, x_axis=None):
        if x_axis is None:
            x_axis = np.linspace(0, .5, 500)

        if not np.all(np.isfinite(data)):
            data = np.nan_to_num(data)

        kde, pdf = self._kde_sklearn(data, x_axis)
        func_ = lambda x: kde.score_samples([[[x]]][0])

        peaks = find_peaks(pdf)[0]

        try:
            max_val = x_axis[ peaks[ pdf[peaks].argmax()] ]
            no_max=False
        except ValueError:
            no_max=True
            max_val = data.mean()

        return func_, max_val, no_max


    def find_descriminator(self, sfam, other, sfam_name):
        x_axis = np.linspace(0, .5, 500)

        # kdeSfam, pdfSfam = self._kde_sklearn(sfam, x_axis)
        # kdeOther, pdfOther = self._kde_sklearn(other, x_axis)
        # funcA = lambda x: kdeSfam.score_samples([[[x]]][0])
        # funcB = lambda x: kdeOther.score_samples([[[x]]][0])
        #
        # sfamPeaks = find_peaks(pdfSfam)[0]
        # otherPeaks = find_peaks(pdfOther)[0]
        #
        # try:
        #     max_sfam = x_axis[ sfamPeaks[pdfSfam[sfamPeaks].argmax()] ]
        #     max_other = x_axis[ otherPeaks[pdfOther[otherPeaks].argmax()] ]
        #
        #     no_max=False
        # except ValueError:
        #     no_max=True
        #     max_sfam = sfam.mean()
        #     max_other = other.mean()

        funcB, max_other, no_max_other = self.find_descriminator_(other, x_axis)

        if len(sfam) == 1:
            descriminator = sfam.iloc[0]+other.std()

            if no_max_other:
                return descriminator, None, None
            else:
                return descriminator, sfam.iloc[0], max_other
        else:
            funcA, max_sfam, no_max_sfam = self.find_descriminator_(sfam, x_axis)

        search_factor = 1
        if max_other <= max_sfam:
            #Something is wierd, but ok
            search_factor = 3

        try:
            descriminator = self._findIntersection(funcA, funcB,
                max_sfam, #sfam.mean(), #-sfam.std()*1,
                max_other+other.std()*search_factor #.3
                )
        except ValueError as e:
            print("Failed", sfam_name)
            descriminator = None

        if no_max_sfam or no_max_other:
            max_sfam = None
            max_other = None

        return descriminator, max_sfam, max_other

    def get_descriminator_kde(self, sfam_name, plot=True):
        one_sfam = self.distances[sfam_name].to_frame().reset_index()
        #one_sfam.columns = [x[0] for x in one_sfam.columns.tolist()]
        one_sfam = one_sfam.drop(columns=["cathDomain"])
        sfam = one_sfam[one_sfam["true_sfam"]==sfam_name][sfam_name]
        other = one_sfam[one_sfam["true_sfam"]!=sfam_name][sfam_name]

        descriminator, max_sfam, max_other = self.find_descriminator(sfam, other, sfam_name)

        print(sfam_name, descriminator)

        prefix=f"{sfam_name}-{self.score_type}"
        self._plot(sfam, other, sfam_name, prefix, vertical_lines=
            {"Descriminator":descriminator,
            f"Max Sfam {sfam_name}":max_sfam,
            f"Max Other {sfam_name}":max_other})

        self.kde_descriminators[sfam_name] = descriminator

    def get_descriminator_logodds(self, sfam_name, plot=True):
        one_sfam = self.distances[sfam_name].to_frame().reset_index("true_sfam")
        #one_sfam.columns = [x[0] for x in one_sfam.columns.tolist()]
        sfam = one_sfam[one_sfam["true_sfam"]==sfam_name][sfam_name]
        other = one_sfam[one_sfam["true_sfam"]!=sfam_name][sfam_name]

        sfam_log_median_score = np.log(sfam.median())
        sfam_log_score = sfam.apply(np.log).rename(f"log({self.score_type})")
        other_log_score = other.apply(np.log).rename(f"log({self.score_type})")
        log_odds = other_log_score-sfam_log_median_score
        sfam_log_odds = sfam_log_score-sfam_log_median_score
        log_odds = pd.concat((log_odds, sfam_log_score), axis=0)
        log_odds = log_odds.rename("Log-odds Score")
        self.log_odds[sfam_name] = log_odds
        self.log_odds_descriminator[sfam_name] = log_odds<0.05

        try:
            descriminator, max_sfam, max_other = self.find_descriminator(sfam_log_score, other_log_score, sfam_name)
        except Exception as e:
            print(e)
            import traceback as tb
            print(tb.format_exc())
            import pdb; pdb.set_trace()

        self._plot(sfam_log_score, other_log_score, sfam_name, f"{sfam_name}-log_{self.score_type}",
            vertical_lines=
                {"Descriminator":descriminator,
                f"Max Sfam {sfam_name}":max_sfam,
                f"Max Other {sfam_name}":max_other})

        self._plot(sfam_log_odds, log_odds, sfam_name, f"{sfam_name}-log_odds", vertical_lines=
            {f"Median log({self.score_type}) ({sfam_name})":sfam_log_median_score})

    def compare_to(self, other="CATH"):
        if other == "CATH":
            cluster_names = self.sfams.tolist()
            other_clusters = [cluster_names.index(true_sfam) for cathDomain, true_sfam in self.domain_group_membership.index]
        elif isinstance(other, Clustering):
            other_clusters = other.domain_group_membership[[c for c in other.domain_group_membership.columns if "Level" in c]]
            other_clusters = other_clusters.astype(str).agg('.'.join, axis=1)

        this_clusters = self.domain_group_membership[[c for c in self.domain_group_membership.columns if "Level" in c]]

        this_clusters = this_clusters.astype(str).agg('.'.join, axis=1)
        this_cluster_names = this_clusters.drop_duplicates().tolist()
        this_clusters_int = [this_cluster_names.index(domain) for domain in this_clusters]

        labels_true = other_clusters
        labels_pred = this_clusters

        return gt.inference.partition_overlap(labels_true, this_clusters_int), \
            metrics.rand_score(labels_true, labels_pred), \
            metrics.adjusted_rand_score(labels_true, labels_pred), \
            metrics.adjusted_mutual_info_score(labels_true, labels_pred), \
            metrics.homogeneity_score(labels_true, labels_pred), \
            metrics.completeness_score(labels_true, labels_pred), \
            #metrics.normalized_mutual_info_score(labels_true, labels_pred), \
            #metrics.v_measure_score(labels_true, labels_pred)

    def silhouette_score(self, metric='euclidean'):
        this_clusters = self.domain_group_membership[[c for c in self.domain_group_membership.columns if "Level" in c]]
        this_clusters = this_clusters.astype(str).agg('.'.join, axis=1)
        return metrics.silhouette_score(self.distances, this_clusters, metric=metric)

    def davies_bouldin_score(self):
        this_clusters = self.domain_group_membership[[c for c in self.domain_group_membership.columns if "Level" in c]]
        this_clusters = this_clusters.astype(str).agg('.'.join, axis=1)
        return metrics.davies_bouldin_score(self.distances, this_clusters)

    def internal_evaluation(self, silhouette_metric="euclidean"):
        return self.silhouette_score(silhouette_metric), self.davies_bouldin_score()

    COMPARISON_ROW_COLUMNS = ["Silhoette Score", "Davies-Boundin Score", "Overlap Score",
                              "Rand Score", "Rand Score Adjusted", "Adjusted Mutual Information", 
                              "Homogeneity Score", "Completeness Score"]
    def make_comparison_table_row(self, silhouette_metric="euclidean", deepurfold_clusters=None):
        row = [self.tool_name, self.model, self.model_input, self.score_type, self.n_clusters,
            *["${:.4f}$".format(s) for s in self.internal_evaluation()],
            " ",
            *["${:.4f}$".format(s) for s in self.compare_to("CATH")], " "]

        if deepurfold_clusters is not None:
            row += ["${:.4f}$".format(s) for s in self.compare_to(deepurfold_clusters)]
        else:
            row += list("-"*8)

        return row

    @classmethod
    def make_comparison_table(cls, x):
        rows = [model.make_comparison_table_row() for model in models]
        columns_sizes = [max(len(r) for r in cols) for cols in zip(*rows)]

        format = " & ".join([f"{{: >{s}}}" for s in columns_sizes])

        with open(os.path.join(self.path_prefix, "cluster_comparisons.tex")) as f:
            for row, size in zip(row, columns_sizes):
                print(format.format(row), end="\\\\\n")

    @staticmethod
    def read_clusters_and_process(cluster_file, process_node=None, process_leaf=None, n_jobs=None):
        df = pd.read_hdf(cluster_file, "table")
        ss = pd.read_hdf("/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/superfamilies_for_paper/All models-umap-all_latent.h5", "table")[["ss_score2", "cath_domain"]]
        from sklearn.preprocessing import MinMaxScaler
        ss = ss.assign(ss=MinMaxScaler().fit_transform(ss["ss_score2"].values.reshape((len(ss["ss_score2"]),1))).flatten()).drop(columns=["ss_score2"])
        df = pd.merge(df.reset_index(), ss, left_on="cathDomain", right_on="cath_domain")
        df = df.set_index(["cathDomain", "true_sfam"]).drop(columns=["cath_domain"])

        if process_node is None and process_leaf is None:
            return df

        if process_node is None:
            process_node = lambda c,n: None

        if process_leaf is None:
            process_leaf = lambda c,n: None

        def process_hierarchy(h, level=None, name=None):
            levels = sorted([c for c in h.columns if "Level" in c], key=lambda n: int(n.split()[1]), reverse=True)+["END"]
            assert level is None or level in levels
            if level is None:
                print("Running start with", levels[0])
                return process_hierarchy(h, levels[0], "root")
            elif level == levels[-1]:
                return process_leaf(h, name)

            hgroups = h.groupby(levels[:levels.index(level)+1], as_index=False)
            next_level = levels[levels.index(level)+1]

            if n_jobs is None:
                children = [process_hierarchy(hierarchy, next_level, level_name) \
                    for level_name, hierarchy in hgroups]
            else:
                children = Parallel(n_jobs=n_jobs)(delayed(process_hierarchy)( \
                    hierarchy, next_level, level_name) for level_name, hierarchy in hgroups)

            return process_node(children, name)

        return process_hierarchy(df)

    @staticmethod
    def convert_clusters_to_json(cluster_file, method):
        # df = pd.read_hdf(cluster_file, "table")
        # ss = pd.read_hdf("/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/superfamilies_for_paper/All models-umap-all_latent.h5", "table")[["ss_score2", "cath_domain"]]
        # from sklearn.preprocessing import MinMaxScaler
        # ss = ss.assign(ss=MinMaxScaler().fit_transform(ss["ss_score2"].values.reshape((len(ss["ss_score2"]),1))).flatten()).drop(columns=["ss_score2"])
        # df = pd.merge(df.reset_index(), ss, left_on="cathDomain", right_on="cath_domain")
        # df = df.set_index(["cathDomain", "true_sfam"]).drop(columns=["cath_domain"])
        #
        # def process_hierarchy(h, level=None, name=None):
        #     levels = sorted([c for c in h.columns if "Level" in c], key=lambda n: int(n.split()[1]), reverse=True)+["END"]
        #     assert level is None or level in levels
        #     if level is None:
        #         print("Running start with", levels[0])
        #         return process_hierarchy(h, levels[0], method)
        #     elif level == levels[-1]:
        #         print("Saving N domains into:", len(h.index), level, name)
        #         data = {"name":str(name), "children":[]}
        #         data["children"] = [{"name":row.Index[0], "superfamily":row.Index[1], "value":10, "ss":row.ss} \
        #             for row in h.itertuples()]
        #         data["ss"] = np.mean([c["ss"] for c in data["children"]])
        #         return data
        #
        #     hgroups = h.groupby(levels[:levels.index(level)+1], as_index=False)
        #     print("Running N groups:", hgroups.ngroups)
        #     next_level = levels[levels.index(level)+1]
        #     # if hgroups.ngroups == 1:
        #     #     level_name, hierarchy = next(iter(hgroups))
        #     #     return process_hierarchy(hierarchy, next_level, level_name)
        #
        #     data = {"name":str(name), "children":[], "ss":[]}
        #     for level_name, hierarchy in hgroups:
        #         print("Running", levels[:levels.index(level)+1], level_name, "->", levels[levels.index(level)+1])
        #         next_h = process_hierarchy(hierarchy, next_level, level_name)
        #         data["children"].append(next_h)
        #         data["ss"].append(next_h["ss"])
        #     data["ss"] = np.mean(data["ss"])
        #
        #
        #     # if level == levels[-1]:
        #     #     data["children"] = data["children"][0]
        #     return data

        # flare = process_hierarchy(df)

        def process_leaf(h, level_name):
            data = {"name":str(level_name), "children":[]}
            data["children"] = [{"name":row.Index[0], "superfamily":row.Index[1], "value":10, "ss":row.ss} \
                for row in h.itertuples()]
            data["ss"] = np.mean([c["ss"] for c in data["children"]])
            return data

        def process_node(children, level_name):
            data = {"name":str(level_name)}
            data["children"] = children
            data["ss"] = np.mean([child["ss"] for child in children])
            return data

        flare = Clustering.read_clusters_and_process(cluster_file, process_node, process_leaf)

        output_file = f"{os.path.splitext(cluster_file)[0]}.json"
        with open(output_file, "w") as f:
            json.dump(flare, f)

        return flare

    @staticmethod
    def align_communities(cluster_file, g):
        from molmimic.parsers.superpose.mtmalign import mTMAlign
        data_dir = "/home/bournelab/data-eppic-cath-features/prepared-cath-structures"

        if not os.path.isdir("alignments"):
            os.makedirs("alignments")

        def align_structures(pdb_files, level_name):
            group_name = "-".join(map(str,level_name)) if isinstance(level_name, (list, tuple)) else str(level_name)
            aln_dir = os.path.abspath(os.path.join("alignments", group_name))
            results_file = os.path.join(aln_dir, "results.pdb")

            if not os.path.isdir(aln_dir):
                os.makedirs(aln_dir)
            elif os.path.isfile(results_file):
                return results_file

            ocwd = os.getcwd()
            os.chdir(aln_dir)
            distances = mTMAlign(work_dir=aln_dir).all_vs_all(pdb_files, out_file=results_file)
            os.chdir(ocwd)
            return distances

        align_groups = []

        def process_leaf(h, level_name):
            pdb_files = [f"{data_dir}/{sfam.replace('.','/')}/{cathDomain}.pdb" for cathDomain, sfam in h.index]
            align_groups.append((pdb_files, level_name))
            return pdb_files

        def process_node(children, level_name):
            if isinstance(level_name, (int, float, str)) or len(level_name) < 3: return []
            pdb_files = [pdb for child in children for pdb in child]
            align_groups.append((pdb_files, level_name))
            return pdb_files

        Clustering.read_clusters_and_process(cluster_file, process_node, process_leaf)

        align_groups = sorted(align_groups, key=lambda x: len(x[0]))

        Parallel(n_jobs=16)(delayed(align_structures)(pdb_files, level_name) for \
            pdb_files, level_name in align_groups)

    @staticmethod
    def lrp_communities(cluster_file, name):
        def process_leaf(h, level_name):
            group_name = "-".join(map(str,level_name)) if isinstance(level_name, (list, tuple)) else str(level_name)
            if os.path.isdir(os.path.join("lrp", group_name)):
                return
            Clustering.run_lrp(h, group_name)

        df = Clustering.read_clusters_and_process(cluster_file)
        Clustering.run_lrp(df)

    @staticmethod
    def find_common_fragments(cluster_file, name, resolution_kmer=4, resolution_radius=6):
        from geometricus import MomentInvariants, SplitType
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import NMF
        import openTSNE
        import pickle

        data_dir = "/home/bournelab/data-eppic-cath-features/prepared-cath-structures"

        invariants_kmer = []
        invariants_radius = []
        shapemers = []
        keys = []

        def process_leaf(h, level_name):
            for cathDomain, sfam in h.index:
                pdb_file = f"{data_dir}/{sfam.replace('.','/')}/{cathDomain}.pdb"
                keys.append(cathDomain)
                invariants = MomentInvariants.from_pdb_file(pdb_file, split_type=SplitType.ALLMER, split_size=16)
                invariants_radius.append(invariants)
                shapemers_ = " ".join([f"k{x[0]}i{x[1]}i{x[2]}i{x[3]}" for x in
                    (np.log1p(invariants.moments) * resolution_kmer).astype(int)])

                invaraints = MomentInvariants.from_pdb_file(pdb_file, split_type=SplitType.RADIUS, split_size=10)
                invariants_radius.append(invaraints)
                shapemers_ += " " + " ".join([f"r{x[0]}i{x[1]}i{x[2]}i{x[3]}" for x in
                    (np.log1p(invariants.moments) * resolution_radius).astype(int)])
                shapemers.append(shapemers_)

        Clustering.read_clusters_and_process(cluster_file, process_leaf=process_leaf)

        with open(f"{os.path.splitext(cluster_file)}_shapemers.txt", "w") as f:
            for cathDomain, shapemer in zip(keys, shapemers):
                print(cathDomain + "\t" + shapemer, file=f)

        corpus = shapemers
        vectorizer = TfidfVectorizer(min_df=2)
        tfidf_matrix = vectorizer.fit_transform(corpus)
        num_topics = 250
        topic_model = NMF(n_components=num_topics,
                    random_state=42,
                    solver='cd', tol=0.0005,
                    max_iter=500,
                    alpha=.1,
                    l1_ratio=.5,
                    verbose=1)
        w_matrix = topic_model.fit_transform(tfidf_matrix)

        scaler = StandardScaler()
        w_matrix_norm = scaler.fit_transform(w_matrix)

        tsne_reducer = openTSNE.TSNE(
            perplexity=50,
            initialization="pca",
            metric="cosine",
            n_jobs=14,
            random_state=42,
            n_iter=1000,
            verbose=True
        )
        reduced = tsne_reducer.fit(w_matrix_norm)

        with open(f"{os.path.splitext(cluster_file)}_topic_modelling_data.pkl", "wb") as f:
            pickle.dump((keys,
                         vectorizer, tfidf_matrix,
                         topic_model, w_matrix,
                         scaler, w_matrix_norm,
                         tsne_reducer, reduced), f)

        kmer_embedder = GeometricusEmbedding.from_invariants(invariants_kmer, resolution=resolution_kmer)
        radius_embedder = GeometricusEmbedding.from_invariants(invariants_radius, resolution=resolution_radius)
        reducer = umap.UMAP(metric="cosine", n_components=2)
        reduced = reducer.fit_transform(np.hstack((kmer_embedder.embedding, radius_embedder.embedding)))

    def find_common_local_sustructures(self, levels=None):
        all_levels = reversed([c for c in self.domain_group_membership.columns if "Level" in c])
        if levels is None:
            levels = all_levels[0]
        groups = self.domain_group_membership.groupby(levels, as_index=False)
        if groups.ngroups == 1 and level != levels[-1]:
            return find_common_local_sustructures(all_levels[:all_levels.index(level)+2])
        for group_name, group in groups:
            pass

        cathDomains = self.domain_group_membership.index.get_level_values('cathDomain')
        superfamilies = self.domain_group_membership.index.get_level_values('true_sfam').drop_duplicates()

        for sfam in superfamilies:
            self.run_lrp(cathDomains, superfamilies)

        mixed_lrp = {}
        for domain1, true_sfam1 in self.domain_group_membership.index:
            for domain2, true_sfam2 in self.domain_group_membership.index:
                if domain1 == domain2: continue
                original = self.lrp(domain1, true_sfam1)
                other = self.lrp(domain2, true_sfam1)
                mixed_lrp = self.compare_lrp(original, other)

        for group_name, group_domains in domain_groups:
            best_sfams = group_domains[[c for c in group_domains.columns if "ELBO" in c]].idxmin(axis=1)
            best_sfams = best_sfams.groupby(best_sfams).groups.keys()
            best_sfams = [s.split()[1] for s in best_sfams]
            find_relevant_voxels(group_domains, best_sfams)

    @staticmethod
    def find_common_shapemers(cluster_file, name):
        from geometricus import GeometricusEmbedding, MomentInvariants, SplitType

        data_dir = "/home/bournelab/data-eppic-cath-features/prepared-cath-structures"

        os.makedirs("shapemer_html", exist_ok=True)

        def process_leaf(h, level_name):
            invariants = []
            for cathDomain, sfam in h.index:
                pdb_file = f"{data_dir}/{sfam.replace('.','/')}/{cathDomain}.pdb"
                k_invariants = MomentInvariants.from_pdb_file(pdb_file, split_type=SplitType.ALLMER, split_size=16)
                r_invariants = MomentInvariants.from_pdb_file(pdb_file, split_type=SplitType.RADIUS, split_size=10)
                invariants += [k_invariants, r_invariants]

            process_invariants(level_name, invariants, leaf=True)

            return invariants

        def process_node(children, level_name):
            if len(children) == 0:
                return []

            invariants = list(itertools.chain.from_iterable(children))

            print(invariants)
            print(level_name)
            print(type(children))

            if not isinstance(level_name, (list, tuple)):
                return []

            process_invariants(level_name, invariants)

            return invariants

        def process_invariants(level_name, invariants, leaf=False):
            embed = GeometricusEmbedding.from_invariants(invariants)

            if not isinstance(level_name, (list, tuple)):
                level_name = [level_name]

            comm_dir = os.path.join("shapemer_html", "_".join(map(lambda x: str(int(x)), level_name)))
            os.makedirs(comm_dir, exist_ok=True)

            shapmers_comm_dir = os.path.join(comm_dir, "by_shapemer")
            os.makedirs(shapmers_comm_dir, exist_ok=True)

            sh_size_file = os.path.join(shapmers_comm_dir, "sizes.json")

            if not os.path.isfile(sh_size_file):
                max_s = None
                max_p = None
                sh_sizes = {}

                title_re = re.compile("(\d+) times across (\d+) proteins")

                for j, shapemer in enumerate(embed.shapemer_keys):
                    if j%10==0: print("s", end="")
                    fig = embed.plot_shapemers(shapemer)


                    m = title_re.search(fig.to_plotly_json()["layout"]["title"]["text"])
                    assert m
                    num_s, num_p = int(m.groups()[0]), int(m.groups()[1])

                    fig.write_html(os.path.join(shapmers_comm_dir, f"{'_'.join(map(str, shapemer))}.html"))


                    sh_sizes["_".join(map(str,shapemer))] = [num_s, num_p]
                    if max_s is None or num_s > max_s[0]:
                        max_s = [num_s, "_".join(map(str,shapemer))]
                    if max_p is None or num_p > max_p[0]:
                        max_p = [num_p, "_".join(map(str,shapemer))]

                sh_sizes["max_s"] = max_s
                sh_sizes["max_p"] = max_p

                with open(sh_size_file, "w") as f:
                    json.dump(sh_sizes, f)

            if leaf:
                for i, domain in enumerate(embed.protein_keys):
                    domain_dir = os.path.join(comm_dir, domain)
                    os.makedirs(domain_dir, exist_ok=True)
                    for shapemer in embed.proteins_to_shapemers[domain]:
                        shapemer_prefix = os.path.join(domain_dir, f"{domain}_{'_'.join(map(str, shapemer))}.html")
                        if not os.path.isfile(shapemer_prefix):
                            embed.plot_shapemer_on_protein(shapemer, domain).write_html(shapemer_prefix)

        Clustering.read_clusters_and_process(cluster_file, process_leaf=process_leaf, process_node=process_node, n_jobs=16)

        # community_embeddings = {}
        # for name, domains in groups:
        #     invariants = []
        #     for cathDomain, sfam in domains.index:
        #         pdb_file = f"{data_dir}/{sfam.replace('.','/')}/{cathDomain}.pdb"
        #         k_invariants = MomentInvariants.from_pdb_file(pdb_file, split_type=SplitType.ALLMER, split_size=16)
        #         r_invariants = MomentInvariants.from_pdb_file(pdb_file, split_type=SplitType.RADIUS, split_size=10)
        #         invariants += [k_invariants, r_invariants]
        #     embedding = GeometricusEmbedding.from_invariants(invariants)
        #     community_embeddings[name] = embedding
        #
        # for comm, embed in community_embeddings.items():
        #     print("Running", comm)
        #     comm_dir = "/home/bournelab/shapemer_html/"+"_".join(map(lambda x: str(int(x)), comm))
        #     os.makedirs(comm_dir, exist_ok=True)
        #     shapmers_comm_dir = os.path.join(comm_dir, "by_shapemer")
        #     os.makedirs(shapmers_comm_dir, exist_ok=True)
        #     for j, shapemer in enumerate(embed.shapemer_keys):
        #         if j%10==0: print("s", end="")
        #         embed.plot_shapemers(shapemer).write_html(os.path.join(shapmers_comm_dir, f"{'_'.join(map(str, shapemer))}.html"))
        #     print()
        #     for i, domain in enumerate(embed.protein_keys):
        #         if i%10==0: print(".", end="")
        #         domain_dir = os.path.join(comm_dir, domain)
        #         os.makedirs(domain_dir, exist_ok=True)
        #         for shapemer in embed.proteins_to_shapemers[domain]:
        #             shapemer_prefix = os.path.join(domain_dir, f"{domain}_{'_'.join(map(str, shapemer))}.html")
        #             if not os.path.isfile(shapemer_prefix):
        #                 embed.plot_shapemer_on_protein(shapemer, domain).write_html(shapemer_prefix)
        #     print()

    @staticmethod
    def run_lrp(g):
        data_dir = "/home/bournelab/data-eppic-cath-features"
        os.makedirs(os.path.join(os.getcwd(), "lrp"), exist_ok=True)
        superfamilies = g.index.get_level_values('true_sfam').drop_duplicates()
        cathDomains = set(g.index.get_level_values('cathDomain'))
        def lrp_cmd(sfam_model):
            sfam_path = os.path.join(os.path.abspath(os.getcwd()), "lrp", sfam_model)
            completed_domains = set([f.split("/")[-2] for f in glob.glob(os.path.join(sfam_path, "*", "*.h5"))])
            domains_to_run = cathDomains-completed_domains
            if domains_to_run == len(cathDomains):
                domain_args = ["--representatives"]
            else:
                domain_args = ["--domains", *domains_to_run]
            return [
                "python",
                "-m", "DeepUrfold.Evaluators.EvaluateDomainStructureVAE",
                "--data_dir", data_dir,
                "--superfamily", *superfamilies,
                #"--representatives",
                #"--domains", *cathDomains,
                "--checkpoint", f"/home/bournelab/urfold_runs/superfamilies_for_paper/{sfam_model}/last.ckpt",
                "--features", "H;HD;HS;C;A;N;NA;NS;OA;OS;SA;S;Unk_atom__is_helix;is_sheet;Unk_SS__residue_buried__is_hydrophobic__pos_charge__is_electronegative",
                "--feature_groups", "Atom Type;Secondary Structure;Solvent Accessibility;Hydrophobicity;Charge;Electrostatics",
                "--return_latent",
                "--gpus", "1",
                #"--accelerator", "None",
                "--batch_size", "256",
                "--num_workers", "64",
                "--no_compute",
                "--lrp",
                "--prefix", f"LRP:model={sfam_model}_input=all_representatives"
            ] + domain_args

        commands = []
        for sfam_model in superfamilies:
            sfam_model_dir = os.path.join(os.path.abspath(os.getcwd()), "lrp", sfam_model)
            if not os.path.isdir(sfam_model_dir):
                os.makedirs(sfam_model_dir)
            command = lrp_cmd(sfam_model)
            commands.append((command, sfam_model_dir))

        def run_command(i, cmd, cwd):
            from DeepUrfold.Analysis.AllVsAll import get_available_gpu
            gpu = str(get_available_gpu(None, i))
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = gpu
            print(cmd)
            subprocess.call(cmd, cwd=cwd, env=my_env)

        Parallel(n_jobs=4)(delayed(run_command)(i, cmd, pwd) for i, (cmd, pwd) in enumerate(commands))

    @staticmethod
    def regroup_lrp(lrp_dir):
        import json
        os.makedirs("lrp_by_domain", exist_ok=True)

        superfamilies = next(os.walk(lrp_dir))[1]
        domains = next(os.walk(os.path.join(lrp_dir, superfamilies[1])))[1]
        # for domain in domains:
        #     if domain == "lightning_logs": continue
        #     if os.path.isfile(os.path.join("lrp_by_domain", f"{domain}.json")): continue
        #
        #     domain_lrp_info = {"pdb":None}

        def process_domain(domain):
            domain_lrp_info = {"pdb":None}
            for superfamily in superfamilies:
                relevance_file = os.path.join(lrp_dir, superfamily, domain,
                    f"{domain}-v_agg=arithmetic_mean__f_agg=npsum.h5")

                if domain_lrp_info["pdb"] is None:
                    with open(relevance_file[:-3]+"-total_relevance.pdb") as f:
                        domain_lrp_info["pdb"] = f.read()

                relevence_df = pd.read_hdf(relevance_file, "table")
                propMap = [{'serial':serial, 'props':props} for \
                    serial, props in relevence_df.to_dict("index").items()]

                domain_lrp_info[superfamily] = propMap

            print(domain, len(domain_lrp_info))

            with open(os.path.join("lrp_by_domain", f"{domain}.json"), "w") as f:
                json.dump(domain_lrp_info, f)

        Parallel(n_jobs=20)(delayed(process_domain)(d) for d in domains if \
            len(d)==7)  # and not os.path.isfile(os.path.join("lrp_by_domain", f"{d}.json")))

    def get_lrp_for_community(g, name):
        superfamilies = g.index.get_level_values('true_sfam').drop_duplicates()
        cathDomains = g.index.get_level_values('cathDomain').tolist()
        sfam_groups = g.index.to_frame().reset_index(drop=True).groupby("true_sfam")
        for sfam_model in superfamilies:
            sfam_model_dir = os.path.join(os.path.abspath(os.getcwd()), "lrp", name, sfam_model)
            if not os.path.isdir(sfam_model_dir):
                raise RuntimeError("Must run lrp first")
            for test_sfam, sfam_group in sfam_groups:
                sfam_test_dir = os.path.join(sfam_model_dir, test_sfam)
                if not os.path.isdir(sfam_test_dir):
                    os.makedirs(sfam_test_dir)
                cathDomains = sfam_group.cathDomain.tolist()
                command = cmd(cathDomains, test_sfam, sfam_model)
                commands.append((command, sfam_test_dir))

    def find_relevant_voxels(self, g=None, other_sfams=None):
        if g is None:
            g = self.domain_group_membership

        superfamilies = g.index.get_level_values('true_sfam').drop_duplicates()
        sfam_groups = g.index.to_frame().groupby("true_sfam")

        for test_sfam, sfam_group in sfam_groups:
            for model_sfam in superfamilies:
                for cathDomain in sfam_group.index.get_level_values('cathDomain'):
                    lrp_results = os.path.join("all_lrp", model_sfam, test_sfam, cathDomain, "*total_relevance.pdb")
                    try:
                        lrp_results = list(glob.glob(lrp_results))[0]
                    except IndexError:
                        continue
                    lrp_results = np.array(list(get_b(lrp_results)))
                    important_voxels = lrp_results[lrp_results>=lrp_results.percentile(99)]



        if g is None:
            g = self.domain_group_membership

        if not os.path.isdir("all_lrp"):
            pass


        pdb_files = [f"{data_dir}/{sfam.replace('.','/')}/{cathDomain}.pdb" for cathDomain, sfam in g.index]

        sfam_groups = self.domain_group_membership.index.to_frame().groupby("true_sfam")
        for sfam, sfam_group in sfam_groups:
            if not os.path.isdir(sfam):
                os.makedirs(sfam)
            for sfam_model in superfamilies:
                subprocess.call(cmd(sfam_group.cathDomain, sfam, sfam_model))

    def align_structures(self, g):
        from molmimic.parsers.superpose.mtmalign import mTMAlign
        data_dir = "/home/bournelab/data-eppic-cath-features/prepared-cath-structures"
        pdb_files = [f"{data_dir}/{sfam.replace('.','/')}/{cathDomain}.pdb" for cathDomain, sfam in g.index]

        group_name = "-".join(map(str,g.iloc[0].index))
        results_file = f"{group_name}.pdb"
        distances = mTMAlign().all_vs_all(pdb_files, out_file=results_file)


        distances.groupby("chain1")["moving_tm_score"].agg(sum)
        centroid = distances.sum(axis=1).argmin()
        distance_to_centroid = distances[:, centroid]
        return distance_to_centroid

        def process_row(row):
            focus = [i for i, res in enumerate(row.chain1_aln) if res not in ["-", "."]]
            focus = {i:f for f, i in enumerate(focus)}
            aln = [focus[i] for i, aln_i in enumerate(row.aln_info) if aln_i in ":."]
            return aln

        def gg(g):
            aln = g[["chain1_aln","chain2_aln","aln_info"]]
            chain1 = aln.iloc[0].chain1_aln.replace("-", "").replace(".", "")
            unique, counts = numpy.unique(g.apply(process_row, axis=1), return_counts=True)
            aliggned = dict(zip(unique, counts))

        #distances.groupby("chain1").apply(lambda g: )

        #return clusters, distances
        #clusters.groupby("structure_clusters").count().
        cluster_count = clusters.groupby("structure_clusters")[["item"]].count()
        cluster_count_spread = pd.merge(cluster_count, clusters[["structure_clusters", "spread", "centroid_item", "centroid_pdb_file"]].drop_duplicates(), on="structure_clusters")
        centroid_item = cluster_count_spread.sort_values(["item","spread"], ascending=[False, True]).iloc[0]
        centroid_distances = distances.loc[(centroid_item["centroid_item"], slice(None))]
        centroid_distances = centroid_distances.to_dict()["distances"]
        centroid_distances[centroid_item["centroid_pdb_file"].split("/")[-1][:7]]=0.0
        return centroid_distances

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--json', action='store_true')
    group.add_argument('--align', action='store_true')
    group.add_argument('--lrp', action='store_true')
    group.add_argument('--lrp-regroup', action='store_true')
    group.add_argument('--shapemers', action='store_true')
    parser.add_argument("cluster_file")
    parser.add_argument("name")
    args = parser.parse_args()
    if args.json:
        Clustering.convert_clusters_to_json(args.cluster_file, args.name)
    elif args.align:
        Clustering.align_communities(args.cluster_file, args.name)
    elif args.lrp:
        Clustering.lrp_communities(args.cluster_file, args.name)
    elif args.shapemers:
        Clustering.find_common_shapemers(args.cluster_file, args.name)
    elif args.lrp_regroup:
        Clustering.regroup_lrp(args.cluster_file)
