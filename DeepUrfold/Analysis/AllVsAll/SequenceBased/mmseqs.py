import os
import re
import json
import uuid
import shutil
import argparse
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from Bio.PDB.Polypeptide import three_to_one

from DeepUrfold.Analysis.AllVsAll.SequenceBased import SequenceBasedAllVsAll
from Prop3D.parsers import mmseqs

def run(i, fasta, hparams, other, total, work_dir, representative_domains):
    print(f"Running hyperparameter set: {i}/{total}")
    prefix = os.path.join(work_dir, os.path.splitext(os.path.basename(fasta))[0])
    id_parser = re.compile(r"cath\|4_3_0\|([a-zA-Z0-9]+)\/")
    hparams.update(other)
    hparams["prefix"] = f"{os.path.splitext(os.path.basename(fasta))[0]}.hyper_params.{i}.tsv"
    hparams["tmp_dir"] = os.path.join(work_dir, str(i), f"tmp-{uuid.uuid4()}")
    os.makedirs(hparams["tmp_dir"], exist_ok=True)
    print("Run", i)
    results_file = mmseqs.EasyCluster(work_dir=work_dir).cluster(fasta, **hparams)
    print("DOne", i)
    results = pd.read_csv(results_file, sep="\t", header=None, names="source target targetID alnScore seqIdentity eVal qStart qEnd qLen tStart tEnd tLen cigar".split())
    results["source"] = results["source"].apply(lambda x: id_parser.search(x).groups()[0] if id_parser.search(x) else None)
    results["target"] = results["source"].apply(lambda x: id_parser.search(x).groups()[0] if id_parser.search(x) else None)
    results = pd.merge(results, representative_domains, left_on="source", right_on="cathDomain")

    n_clusters = len(results[["source", "superfamily"]].drop_duplicates())
    n_sfams = len(results["superfamily"].drop_duplicates())

    with open(f"{prefix}.hyperparameter_combos_{i}.tsv", "w") as f:
        print(f"{n_clusters}\t{n_sfams}\t{hparams}", file=f)
        print(f"{n_clusters}\t{n_sfams}\t{hparams}")

class MMSeqsAllVsAll(SequenceBasedAllVsAll):
    NJOBS = 60
    CLUSTERED = True
    METHOD = "MMSeqs"
    DATA_INPUT = "CATH S35"
    SCORE_INCREASING = True
    MODEL_TYPE = "Local Alignment"
    SCORE_METRIC = "% Sequence Identity"

    def create_superfamily_datasets(self):
        """No extra processing"""
        return None

    def train_all(self, *args, **kwds):
        """No Trianing needed for MMSeqs"""
        return None

    def infer_all(self, model, combined_sequences):
        # self.try_all(combined_sequencees, self.representative_domains, threads=60)
        # assert 0
        if False and os.path.isfile("mmseqs_allvsall_results.hdf"):
            results = pd.read_hdf("mmseqs_allvsall_results.hdf", "table")
        else:
            #cluterer = mmseqs.Easy
            cluterer = mmseqs.MMSeqs(work_dir=self.work_dir)
            _, results_file = cluterer.all_vs_all(combined_sequences)
            #try_all(combined_sequencees, self.representative_domains, threads=20)#,aggressive=True,  , cluster_reassign=True,)  #cov_mode=2, #,
            #assert 0
            #results = pd.read_csv(results_file, delimeter="\t")
            results = pd.read_csv(results_file, sep="\t", header=None, names="source target alnScore seqIdentity eVal qStart qEnd qLen tStart tEnd tLen cigar".split())
            results["source"] = results["source"].apply(lambda x: self.id_parser.search(x).groups()[0] if self.id_parser.search(x) else None)
            results = pd.merge(results, self.representative_domains, left_on="source", right_on="cathDomain")
            results = results.rename(columns={"superfamily":"source_superfamily"})

            results["target"] = results["target"].apply(lambda x: self.id_parser.search(x).groups()[0] if self.id_parser.search(x) else None)
            results = pd.merge(results, self.representative_domains, left_on="target", right_on="cathDomain")
            results = results.rename(columns={"superfamily":"target_superfamily"})

            results = results.rename(columns={"source":"cathDomain", "source_superfamily":"true_sfam"})
            domain_groups = results.groupby(["cathDomain", "true_sfam", "target_superfamily"])

            results = pd.DataFrame(np.nan, index=self.representative_domains["cathDomain"],
                columns=sorted(self.superfamily_datasets.keys()))
            results = pd.merge(results, self.representative_domains, left_index=True, right_on="cathDomain")
            results = results.rename(columns={"superfamily":"true_sfam"})
            results = results.set_index(["cathDomain", "true_sfam"])

            for (cathDomain, true_sfam, test_superfam), group in domain_groups:
                distance = group["seqIdentity"].median()
                results.loc[(cathDomain, true_sfam), test_superfam] = distance

            import pdb; pdb.set_trace()

            results.to_hdf("mmseqs_allvsall_results.hdf", "table")

        print(results)

        results = results.fillna(0.0)
        results = results.replace([np.inf, -np.inf], [1, 0])
        results = results.astype(np.float64)

        return results


        qTest = dGroups["seqIdentity"].quantile(0.95)
        results = qTest.to_frame().reset_index().pivot(index=["cathDomain", "true_sfam"], columns="target_superfamily", values="seqIdentity")
        results.to_hdf("mmseqs_seqID_ava.hdf", "table")

        assert 0

        clusters = results[["cluster-representative"]].drop_duplicates()
        clusters = clusters.assign(cluster=range(len(clusters)))

        results = pd.merge(results, clusters, on="cluster-representative")
        results = results.drop(columns=["cluster-representative"]).rename(columns={"cluster-member":"cathDomain"})
        results = pd.merge(results, self.representative_domains, left_index=True, right_on="cathDomain")
        results = results.rename(columns={"superfamily":"true_sfam"})
        results = results.reset_index(["cathDomain", "true_sfam"])

        return results

    def _plot(self, sfam, other, sfam_name, prefix="", vertical_lines={}): #descriminator=None, max_sfam=None, max_other=None):
        fig = plt.figure(figsize=(10, 6), dpi=300)
        ax = fig.subplots(1, 1)
        if sfam is not None:
            sns.kdeplot(sfam, label=f"True {sfam_name}", ax=ax)
        sns.kdeplot(other, label="Other Superfamilies", ax=ax)

        colors = sns.color_palette("hls", 8)
        for i, (name, value) in enumerate(vertical_lines.items()):
            if value is None: continue
            ax.axvline(value, label="{} ({:.4f})".format(name, value), color=colors[i])

        plt.legend()
        plt.savefig(f"{prefix}_kde.png")



    def try_all(self, fastaFile, representative_domains, **kwds):
        aggressive_values = dict(
            sensitivity=[6, 7.5, 8, 9.5, 9, 9.5, 10],
            min_ungapped_score=range(31),
            min_covered=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            max_evalue=[10, 100, 1000, float("inf")],
            min_seq_id=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            cluster_mode=[0,1,2,3])

        keys, values = zip(*aggressive_values.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
        print("Trying", len(permutations_dicts), "hyperparameter combinations")

        prefix = os.path.join(self.work_dir, os.path.splitext(os.path.basename(fastaFile))[0])
        with open(f"{prefix}.hyperparameter_combos", "w") as f:
            print("#n_clusters\tn_sfams\thyperparameters", file=f)

        os.makedirs(prefix, exist_ok=True)

        tmp_dir = os.path.join(self.work_dir, "tmp4")
        os.makedirs(tmp_dir, exist_ok=True)

        hp_dir = os.path.join(self.work_dir, "hyper_params")
        os.makedirs(hp_dir, exist_ok=True)

        Parallel(n_jobs=4)(delayed(run)(i, fastaFile, hyperparameters, kwds, len(permutations_dicts), self.work_dir, representative_domains) for i, hyperparameters in enumerate(permutations_dicts))

        #[run(i, fastaFile, hyperparameters, kwds, len(permutations_dicts), self.work_dir, representative_domains) for i, hyperparameters in enumerate(permutations_dicts)]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("-w", "--work_dir", default=os.getcwd(), required=False)
    parser.add_argument("-p", "--permutation_dir", default="/home/bournelab/urfold_runs/multiple_loop_permutations/sh3_3", required=False)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("ava_superfamily", nargs="+")
    args = parser.parse_args()
    runner = MMSeqsAllVsAll(args.ava_superfamily, args.data_dir, permutation_dir=args.permutation_dir,
        work_dir=args.work_dir, force=args.force)
    runner.run()
