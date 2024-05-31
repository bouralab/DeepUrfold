import os
import json
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from Bio.PDB.Polypeptide import three_to_one

from DeepUrfold.Analysis.AllVsAll.SequenceBased import SequenceBasedAllVsAll
from DeepUrfold.DataModules import DomainStructureDataModule
from Prop3D.parsers.USEARCH import AllPairsLocal, AllPairsGlobal

class UClustAllVsAll:
    NJOBS = 4

    def create_superfamily_datasets(self):
        """No extra processing"""
        return None

    def train_all(self, *args, **kwds):
        """No Trianing needed for MaxCluster"""
        return None

    def process_results(self, uc_file):
        results = AllPairsLocal.parse_uc_file(uc_file)

        results["label_query"] = results["label_query"].apply(lambda x: self.id_parser.search(x).groups()[0] if self.id_parser.search(x) else None)
        results = pd.merge(results, self.representative_domains, left_on="label_query", right_on="cathDomain")
        results = results.rename(columns={"cathDomain":"query_cathDomain", "superfamily":"true_sfam"})

        results["label_target"] = results["label_target"].apply(lambda x: self.id_parser.search(x).groups()[0] if self.id_parser.search(x) else None)
        results = pd.merge(results, self.representative_domains, left_on="label_target", right_on="cathDomain")
        results = results.rename(columns={"cathDomain":"target_cathDomain", "superfamily":"target_superfamily"})

        #results = results.rename(columns={"source":"cathDomain", "query_superfamily":"true_sfam"})

        print(results)
        print(results.columns)
        domain_groups = results.groupby(["query_cathDomain", "true_sfam", "target_superfamily"])

        results = pd.DataFrame(np.nan, index=self.representative_domains["cathDomain"],
            columns=sorted(self.superfamily_datasets.keys()))
        results = pd.merge(results, self.representative_domains, left_index=True, right_on="cathDomain")
        results = results.rename(columns={"superfamily":"true_sfam"})
        results = results.set_index(["cathDomain", "true_sfam"])

        for (cathDomain, true_sfam, test_superfam), group in domain_groups:
            distance = group["pctId"].median()
            results.loc[(cathDomain, true_sfam), test_superfam] = distance

        results = results.fillna(0.0)
        results = results.replace([np.inf, -np.inf], [1, 0])
        results = results.astype(np.float64)

        import pdb; pdb.set_trace()

        return results

class UClustAllVsAllLocal(SequenceBasedAllVsAll, UClustAllVsAll):
    METHOD = "USEARCH local"
    MODEL_TYPE = "Local Alignment"
    DATA_INPUT = "CATH S35"
    SCORE_INCREASING = True
    SCORE_METRIC = "% Sequence Identity"
    NJOBS = 4

    def create_superfamily_datasets(self):
        """No extra processing"""
        return None

    def train_all(self, *args, **kwds):
        """No Trianing needed for MaxCluster"""
        return None

    def infer_all(self, model, combined_sequences):
        results_file = f"{os.path.splitext(combined_sequences)[0]}_local.uc"
        if self.force or not os.path.isfile(results_file):
            aligner = AllPairsLocal(work_dir=self.work_dir)
            results_file = aligner(fasta_file=combined_sequences, acceptall=True,
                uc=results_file,
                alnout=f"{os.path.splitext(combined_sequences)[0]}_local.aln",
                threads=self.NJOBS)
            results_file = results_file["uc"]

        results = self.process_results(results_file)
        results.to_hdf("uclust_local_allvsall_results.hdf", "table")

        return results

class UClustAllVsAllGlobal(SequenceBasedAllVsAll, UClustAllVsAll):
    METHOD = "USEARCH global"
    MODEL_TYPE = "Global Alignment"
    DATA_INPUT = "CATH S35"
    SCORE_INCREASING = True
    SCORE_METRIC = "% Sequence Identity"
    NJOBS = 4

    def create_superfamily_datasets(self):
        """No extra processing"""
        return None

    def train_all(self, *args, **kwds):
        """No Trianing needed for MaxCluster"""
        return None

    def infer_all(self, model, combined_sequences):
        results_file = f"{os.path.splitext(combined_sequences)[0]}_global.uc"
        if self.force or not os.path.isfile(results_file):
            aligner = AllPairsGlobal(work_dir=self.work_dir)
            results_file = aligner(fasta_file=combined_sequences, acceptall=True,
                uc=f"{os.path.splitext(combined_sequences)[0]}_global.uc",
                alnout=f"{os.path.splitext(combined_sequences)[0]}_global.aln",
                threads=self.NJOBS)
            results_file = results_file["uc"]

        results = self.process_results(results_file)
        results.to_hdf("uclust_global_allvsall_results.hdf", "table")

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("-w", "--work_dir", default=os.getcwd(), required=False)
    parser.add_argument("-p", "--permutation_dir", default="/home/bournelab/urfold_runs/multiple_loop_permutations/sh3_3", required=False)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-l", "--local", action="store_true")
    parser.add_argument("ava_superfamily", nargs="+")
    args = parser.parse_args()

    if args.local:
        runner = UClustAllVsAllLocal(args.ava_superfamily, args.data_dir, permutation_dir=args.permutation_dir,
            work_dir=args.work_dir, force=args.force)
    else:

        runner = UClustAllVsAllGlobal(args.ava_superfamily, args.data_dir, permutation_dir=args.permutation_dir,
            work_dir=args.work_dir, force=args.force)

    runner.run()
