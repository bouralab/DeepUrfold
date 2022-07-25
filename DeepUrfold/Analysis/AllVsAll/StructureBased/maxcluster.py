import os
import json
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from Bio.PDB.Polypeptide import three_to_one

from DeepUrfold.Analysis.AllVsAll.StructureBased import StructureBasedAllVsAll
from DeepUrfold.DataModules.DomainStructureDataModule import DomainStructureDataModule
from molmimic.parsers.MaxCluster import MaxCluster

class MaxClusterAllVsAll(StructureBasedAllVsAll):
    NJOBS = 60
    METHOD = "MaxCluster"
    DATA_INPUT = "CATH Superfamilies"
    SCORE_INCREASING = True
    MODEL_TYPE = "Structural Alignment"
    SCORE_METRIC = "TM-score"

    def train_all(self, *args, **kwds):
        """No Trianing needed for MaxCluster"""
        return None

    def infer_all(self, model, combined_structures):
        print(combined_structures)

        results_file = os.path.join(self.work_dir, "all_distances.csv")
        if True or not os.path.isfile(results_file):
            print(combined_structures["structure_file"])
            combined_structures_paths = combined_structures["structure_file"].tolist()
            print(combined_structures_paths)
            mc = MaxCluster(work_dir=self.work_dir)
            results_file = mc.all_vs_all(combined_structures_paths, C=0, sequence_independent=True, distributed=True, cores=100, work_dir=self.work_dir)

        #import pdb; pdb.set_trace()
        #clusters = mc.get_clusters(results)
        #distances = mc.get_distances(results, combined_structures_paths)

        distances = pd.read_csv(results_file)
        distances["PDB1"] = distances["PDB1"].str[:7]
        distances["PDB2"] = distances["PDB2"].str[:7]

        rev = distances.copy(deep=True)
        rev = rev.rename(columns={"PDB1":"_PDB2"}).rename(columns={"PDB2":"PDB1"}).rename(columns={"_PDB2":"PDB2"})
        rev = rev.drop(columns=["TM-Score1"]).rename(columns={"TM-Score2":"TM-Score"})

        distances = distances.rename(columns={"TM-Score1":"TM-Score"})
        distances = pd.concat((distances, rev))

        # rev = distances.copy(deep=True)
        # rev = rev.rename(columns={"PDB1":"_PDB2"}).rename(columns={"PDB2":"PDB1"}).rename(columns={"_PDB2":"PDB2"})
        # distances = pd.concat((distances, rev))


        #import pdb; pdb.set_trace()

        results = pd.DataFrame(np.nan, index=self.representative_domains["cathDomain"],
            columns=sorted(self.superfamily_datasets.keys()))
        results = pd.merge(results, self.representative_domains, left_index=True, right_on="cathDomain")
        results = results.rename(columns={"superfamily":"true_sfam"})
        results = results.set_index(["cathDomain", "true_sfam"])

        distances = pd.merge(distances, self.representative_domains, how="outer", left_on="PDB1", right_on="cathDomain")
        distances = distances.rename(columns={"cathDomain":"cathDomain1", "superfamily":"superfamily1"})
        distances = pd.merge(distances, self.representative_domains, how="outer", left_on="PDB2", right_on="cathDomain")
        distances = distances.rename(columns={"cathDomain":"cathDomain2", "superfamily":"superfamily2"})
        print(distances)
        domain_groups = distances.groupby(["cathDomain1", "superfamily1", "superfamily2"])

        for (cathDomain, true_sfam, test_superfam), group in domain_groups:
            distance = group["TM"].median()
            results.loc[(cathDomain, true_sfam), test_superfam] = distance

        import pdb; pdb.set_trace()

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    #parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("-w", "--work_dir", default=os.getcwd(), required=False)
    parser.add_argument("-p", "--permutation_dir", default="/home/bournelab/urfold_runs/multiple_loop_permutations/sh3_3", required=False)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("ava_superfamily", nargs="+")
    #parser = DomainStructureDataModule.add_data_specific_args(parser)
    parser = DomainStructureDataModule.add_data_specific_args(parser, eval=True)
    parser.set_defaults(
        data_dir="/home/bournelab/data-eppic-cath-features/",
        all_domains=True)
    args = parser.parse_args()
    runner = MaxClusterAllVsAll(args.ava_superfamily, args.data_dir, args, permutation_dir=args.permutation_dir,
        work_dir=args.work_dir, force=args.force)
    runner.run()
