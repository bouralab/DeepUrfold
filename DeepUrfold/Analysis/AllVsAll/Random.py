import os
import argparse

import numpy as np
import pandas as pd

from DeepUrfold.Analysis.AllVsAll import AllVsAll
from DeepUrfold.Analysis.Clustering import Clustering
from DeepUrfold.DataModules.DistributedDomainStructureDataModule import DistributedDomainStructureDataModule

class RandomAllVsAll(AllVsAll):
    METHOD = "RandomSBM"
    DATA_INPUT = "CATH S35"
    SCORE_INCREASING = True
    MODEL_TYPE = "Uniform"
    SCORE_METRIC = "Value"

    def __init__(self, superfamilies, data_dir, old_flare=None, no_sbm=False, permutation_dir=None, work_dir=None, force=False):
        super().__init__(superfamilies, data_dir, permutation_dir=permutation_dir, work_dir=work_dir, force=force)
        if no_sbm:
            self.clusterer = RandomClustering
            self.METHOD = "Random20"
        self.old_flare = old_flare

    def create_superfamily_datasets(self):
        self.superfamily_datasets = {sfam:True for sfam in self.superfamilies}

    def create_combined_dataset(self):
        return None

    def train_all(self):
        return {sfam:True for sfam in self.superfamily_datasets.keys()}

    def infer_all(self, models, combined_datatset):
        results = pd.DataFrame(
            np.random.rand(len(self.representative_domains), len(self.superfamilies)),
            index=self.representative_domains["cathDomain"],
            columns=sorted(self.superfamily_datasets.keys()))
        results = results.reset_index()
        results = pd.merge(results, self.representative_domains, on="cathDomain")
        results = results.rename(columns={"superfamily": "true_sfam"})
        results = results.set_index(["cathDomain", "true_sfam"])
        print(results)

        results.to_hdf(f"{self.METHOD}-{self.DATA_INPUT}-all_vs_all.hdf", "table")

        return results

class RandomClustering(Clustering):
    """Force a uniform clustering in 20 communties, don't use random weights"""
    N_COMMUNITIES = 20

    def find_communities(self,nested=True, overlap=True, deg_corr=True, kde_descriminator=False, log_odds_descriminator=False, weighted=True, score_type="elbo", force=False, prefix=None, old_flare=None, downsample=False):
        self.cluster_data = pd.DataFrame(
            {
                "l_0": 0.,
                "l_1": np.random.choice(self.N_COMMUNITIES, len(self.distances))
            },
            index=self.distances.index)
        
        self.calculate_cluster_metrics(old_flare=old_flare)

if __name__ == "__main__":
    sfams = "1.10.10.10 1.10.238.10 1.10.490.10 1.10.510.10 1.20.1260.10 2.30.30.100 2.40.50.140 2.60.40.10 3.10.20.30 3.30.230.10 3.30.300.20 3.30.310.60 3.30.1360.40 3.30.1370.10 3.30.1380.10 3.40.50.300 3.40.50.720 3.80.10.10 3.90.79.10 3.90.420.10".split()

    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    #parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("-w", "--work_dir", default=os.getcwd(), required=False)
    parser.add_argument("--old_flare", default=None, required=False)
    parser.add_argument("-p", "--permutation_dir", default="/home/bournelab/urfold_runs/multiple_loop_permutations/sh3_3", required=False)
    parser.add_argument("--no_sbm", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("ava_superfamily", nargs="+", default=sfams)
    #parser = DomainStructureDataModule.add_data_specific_args(parser)
    parser = DistributedDomainStructureDataModule.add_data_specific_args(parser, eval=True)
    parser.set_defaults(
        all_domains=True)
    args = parser.parse_args()
    runner = RandomAllVsAll(args.ava_superfamily, args.data_dir, old_flare=args.old_flare, no_sbm=args.no_sbm, permutation_dir=args.permutation_dir,
        work_dir=args.work_dir, force=args.force)
    runner.run()
