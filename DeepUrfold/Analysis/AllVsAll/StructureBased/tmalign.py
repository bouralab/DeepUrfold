import os
import json
import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from Bio.PDB.Polypeptide import three_to_one

from DeepUrfold.Analysis.AllVsAll.StructureBased import StructureBasedAllVsAll
from DeepUrfold.DataModules.DomainStructureDataModule import DomainStructureDataModule
from Prop3D.parsers.superpose.tmalign import TMAlign

class TMAlignAllVsAll(StructureBasedAllVsAll):
    NJOBS = 60
    METHOD = "TM-Align"
    DATA_INPUT = "CATH Superfamilies"
    SCORE_INCREASING = True
    MODEL_TYPE = "Structural Alignment"
    SCORE_METRIC = "TM-score"

    def __init__(self, superfamilies, data_dir, hparams, rmsd=False, cp=False, superpose=False, permutation_dir=None, work_dir=None, force=False):
        super().__init__(superfamilies, data_dir, hparams, permutation_dir=permutation_dir, work_dir=work_dir, force=force)
        self.rmsd = rmsd
        self.cp = cp
        self.superpose = superpose
        if rmsd:
            self.SCORE_METRIC = "RMSD"
            self.SCORE_INCREASING = False
            self.METHOD = "MaxClusterRMSD"

        if cp:
            self.METHOD += "CP"

    def train_all(self, *args, **kwds):
        """No Trianing needed for MaxCluster"""
        return None

    def infer_all(self, model, combined_structures):
        print(combined_structures)

        if not self.cp:
            results_file = os.path.join(self.work_dir, "all_distances.csv")
        else:
            results_file = os.path.join(self.work_dir, "all_distances_cp.csv")
        if not os.path.isfile(results_file):
            print(combined_structures["structure_file"])
            combined_structures_paths = combined_structures["structure_file"].tolist()
            print(combined_structures_paths)
            mc = TMAlign(work_dir=self.work_dir)
            if self.superpose:
                distances = mc.all_vs_all(combined_structures_paths, table_out_file=results_file, cp=self.cp, out_file="all_vs_all", distributed=96)
            else:
                distances = mc.all_vs_all(combined_structures_paths, table_out_file=results_file, cp=self.cp, distributed=96)
        else:
            distances = pd.read_csv(results_file)
        #assert 0, results_file
        #import pdb; pdb.set_trace()
        #clusters = mc.get_clusters(results)
        #distances = mc.get_distances(results, combined_structures_paths)

        print(results_file)

        #distances = pd.read_csv(results_file)
        distances["chain1"] = distances["chain1"].apply(lambda s: os.path.basename(s)).str[:7]
        distances["chain2"] = distances["chain2"].apply(lambda s: os.path.basename(s)).str[:7]

        rev = distances.copy(deep=True)
        rev = rev.rename(columns={"chain1":"_chain2"}).rename(columns={"chain2":"chain1"}).rename(columns={"_chain2":"chain2"})

        if not self.rmsd:
            rev = rev.drop(columns=["moving_tm_score"]).rename(columns={"fixed_tm_score":"TM-Score"})
            distances = distances.rename(columns={"moving_tm_score":"TM-Score"}).drop(columns=["fixed_tm_score"])
        else:
            rev = rev.drop(columns=["moving_tm_score", "fixed_tm_score"])
            distances = distances.drop(columns=["moving_tm_score", "fixed_tm_score"])

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

        distances = pd.merge(distances, self.representative_domains, how="outer", left_on="chain1", right_on="cathDomain")
        distances = distances.rename(columns={"cathDomain":"cathDomain1", "superfamily":"superfamily1"})
        distances = pd.merge(distances, self.representative_domains, how="outer", left_on="chain2", right_on="cathDomain")
        distances = distances.rename(columns={"cathDomain":"cathDomain2", "superfamily":"superfamily2"})
        print(distances)
        domain_groups = distances.groupby(["cathDomain1", "superfamily1", "superfamily2"])

        score_type = "rmsd" if self.SCORE_METRIC=="RMSD" else "TM-Score"
        for (cathDomain, true_sfam, test_superfam), group in domain_groups:
            distance = group[score_type].median()
            results.loc[(cathDomain, true_sfam), test_superfam] = distance

        results = results.fillna(0.0)
        results = results.replace([np.inf, -np.inf], [1, 0])
        results = results.astype(np.float64)

        import pdb; pdb.set_trace()

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    #parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("-w", "--work_dir", default=os.getcwd(), required=False)
    parser.add_argument("-p", "--permutation_dir", default="/home/bournelab/urfold_runs/multiple_loop_permutations/sh3_3", required=False)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-r", "--rmsd", action="store_true")
    parser.add_argument("-c", "--cp", action="store_true")
    parser.add_argument("--save_superpose", action="store_true")
    parser.add_argument("ava_superfamily", nargs="+")
    #parser = DomainStructureDataModule.add_data_specific_args(parser)
    parser = DomainStructureDataModule.add_data_specific_args(parser, eval=True)
    parser.set_defaults(
        data_dir="/home/bournelab/data-eppic-cath-features/",
        all_domains=True)
    args = parser.parse_args()
    runner = TMAlignAllVsAll(args.ava_superfamily, args.data_dir, args, rmsd=args.rmsd, cp=args.cp, superpose=args.save_superpose, permutation_dir=args.permutation_dir,
        work_dir=args.work_dir, force=args.force)
    runner.run()
