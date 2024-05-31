import os
import re
import glob
import time
import argparse
import subprocess
import multiprocessing

import torch
import pandas as pd
from Bio import SeqIO
from joblib import Parallel, delayed

from DeepUrfold.Analysis.AllVsAll import AllVsAll
from DeepUrfold.DataModules.DistributedDomainStructureDataModule import DistributedDomainStructureDataModule
from Prop3D.util.pdb import remove_ter_lines

def fix_ter(pdb_file):
    if not os.path.isfile(pdb_file+".noter") or os.stat(pdb_file+".noter").st_size == 0:
        print(f"Making file {pdb_file}.noter")
        remove_ter_lines(pdb_file, pdb_file+".noter")
    return pdb_file+".noter"

class StructureBasedAllVsAll(AllVsAll):
    NJOBS = 20
    NJOBS_CPU = None
    GPU = False
    METHOD = None

    def __init__(self, superfamilies, data_dir, hparams, permutation_dir=None, work_dir=None, force=False, downsample_sbm=False, cluster=True):
        super().__init__(superfamilies, data_dir, permutation_dir=permutation_dir, work_dir=work_dir, force=force, downsample_sbm=downsample_sbm, cluster=cluster)
        self.hparams = hparams

    def create_superfamily_datasets(self):
        self.superfamily_datasets = dict(self.create_superfamily_dataset(superfamily, self.hparams) \
            for superfamily in self.superfamilies)

    def create_superfamily_dataset(self, superfamily, hparams):
        hparams.superfamily = superfamily
        print(hparams.superfamily)
        dataset = DistributedDomainStructureDataModule(hparams, eval=True)
        dataset.setup(stage="test")

        domains = [d.split("/")[-1] for d in dataset.test_dataset.order]
        return superfamily, domains
        #dataset.test_dataset.return_structure = True
        #input_data = dataset.test_dataset.data #[["cathDomain", "structure_file"]]
        #input_data["structure_file"] = input_data["structure_file"].apply(fix_ter)

        #
        # input_data = pd.DataFrame([(s.name, s.save_pdb(f"{s.name}.pdb"), superfamily) \
        #     for s in dataset.test_dataset], names=["cathDomain", "structure_file", "superfamily"])
        #
        # return superfamily, input_data

    def create_combined_dataset(self):
        representatives = self.get_representative_domains()
        #assert representatives.shape[0] == 3674, representatives.shape[0]

        return  representatives
        combined = pd.DataFrame([(superfamily, cathDomain) for superfamily, domains \
            in self.superfamily_datasets.items() for cathDomain in domains],
            columns=["superfamily", "cathDomain"])
        combined = combined[combined["cathDomain"].isin(representatives["cathDomain"])]

        assert combined.shape[0] == 3674


        # for sfam, dset in self.superfamily_datasets.items():



        #combined["structure_file"] = combined["structure_file"].apply(fix_ter)
        return combined

    def get_multiple_loop_permutations(self, permutation_dir):
        permuted_seqs = os.path.join(self.work_dir, "permuted_sequences.fasta")

        if not self.force and os.path.isfile(permuted_seqs):
            return permuted_seqs

        with open(permuted_seqs, "w") as new:
            for pir in glob.glob(os.path.join(permutation_dir, "*_template.pir")):
                reading = False
                name = None
                sequence = ""
                with open(pir) as f:
                    for line in f:
                        if not reading and line.startswith(">") and "target" in line:
                            reading=True
                            name = line[4:-7]
                            next(f)
                        elif reading and line.startswith(">"):
                            break
                        elif reading:
                            sequence += line.rstrip().replace("*", "")
                print(f">{name}\n{sequence}", file=new)
        return permuted_seqs

    def test_multiple_loop_permutations(self, permuted_seqs):
        self.search_all(permuted_seqs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    #parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("superfamily", nargs="+")
    parser = DomainStructureDataModule.add_data_specific_args(parser, eval=True)
    parser.set_defaults(
        data_dir="/home/bournelab/data-eppic-cath-features/",
        all_domains=True
        )
    args = parser.parse_args()
    HMMAllVsAll(args.superfamily, args.data_dir, representatives=args.representatives)
