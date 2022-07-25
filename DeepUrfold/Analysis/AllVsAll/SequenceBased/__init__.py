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
from molmimic.parsers.cath import CATHApi
from molmimic.parsers.muscle import Muscle


def parallel_align(obj, i, model_name, sequences, out_file=None):
    result = obj.create_superfamily_alignment(i, model_name, sequences, out_file=out_file)
    return result

class SequenceBasedAllVsAll(AllVsAll):
    NJOBS = 20
    NJOBS_CPU = None
    GPU = False
    METHOD = None

    def __init__(self, superfamilies, data_dir, permutation_dir=None, work_dir=None, force=False, downsample_sbm=False):
        super().__init__(superfamilies, data_dir, permutation_dir=permutation_dir, work_dir=work_dir, force=force, downsample_sbm=downsample_sbm)
        self.raw_sequences = {}
        self.combined_datatset = {}

    def initialize_data(self):
        self.download_cath_sequences()
        self.combined_dataset = self.combine_filter_sequences("combined_superfamilies.fasta", ungap=True)
        self.create_superfamily_datasets()
        print(self.combined_dataset)

    def combine_filter_sequences(self, new_file, ungap=False, representatives=True):
        combined_fasta = os.path.join(self.work_dir, new_file)
        superfamily_datasets = self.superfamily_datasets if not ungap else self.raw_sequences
        print("raw2", self.raw_sequences)
        print(superfamily_datasets)
        with open(combined_fasta, "w") as f:
            for superfamily, fasta_file in superfamily_datasets.items():
                self.filter_sequences(fasta_file, f, ungap=ungap, representatives=representatives)
        return combined_fasta

    def filter_sequences(self, sequences, out_file, ungap=False, representatives=True):
        for cathDomain, sequence in self.iterate_sequences(sequences, representatives=representatives):
            if ungap:
                sequence.seq = sequence.seq.ungap("-").ungap(".").upper()
            SeqIO.write(sequence, out_file, "fasta")

    def iterate_sequences(self, sequences, representatives=True):
        for sequence in SeqIO.parse(sequences, "fasta"):
            m = self.id_parser.search(sequence.id)
            if not representatives or (m and m.groups()[0] in self.representative_domains["cathDomain"].values):
                cathDomain = m.groups()[0] if m else sequence.id
                yield cathDomain, sequence

    def download_cath_sequences(self):
        for superfamily in self.superfamilies:
            superfamily, small_fasta_file = \
                self.download_cath_sequences_for_superfamily(superfamily)
            # if sum(1 for _ in self.iterate_sequences(small_fasta_file)) < 2:
            #     continue
            self.raw_sequences[superfamily] = small_fasta_file
        print("raw", self.raw_sequences)

    def complete_superfamily_dataset(self, superfamily):
        return None

    def create_superfamily_datasets(self):
        n_jobs = self.NJOBS_CPU if self.NJOBS_CPU is not None else self.NJOBS
        superfamily_datasets = {sfam:self.complete_superfamily_dataset(sfam) \
            for sfam in self.raw_sequences.keys()}
        self.superfamily_datasets = {sfam: model for sfam, model in superfamily_datasets.items() \
            if model is not None}
        new_superfamily_datasets = dict(Parallel(n_jobs=n_jobs)(delayed(parallel_align)(\
            self, i, superfamily, alignment) for i, (superfamily, alignment) in \
            enumerate(sorted(self.raw_sequences.items(), key=lambda x:x[0]))))
        self.superfamily_datasets.update(new_superfamily_datasets)

    def download_cath_sequences_for_superfamily(self, superfamily):
        small_fasta_file = os.path.join(self.work_dir, f"{superfamily}.fasta")

        if self.force or not os.path.isfile(small_fasta_file):
            cath = CATHApi(work_dir=self.work_dir)
            fasta_file = cath.get_superfamily_sequences(superfamily)
            with open(small_fasta_file, "w") as f:
                self.filter_sequences(fasta_file, f)

        return superfamily, small_fasta_file

    def create_superfamily_alignment(self, i, superfamily, sequences, out_file=None):
        if out_file is None:
            out_file = os.path.join(self.work_dir, f"{os.path.splitext(os.path.basename(sequences))[0]}.aln.fasta")

        if self.force or not os.path.isfile(out_file):
            muscle = Muscle(work_dir=self.work_dir)
            out_file = muscle(in_file=sequences, out_file=out_file)
        return superfamily, out_file

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
    parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("superfamily", nargs="+")
    args = parser.parse_args()
    HMMAllVsAll(args.superfamily, args.data_dir, representatives=args.representatives)
