import os
import glob
import time
import argparse

import numpy as np
import pandas as pd
from Bio import SeqIO
from evcouplings.utils import read_config_file
from evcouplings.utils.pipeline import execute
from evcouplings.couplings import CouplingsModel

from DeepUrfold.Analysis.AllVsAll.SequenceBased import SequenceBasedAllVsAll

class EVcouplingsAllVsAll(SequenceBasedAllVsAll):
    NJOBS = 12
    METHOD = "EVCouplings"
    MODEL_TYPE = "Potts model"
    SCORE_METRIC = "Hamiltonian"
    SCORE_INCREASING = True
    DATA_INPUT = "EVcouplings alignment"

    def create_superfamily_alignment(self, i, superfamily, sequences, out_file=None):
        prefix = f"evcouplings_output/{superfamily}"
        alignment_file = os.path.join(prefix, "align", f"{superfamily}.a2m")

        if not self.force and os.path.isfile(alignment_file):
            return superfamily, alignment_file

        print("Creating alingment for", superfamily)

        focus_seq = next(SeqIO.parse(sequences, "fasta"))
        m = self.id_parser.search(focus_seq.id)
        focus_seq_id = m.groups()[0] if m else focus_seq_id.id
        focus_seq_file = os.path.join(self.work_dir, f"focus_seq_{focus_seq_id}_{superfamily}.fasta")
        with open(focus_seq_file, "w") as f:
            SeqIO.write(focus_seq, f, "fasta")

        config = read_config_file(os.path.join(os.path.dirname(__file__), "deepsequence_config.txt"))
        config["global"]["prefix"] = f"evcouplings_output/{superfamily}"
        config["global"]["sequence_id"] = focus_seq_id
        config["global"]["sequence_file"] = focus_seq_file

        try:
            outcfg = execute(**config)
        except SystemExit as e:
            if not "billiard" in str(e):
                raise

        if not os.path.isfile(alignment_file):
            raise RuntimeError(f"EVcouplings could not run, alignment_file does not exist: {alignment_file}")

        return superfamily, alignment_file

    def train(self, i, model_name, model_input, gpu=0):
        #Training is donw during alignment generation

        model_file = os.path.join(self.work_dir, "evcouplings_output", model_name, "couplings", f"{model_name}.model")
        print(model_file)

        if not os.path.isfile(model_file):
            raise RuntimeError(f"Must create alingment, which fits the evcouplings model simultaneosly for {model_name}")

        return model_name, model_file

    def completed_inference(self, model_name, model_input):
        return None

    def infer(self, i, model_name, model_path, combined_sequences):
        c = CouplingsModel(model_path)
        input_sequences, input_test_split, domain_order = self.create_input_sequences(combined_sequences, c.L)

        for i in range(len(input_test_split)-1):
            assert input_test_split[i]<input_test_split[i+1] and input_test_split[i] < len(input_sequences)

        all_sequence_scores = self.score_sequences(model_path, input_sequences)
        sequence_scores = np.array([np.max(all_sequence_scores[input_test_split[i]:input_test_split[i+1]]) \
            for i in range(len(input_test_split)-1)])

        results = pd.DataFrame({"cathDomain":domain_order, model_name:sequence_scores})
        results = pd.merge(results, self.representative_domains, on="cathDomain")
        results = results.rename(columns={"superfamily":"true_sfam"})
        results = results.set_index(["cathDomain", "true_sfam"])

        print(results.columns)

        return results

    def create_input_sequences(self, combined_sequences, L):
        domain_order = []
        test_sequences = []
        n_seqs_per_test = []
        for cathDomain, sequence in self.iterate_sequences(combined_sequences):
            sequence = str(sequence.seq).replace("X", "G")
            if len(sequence)>L:
                s_test = [sequence[i:i+L] for i in range(len(sequence)-L)]
                assert len(s_test)>0, "Long"
            elif len(sequence)<L:
                s_test = ["G"*i+sequence+"G"*(L-len(sequence)-i) for i in range(L-len(sequence))]
                assert len(s_test)>0, "Short"
            else:
                s_test = [sequence]

            test_sequences += s_test
            domain_order.append(cathDomain)
            n_seqs_per_test.append(len(s_test))
        n_seqs_per_test = [0]+list(np.cumsum(n_seqs_per_test))
        return test_sequences, n_seqs_per_test, domain_order

    def score_sequences(self, model_path, sequences):
        c = CouplingsModel(model_path)
        result = c.hamiltonians(sequences)[:,0]
        print(result)
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("-w", "--work_dir", default=os.getcwd(), required=False)
    parser.add_argument("-p", "--permutation_dir", default="/home/bournelab/urfold_runs/multiple_loop_permutations/sh3_3", required=False)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("superfamily", nargs="+")
    args = parser.parse_args()
    runner = EVcouplingsAllVsAll(args.superfamily, args.data_dir, permutation_dir=args.permutation_dir,
        work_dir=args.work_dir, force=args.force)
    runner.run()
