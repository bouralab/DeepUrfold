import os
import argparse

import numpy as np
import pandas as pd

from DeepUrfold.Analysis.AllVsAll.SequenceBased import SequenceBasedAllVsAll
from DeepUrfold.Analysis.AllVsAll.SequenceBased.EVcouplings import EVcouplingsAllVsAll
from molmimic.parsers.hmmer import HMMER

class HmmerAllVsAll(SequenceBasedAllVsAll):
    METHOD = "HMMER"
    DATA_INPUT = "CATH S35"
    SCORE_INCREASING = True
    MODEL_TYPE = "HMM"
    SCORE_METRIC = "Bitscore"

    def __init__(self, superfamilies, data_dir, permutation_dir=None, work_dir=None, force=False):
        super().__init__(superfamilies, data_dir, permutation_dir=permutation_dir, work_dir=work_dir, force=force)
        self.hmmer = HMMER(work_dir=work_dir)

    def train_all(self):
        hmm_file = os.path.join(self.work_dir, "combined_superfamily_models.hmm")
        if self.force or not os.path.isfile(hmm_file):
            hmm_file = self.hmmer.build_combined_hmms(*self.superfamily_datasets.values(), name="combined_superfamily_models")
        return hmm_file

    def infer_all(self, models, combined_sequences):
        db_search_results_file = os.path.join(self.work_dir, "hmmer_combined_search_results.out")
        if True or self.force or not os.path.isfile(db_search_results_file):
            self.hmmer.search(hmmfile=models, seqdb=combined_sequences, o=db_search_results_file, T=-1e12, max=True)
        results = self.hmmer.parse_hmmsearch_results(db_search_results_file, id_parser=self.id_parser)

        print(results)

        order = pd.DataFrame({"cathDomain":self.representative_domains["cathDomain"]})
        order = order[order["cathDomain"].isin(results.index)]["cathDomain"]

        results = results.reindex(order)
        results = results[sorted(self.superfamily_datasets.keys())]

        print(results)

        results = pd.merge(results, self.representative_domains, left_index=True, right_on="cathDomain")
        results = results.rename(columns={"superfamily":"true_sfam"})
        results = results.set_index(["cathDomain", "true_sfam"])

        import pdb; pdb.set_trace()

        results = results.fillna(0.0)
        results = results.replace([np.inf, -np.inf], [1, 0])
        results = results.astype(np.float64)

        return results

class EVHmmerAllVsAll(EVcouplingsAllVsAll, HmmerAllVsAll):
    METHOD = "HMMER"
    DATA_INPUT = "EVcouplings alignment"
    SCORE_INCREASING = True
    MODEL_TYPE = "HMM"
    SCORE_METRIC = "Bitscore"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("-w", "--work_dir", default=os.getcwd(), required=False)
    parser.add_argument("-p", "--permutation_dir", default="/home/bournelab/urfold_runs/multiple_loop_permutations/sh3_3", required=False)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-e", "--evcouplings", action="store_true")
    parser.add_argument("superfamily", nargs="+")
    args = parser.parse_args()

    if args.evcouplings:
        runner = EVHmmerAllVsAll(args.superfamily, args.data_dir, permutation_dir=args.permutation_dir,
            work_dir=args.work_dir, force=args.force)
    else:
        runner = HmmerAllVsAll(args.superfamily, args.data_dir, permutation_dir=args.permutation_dir,
            work_dir=args.work_dir, force=args.force)

    runner.run()
