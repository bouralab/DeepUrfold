import os
import glob
import time
import argparse
import multiprocessing

import torch
import numpy as np
from Bio import SeqIO

from DeepUrfold.Analysis.AllVsAll.SequenceBased.EVcouplings import EVcouplingsAllVsAll
from molmimic.parsers.container import Container
from molmimic.parsers.muscle import Muscle

class DeepSequence(Container):
    IMAGE = "deepsequence" #"docker://edraizen/deepsequence"
    GPUS = True
    PARAMETERS = [
        ("python_file", "path:in", ["{}"]),
        ("alignment_file", "path:in"),
        (":weights", "str"),
        (":test_sequences", "path:in")
    ]
    DETACH = True
    ARG_START = "--"
    EXTRA_CONTAINER_KWDS = {"shm_size":"50g"}

    def train(self, alignment_file, python_file=None):
        if python_file is None:
            python_file = os.path.join(os.path.dirname(__file__), "run_deepsequence.py")

        if not os.path.isfile(python_file):
            raise RuntimeError(f"Python file not found: {python_file}")

        self(python_file=python_file, alignment_file=alignment_file)

        return self.get_model_params(alignment_file)

    def infer(self, alignment_file, weights, test_sequences, python_file=None):
        if python_file is None:
            python_file = os.path.join(os.path.dirname(__file__), "run_deepsequence.py")

        if not os.path.isfile(python_file):
            raise RuntimeError(f"Python file not found: {python_file}")

        self(python_file=python_file, alignment_file=alignment_file, weights=weights,
            test_sequences=test_sequences)

        return self.get_results(test_sequences, weights)

    @classmethod
    def get_model_params(cls, alignment_file, work_dir=None):
        if work_dir is None:
            if hasattr(cls, "work_dir"):
                work_dir = cls.work_dir
            else:
                work_dir = os.getcwd()

        #Final parameters look like this:
        #vae_output_encoder-1500-1500_Nlatent-30_decoder-100-500_alignment_file-3.30.300.20.aln.fasta_bs-100_conv_pat-True_d_c_size-40_final_decode_nonlin-sigmoid_final_pwm_scale-True_logit_p-0.001_n_pat-4_r_seed-12345_sparsity-logit_v.pkl
        for v_file in glob.glob(os.path.join(work_dir, "params", f"*_alignment_file-{os.path.basename(alignment_file)}_*_v.pkl")):
            prefix = v_file[:-6]
            if os.path.isfile(f"{prefix}_params.pkl") and os.path.isfile(f"{prefix}_m.pkl"):
                #All 3 param files exists
                break
        else:
            raise RuntimeError(f"Cannot find parameter files for {alignment_file}")

        return os.path.basename(prefix)

    @classmethod
    def get_results(cls, sequences, parameter_file_prefix, work_dir=None):
        if work_dir is None:
            if hasattr(cls, "work_dir"):
                work_dir = cls.work_dir
            else:
                work_dir = os.getcwd()

        #Results look like this:
        #combined_sequences_vae_output_encoder-1500-1500_Nlatent-30_decoder-100-500_alignment_file-3.30.300.20.aln.fasta_bs-100_conv_pat-True_d_c_size-40_final_decode_nonlin-sigmoid_final_pwm_scale-True_logit_p-0.001_n_pat-4_r_seed-12345_sparsity-logit_scores.npy
        for results_file in glob.glob(os.path.join(work_dir, "params", f"*.npy")):
            prefix = v_file[:-6]
            if os.path.basename(os.path.splitext(sequences)[0]) in results_file and \
               os.path.basename(os.path.splitext(parameter_file_prefix)[0]) in results_file:
                break
        else:
            raise RuntimeError(f"Cannot find results files for {sequences} and {parameter_file_prefix}")

        results = np.load(results_file)
        results = pd.DataFrame(results, columns=["logpx_i", "KLD_latent", "logpxz"])
        return results

class DeepSequenceAllVsAll(EVcouplingsAllVsAll):
    NJOBS_CPU = 12
    NJOBS = torch.cuda.device_count() #Number of GPUs
    GPU = True
    METHOD = "DeepSequence"
    MODEL_TYPE = "VAE"
    SCORE_METRIC = "ELBO"
    SCORE_INCREASING = False
    DATA_INPUT = "EVcouplings alignment"

    def train_gpu(self, i, model_name, model_input, gpu=0):
        print("Train gpu", gpu, model_name, i)

        if not self.force:
            try:
                trained_model_prefix = DeepSequence.get_model_params(model_input, work_dir=self.work_dir)
                print(model_name, trained_model_prefix)
                return model_name, trained_model_prefix
            except RuntimeError:
                #Train as usual
                pass

        print("FAILING", i, model_name, model_input, gpu)



        svi.enable_gpus(gpu)
        trained_model_prefix = svi.train(alignment_file=model_input)

        print(model_name, trained_model_prefix)

        return model_name, trained_model_prefix

    def completed_inference(self, model_name, model_input):
        return None

    def infer_gpu(self, i, model_name, model_path, combined_sequences, gpu=0):
        c = CouplingsModel(model_path)
        input_sequences, input_test_split, domain_order = self.create_input_sequences(combined_sequences, c.L)

        all_sequence_scores = self.score_sequences((model_path, gpu), input_sequences)

        indices = np.cumsum([0]+input_test_split)

        import pdb; pdb.set_trace()

        results = pd.concat([all_sequence_scores.iloc[indices[i]:indices[i+1], :]["logpx_i"].median() \
            for i in range(len(indices)-1)])

        sequence_scores = np.concatenate([np.median(scores) for scores in \
            np.split(all_sequence_scores, input_test_split)])

        results = pd.DataFrame({"cathDomains":domain_order, "E":sequence_scores})
        results = pd.merge(results, self.representative_domains, on="cathDomain")
        results = results.rename(columns={"superfamily":"true_sfam"})
        results = results.set_index(["cathDomain", "true_sfam"])

        return results
        self.infer(i, model_name, model_path, combined_sequences)
        print(i, model_name, model_path, combined_sequences, gpu)

        results = None
        if not self.force:
            try:
                results = DeepSequence.get_results(combined_sequences, model_path, work_dir=self.work_dir)
            except RuntimeError:
                #Infer as usual
                pass

        if results is None:
            self.muscle = Muscle(work_dir=work_dir)
            aligned_sequences = self.muscle.align_sequences_to_alignment(combined_sequences, self.superfamily_datasets[model_name], n_jobs=20)
            svi = DeepSequence(work_dir=self.work_dir)

        results = results[["logpx_i"]]
        results = results.assign(cathDomain=cathDomain)
        results = pd.merge(results, self.representative_domains, on="cathDomain")
        results = results.rename(columns={"superfamily":"true_sfam"})
        results = results.set_index(["cathDomain", "true_sfam"])

        return results

    def score_sequences(self, model_info, sequences):
        model_path, gpu = model_info

        try:
            svi = DeepSequence(work_dir=self.work_dir)
        except RuntimeError:
            print("FAILING", i, model_name, model_input, gpu)
            raise

        svi.enable_gpus(gpu)
        return svi.infer(alignment_file=self.superfamily_datasets[model_name],
            weights=model_path, test_sequences=aligned_sequences)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("-w", "--work_dir", default=os.getcwd(), required=False)
    parser.add_argument("-p", "--permutation_dir", default="/home/bournelab/urfold_runs/multiple_loop_permutations/sh3_3", required=False)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("superfamily", nargs="+")
    args = parser.parse_args()
    runner = DeepSequenceAllVsAll(args.superfamily, args.data_dir, permutation_dir=args.permutation_dir,
        work_dir=args.work_dir, force=args.force)
    runner.run()
