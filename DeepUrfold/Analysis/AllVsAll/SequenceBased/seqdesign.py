import os
import re
import glob
import time
import shutil
import argparse
import multiprocessing
from datetime import datetime

import numpy as np
import pandas as pd


from DeepUrfold.Analysis.AllVsAll.SequenceBased.deepsequence import DeepSequenceAllVsAll
from molmimic.parsers.container import Container
from molmimic.parsers.muscle import Muscle

import torch

class SeqDesign(Container):
    IMAGE = "seqdesign" #"docker://edraizen/seqdesign"
    GPUS = True
    ENTRYPOINT = "calc_logprobs_seqs_fr"
    PARAMETERS = [
        ("sess", "str"),
        (":dropout-p:1.0", "str"),
        (":num-samples:1", "str"),
        ("input", "path:in"),
        ("output", "path:out"),
        (":minibatch-size:512", "str")
    ]
    DETACH = True
    ARG_START = "--"
    EXTRA_CONTAINER_KWDS = {"shm_size":"50g"}

    def __call__(self, *args, **kwds):
        if self.ENTRYPOINT == "calc_logprobs_seqs_fr":
            tmpdir = self.tempfile(delete=True)
            os.makedirs(tmpdir)

            tmpdir_local = os.path.basename(tmpdir)
            with open(f"{tmpdir}.sh", "w") as new_entrypoint:
                print(f"cd {tmpdir_local} && calc_logprobs_seqs_fr \"$@\"", file=new_entrypoint)

            self.ENTRYPOINT = ["bash", f"/data/{tmpdir_local}.sh"]
            super().__call__(*args, **kwds)
            self.ENTRYPOINT = "calc_logprobs_seqs_fr"
            try:
                shutil.rmtree()
            except Exception:
                pass

            try:
                os.remove(tmpdir)
            except Exception:
                pass
        else:
            super().__call__(*args, **kwds)

    def train(self, dataset, batch_size=1024, gpu=0, workers=60, restore=None):
        new_params = [
            ("dataset", "path:in"),
            (f":batch-size:{batch_size}", "str"),
            (f":num-data-workers:{workers}", "str"),
        ]
        if restore is not None:
            new_params.append((f":restore", "path:in"))

        with self.custom_entrypoint("run_autoregressive_fr", new_params):
            if restore is not None:
                self(dataset, restore=restore)
            else:
                self(dataset)

    def infer(self, *args, **kwds):
        return self(*args, **kwds)

class SeqDesignAllVsAll(DeepSequenceAllVsAll):
    #Subclass DeepSequence to get correct alignment
    NJOBS = torch.cuda.device_count() #Number of GPUs
    METHOD = "SeqDesign"
    MODEL_TYPE = "Auto-regressive model"
    SCORE_METRIC = "Bitscore"
    SCORE_INCREASING = False
    DATA_INPUT = "EVcouplings alignment"

    def create_superfamily_alignment(self, i, superfamily, sequences, out_file=None):
        input_file = os.path.join(self.work_dir, f"SeqDesign_{superfamily}_train.fasta")

        if not self.force and os.path.isfile(input_file):
           return superfamily, input_file

        superfamily, alignment_file = super().create_superfamily_alignment(i, superfamily, sequences, out_file=out_file)

        if alignment_file is None:
            return superfamily, None

        weight_file = self.calc_weights(alignment_file, input_file+".weights")

        with open(input_file, "w") as f:
            self.filter_sequences(weight_file, f, ungap=True, representatives=False)

        return superfamily, input_file

    def calc_weights(self, alignment_file, output_file, cutoff=0.2):
        from evcouplings.align.alignment import Alignment
        with open(alignment_file) as f:
            aln = Alignment.from_file(f)

        print("Calc weights", alignment_file)
        aln.set_weights(1-cutoff)
        print("Done Calc weights", alignment_file)
        aln.ids = np.array([f"{name}:{weight}" for name, weight in zip(aln.ids, aln.weights)])

        with open(output_file, "w") as f:
            aln.write(f)

        return output_file

    def completed_trained_model(self, model_name, model_input, should_restore=False):
        model_dir_prefix = os.path.join(self.work_dir, "sess")
        prefix = f"{os.path.basename(os.path.splitext(model_input)[0])}_v3-pt_channels-48_rseed-42_"

        versions = [d[len(prefix):]for d in next(os.walk(model_dir_prefix))[1] \
            if d.startswith(prefix)]

        restore = None
        if len(versions) > 0:
            newest = max(versions, key=lambda x: datetime.strptime(x, '%y%b%d_%I%M%p'))
            prefix += newest
            model_file_prefix = os.path.join(os.path.join(model_dir_prefix), prefix, f"{prefix}.ckpt-")

            if os.path.isfile(model_file_prefix+"250000.pth"):
                return model_file_prefix+"250000.pth"
            elif not should_restore:
                return None

            start_iterations = [int(model_file[:-4].split("-")[-1]) for model_file \
                    in glob.glob(model_file_prefix+"*.pth")]

            if len(start_iterations) > 0:
                start_iteration = max(start_iterations)
                restore = model_file_prefix+f"{start_iteration}.pth"

            return restore

        #Either not trained yet or nothing to restore
        return None

    def train_gpu(self, i, model_name, model_input, gpu=0):
        print("Start train", model_name, "on gpu", gpu)
        if model_input is None:
            return model_name, self.completed_trained_model(model_name, model_input)

        restore = self.completed_trained_model(model_name, model_input, should_restore=True)

        model = SeqDesign(work_dir=self.work_dir)
        model.enable_gpus(gpu)
        try:
            model.train(dataset=model_input, restore=restore, batch_size=512)
        except Exception:
            model.train(dataset=model_input, restore=restore, batch_size=512)

        return model_name, self.completed_trained_model(model_name, model_input)

    def completed_inference(self, model_name):
        output_file = os.path.join(self.work_dir, f"SeqDesing_{model_name}_all_vs_all_results.csv")
        if os.path.isfile(output_file):
            results =  pd.read_csv(output_file, index_col=None)
            results = pd.read_csv(output_file, index_col=None)
            results["name"] = results["name"].str[11:18]
            results = results.rename(columns={"name":"cathDomain"})
            results = pd.merge(results, self.representative_domains, on="cathDomain")
            results = results.rename(columns={"superfamily":"true_sfam"})
            results = results.set_index(["cathDomain", "true_sfam"])
            results = results.drop(columns=["mean", "forward", "reverse", "sequence"])
            results = results.rename(columns={"bitperchar":model_name})
            return results
        return None

    def infer_gpu(self, i, superfamily, model_path, combined_sequences, gpu=0):
        output_file = os.path.join(self.work_dir, f"SeqDesing_{superfamily}_all_vs_all_results.csv")

        if self.force or not os.path.isfile(output_file):
            #model_prefix = os.path.basename(model_path).split(".ckpt")[0]
            #model_prefix += f"/{model_prefix}"
            model_prefix = os.path.basename(os.path.dirname(model_path))
            model_file = f"{model_prefix}/{os.path.splitext(os.path.basename(model_path))[0]}"
            model = SeqDesign(work_dir=self.work_dir)
            model.enable_gpus(gpu)
            model(sess=model_file, input=combined_sequences, output=output_file)

        results = self.completed_inference(superfamily)
        if results is None:
            raise RuntimeError

        return superfamily, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("-w", "--work_dir", default=os.getcwd(), required=False)
    parser.add_argument("-p", "--permutation_dir", default="/home/bournelab/urfold_runs/multiple_loop_permutations/sh3_3", required=False)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("superfamily", nargs="+")
    args = parser.parse_args()
    runner = SeqDesignAllVsAll(args.superfamily, args.data_dir, permutation_dir=args.permutation_dir,
        work_dir=args.work_dir, force=args.force)
    runner.run()
