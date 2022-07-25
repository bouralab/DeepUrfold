import os
import sys
import glob
import json
import argparse
import subprocess

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from Bio.PDB.Polypeptide import three_to_one

from DeepUrfold.Analysis.AllVsAll.StructureBased import StructureBasedAllVsAll
from DeepUrfold.DataModules.DomainStructureDataModule import DomainStructureDataModule
from molmimic.common.Structure import Structure
from molmimic.util.pdb import remove_ter_lines

def parallel_process_domain(obj, cath_domain, pdb_file, features_file):
    return obj.process_domain(cath_domain, pdb_file, features_file)

class Struct2Seq(StructureBasedAllVsAll):
    METHOD = "Struct2Seq"
    GPU = True
    NJOBS_CPU = 20
    NJOBS = 4
    DATA_INPUT = "CATH Superfamiies"
    SCORE_INCREASING = False
    MODEL_TYPE = "Graph NN"
    SCORE_METRIC = "Bitscore"

    def create_superfamily_dataset(self, superfamily, hparams):
        input_file = f"Struct2Seq_{superfamily}_input.jsonl"
        split_file = f"Struct2Seq_{superfamily}_splits.json"
        if not self.force and os.path.isfile(input_file) and os.path.isfile(split_file):
            return superfamily, (input_file, split_file)

        hparams.superfamily = superfamily
        dataset = DomainStructureDataModule(hparams, eval=True)
        dataset.setup(stage="test")
        with open(input_file, "w") as f:
            for entry in Parallel(n_jobs=self.NJOBS_CPU)(delayed(parallel_process_domain)(\
              self, row.cathDomain, row.structure_file, row.feature_file) for row in \
              dataset.test_dataset.data.itertuples()):
                f.write(json.dumps(entry) + '\n')

        train_dir = os.path.dirname(dataset.test_file)
        train_file = os.path.join(train_dir, "DomainStructureDataset-train-C.A.T.H.S35-split0.8.h5")
        valid_file = os.path.join(train_dir, "DomainStructureDataset-valid-C.A.T.H.S35-split0.1.h5")
        test_file = os.path.join(train_dir, "DomainStructureDataset-test-C.A.T.H.S35-split0.1.h5")

        data_splits = {}
        for n, f in (('train', train_file), ('validation', valid_file), ('test', test_file)):
            assert os.path.isfile(f)
            try:
                df = pd.read_hdf(f, "table")
                data_splits[n] = df["cathDomain"].tolist()
            except KeyError as e:
                data_splits[n] = []

        with open(split_file, "w") as f:
            f.write(json.dumps(data_splits))

        print(superfamily, input_file)
        return superfamily, (input_file, split_file)

    def process_domain(self, cath_domain, pdb_file, features_file, target_atoms = ['N', 'CA', 'C', 'O']):
        """ Parse mmtf file to extract C-alpha coordinates """
        if not os.path.isfile(pdb_file+".noter") or os.stat(pdb_file+".noter").st_size == 0:
            print(f"Making file {pdb_file}.noter")
            remove_ter_lines(pdb_file, pdb_file+".noter")

        pdb_file = pdb_file+".noter"

        features_path = os.path.dirname(features_file)

        structure = Structure(pdb_file, cath_domain, features_path=features_path)

        # Build a dictionary
        mmtf_dict = {}
        mmtf_dict['seq'] = ""
        mmtf_dict['coords'] = {code:[] for code in target_atoms}
        mmtf_dict['num_chains'] = 1
        mmtf_dict['name'] = cath_domain
        mmtf_dict['CATH'] = [cath_domain]

        for residue in structure.structure.get_residues():
            resname = "{}{}".format(*residue.get_id()[1:]).strip()
            mmtf_dict['seq'] += three_to_one(residue.get_resname())

            for atom in residue:
                if atom.get_name() in target_atoms:
                    coord = np.around(atom.coord, decimals=4).tolist()
                    mmtf_dict['coords'][atom.get_name()].append(coord)


        return mmtf_dict

    def create_combined_dataset(self):
        input_file = f"Struct2Seq_combined_input.jsonl"
        input_names_file = f"Struct2Seq_combined_input_names.json"
        if not self.force and os.path.isfile(input_file) and os.path.isfile(input_names_file):
            return (input_file, input_names_file)

        representatives = self.representative_domains["cathDomain"].tolist()
        with open(input_file, "w") as combined:
            for model_name, model_input in self.superfamily_datasets.items():
                with open(model_input[0]) as f:
                    for i, line in enumerate(f):
                        entry = json.loads(line)
                        if entry["name"] in representatives:
                            combined.write(json.dumps(entry) + '\n')

        data_splits = {"test": self.representative_domains["cathDomain"].tolist()}
        with open(input_names_file, "w") as f:
            f.write(json.dumps(data_splits))

        return (input_file, input_names_file)

    def completed_trained_model(self, model_name, model_input):
        model_file_prefix = os.path.join(os.getcwd(), "log", model_name, "checkpoints", "epoch100_*.pt")
        model_file = list(glob.glob(model_file_prefix))
        print(model_file)
        if len(model_file)>0:
            return model_name, model_file[0]
        return None

    def train_gpu(self, i, model_name, model_input, gpu=0):
        model_file_prefix = os.path.join(os.getcwd(), "log", model_name, "checkpoints", "epoch100_*.pt")

        completed = self.completed_trained_model(model_name, model_input)
        if completed is not None:
            return completed

        f = os.path.join(os.getcwd(), "neurips19-graph-protein-design", "experiments", "train_s2s.py")
        cmd = [sys.executable, f, "--cuda",
            "--file_data", model_input[0],
            "--file_splits", model_input[1],
            "--batch_tokens=6000",
            "--features", "full",
            "--name", model_name]
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
        my_env["PYTHONPATH"] = f"{my_env.get('PYTHONPATH', '')}:{os.path.join(os.getcwd(), 'neurips19-graph-protein-design')}"

        subprocess.call(cmd, env=my_env)

        model_file = list(glob.glob(model_file_prefix))
        if len(model_file)>0:
            return model_name, model_file[0]
        else:
            with open(model_input[1]) as f:
                data_splits = json.load(f)

            if len(data_splits["test"]) == 0:
                raise RuntimeError("No domains to test")

            if len(data_splits["validation"]) == 0:
                data_splits["validation"] = data_splits["test"]

            if len(data_splits["train"]) == 0:
                data_splits["train"] = data_splits["validation"]

            with open(model_input[1], "w") as f:
                f.write(json.dumps(data_splits))

            return self.train_gpu(i, model_name, model_input, gpu=gpu)

    def completed_inference(self, model_name):
        results_file = os.path.join(self.work_dir, f"Struct2Seq_{model_name}_results.h5")
        if not self.force and os.path.isfile(results_file):
            df =  pd.read_hdf(results_file, "table")
            #df = df.astype(np.float64)
            #df.to_hdf(results_file, "table")
            return df
        return None

    def infer_gpu(self, i, model_name, model_path, combined_dataset, gpu=0):
        results_file = os.path.join(os.path.dirname(model_path[0]), f"Struct2Seq_{model_name}_results.h5")
        # if not self.force and os.path.isfile(results_file):
        #     df = pd.read_hdf(results_file, "table")
        #     return df

        f = os.path.join(os.getcwd(), "neurips19-graph-protein-design", "experiments", "test_s2s_loop.py")

        cmd = [sys.executable, f, "--cuda",
            "--restore", model_path[1],
            "--file_data", combined_dataset[0],
            "--file_splits", combined_dataset[1],
            "--batch_tokens=1",
            "--features", "full",
            "--name", model_name]
        my_env = os.environ.copy()
        my_env["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
        my_env["PYTHONPATH"] = f"{my_env.get('PYTHONPATH', '')}:{os.path.join(os.getcwd(), 'neurips19-graph-protein-design')}"
        p = subprocess.Popen(cmd, env=my_env, stdout=subprocess.PIPE)
        stdout, stderr = p.communicate()

        s2s_output = [[kv.split("=")[1] for kv in p.split(";")] for p in \
            stdout.decode('utf-8').splitlines() if p.startswith("name")]
        domains, perplexities = zip(*s2s_output)
        perplexities = [float(p) for p in perplexities]
        print(domains)
        print(perplexities)

        results = pd.DataFrame({"cathDomain": domains, model_name:perplexities})
        results = pd.merge(results, self.representative_domains, on="cathDomain")
        results = results.rename(columns={"superfamily":"true_sfam"})
        print(results)
        results = results.set_index(["cathDomain", "true_sfam"])

        results.to_hdf(results_file, "table")


        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    #parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("-w", "--work_dir", default=os.getcwd(), required=False)
    parser.add_argument("-p", "--permutation_dir", default="/home/bournelab/urfold_runs/multiple_loop_permutations/sh3_3", required=False)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("ava_superfamily", nargs="+")
    parser = DomainStructureDataModule.add_data_specific_args(parser, eval=True)
    parser.set_defaults(
        data_dir="/home/bournelab/data-eppic-cath-features/",
        all_domains=True)
    args = parser.parse_args()
    print(args)
    runner = Struct2Seq(args.ava_superfamily, args.data_dir, args, permutation_dir=args.permutation_dir,
        work_dir=args.work_dir, force=args.force)
    runner.run()
