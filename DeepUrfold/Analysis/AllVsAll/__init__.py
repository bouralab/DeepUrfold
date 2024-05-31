import os
import re
import glob
import time
import argparse
import subprocess
import multiprocessing

#Must be imported before torch
from DeepUrfold.Analysis.StochasticBlockModel import StochasticBlockModel

import h5pyd
import torch
import pandas as pd
from Bio import SeqIO
from joblib import Parallel, delayed

from Prop3D.parsers.cath import CATHApi
from Prop3D.parsers.muscle import Muscle

from DeepUrfold.Analysis.StochasticBlockModel import StochasticBlockModel

total_gpus = torch.cuda.device_count()
# manager = multiprocessing.Manager()
# AVAILABLE_GPUS = manager.list([True]*torch.cuda.device_count())

def get_available_gpu(obj, model_name, i, should_wait=True, wait_time=1):
    """Find free GPU to use

    Bug: If function immediately returns without running, the next functions to
    run have a wierd time delay and some GPUs say they are availabvel even when
    they are not
    """
    #global AVAILABLE_GPUS
    #print(model_name, i, "AVAILABLE_GPUS", AVAILABLE_GPUS, wait_time*2)

    if i < obj.NJOBS & obj.NJOBS<=total_gpus:
        return i

    if should_wait and i>0:
        time.sleep(wait_time*(10*(i%obj.NJOBS)+1)+20) #Wait a second to get correct GPU



    # print(model_name, i, "AVAILABLE_GPUS", AVAILABLE_GPUS, "use")
    #
    # if AVAILABLE_GPUS[i%n_gpus]:
    #     gpu = i%n_gpus
    #     AVAILABLE_GPUS[gpu] = False
    #     return gpu
    #
    # while True:
    #     try:
    #         gpu = AVAILABLE_GPUS.index(True)
    #         AVAILABLE_GPUS[gpu] = False
    #         break
    #     except ValueError:
    #         time.sleep(wait_time*2) #Wait a second to get correct GPU
    cmd = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv"]
    result = subprocess.check_output(cmd).decode("utf-8").splitlines()[1:]
    print(model_name, i, "AVAILABLE_GPUS", result)


    gpu = min([[float(v.replace(" MiB", "")) for v in line.split(", ")] \
        for line in result], key = lambda x: x[1])

    if gpu[1] < 15:
        return int(gpu[0])

    #One will open?
    return get_available_gpu(obj, model_name, i, wait_time=wait_time)

def use_gpu(func):
    def inner(obj, i, model_name, *args, **kwargs):
        if obj.GPU:
            kwargs["gpu"] = get_available_gpu(obj, model_name, i, should_wait=obj.GPU_WAIT, wait_time=i%(obj.NJOBS))
        return func(obj, i, model_name, *args, **kwargs)
    return inner

@use_gpu
def parallel_train(obj, i, model_name, model_input, gpu=None):
    if obj.GPU and gpu is not None:
        result = obj.train_gpu(i, model_name, model_input, gpu=gpu)
    else:
        result = obj.train(i, model_name, model_input)

    return result

@use_gpu
def parallel_infer(obj, i, model_name, model_path, combined_datatset, gpu=None):
    if obj.GPU and gpu is not None:
        result = obj.infer_gpu(i, model_name, model_path, combined_datatset, gpu=gpu)
    else:
        result = obj.infer(i, model_name, model_path, combined_datatset)

    return result

def parallel_completed_trained_model(obj, model_name, model_input):
    return obj.completed_trained_model(model_name, model_input)

def parallel_completed_inference(obj, model_name):
    return obj.completed_inference(model_name)

AVAILABLE_AVA_COMPARATORS = {}

def get_representative_domains(superfamilies, data_dir):
    all_representatives = None

    print(superfamilies)

    with h5pyd.File(data_dir, use_cache=False) as f:
        for superfamily in superfamilies:
            reps_ = f[superfamily.replace(".","/")]["representatives"]
            reps = []
            if len(reps_)>0:
                reps += list(reps_.keys())
            elif len(reps_.attrs.get("missing_domains", [])) > 0:
                reps += list(reps_.attrs["missing_domains"])
            reps = pd.DataFrame({"cathDomain":reps, "superfamily":superfamily})
            if len(reps)>0:
                if all_representatives is None:
                    all_representatives = reps
                else:
                    all_representatives = pd.concat((all_representatives, reps))
    all_representatives.reset_index()
    return all_representatives

class AllVsAll(object):
    NJOBS = 20
    NJOBS_CPU = None
    GPU = False
    METHOD = None
    MODEL_TYPE = None
    DATA_INPUT = None
    SCORE_METRIC = None
    SCORE_INCREASING = True
    SCORE_FN = None
    GPU_WAIT = True

    @classmethod
    def __init_subclass__(cls, *args, **kwds):
        global AVAILABLE_AVA_COMPARATORS
        AVAILABLE_AVA_COMPARATORS[cls.__name__.rsplit(".", 1)[-1]] = cls

    def __init__(self, superfamilies, data_dir, permutation_dir=None, work_dir=None, force=False, downsample_sbm=False, cluster=True):
        if work_dir is None:
            work_dir = os.getcwd()

        assert self.METHOD is not None, "Must set METHOD when subclassing"
        self.method = self.METHOD
        self.superfamilies = superfamilies
        self.data_dir = data_dir
        self.permutation_dir = permutation_dir
        self.work_dir = work_dir
        self.force = force
        self.downsample_sbm = downsample_sbm
        self.cluster = cluster
        self.clusterer = StochasticBlockModel

        self.id_parser = re.compile(r"cath\|4_3_0\|([a-zA-Z0-9]+)\/")

        self.representative_domains = self.get_representative_domains()
        self.superfamily_datasets = {}
        self.raw_sequences = {}

        if permutation_dir is not None:
            self.permuted_seqs = self.get_multiple_loop_permutations(permutation_dir)
        else:
            self.permuted_seqs = None

    def run(self):
        self.initialize_data()
        self.models = self.train_all()
        self.results = self.infer_all(self.models, self.combined_dataset)
        if isinstance(self.results, pd.DataFrame):
            self.results.to_hdf(f"{self.method}-results.h5", "table")

        #if self.permuted_seqs is not None:
        #    self.test_multiple_loop_permutations(self.permuted_seqs)

        if self.cluster:
            self.detect_communities()

    def initialize_data(self):
        self.create_superfamily_datasets()
        self.combined_dataset = self.create_combined_dataset()

    def completed_trained_model(self, model_name, model_input):
        return None

    def train_all(self):
        """Subclass this method to train all superfamilies in one model

        Returns
        ----------
        A path to the model or dictionary of model names and path each model
        """
        print("Train all, cores=", self.NJOBS)
        # done_sfams = {model_name:self.completed_trained_model(model_name, model_input) \
        #     for model_name, model_input in self.superfamily_datasets.items()}

        check_sfams = Parallel(n_jobs=self.NJOBS)(delayed(parallel_completed_trained_model)(\
            self, model_name, model_input) for model_name, model_input in \
            self.superfamily_datasets.items())
        done_sfams = dict(zip(self.superfamily_datasets.keys(), check_sfams))

        num_sfams = len(done_sfams)
        done_sfams = {model_name: model for model_name, model in done_sfams.items() \
            if model is not None}
        print(f"Training {num_sfams-len(done_sfams)}/{num_sfams}")
        train_sfams = ((superfamily, alignment) for superfamily, alignment in \
            self.superfamily_datasets.items() if superfamily not in done_sfams)
        new_sfams = dict(Parallel(n_jobs=self.NJOBS)(delayed(parallel_train)(self, i, superfamily, alignment) \
            for i, (superfamily, alignment) in enumerate(train_sfams)))
        done_sfams.update(new_sfams)


        return done_sfams

    def train(self, i, model_name, model_input):
        """Subclass this method to train an individual superfamilies. Modifications
        to object properties will not be saved.

        Returns
        ----------
        Thae name and path to the trained model
        """
        raise NotImplementedError

    def train_gpu(self, i, model_name, model_input, gpu=0):
        """Subclass this method to train an individual superfamilies on a specific
        GPU. Modifications to object properties will not be saved.

        Returns
        ----------
        Thae name and path to the trained model
        """
        raise NotImplementedError

    def completed_inference(self, model_name):
        return None

    def infer_all(self, models, combined_datatset):
        """Subclass this method to infer by all superfamilies in one model

        Returns
        ----------
        An d x s pd.DataFrame with the name of the cath domain as indices and
        superfamilies as columns
        """
        # done_sfams = {superfamily: self.completed_inference(superfamily) \
        #     for superfamily in self.superfamily_datasets.keys()}

        check_sfams = Parallel(n_jobs=self.NJOBS)(delayed(parallel_completed_inference)(\
            self, superfamily) for superfamily in self.superfamily_datasets.keys())
        done_sfams = dict(zip(self.superfamily_datasets.keys(), check_sfams))

        num_sfams = len(done_sfams)
        done_sfams = {model_name: infer_results for model_name, infer_results in \
            done_sfams.items() if infer_results is not None}

        print(f"Infering {num_sfams-len(done_sfams)}/{num_sfams}")

        infer_sfams = (superfamily for superfamily in self.superfamily_datasets.keys() \
            if superfamily not in done_sfams)

        new_sfams = dict(Parallel(n_jobs=self.NJOBS)(delayed(parallel_infer)(\
            self, i, superfamily, self.models[superfamily], combined_datatset) \
            for i, superfamily in enumerate(infer_sfams)))
        done_sfams.update(new_sfams)
        results = pd.concat(done_sfams.values(), axis=1)
        try:
            results.index.get_level_values('true_sfam')
        except KeyError:
            results_ = pd.merge(results.reset_index(), self.get_representative_domains(), on="cathDomain")
            assert len(results_)==len(results)
            results = results_.rename(columns={"superfamily":"true_sfam"}).set_index(["cathDomain", "true_sfam"])
        print(results)

        if results.dropna().shape[0] != results.shape[0]:
            import pdb; pdb.set_trace()

        results.to_hdf(f"{self.METHOD}-{self.DATA_INPUT}-all_vs_all.hdf", "table")
        return results


        results = None
        for i, (model_name, model_path) in enumerate(models.items()):
            output = self.infer(i, model_name, model_path, combined_datatset)
            if results is None:
                result = outputs
            else:
                results = pd.merge(results, output, left_index=True, right_index=True)


        return results

    def infer(self, i, model_name, model_path, combined_datatset):
        """Subclass this method to infer by individual superfamily

        Returns
        ----------
        A pd.Series with indices of (cathDomain, true_sfam)s an similarity scores as values
        """
        raise NotImplementedError

    def infer_gpu(self, i, model_name, model_path, combined_datatset, gpu=0):
        """Subclass this method to infer by individual superfamily on a specific GPU

        Returns
        ----------
        A pd.Series with indices of (cathDomain, true_sfam)s an similarity scores as values
        """
        raise NotImplementedError

    def create_superfamily_datasets(self):
        raise NotImplementedError

    def create_combined_dataset(self):
        raise NotImplementedError


    def get_representative_domains(self):
        return get_representative_domains(self.superfamilies, self.data_dir)

        if superfamilies is None:
            superfamlies = self.superfamilies

        all_representatives = None

        print(self.superfamilies)

        with h5pyd.File(self.data_dir, use_cache=False) as f:
            for superfamily in self.superfamilies:
                reps_ = f[superfamily.replace(".","/")]["representatives"]
                reps = []
                if len(reps_)>0:
                    reps += list(reps_.keys())
                elif len(reps_.attrs.get("missing_domains", [])) > 0:
                    reps += list(reps_.attrs["missing_domains"])
                reps = pd.DataFrame({"cathDomain":reps, "superfamily":superfamily})
                if len(reps)>0:
                    if all_representatives is None:
                        all_representatives = reps
                    else:
                        all_representatives = pd.concat((all_representatives, reps))
        all_representatives.reset_index()
        return all_representatives

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

    def detect_communities(self):
        clusterer = self.clusterer(self.results, self.METHOD, self.MODEL_TYPE, self.DATA_INPUT,
            self.SCORE_METRIC, increasing=self.SCORE_INCREASING)
        clusterer.find_communities(score_type=self.SCORE_FN, downsample=self.downsample_sbm, old_flare=getattr(self, 'old_flare', None))
        with open(os.path.join(self.work_dir, f"{self.METHOD}-{self.DATA_INPUT}.tex"), "w") as f:
            self.comparasion_table_row = clusterer.make_comparison_table_row()
            print(self.comparasion_table_row, file=f)

    @staticmethod
    def add_argpase_options(parserr):
        parser.add_argument("-d", "--data_dir", default="/home/ed4bu/deepurfold-paper-2.h5", required=False)
        parser.add_argument("superfamily", nargs="+")
        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("--no-cluster", default=False, action="store_true")
    parser.add_argument("superfamily", nargs="+")
    args = parser.parse_args()
    HMMAllVsAll(args.superfamily, args.data_dir, representatives=args.representatives, cluster=not args.no_claster)
