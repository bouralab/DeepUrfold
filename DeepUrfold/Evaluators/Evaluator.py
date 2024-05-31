import os
import sys
import re
import argparse
import glob
import subprocess
from functools import partial
from datetime import datetime

import torch
import numpy as np

import MinkowskiEngine as ME
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

from DeepUrfold.Metrics.ClassifactionMetricSaver import ClassifactionMetricSaver
from DeepUrfold.Metrics.RawMetricSaver import RawMetricSaver
from DeepUrfold.Metrics.LRPSaver import LRPSaver
from DeepUrfold.Metrics.relevance import AtomicRelevance
from DeepUrfold.DataModules.DomainStructureDataModule import DomainStructureDataModule
from DeepUrfold.Models.DomainStructureSegmentationModel import DomainStructureSegmentationModel
from DeepUrfold.util import str2bool, str2boolorlist, str2boolorval

def simulate_args_from_namespace(n, parser):
    """ check an argparse namespace against a module's get_args method.
    Ideally, there would be something built in to argparse, but no such luck.
    This tries to reconstruct the arg list that argparse.parse_args would expect
    """
    arg_list = [[k, v] for k, v in sorted(vars(n).items())]
    argparse_formatted_list = []
    for l in arg_list:
        ####  deal with flag arguments (store true/false)
        # if l[1] == True:
        #     argparse_formatted_list.append("--{}".format(l[0]))
        # elif l[1] == False or l[1] is None:
        #     pass  # dont add this arg
        # # add positional argments
        # elif l[0] in positional:
        #     argparse_formatted_list.append(str(l[0]))
        # # add the named arguments
        # else:
        if  l[1] == False or l[1] is None:
            continue

        default = parser.get_default(l[0])
        if default is None or l[1] != default:
            argparse_formatted_list.append("--{}".format(l[0]))
            argparse_formatted_list.append(str(l[1]))
    #print(argparse_formatted_list)
    return argparse_formatted_list


class DomainStructureEvaluator(object):
    DATAMODULE = DomainStructureDataModule
    MODEL = DomainStructureSegmentationModel
    LRP_VARS = {"data":0}

    def __init__(self, args=None, prefix=None):
        assert self.DATAMODULE is not None, "Must subclass and set DATASET_CLS"
        assert self.MODEL is not None, "Must subclass and set MODEL"

        self.parser = argparse.ArgumentParser()
        checkpoint = self.parser.add_mutually_exclusive_group(required=True)
        checkpoint.add_argument("--checkpoint")
        checkpoint.add_argument("--model_dir")


        self.parser = self.MODEL.add_model_specific_args(self.parser)
        self.parser = self.DATAMODULE.add_data_specific_args(self.parser, eval=True)
        self.parser = self.__add_evaluator_args()

        self.parser.set_defaults(prefix=None)

        #Create Hyperparameters
        self.hparams = self.parser.parse_args(args)

        # if self.hparams.distributed_backend == "horovod":
        #     self.hparams.gpus = 1
        #     self.hparams.num_nodes = 1
        # elif self.hparams.gpus == 1 and self.hparams.distributed_backend is not None:
        #     self.hparams.distributed_backend = None

        if isinstance(self.hparams.gpus, str) and "," in self.hparams.gpus:
            try:
                self.hparams.gpus = list(map(int, self.hparams.gpus.rstrip(',').split(',')))
            except ValueError:
                raise ValueError("Invalid GPU type")

        self.hparams.max_epochs = 1

        print(self.hparams.prefix)

        self.prefix = "{}-{}-".format(self.DATAMODULE.DATASET.__name__.split(".")[-1],
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) if self.hparams.prefix is None else self.hparams.prefix
        print("Eval prefix", self.prefix)

        if self.hparams.checkpoint is not None:
            self.run(checkpoint=self.hparams.checkpoint, prefix=self.hparams.prefix)
        elif self.hparams.model_dir is not None:
            if self.hparams.cross_model:
                self.hparams.representatives = True
                self.run_all_vs_all()
            else:
                print("Run all")
                self.hparams.representatives = True
                self.run_all()
        else:
            raise RuntimeError("checkpoint or model_dir must not be None")

    def run(self, checkpoint, prefix=None):
        prefix = prefix if prefix is not None else self.hparams.prefix

        #if self.hparams.gpus > 0:
        #    torch.cuda.empty_cache()

        #Create Data Module
        self.data_module = self.DATAMODULE(self.hparams, eval=True)

        #Update hparams from data module
        #self.hparams = self.data_module.hparams
        #print(self.hparams)

        #self.model.freeze()
        if self.hparams.batch_size != self.data_module.h_params.batch_size:
            self.hparams.batch_size = self.data_module.h_params.batch_size

        self.hparams.samplewise_loss = not self.hparams.batchwise_loss

        self.atomic_relevance = None
        if self.hparams.lrp:
            self.hparams.return_latent = True
            self.hparams.return_var = True
            self.hparams.raw = self.MODEL.LATENT_VARS
            print("Running LRP with aggregators", self.hparams.atomic_relevance)
            self.atomic_relevance = AtomicRelevance(
                self.hparams.features,
                self.hparams.data_dir, #*self.hparams.superfamily.split(".")),
                self.hparams.atomic_relevance[0],
                self.hparams.atomic_relevance[1],
                volume=self.hparams.input_size)
        elif isinstance(self.hparams.raw, bool) and self.hparams.raw:
            if self.hparams.return_latent:
                self.hparams.raw = self.model.LATENT_VARS #FIXME: hardocded vlaues for VAE...
            elif self.hparams.return_reconstruction:
                self.hparams.raw = ["reconstruction"]

        print("Raw saver?", self.hparams.raw)
        if self.hparams.classification:
            print("Running Classification Metrics")
            self.hparams.raw = ["reconstruction"]
            self.metric_saver = ClassifactionMetricSaver(
                self.hparams,
                prefix=prefix,
                num_classes=len(self.hparams.features),
                per_sample=self.hparams.samplewise_loss)
        elif isinstance(self.hparams.lrp, bool) and self.hparams.lrp:
            self.metric_saver = LRPSaver(
                self.hparams,
                self.data_module,
                all_metrics=self.hparams.raw,
                lrp_metric=self.LRP_VARS,
                atomic_relevance=self.atomic_relevance,
                prefix=prefix,
                per_sample=self.hparams.samplewise_loss)
        elif isinstance(self.hparams.raw, bool) and self.hparams.raw or isinstance(self.hparams.raw, (list, tuple)):
            self.metric_saver = RawMetricSaver(self.hparams, self.data_module,
                metrics=self.hparams.raw,
                prefix=prefix,
                per_sample=self.hparams.samplewise_loss,
                save_labels=True)
        else:
            self.metric_saver = None

        if self.metric_saver is not None:
            kwds = {"callbacks": [self.metric_saver]}
        else:
            kwds = {}

        #assert len(kwds) > 0, kwds

        #self.metric_savers = {metric:ClassifactionMetricSaver(self.hparams, num_classes=len(self.hparams.features)) for metric in self.hparams.metric}
        #kwds = {"callbacks": list(self.metric_savers.values())}

        #if self.hparams.viz:
            #self.model.hparams.sample_during_eval = self.hparams.sample_during_eval
            #self.model.hparams.return_latent = True
            # self.metric_saver = MetricSaver(self.data_module, metric="Z", prefix=self.prefix)
            # kwds = {"callbacks": [self.metric_saver]}
        # elif self.hparams.representatives:
        #     kwds = {"callbacks": [ELBOSaver(self.data_module, prefix=self.prefix)]}
        # else:
        #     kwds = {}

        #Create model and update parameters
        model_params = vars(self.hparams)
        print(model_params)
        model_params.pop("checkpoint", None)
        self.model = self.MODEL.load_from_checkpoint(checkpoint, **model_params)

        if self.hparams.summary:
            print(self.model)
            sys.exit()

        if isinstance(self.hparams.lrp, bool) and self.hparams.lrp:
            self.metric_saver.set_model(self.lrp_model())

        #model_params.update(kwds)
        #self.hparams.update(self.model.hparams)
        #self.data_module.hparams.update(self.model.hparams)
        #self.data_module.hparams_initial = self.hparams
        #Create trainier and train mode

        # datamodule_hparams = self.data_module.hparams_initial
        # lightning_hparams = self.hparams
        # inconsistent_keys = []
        # for key in lightning_hparams.keys() & datamodule_hparams.keys():
        #     lm_val, dm_val = lightning_hparams[key], datamodule_hparams[key]
        #     if type(lm_val) != type(dm_val):
        #         print("inconsistent_keys", key, lm_val, dm_val)
        #     elif isinstance(lm_val, torch.Tensor) and id(lm_val) != id(dm_val):
        #         print("inconsistent_keys", key, lm_val, dm_val)
        #     elif lm_val != dm_val:
        #         print("inconsistent_keys", key, lm_val, dm_val)

        self.trainer = Trainer.from_argparse_args(self.hparams, gpus=self.hparams.gpus, **kwds)


    def test(self):
        self.trainer.test(self.model, datamodule=self.data_module)
        self.metric_saver.process(no_compute=self.hparams.no_compute)

    def lrp_model(self):
        """Specify which model LRP should use and which variable(s) to save.
        Overwrite this method to change"""
        return self, ["data"]

    def viz(self, metric=None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style='white', rc={'figure.figsize':(14,10)})

        if metric is None:
            metric = list(self.matric_savers.keys())[0]

        if metric == "elbo":
            self.test()
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(self.matric_savers[metric].get_numpy())
            x, y = embedding[:, 0], embedding[:, 1]
        elif metric == "roc":
            pass

        plt.scatter(embedding[:, 0], embedding[:, 1])
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the Penguin dataset', fontsize=24)

        plt.savefig(f'{self.prefix}-umap.pdf')

    def __add_evaluator_args(self):
        self.parser = Trainer.add_argparse_args(self.parser)

        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
        num_gpus = torch.cuda.device_count()
        self.parser.add_argument("--summary", type=str2bool, nargs='?',
                           const=True, default=False)
        self.parser.add_argument("--viz", type=str2bool, nargs='?',
                           const=True, default=False)
        #self.parser.add_argument("--metric", default="classification")
        self.parser.add_argument("--batchwise_loss", type=str2bool, nargs='?',
                           const=True, default=False)
        self.parser.add_argument("--no_compute", type=str2bool, nargs='?',
                           const=True, default=False)

        self.parser.add_argument("--cross_model", type=str2bool, nargs='?',
                           const=True, default=False)
        self.parser.add_argument("--gt0", type=str2bool, nargs='?',
                           const=True, default=False)
        
        self.parser.add_argument("--force_rotate", type=str2bool, nargs='?',
                           const=True, default=False)
        self.parser.add_argument("--n_copies_per_domain", type=int, default=1)

        metrics = self.parser.add_mutually_exclusive_group(required=True)
        metrics.add_argument("--classification", type=str2bool, nargs='?',
                           const=True, default=False)
        metrics.add_argument("--lrp", type=partial(str2boolorval, default="GRule"), nargs='?',
                           const=True, default=False)
        metrics.add_argument("--raw", type=str2boolorlist, nargs='*',
                           default=False)
        metrics.add_argument("--visualize_tensors", type=str2bool, nargs='?',
                           const=True, default=False)

        self.parser.add_argument("--atomic_relevance", default=["arithmetic_mean", "sum"], nargs=2)

        self.parser.set_defaults(
            num_nodes=num_nodes,
            gpus=num_gpus,
            nb_sanity_val_steps=0,
            val_percent_check=0,
            test_percent_check=1,
            early_stop_callback=False,
            num_sanity_val_steps=0,
            #distributed_backend=None, #"ddp" if num_gpus>0 else None,
            batch_size=32,
            max_epochs=1
        )

        return self.parser

    def run_all(self, all_vs_all=False):
        original_args = sys.argv[:]

        #DomainStructureDataset-2021-01-08_21-20-56-3.40.50.300-last.ckpt
        file_re = re.compile(r"DomainStructureDataset-.+-(?P<superfamily>\d+\.\d+\.\d+\.\d+)-last.ckpt")

        if "*" in self.hparams.model_dir:
            if not self.hparams.model_dir.endswith(".ckpt"):
                #Assume direcotry structure is complete, otherwise user would pass it
                model_dir = os.path.join(self.hparams.model_dir, "*.ckpt")
            else:
                model_dir = self.hparams.model_dir
        else:
            #An actual direcotry
            model_dir = os.path.join(self.hparams.model_dir, "*.ckpt")

        self.hparams.model_dir = None

        model_files = list(glob.glob(model_dir))
        print(model_dir)
        for model_file in model_files: #glob.iglob(model_dir): #os.listdir(self.hparams.model_dir):
            print(model_file)

            m = file_re.match(os.path.basename(model_file))
            if m:
                model_sfam = m.group("superfamily")
                #print("Running", model_sfam)


                #Reset model_file and prefix since pytorch_lightning will restart the program with those params if ddp
                self.hparams.checkpoint = model_file

                input_files = model_files if all_vs_all else [model_file]

                for input_file in input_files:
                    m2 = file_re.match(os.path.basename(model_file))
                    if m2:
                        input_sfam = m2.group("superfamily")
                        print("Running", model_sfam, "with input", input_sfam)
                        self.hparams.prefix = f'model={model_sfam}_input={input_sfam}'
                        self.hparams.superfamily = input_sfam

                        command = [sys.executable]+original_args[:1]+simulate_args_from_namespace(self.hparams, self.parser)
                        print(command)
                        subprocess.call(command )
                        # self.run(model_file, prefix=f'model={sfam}_input={sfam}')
                        # try:
                        #     self.test()
                        # except Exception as e:
                        #     raise
                        #
                        # self.metric_saver.reset()
                        #
                        # del self.data_module, self.model, self.metric_saver, self.trainer
                        # torch.cuda.empty_cache()

                #

    def run_all_vs_all(self):
        models = {}

        file_re = re.compile(r"DomainStructureDataset-.+-(?P<superfamily>\d+\.\d+\.\d+\.\d+)_epoch=(?P<epoch>\d+)-val_loss=(?P<val_loss>\d*\.\d+)\.ckpt")
        for model_file in os.listdir(self.hparams.model_dir):
            m = file_re.match(model_file)
            if m:
                sfam = m.group("superfamily")
                file_info = {"file":os.path.join(self.hparams.model_dir, model_file),
                    "epoch": int(m.group_dict["epoch"]), "val_loss":float(m.group_dict["epoch"])}

                try:
                    if models[superfamily]["val_loss"] < file_info["val_loss"]:
                        models[superfamily] = file_info
                except KeyError:
                    models[superfamily] = file_info

        superfamilies = models.keys()
        for model in models:
            for sfam in superfamilies:
                self.hparams.superfamily = sfam
                self.run(model["file"], prefix=f'model={model["superfamily"]}_input={sfam}')

    def run_lrp(self):
        self.model.hparams.return_latent = True
        self.model.hparams.return_var = True
        self.hparams.metric = "raw"
        self.innvestigator = InnvestigateModel(self.model.encoder)
        #self.run(checkpoint, prefix=prefix)
        for (mean, log_var) in zip(self.metric.metric_values["means"], self.metric.metric_values["log_var"]):
            mean = ME.SparseTensor(torch.unsqueeze(mean, 0), torch.zeros(1,3))
            log_var = ME.SparseTensor(torch.unsqueeze(log_var, 0), torch.zeros(1,3).int())
            _, relevance = self.innvestigator.innvestigate((mean, log_var), autoencoder_in=True, autoencoder_out=False, no_recalc=True)

        """
        import torch
        means = torch.load("model=2.60.40.10_input=2.60.40.10-means-data.pt")
        log_var = torch.load("model=2.60.40.10_input=2.60.40.10-log_var-data.pt")
        from DeepUrfold.Models.DomainStructureVAE import DomainStructureVAE
        model = DomainStructureVAE.load_from_checkpoint("/project/ppi_workspace/urfold_compare/nussinov_superfams/DomainStructureDataset-2021-01-14_08-11-41-2.60.40.10-last.ckpt")
        from pytorch_lrp.innvestigator import InnvestigateModel
        from pytorch_lrp.minkowski_lrp import *
        import MinkowskiEngine as ME
        mean1 = ME.SparseTensor(torch.unsqueeze(means[0], 0), torch.zeros(1,3).int())
        var1 = ME.SparseTensor(torch.unsqueeze(log_var[0], 0), torch.zeros(1,3).int())
        innvestigator = InnvestigateModel(model.encoder)
        innvestigator.innvestigate((mean1, var1), autoencoder_in=True, autoencoder_out=False, no_recalc=True)
"""
