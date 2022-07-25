import os
import argparse
import glob
from datetime import datetime

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
#setattr(WandbLogger, 'name', property(lambda self: self._name))

from scipy.stats import special_ortho_group

from DeepUrfold.DataModules.DomainStructureDataModule import DomainStructureDataModule
from DeepUrfold.Models.DomainStructureSegmentationModel import DomainStructureSegmentationModel

class StructureRotater(Callback):
    def __init__(self, dm):
        self.dm = dm
        self.rvs = None

    # def on_train_start(self, trainer, pl_module):
    #     self.rvs = special_ortho_group.rvs(3)
    #     self.dm.train_dataset.set_rotation_matrix(self.rvs)
    #
    # def on_validation_start(self, trainer, pl_module):
    #     self.dm.valid_dataset.set_rotation_matrix(self.rvs)
    #
    # def on_test_start(self, trainer, pl_module):
    #     self.dm.test_dataset.set_rotation_matrix(self.rvs)

    def on_train_epoch_start(self, trainer, pl_module):
        self.dm.train_dataset.set_rotation_matrix(self.rvs)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.dm.valid_dataset.set_rotation_matrix(self.rvs)

    def on_test_epoch_start(self, trainer, pl_module):
        self.dm.test_dataset.set_rotation_matrix(self.rvs)

class DomainStructureTrainer(object):
    DATAMODULE = DomainStructureDataModule
    MODEL = DomainStructureSegmentationModel

    def __init__(self, args=None, prefix=None, wandb_logger=True, checkpoint=True):
        assert self.DATAMODULE is not None, "Must subclass and set DATASET_CLS"
        assert self.MODEL is not None, "Must subclass and set MODEL"

        self.prefix = "{}-{}-".format(self.DATAMODULE.DATASET.__name__.split(".")[-1],
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S')) if prefix is None else prefix

        self.parser = argparse.ArgumentParser()
        self.parser = self.MODEL.add_model_specific_args(self.parser)
        self.parser = self.DATAMODULE.add_data_specific_args(self.parser)
        self.parser = self.__add_trainer_args()

        self.parser.set_defaults(prefix=self.prefix)

        #Create Hyperparameters
        self.hparams = self.parser.parse_args(args) #W&B Sweep has precoded path, but we ignore it with known args. Orig: parse_args(args)

        if self.hparams.prefix:
            self.prefix += self.hparams.prefix

        if self.hparams.distributed_backend == "horovod":
            self.hparams.gpus = 1
            self.hparams.num_nodes = 1
        elif self.hparams.gpus == 1 and self.hparams.distributed_backend is not None:
            self.hparams.distributed_backend = None

        if isinstance(self.hparams.gpus, str) and "," in self.hparams.gpus:
            try:
                self.hparams.gpus = list(map(int, self.hparams.gpus.rstrip(',').split(',')))
            except ValueError:
                raise ValueError("Invalid GPU type")

        if (isinstance(self.hparams.gpus, int) and isinstance(self.hparams.gpus, int) > 0) or \
          (isinstance(self.hparams.gpus, (list,tuple)) and len(self.hparams.gpus) > 0):
            torch.cuda.empty_cache()

        if self.hparams.data_dir is None:
            if "Distributed" in self.DATAMODULE.__name__:
                raise RuntimeError("Set data_dir to HSDS h5 file")
            elif not os.path.isdir(self.hparams.data_dir):
                if os.path.isdir("./data_eppic_cath_features"):
                    self.hparams.data_dir = "./data_eppic_cath_features"
                elif os.path.isdir("./data-eppic-cath-features"):
                    self.hparams.data_dir = "./data-eppic-cath-features"
                else:
                    raise RuntimeError(f"Cannot find data_dir '{self.hparams.data_dir}'")
            #self.hparams.data_dir = "/project/ppi_workspace/data_eppic_cath_features"

        #Create Data Module
        self.data_module = self.DATAMODULE(self.hparams)


        kwds = {}

        args_copy = vars(self.hparams)
        args_copy = {k:frozenset(v) if isinstance(v, list) else v for k, v in args_copy.items()}

        #Initialize w&b first to allow it to change parameters.
        if not self.hparams.no_wandb:
            kwds["logger"] = WandbLogger(name=self.prefix, project=self.prefix, allow_val_change=True)

        new_args = vars(self.hparams)
        new_args = {k:frozenset(v) if isinstance(v, list) else v for k, v in new_args.items()}

        #Create model
        # self.model = self.MODEL(self.hparams)
        self.last_checkpoint_file = None
        if self.hparams.reload_checkpoint:
            last_checkpoint_file = self.hparams.last_checkpoint_file if self.hparams.last_checkpoint_file is not None \
                else os.path.join(os.path.getcwd(), f"*{self.prefix}*last.ckpt")
            last_checkpoint_file = list(glob.glob(last_checkpoint_file))
            if self.hparams.reload_checkpoint and len(last_checkpoint_file) > 0:
                #Reload checkpoint file to load previous model
                self.model = self.MODEL.load_from_checkpoint(last_checkpoint_file[0])
                self.last_checkpoint_file = last_checkpoint_file[0]
                self.hparams.resume_from_checkpoint = last_checkpoint_file[0]
            else:
                #It not found, create new model
                self.model = self.MODEL(self.hparams)
        else:
            #Create nw model
            self.model = self.MODEL(self.hparams)

        kwds["callbacks"] = [StructureRotater(self.data_module)]

        if not self.hparams.no_early_stopping:
            kwds["callbacks"].append(EarlyStopping(monitor='val_loss', patience=8))

        if checkpoint:
            self.model_checkpoint = ModelCheckpoint(
                dirpath=os.getcwd(),
                filename=self.prefix+'_{epoch:02d}-{val_loss:.2f}',
                save_last=True,
                save_top_k=1,
                monitor='val_loss'
            )
            kwds["callbacks"].append(self.model_checkpoint)

        # if hasattr(self.hparams, "resume_from_checkpoint") and isinstance(self.hparams.resume_from_checkpoint, str) and os.path.exists(self.hparams.resume_from_checkpoint):
        #     kwds["ckpt_path"] = self.hparams.resume_from_checkpoint
        #     self.hparams.resume_from_checkpoint = None
        #     #print(kwds["checkpoint_callback"])

        #Create trainier and train model
        self.trainer = Trainer.from_argparse_args(self.hparams, **kwds)

        if self.last_checkpoint_file is not None: #hasattr(self.hparams, "resume_from_checkpoint") and os.path.exists(self.hparams.resume_from_checkpoint):
            print("Resuming from", self.hparams.resume_from_checkpoint)
            print("At epoch", self.trainer.current_epoch)
            print("At loop epoch", self.trainer.fit_loop.epoch_progress.current.completed)

            model = torch.load(self.last_checkpoint_file, map_location="cpu")
            last_epoch = model["epoch"]
            del model

            self.trainer.fit_loop.epoch_progress.current.completed = last_epoch

            print("Resuming from", self.hparams.resume_from_checkpoint)
            print("At epoch", self.trainer.current_epoch)
            print("At loop epoch", self.trainer.fit_loop.epoch_progress.current.completed)


    def fit(self):
        self.trainer.fit(self.model, self.data_module)

    def __add_trainer_args(self):
        self.parser = Trainer.add_argparse_args(self.parser)

        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            distributed_backend = "ddp"
            if num_nodes>1:
                batch_size = 1 #self.parser.get_default('batch_size')/2
            else:
                batch_size = 1 #self.parser.get_default('batch_size')
        else:
            distributed_backend = None
            batch_size = self.parser.get_default('batch_size')

        self.parser.set_defaults(
            num_nodes=num_nodes,
            gpus=num_gpus,
            nb_sanity_val_steps=0,
            val_percent_check=0.1,
            test_percent_check=0.2,
            early_stop_callback=False,
            num_sanity_val_steps=0,
            distributed_backend=distributed_backend,
            batch_size=batch_size,
            max_epochs=100
        )

        self.parser.add_argument("--reload_checkpoint", action="store_true", default=False)
        self.parser.add_argument("--last_checkpoint_file", default=None)
        self.parser.add_argument("--no_early_stopping", action="store_true", default=False)
        self.parser.add_argument("--no_wandb", default=False, action="store_true")

        return self.parser
