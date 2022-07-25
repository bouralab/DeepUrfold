import os
import argparse
from math import ceil
from functools import partial
import subprocess

import torch
import pandas as pd
import pytorch_lightning as pl
import MinkowskiEngine as ME
from torch.utils.data import DataLoader

from DeepUrfold.Datasets.DomainStructureDataset import DomainStructureDataset
from DeepUrfold.Datasets.DistributedDomainStructureDataset import DistributedDomainStructureDataset

from DeepUrfold.util import str2bool
from DeepUrfold.Models.van_der_waals_surface import search_algorithms as space_fill_algorithms

from molmimic.generate_data.get_cath_representatives import get_representatives

default_atom_features = "H;HD;HS;C;A;N;NA;NS;OA;OS;SA;S;Unk_atom__is_helix;is_sheet;Unk_SS__residue_buried__is_hydrophobic__pos_charge__is_electronegative"
default_feature_groups = "Atom Type;Secondary Structure;Solvent Accessibility;Hydrophobicity;Charge;Electrostatics"


#from molmimic.common.features import atom_features
# atom_features = [
#      'C', 'CA', 'N', 'O', 'OH', 'C_elem', 'N_elem', 'O_elem', 'S_elem',
#       'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
#       'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR',
#       'is_helix', 'is_sheet', 'residue_buried', 'residue_exposed','atom_is_buried',
#       'atom_exposed',  'is_hydrophbic', 'pos_charge',  'is_electronegative'
# ]

atom_features = [
      'C', 'A', 'N', 'OA', 'OS', 'C_elem', 'N_elem', 'O_elem', 'S_elem',
      'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
      'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR',
      'is_helix', 'is_sheet', 'residue_buried', 'is_hydrophobic', 'pos_charge',
      'is_electronegative'
]

def num_or_all(v):
    import argparse
    print(num_or_all, v, type(v))
    if isinstance(v, bool):
       return v
    elif isinstance(v, str):
        return v[0].lower()=="t"

    try:
        return int(v)
    except:
        raise argparse.ArgumentTypeError('Int value expected.')

def multigpu_sparse_collate(data, num_gpus=1):
    """Make sure coords is not a tensor so they don;t get put on GPU
    """
    minibatch_len = int(ceil(len(data)/num_gpus))
    batches = list(zip(*(ME.utils.batch_sparse_collate(data[i:i+minibatch_len]) \
        for i in range(0, len(data), minibatch_len))))
    del data

    if num_gpus == 1:
        #Make sure batch is a list of minibatches
        batches = [batches]

    return batches

def ddp_sparse_collate(data):
    print("DDP DATA: {}".format(data))
    out = ME.utils.batch_sparse_collate(data)
    print("DDP COLLATE: {}".format(out))
    return out

def collate_coordinates(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    lengths = torch.tensor([t[0].shape[0] for t in batch])

    ## padd
    combined = list(zip(*batch))
    coords_, feats = combined[:2]

    coords = torch.nn.utils.rnn.pad_sequence(
        coords_,
        batch_first=True,
        padding_value=float("nan"))
    feats = torch.nn.utils.rnn.pad_sequence(
        feats,
        batch_first=True,
        padding_value=float("nan"))

    if len(combined)>2:
        # labels = torch.nn.utils.rnn.pad_sequence(
        #     combined[2],
        #     batch_first=True,
        #     padding_value=float("nan"))
        labels = ME.utils.sparse_collate(*zip(*combined[2]))
        if len(combined)>3:
            voxel_map = combined[3]

            return coords, feats, labels, voxel_map
        else:
            return coords, feats, labels, lengths

    else:
        labels = None
        return coords, feats, labels, lengths

def batch_sparse_collate(data):
    return ME.utils.batch_sparse_collate(data)

class MinkowskiDataLoader(DataLoader):
    def __init__(self, *args, gpus=0, distributed_backend=None, **kwds):
        collate_fn = ME.utils.batch_sparse_collate
        if isinstance(gpus, (list, tuple)):
            self.gpus = len(gpus)
        elif isinstance(gpus, int):
            self.gpus = gpus
        else:
            self.gpus = 0

        if self.gpus > 1 and distributed_backend in ["dp", "ddp2"]:
            collate_fn = partial(multigpu_sparse_collate, num_gpus=self.gpus)

            #Adjust batch size
            #if self.gpus > 0 and distributed_backend in ["dp", "ddp2"]:
            kwds["batch_size"] = kwds.get("batch_size", 1) * self.gpus

        kwds["collate_fn"] = collate_fn

        super().__init__(*args, **kwds)

    def __iter__(self):
        it = super().__iter__()
        for minibatches in it:
            yield minibatches
        it._shutdown_workers()

    # def __iter__(self):
    #     print("START: {}".format(subprocess.check_output(["free", "-h"])))
    #     it = super().__iter__()
    #     for minibatches in it:
    #         inputs, labels = [], []
    #         if self.gpus <= 1:
    #             minibatches = [minibatches]
    #
    #         coords, inputs, labels = zip(*minibatches)
    #
    #         del minibatches
    #
    #         # for batch in minibatches:
    #         #     coords, feats, _labels = batch
    #         #     input = ME.SparseTensor(feats.float(), coords.int())
    #         #     # _labels = ME.SparseTensor(
    #         #     #     # coords=coords,  not required
    #         #     #     feats=_labels,
    #         #     #     coords_manager=input.coords_man,  # must share the same coordinate manager
    #         #     #     coords_key=input.coords_key  # For inplace, must share the same coords key
    #         #     # )
    #         #     #print("MINIBATCH SIZE IS {} {}".format(input.F.size(), input.coords.size()))
    #         #     inputs.append(input)
    #         #     labels.append(_labels)
    #
    #         print("DATA: {}".format(subprocess.check_output(["free", "-h"])))
    #
    #         if self.gpus <= 1:
    #             yield coords[0], inputs[0], labels[0]
    #         else:
    #             yield coords, inputs, labels

class DistributedDomainStructureDataModule(pl.LightningDataModule):
    DATASET = DistributedDomainStructureDataset

    def __init__(self, h_params, eval=False):
        assert self.DATASET is not None, "Must subclass and set DATASET"
        super().__init__()

        self.h_params = h_params
        self.data_dir = self.h_params.data_dir
        self.train_dir = os.path.join(self.data_dir, "train_files")

        if isinstance(self.h_params.features, (tuple, list)) and \
          len(self.h_params.features)==1 and \
          isinstance(self.h_params.features[0], str):
            self.h_params.features = self.h_params.features[0] #.replace(";", " ").split()
        elif isinstance(self.h_params.features, (tuple, list)) and \
          len(self.h_params.features)>1:
            pass
        elif not isinstance(self.h_params.features, str):
            raise RuntimeError("Invlaid feature type")
            #self.h_params.features = self.h_params.features #.replace(";", " ").split()

        self.h_params.original_features = self.h_params.features

        if isinstance(self.h_params.features, str):
            self.h_params.features_original = self.h_params.features
            self.h_params.features = self.h_params.features.replace("__", " ").replace(";", " ").split()

        if isinstance(self.h_params.features_to_drop, (list, tuple)) and \
          len(self.h_params.features_to_drop) > 0:
            self.h_params.features = [feature for feature in self.h_params.features if \
                feature not in self.h_params.features_to_drop]

        if self.h_params.nClasses is None or self.h_params.autoencoder:
            self.h_params.nClasses = len(self.h_params.features)
        else:
            try:
                self.h_params.nClasses = int(self.h_params.nClasses)
            except ValueError:
                if self.h_params.nClasses in ["bool", "bool1"]:
                    self.h_params.nClasses = 1
                elif self.h_params.nClasses == "bool2":
                    self.h_params.nClasses = 2
                elif self.h_params.label_type == "sfam":
                    self.h_params.nClasses = 6119
                else:
                    self.h_params.nClasses = 1

        cluster_levels = self.DATASET.hierarchy

        if self.h_params.superfamily is not None:
            if isinstance(self.h_params.superfamily, str):
                self.h_params.superfamily = [self.h_params.superfamily]

            self.train_keys = [superfamily.replace("/",".").split(".") \
                for superfamily in self.h_params.superfamily]

            #cluster_levels = cluster_levels[4:]
            if self.h_params.cluster_level not in cluster_levels[4:]:
                self.h_params.cluster_level = "S35"
        else:
            #Needs to be a list to pass into dataset
            self.h_params.superfamily = [None]

        if self.h_params.over_sample:
            self.h_params.balance = "oversample"
        elif self.h_params.under_sample:
            self.h_params.balance = "undersample"
        else:
            self.h_params.balance = None

        self.cluster_level = ".".join(cluster_levels[:cluster_levels.index(self.h_params.cluster_level)+1])

        prefix = self.DATASET.__name__.split(".")[-1]

        #if not eval and len(self.h_params.superfamily) > 1:
        #    raise RuntimeError("Multiple superfamily training not supported yet.")

            #self.train_dir = self.train_dir[0]

        # if self.h_params.representatives:
        #     self.train_file = ""
        #     self.valid_file = ""
        #     test_files = [os.path.join(train_dir, f"{prefix}-representatives.h5") \
        #          for train_dir in self.train_dir]
        #     if len(test_files) == 1:
        #         self.test_file = test_files[0]
        #     else:
        #         self.test_file = None
        #         if self.h_params.superfamily is not None:
        #             for sfam, test_file in zip(self.h_params.superfamily, test_files):
        #                 if not os.path.isfile(test_file):
        #                     get_representatives(sfam, self.data_dir)
        #                 test_file = pd.read_hdf(test_file, "table")
        #                 test_file = test_file.assign(superfamily=sfam)
        #                 if self.test_file is None:
        #                     self.test_file = test_file
        #                 else:
        #                     self.test_file = pd.concat((self.test_file, test_file))
        #             if self.test_file is None:
        #                 raise RuntimeError(f"Unable to load or create full train files: {full_files}")
        #         else:
        #             raise RuntimeError("Unable to find or create CATH representatives")
        #         print(self.test_file.columns)
        # elif self.h_params.domains is not None or self.h_params.all_domains is not None:
        #     self.train_file = ""
        #     self.valid_file = ""
        #     full_files = [os.path.join(train_dir, f"{prefix}-full-train.h5") \
        #         for train_dir in self.train_dir]
        #     if len(full_files) == 1:
        #         self.test_file = full_files[0]
        #     else:
        #         self.test_file = None
        #         for sfam, full_file in zip(self.h_params.superfamily, full_files):
        #             if not os.path.isfile(full_file):
        #                 raise RuntimeError(f"Unable to find or create full train file: {full_file}")
        #
        #             test_file = pd.read_hdf(full_file, "table")
        #             if self.h_params.domains is not None:
        #                 test_file = test_file[test_file["cathDomain"].isin(self.h_params.domains)]
        #             test_file = test_file.assign(superfamily=sfam)
        #             if self.test_file is None:
        #                 self.test_file = test_file
        #             else:
        #                 self.test_file = pd.concat((self.test_file, test_file))
        #         if self.test_file is None:
        #             raise RuntimeError(f"Unable to load or create full train files: {full_files}")
        # elif self.h_params.test_file is not None:
        #     self.train_file = ""
        #     self.valid_file = ""
        #     self.test_file = self.h_params.test_file
        # elif self.h_params.all_domains:
        #     full_file = os.path.join(self.train_dir, f"{prefix}-full-train.h5")
        #     if not os.path.isfile(full_file):
        #         raise RuntimeError("Unable to find or create full train file")
        #     self.test_file = full_file
        #     self.train_file = ""
        #     self.valid_file = ""
        # else:
        #     self.train_file = ""
        #     self.valid_file = ""
        #     self.test_file = [os.path.join(train_dir, "{}-test-{}-split{:.1f}.h5".format(
        #         prefix, self.cluster_level, (1-self.h_params.split_size)/2)) \
        #         for train_dir in self.train_dir]

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()

        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.label_encoder = None


    # def prepare_data(self):
    #     # download
    #     files = [os.path.isfile(f) for f in (self.train_file, self.valid_file, self.test_file) if isinstance(f, str) and f!=""]
    #     if len(files)>0 and not any(files):
    #         raise RuntimeError("Must build dataset first. Files note found: {}".format((self.train_file, self.valid_file, self.test_file)))
    #         if self.h_params.superfamily is not None:
    #             self.DATASET.from_superfamily()

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = self.DATASET.create_multiple(
                self.data_dir,
                [s.replace(".","/") for s in self.h_params.superfamily],
                balance = self.h_params.balance,
                cluster_level=self.h_params.cluster_level,
                use_features=self.h_params.features,
                nClasses=self.h_params.nClasses,
                volume=self.h_params.input_size,
                expand_surface_data_loader=self.h_params.expand_surface_data_loader,
                compare_kdtree=self.h_params.compare_kdtree,
                remove_loops=self.h_params.remove_loops
            )
                #representatives=self.h_params.representatives,
                #domains=self.h_params.domains,
                #all_domains=self.h_params.all_domains
                #)

            if len(self.train_dataset) == 0:
                #Selected split is empty bc there is dataset in h5 file
                if not self.h_params.ignore_splits_on_error:
                    raise RuntimeError("Training dataset is empty. Use --ignore_splits_on_error to use the validation/test datset")

                self.train_dataset = None
                stage = None

            self.valid_dataset = self.DATASET.create_multiple(
                self.data_dir,
                [s.replace(".","/") for s in self.h_params.superfamily],
                #balance = self.h_params.balance,
                cluster_level=self.h_params.cluster_level,
                use_features=self.h_params.features,
                nClasses=self.h_params.nClasses,
                validation=True,
                volume=self.h_params.input_size,
                expand_surface_data_loader=self.h_params.expand_surface_data_loader,
                compare_kdtree=self.h_params.compare_kdtree,
                remove_loops=self.h_params.remove_loops
            )
                # representatives=self.h_params.representatives,
                # domains=self.h_params.domains,
                # all_domains=self.h_params.all_domains
                #)
            if len(self.valid_dataset) == 0:
                #Selected split is empty bc there is dataset in h5 file
                if not self.h_params.ignore_splits_on_error:
                    raise

                self.valid_dataset = None
                stage = None

            # Optionally...
            #self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:

            if hasattr(self.h_params, "domains") and isinstance(self.h_params.domains, (list, tuple)):
                #Can be domains from any superfamily, create one general enoding
                self.label_encoder = self.DATASET.encode_labels(self.h_params.domains)
                print("Created label encoder", self.label_encoder.classes_ )

            self.test_dataset = self.DATASET.create_multiple(
                self.data_dir,
                [s.replace(".","/") if isinstance(s, str) else s for s in self.h_params.superfamily],
                #balance = self.h_params.balance,
                cluster_level=self.h_params.cluster_level,
                use_features=self.h_params.features,
                nClasses=self.h_params.nClasses,
                volume=self.h_params.input_size,
                rotate=False,
                test=True,
                expand_surface_data_loader=self.h_params.expand_surface_data_loader,
                compare_kdtree=self.h_params.compare_kdtree,
                space_fill_algorithm=self.h_params.space_fill_algorithm,
                representatives=self.h_params.representatives if hasattr(self.h_params, "domains") else False,
                domains=self.h_params.domains if hasattr(self.h_params, "domains") else None,
                all_domains=self.h_params.all_domains if hasattr(self.h_params, "all_domains") else False,
                remove_loops=self.h_params.remove_loops  if hasattr(self.h_params, "remove_loops") else False,
                label_encoder_classes=self.label_encoder.classes_ if self.label_encoder is not None else None
            )
            if len(self.test_dataset) == 0:
                if not self.h_params.ignore_splits_on_error:
                    raise RuntimeError

                self.test_dataset = None
                stage = None

            if self.h_params.ignore_splits_on_error:
                self.train_dataset = self.valid_dataset = self.test_dataset
                self.h_params.val_check_interval = self.h_params.max_epochs-1

            if self.label_encoder is None:
                self.label_encoder = self.test_dataset.embedding

    def _get_dataloader(self, dataset, workers_split=1, shuffle=True, drop_last=True):
        total_workers = self.h_params.num_workers #-2*int(self.h_params.sweep)-1

        # if isinstance(self.h_params.gpus, (list, tuple)):
        #     pin_memory = len(self.h_params.gpus) > 0
        #     #total_workers -= len(self.h_params.gpus)
        # elif isinstance(self.h_params.gpus, (int, float)):
        #     pin_memory = self.h_params.gpus > 0
        #     #total_workers -= self.h_params.gpus
        # else:
        #     pin_memory = False

        ngpus = len(self.h_params.gpus) if isinstance(self.h_params.gpus, (list, tuple)) else self.h_params.gpus

        batch_size = max(1, min(self.h_params.batch_size, int(len(dataset)/ngpus)))

        print(dataset.key, len(dataset), self.h_params.batch_size, batch_size, shuffle)

        assert batch_size>0

        if batch_size <= len(dataset):
            drop_last = False

        if len(dataset)<self.h_params.log_every_n_steps or batch_size <= len(dataset):
            self.h_params.log_every_n_steps = 1

        return DataLoader(
            dataset,
            batch_size=batch_size, #self.h_params.batch_size,
            shuffle=shuffle,
            num_workers=total_workers, #int(workers_split*total_workers),
            collate_fn=batch_sparse_collate if self.h_params.expand_surface_data_loader else collate_coordinates,
            #gpus=self.h_params.gpus,
            #distributed_backend=self.h_params.distributed_backend,
            drop_last=drop_last,
            pin_memory=False #pin_memory
        )

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, workers_split=self.h_params.split_size)

    def val_dataloader(self):
        return self._get_dataloader(self.valid_dataset, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, shuffle=False, drop_last=False)

    @staticmethod
    def add_data_specific_args(parent_parser, eval=False):
        parser = argparse.ArgumentParser(
            parents=[parent_parser],
            add_help=False)

        data = parser.add_argument_group('data')

        data.add_argument("--superfamily", nargs='+',
            help="CATH superfamily code. If specified, it will create a superfamily-specific model")
        data.add_argument("--data_dir", default="")

        if not eval:
            data.add_argument("--cluster_level", default="S35", choices=DomainStructureDataset.hierarchy[4:])
            data.add_argument("--split_size", type=float, default=0.8)
            data.add_argument("--train_file", default=None)
            data.add_argument("--test_file",  default=None)
            data.add_argument("--valid_file", default=None)
        else:
            data_types = data.add_mutually_exclusive_group()
            data_types.add_argument("--representatives", type=num_or_all, nargs="?", const=True, default=False,
                help="Only use representative structures. Pass in number to use else use all. If argument not passed, the regular test data will be used")
            data_types.add_argument("--cluster_level", default="S35", choices=DomainStructureDataset.hierarchy[4:])
            data_types.add_argument("--domains", default=None, nargs="+")
            data_types.add_argument("--test_file", default=None)
            data_types.add_argument("--all_domains", type=str2bool, nargs='?',
                const=True, default=False)
            data.add_argument("--split_size", type=float, default=0.8)

        data.add_argument("--ignore_splits_on_error", type=str2bool, nargs='?',
            const=True, default=False, help="If there is only S35 cluster (or other level), use any of the train/test/valid that exists that contains all inputs")

        data.add_argument("--input_size", type=int, default=256,
                          help="size of training images, default is 256")
        data.add_argument("--batch_size", type=int, default=8,
                          help="batch size for training, default is 8")
        data.add_argument("--expand_surface_data_loader", type=str2bool, nargs='?',
            const=True, default=True, help="Expand atoms to van der walls spheres during data loader. If False, atoms will be expanded during the model's forward function")
        data.add_argument("--compare_kdtree", type=str2bool, nargs='?',
            const=True, default=False, help="Compare space-filling model of kd-tree in dataloader vs KNN in forward function")
        data.add_argument("--space_fill_algorithm", default="kdtree", choices=space_fill_algorithms)
        data.add_argument("--remove_loops", type=str2bool, nargs='?',
            const=True, default=False, help="Remove loops between SS during training")

        balance = data.add_mutually_exclusive_group()
        balance.add_argument("--over_sample", type=str2bool, nargs='?',
            const=True, default=False, help="Balance datasets by oversampling when using multiple superfamlies")
        balance.add_argument("--under_sample", type=str2bool, nargs='?',
            const=True, default=False, help="Balance datasets by oversampling when using multiple superfamlies")

        data.add_argument("--features", nargs="+", default=default_atom_features)
        data.add_argument("--features_to_drop", nargs="+")
        data.add_argument("--feature_groups", default=default_feature_groups, nargs="+")
        data.add_argument("--num_workers", type=int, default=int(os.environ.get(
            "SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))-1)

        nClasses = data.add_mutually_exclusive_group()
        nClasses.add_argument("--nClasses", default=None)
        nClasses.add_argument("--autoencoder", type=str2bool, nargs='?',
            const=True, default=False, help="Use same number of outputs as number of features")

        return parser
