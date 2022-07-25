import os
from functools import wraps
from typing import Any, Callable, Optional

import torch
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import Callback
from torchmetrics import Metric
from pytorch_lightning.trainer.connectors.callback_connector import CallbackConnector

from scipy.stats.mstats import gmean as _gmean
from scipy.stats import hmean as _hmean

from molmimic.common.voxels import ProteinVoxelizer

from pytorch_lrp.innvestigator import InnvestigateModel

class Relevance(Metric, Callback):
    """
     Args:
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False. default: True
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
        process_group:
            Specify the process group on which synchronization is called. default: None (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    """
    def __init__(
        self,
        pl_model,
        autoencoder=False,
        target_class=None,
        rule="LayerNumRule",
        pl_trainer=None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.autoencoder = autoencoder
        self.target_class = target_class
        self.rule = rule

        self.lrp_innvestigator = InnvestigateModel(pl_model)

        if pl_trainer is not None:
            self.setup_callback(pl_trainer)

        self.add_state("relevance", default=[], dist_reduce_fx=None)

    def setup_callback(self, pl_trainer):
        pl_trainer.callbacks.append(self)
        pl_trainer.callbacks = CallbackConnector._reorder_callbacks(all_callbacks)

    def on_test_batch_end(self, trainer, pl_module, result, batch, batch_idx, dataloader_idx):
        _, relevances = self.lrp_innvestigator.innvestigate(result, no_recalc=True, autoencoder_in=True, rule=self.rule)
        self(relevances)

    def update(self, data: torch.Tensor, *args, **kwds):
        """
        Update state with predictions and targets.

        Args:
            data: Predictions from model (probabilities, or labels)
        """
        #preds, target, mode = _auroc_update(preds, target)
        self.data.append(data)

    def compute(self) -> torch.Tensor:
        """
        Computes AUROC based on inputs passed in to ``update`` previously.
        """
        return torch.cat(self.data, dim=0)

def replace_nan(func):
    @wraps(func)
    def wrapper(*args, **kwds):
        nan = kwds.pop("nan", 0.0)
        return np.nan_to_num(func(np.nan_to_num(*args, nan=nan), **kwds))
    return wrapper

@replace_nan
def gmean(a, axis=0, nan=1e-8):
    """Geometric Mean"""
    a[a==0] = nan
    return _gmean(a, axis=axis)

@replace_nan
def hmean(a, axis=0):
    """Harmonic Mean"""
    return _hmean(a, axis=axis)

@replace_nan
def geometric_log_mean(a, axis=0):
    return a.log(axis=axis).sum(axis=axis)/len(a)

@replace_nan
def arithmetic_mean(a, axis=0):
    return np.mean(a, axis=axis)

@replace_nan
def median(a, axis=0):
    return np.median(a, axis=axis)

@replace_nan
def npsum(a, axis=0):
    return np.sum(a, axis=axis)


aggragators = {
    "gmean": gmean,
    "hmean": hmean,
    "arithmetic_mean": arithmetic_mean,
    "median": median,
    "max": np.max,
    "min": np.min,
    "sum": npsum}

class AtomicRelevance(object):
    def __init__(
        self,
        output_labels,
        features_path,
        voxel_aggregate_fn: Callable = gmean,
        feature_aggregate_fn: Callable = npsum,
        volume=256
        # compute_on_step: bool = True,
        # dist_sync_on_step: bool = False,
        # process_group: Optional[Any] = None,
        # dist_sync_fn: Callable = None,
    ):
        # super().__init__(
        #     compute_on_step=compute_on_step,
        #     dist_sync_on_step=dist_sync_on_step,
        #     process_group=process_group,
        #     dist_sync_fn=dist_sync_fn,
        # )

        self.output_labels = output_labels
        self.features_path = features_path
        self.volume = volume

        if isinstance(voxel_aggregate_fn, str) and voxel_aggregate_fn in aggragators:
            voxel_aggregate_fn = aggragators[voxel_aggregate_fn]

        if isinstance(feature_aggregate_fn, str) and feature_aggregate_fn in aggragators:
            feature_aggregate_fn = aggragators[feature_aggregate_fn]

        self.voxel_aggregate_fn = voxel_aggregate_fn
        self.feature_aggregate_fn = feature_aggregate_fn

        #self.add_state("data", default=[], dist_reduce_fx=None)

    def update(self, relevance: torch.Tensor, pdb_file, cath_domain, superfamily, *args, **kwds):
        """
        Update state with predictions and targets.

        Args:
            data: Predictions from model (probabilities, or labels)
        """
        superfamily = superfamily.split(".")
        #preds, target, mode = _auroc_update(preds, target)
        voxelizer = ProteinVoxelizer(pdb_file+".noter", cath_domain, rotate=None,
            features_path=os.path.join(self.features_path, *superfamily),
            residue_feature_mode=None, use_features=self.output_labels, volume=self.volume)

        inputs, labels, atom2voxels = voxelizer.voxels_from_pdb(
            autoencoder=True,
            only_surface=False,
            use_deepsite_features=True,
            return_voxel_map=True,
            use_numpy=True)

        assert(len(inputs[0]), len(relevance.index)), "Not even same lengs coords"
        a = set(map(tuple, inputs[0].astype(int)))
        b = set(map(tuple, relevance.index.to_list()))

        assert a==b, f"{len(a)} {len(b)} {len(a.intersection(b))} {len(a-b)} {list(a-b)[0]} {len(b-a)} {list(b-a)[0]} " #"#f"Coords from {pdb_file} does not equal relevance coords: {inputs[0]} {relevance.index.to_list()}"


        relevance_structure = voxelizer.copy(empty=True)


        voxel_expander = lambda a: atom2voxels[a.serial_number]

        for atom in voxelizer.get_atoms():
            atom = voxelizer._remove_altloc(atom)
            idx = atom.serial_number
            grids = list(voxel_expander(atom))
            rel_for_atom = self.voxel_aggregate_fn(relevance.loc[grids, :])[None] #grid_idx
            relevance_structure.atom_features.loc[idx, self.output_labels] = rel_for_atom

        total_relevance = self.feature_aggregate_fn(relevance_structure.atom_features, axis=1)

        relevance_structure.add_features(total_relevance=total_relevance)

        output_labels = self.output_labels+["total_relevance"]

        cath_domain_dir = os.path.join(os.getcwd(), cath_domain)
        if not os.path.isdir(cath_domain_dir):
            os.makedirs(cath_domain_dir)

        name=f"v_agg={self.voxel_aggregate_fn.__name__}__f_agg={self.feature_aggregate_fn.__name__}"
        relevance_structure.write_features(features=output_labels, name=f"{cath_domain}-{name}.h5", work_dir=cath_domain_dir)
        relevance_structure.write_features_to_pdb(output_labels, name=name, work_dir=cath_domain_dir, other=relevance_structure)

    def compute(self) -> torch.Tensor:
        """
        Computes AUROC based on inputs passed in to ``update`` previously.
        """
        return None
