import os
from pathlib import Path
from functools import wraps
from typing import Any, Callable, Optional

import torch
import h5py
import numpy as np
import pandas as pd

from torchmetrics import Metric
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.connectors.callback_connector import CallbackConnector

from biotite.structure.io.pdb import PDBFile
from biotite.structure import apply_residue_wise, get_residue_starts
from biotite.structure.io.pdbx import PDBxFile, set_structure, get_structure

from scipy.stats.mstats import gmean as _gmean
from scipy.stats import hmean as _hmean
from sklearn.preprocessing import MinMaxScaler

from Prop3D.common.DistributedVoxelizedStructure import DistributedVoxelizedStructure

from pytorch_lrp.innvestigator import InnvestigateModel

import seaborn as sns
import matplotlib.pyplot as plt

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
    "sum": npsum,
    "npsum": npsum,
    "None": None}

class AtomicRelevance(object):
    def __init__(
        self,
        output_labels,
        data_dir,
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
        self.data_dir = data_dir
        self.volume = volume

        if isinstance(voxel_aggregate_fn, str):
            assert voxel_aggregate_fn in aggragators
            voxel_aggregate_fn = aggragators[voxel_aggregate_fn]

        if isinstance(feature_aggregate_fn, str):
            assert feature_aggregate_fn in aggragators
            feature_aggregate_fn = aggragators[feature_aggregate_fn]

        self.voxel_aggregate_fn = voxel_aggregate_fn
        self.feature_aggregate_fn = feature_aggregate_fn

        #self.add_state("data", default=[], dist_reduce_fx=None)

    def update(self, relevance: torch.Tensor, cath_domain, superfamily, checkpoint, *args, **kwds):
        """
        Update state with predictions and targets.

        Args:
            data: Predictions from model (probabilities, or labels)
        """
        superfamily = superfamily.split(".")
        #preds, target, mode = _auroc_update(preds, target)
        # voxelizer = ProteinVoxelizer(pdb_file+".noter", cath_domain, rotate=None,
        #     features_path=os.path.join(self.features_path, *superfamily),
        #     residue_feature_mode=None, use_features=self.output_labels, volume=self.volume)

        voxelizer = DistributedVoxelizedStructure(
            self.data_dir, "/".join(superfamily), cath_domain, volume=self.volume, rotate=None,
            use_features=self.output_labels, replace_na=True)

        indices, labels, _, atom2voxels, _, _ = voxelizer.map_atoms_to_voxel_space(
            autoencoder=True,
            return_voxel_map=True,
            return_serial=True,
            return_b=True,
            nClasses=len(self.output_labels))

        # inputs, labels, atom2voxels = voxelizer.voxels_from_pdb(
        #     autoencoder=True,
        #     only_surface=False,
        #     use_deepsite_features=True,
        #     return_voxel_map=True,
        #     use_numpy=True)

        assert(len(indices), len(relevance.index)), "Not even same lengs coords"
        a = set(map(tuple, indices.astype(int)))
        b = set(map(tuple, relevance.index.to_list()))

        assert a==b, f"{len(a)} {len(b)} {len(a.intersection(b))} {len(a-b)} {list(a-b)[0]} {len(b-a)} {list(b-a)[0]} " #"#f"Coords from {pdb_file} does not equal relevance coords: {inputs[0]} {relevance.index.to_list()}"

        relevance_structure = voxelizer.copy(empty=True)

        new_dtype = np.dtype([(name, np.float32) for name in relevance_structure.features.dtype.names])

        voxel_expander = lambda a: atom2voxels[a] #[a["serial_number"]]

        cath_domain_dir = os.path.join(os.getcwd(), cath_domain)
        if not os.path.isdir(cath_domain_dir):
            os.makedirs(cath_domain_dir)

        if self.voxel_aggregate_fn is None:
            print(relevance.values)
            total_relevance = self.feature_aggregate_fn(relevance.values, axis=1)

            print(total_relevance)

            #Save scaled relevances
            scaler = MinMaxScaler()
            scaled_relevance = np.round(scaler.fit_transform(total_relevance.flatten().reshape(-1,1), 4)).flatten()

            most_relevant_idx = np.where(total_relevance>=np.quantile(total_relevance, 0.75))[0]
            most_relevent_voxels = indices[most_relevant_idx]
            most_relevent_scores = total_relevance[most_relevant_idx]
            most_relevent_scores_scaled = scaled_relevance[most_relevant_idx]
            
            print("Scaled", most_relevent_scores_scaled)

            assert len(indices)==len(total_relevance)

            voxel_file= Path(cath_domain_dir) / f"{superfamily[0].replace('/', '_')}-{cath_domain}f_agg={self.feature_aggregate_fn.__name__}-total-relevance.h5"
            with h5py.File(voxel_file, "w") as f:
                f.create_dataset("indices", data=indices)
                f.create_dataset("relevance", data=total_relevance)
                f.create_dataset("scaled_relevance", data=scaled_relevance)
                f.create_dataset("best_indices", data=most_relevent_voxels)
                f.create_dataset("best_relevance", data=most_relevent_scores)
                f.create_dataset("best_relevance_scaled", data=most_relevent_scores_scaled)

            print("Saved")
            
            return


        def agg_relevance_for_atom(atom_idx):
            grids = list(voxel_expander(atom_idx))
            rel_for_atom = self.voxel_aggregate_fn(relevance.loc[grids, :])[None]
            return rel_for_atom

        relavance_values = np.concatenate([agg_relevance_for_atom(idx) for idx in relevance_structure.data["serial_number"]])
        total_relevance = self.feature_aggregate_fn(relavance_values, axis=1)
        #relavance_values = np.concatenate((relavance_values,total_relevance), axis=1)
        #relavance_values = relavance_values.astype(new_dtype).view(np.recarray)
        relavance_values = np.core.records.fromarrays(relavance_values.T, dtype=new_dtype)
        

        relevance_structure.features = relavance_values
        relevance_structure.add_features(total_relevance=total_relevance)

        # for atom in voxelizer.get_atoms():
        #     idx = atom["serial_number"]
        #     grids = list(voxel_expander(atom))
        #     rel_for_atom = self.voxel_aggregate_fn(relevance.loc[grids, :])[None] #grid_idx
        #     import pdb; pdb.set_trace()
        #     relevance_structure.features[idx][self.output_labels] = rel_for_atom

        #total_relevance = self.feature_aggregate_fn(relevance_structure.features, axis=1)

        #

        output_labels = self.output_labels+["total_relevance"]

        

        name=f"v_agg={self.voxel_aggregate_fn.__name__}__f_agg={self.feature_aggregate_fn.__name__}"
        relevance_structure.write_features(features=output_labels, path=f"{cath_domain}-{name}.h5", work_dir=cath_domain_dir)
        rel_files = relevance_structure.write_features_to_pdb(["total_relevance"], name=name, work_dir=cath_domain_dir, other=relevance_structure)

        #Read in protein, set new field of resi+ins_code
        pdb_file = Path(cath_domain_dir) / rel_files["total_relevance"]
        protein = PDBFile.read(str(pdb_file)).get_structure(model=1, extra_fields=["b_factor", "atom_id"], altloc='first')
        all_res_ids = np.core.defchararray.add(protein.res_id.astype(str), protein.ins_code)
        res_ids = np.unique(all_res_ids)

        #Sum up relevance values for all atoms in each residue, extract residues with a score > 0.75 of quantile
        # total_per_residue = apply_residue_wise(protein, protein.b_factor, np.nansum)
        # threshold = np.quantile(total_per_residue, 0.75)
        # best_res_ids = res_ids[total_per_residue >= np.quantile(total_per_residue, 0.75)]
        # best_structure = protein[np.isin(all_res_ids, best_res_ids)]

        # new_pdb_file = PDBFile()
        # new_pdb_file.set_structure(best_structure)
        # new_pdb_file.lines.insert(0, f"REMARK Scores={';'.join([f'{i:0.4f}' for i in total_per_residue])}")
        # new_pdb_file.lines.insert(0, f"REMARK Threshold={threshold:0.4f} (75% quantile)")
        # new_pdb_file.lines.insert(0, f"REMARK Residues={';'.join(best_res_ids)}")
        # new_pdb_file.write(str(pdb_file.with_suffix(".75pctquntile.pdb")))

        #Save full relevances
        pdbx_file = PDBxFile()
        set_structure(pdbx_file, protein, data_block="structure")
        pdbx_file.write(str(pdb_file.with_suffix(".cif")))

        #Sum up relevance values for all atoms in each residue, extract residues with a score > 0.75 of quantile
        total_per_residue_sum = apply_residue_wise(protein, protein.b_factor, np.nansum)
        threshold = np.quantile(total_per_residue_sum, 0.75)
        best_res_ids = res_ids[total_per_residue_sum >= threshold]
        best_structure = protein[np.isin(all_res_ids, best_res_ids)]
        small_pdbx_file = PDBxFile()
        set_structure(small_pdbx_file, best_structure, data_block="structure")
        small_pdbx_file.write(str(pdb_file.with_suffix(".75percentile.cif")))

        #Save scaled relevances
        scaler = MinMaxScaler()
        new_bfactor = np.round(scaler.fit_transform(protein.b_factor.reshape(-1,1)).flatten(), 4)
        protein.set_annotation("b_factor", new_bfactor)
        new_pdb_file = PDBFile()
        new_pdb_file.set_structure(protein)
        new_pdb_file.write(str(pdb_file.with_suffix(".scaled.pdb")))

        small_protein = protein[np.isin(protein.atom_id, best_structure.atom_id)]
        small_pdb_file = PDBFile()
        small_pdb_file.set_structure(small_protein)
        small_pdb_file.lines.insert(0, f"REMARK Scores={';'.join([f'{i:0.4f}' for i in total_per_residue_sum])}")
        small_pdb_file.lines.insert(0, f"REMARK Threshold={np.quantile(protein.b_factor, 0.75):0.4f} (75% quantile)")
        small_pdb_file.lines.insert(0, f"REMARK 50th perecentile={np.quantile(protein.b_factor, 0.5):0.4f}")
        small_pdb_file.lines.insert(0, f"REMARK 80th perecentile={np.quantile(protein.b_factor, 0.8):0.4f}")
        small_pdb_file.lines.insert(0, f"REMARK Residues={';'.join(best_res_ids)}")
        small_pdb_file.write(str(pdb_file.with_suffix(".75percentile.scaled.pdb")))

        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        sns.kdeplot(protein.b_factor)
        atom_threshold = np.quantile(protein.b_factor, 0.80)
        ax.axvspan(atom_threshold, protein.b_factor.max(), facecolor='gray', alpha=0.5)
        ax.plot([], [], ' ', label=f"Min: {protein.b_factor.min()}")
        ax.plot([], [], ' ', label=f"Max: {protein.b_factor.max()}")
        ax.plot([], [], ' ', label=f"80th percentile: {atom_threshold}")
        ax.plot([], [], ' ', label=f"50th perecentile:{np.quantile(protein.b_factor, 0.5):0.4f}")
        ax.plot([], [], ' ', label=f"75th perecentile={np.quantile(protein.b_factor, 0.75):0.4f}")
        ax.set_title(f"{cath_domain} through {checkpoint} model (Atom wise)")
        plt.legend()
        plt.savefig(str(pdb_file.with_suffix(".atom_wise.pdf")), dpi=300)
        plt.clf()

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        sns.kdeplot(total_per_residue_sum)
        ax.axvspan(threshold, total_per_residue_sum.max(), facecolor='gray', alpha=0.5)
        ax.plot([], [], ' ', label=f"Min: {total_per_residue_sum.min()}")
        ax.plot([], [], ' ', label=f"Max: {total_per_residue_sum.max()}")
        ax.plot([], [], ' ', label=f"75th percentile: {threshold}")
        ax.set_title(f"{cath_domain} through {checkpoint} model (Residue wise)")
        plt.legend()
        plt.savefig(str(pdb_file.with_suffix(".residue_wise.pdf")), dpi=300)
        plt.clf()

    def compute(self) -> torch.Tensor:
        """
        Computes AUROC based on inputs passed in to ``update`` previously.
        """
        return None
