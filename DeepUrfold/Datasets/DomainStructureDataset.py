import os
import sys
import shutil
from glob import glob
from contextlib import closing
import urllib.request as request
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.stats import special_ortho_group

from molmimic.common.voxels import ProteinVoxelizer
from molmimic.util.pdb import remove_ter_lines
from molmimic.util.toil import map_job

from toil.realtimeLogger import RealtimeLogger

from DeepUrfold.Datasets.Dataset import Dataset

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))

def run_superfamily(job, sfam, data_dir, memory="72G", **kwds):
    return DomainStructureDataset.from_superfamily(sfam, data_dir, **kwds)

def merge_superfamilies(job, data_dir, memory="72G", **kwds):
    import dask
    import dask.dataframe as dd
    from multiprocessing.pool import Pool

    cores = kwds.get("cores", job.fileStore.jobStore.config.maxCores)
    if cores > 1:
        cores -= 1

    dask.config.set(scheduler="processes")
    dask.config.set(pool=Pool(cores))

    dataset = "DomainStructureDataset-full-train.h5"

    all_rows = glob(str(os.path.join(data_dir, "train_files", "*", "*",
        "*", "*", dataset)))

    ddf = dd.read_hdf(all_rows, "table")
    ddf = ddf.repartition(npartitions=cores)

    out_file = str(os.path.join(data_dir, "train_files", dataset))

    data_columns = ["cathDomain"]

    ddf.to_hdf(out_file, "table", format="table", complevel=9,
            data_columns=data_columns, complib="bzip2", min_itemsize=1024)

def get_superfamilies():
    superfamily_file = os.path.join(os.getcwd(), "cath-superfamily-list.txt")
    if not os.path.isfile(superfamily_file):
        superfamily_url = "ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/cath-superfamily-list.txt"
        download_file(superfamily_url, superfamily_file)

    return pd.read_csv(superfamily_file, sep="\t", skiprows=1, comment="#",
        names=["CATH_ID", "S35_REPS", "DOMAINS", "NAME"])["CATH_ID"]

def download_file(url, local_filename=None):
    if local_filename is None:
        local_filename = url.split('/')[-1]

    with closing(request.urlopen(url)) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r, f)

    return local_filename

def get_cath_clusters():
    cluster_file = os.path.join(os.getcwd(), "cath-domain-list.txt")
    if not os.path.isfile(cluster_file):
        cath_clusters_url = "ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/cath-classification-data/cath-domain-list.txt"
        download_file(cath_clusters_url, cluster_file)

    cath_clusters = pd.read_csv(cluster_file, delim_whitespace=True, names=[
        "cathDomain", "C", "A", "T", "H",
        "S35", "S60", "S95", "S100", "S100_count",
        "dlength", "resolution"], comment="#")

    cath_clusters = cath_clusters.drop(columns=["S100_count", "dlength", "resolution"])

    return cath_clusters

class DomainStructureDataset(Dataset):
    hierarchy = ["C", "A", "T", "H", "S35", "S60", "S95", "S100"]
    rvs = np.eye(3)

    def __init__(self, data_file, data_key="table", use_features=None, split_level="H",
      domain_key="cathDomain", structure_key="structure_file", feature_key="feature_file",
      truth_key=None, data_dir=None, use_domain_index=False, volume=256, nClasses=1,
      rotate=True, test=False, expand_surface_data_loader=True, compare_kdtree=False, space_fill_algorithm="kdtree"):
        super().__init__(data_file)
        self.use_features = use_features
        self.split_level = split_level
        self.domain_key = domain_key
        self.structure_key = structure_key
        self.feature_key = feature_key
        self.truth_key = truth_key
        self.use_domain_index = use_domain_index
        self.use_features = use_features
        self.volume = volume
        self.nClasses = nClasses
        self.rotate = rotate
        self.test = test
        self.expand_surface_data_loader = expand_surface_data_loader
        self.compare_kdtree = compare_kdtree
        self.space_fill_algorithm = space_fill_algorithm

        #Remove entries from empty PDB files
        skip_domains = ["5jjlB01", "1qzhB00", "4kdaL00", "2ja7G02"]
        if use_domain_index:
            self.data = self.data[~self.data.index.isin(skip_domains)]
        else:
            self.data = self.data[~self.data[self.domain_key].isin(skip_domains)]

        data_dir = os.getcwd() if data_dir is None else data_dir
        fix_path = lambda f: os.path.join(data_dir, f.split("data_eppic_cath_features/")[-1])

        if True: #not os.path.isfile(self.data.iloc[0][structure_key]):
            test_file = fix_path(self.data.iloc[0][structure_key])
            if os.path.isfile(test_file):
                self.data[structure_key] = self.data[structure_key].apply(lambda f: fix_path(f))
                self.data[feature_key] = self.data[feature_key].apply(lambda f: fix_path(f))
            else:
                raise FileNotFoundError(f"You must set 'data_dir' parameter and make sure training files exist in that directory ({test_file} not found)")

        if self.test:
            self.cath_domain_embedding = preprocessing.LabelEncoder().fit(self.data[domain_key])

    def reset_rotation_matrix(self):
        self.rvs = special_ortho_group.rvs(3)
        return self.rvs

    def __getitem__(self, index, voxel_map=False):
        voxelizer, indices, data, truth, _voxel_map, serial, b_factors = [None]*7
        try:
            voxelizer, indices, data, truth, _voxel_map, serial, b_factors = self.get_structure_and_voxels(index)
            #data = np.nan_to_num(data)

            # if truth is not None:
            #     truth = np.nan_to_num(truth)

            # idxs = sys.getsizeof(indices)
            # ds = sys.getsizeof(data)
            # ts = sys.getsizeof(truth)
            # print("i =", idxs, "; d =", ds, "; t =", ts, "; is+ds =", idxs+ds, "; is+ts =", idxs+ts, "; is+ds+ts =", idxs+ds+ts)
            #b_factors = b_factors[:, np.newaxis]
            #idx = torch.FloatTensor([[index]]).repeat(b_factors.shape[0], 1)
            #serials = np.array([[",".join(map(str, s))] for s in serial])
            #truth = np.concatenate((truth, b_factors, idx), axis=1)

            i, d = torch.from_numpy(indices), torch.from_numpy(data)
            if self.expand_surface_data_loader:
                i = i.int()
                d = d.float()

            if truth is not None:
                if self.compare_kdtree and isinstance(truth, (list, tuple)):
                    t=truth
                else:
                    t = torch.from_numpy(truth).float()

            del voxelizer, indices, data, serial, b_factors

            #assert _voxel_map is not None

            if self.compare_kdtree and _voxel_map is not None:
                if truth is None:
                    return i, d, None, _voxel_map
                else:
                    return i, d, t, _voxel_map
            else:
                del _voxel_map

            if truth is None:
                return i, d
            else:
                return i, d, t
        except (KeyboardInterrupt, SystemExit):
            #del voxelizer, indices, data, truth, _voxel_map, serials, b_factors
            raise
        except:
            #del voxelizer, indices, data, truth, _voxel_map, serials, b_factors
            raise
            trace = traceback.format_exc()
            print("Error:", trace)

    def get_structure_and_voxels(self, index, truth_residues=None):
        datum = self.data.iloc[index]

        if self.use_domain_index:
            cath_domain = datum.name
        else:
            cath_domain = datum[self.domain_key]
        pdb_file = datum[self.structure_key]
        features_file = datum[self.feature_key]

        if not os.path.isfile(pdb_file+".noter") or os.stat(pdb_file+".noter").st_size == 0:
            remove_ter_lines(pdb_file, pdb_file+".noter")

        pdb_file = pdb_file+".noter"

        features_path = os.path.dirname(features_file)

        if self.rotate is None or (isinstance(self.rotate, bool) and not self.rotate):
            rotate = False
        elif isinstance(self.rotate, np.ndarray) or (isinstance(self.rotate, str) and self.rotate in ["random", "pai"]):
            rotate = self.rotate
        elif isinstance(self.rotate, bool) and self.rotate:
            #Use Dataset's random roation matrix
            rotate = self.rvs
        else:
            raise RuntimeError("Invalid rotation parameter. It must be True to use this Dataset's random roation matrix updated during each epoch, " + \
                "(None or False) for no rotation, 'random' for a random rotation matrix from the voxelizer updated during every initalization, " + \
                "'pai' to rotate to the structure's princple axes, or a rotation matrix given as a numpy array.")

        voxelizer = ProteinVoxelizer(pdb_file, cath_domain,
            features_path=features_path,
            rotate=rotate,
            use_features=self.use_features,
            volume=self.volume,
            replace_na=True)

        if truth_residues is None and self.truth_key is not None:
            truth_residues = datum[self.truth_key].split(",")
            truth_residues = [voxelizer.get_residue_from_resseq(r) for r in truth_residues]
            truth_residues = [r for r in truth_residues if r is not None]
            autoencoder = False
        else:
            autoencoder = True

        if not self.expand_surface_data_loader:
            indices, data, truth = voxelizer.get_atoms_and_features(
                autoencoder=autoencoder,
                truth_residues=truth_residues,
                use_deepsite_features=True,
                only_surface=False,
                return_voxel_map=True,
                return_b=True,
                return_serial=True,
                nClasses=self.nClasses
                )
            voxel_map, serial, b_factors = None, None, None

            if self.compare_kdtree:
                indices_kdtree, data_kdtree, _, _voxel_map, _, _ = voxelizer.map_atoms_to_voxel_space(
                    autoencoder=autoencoder,
                    truth_residues=truth_residues,
                    use_deepsite_features=True,
                    only_surface=False,
                    return_voxel_map=True,
                    return_b=True,
                    return_serial=True,
                    nClasses=self.nClasses
                    )
                truth = [indices_kdtree, data_kdtree]
                # voxels_serial_map = defaultdict(list)
                # for atom_grid, atoms in voxel_map.items():
                #     for atom in atoms:
                #         voxels_serial_map[atom].append(atom_grid)

                max_atoms = max(_voxel_map.keys()) #max(voxels_serial_map.keys())
                max_voxels = max(len(atoms) for atoms in _voxel_map.values()) #max(len(atoms) for atoms in voxels_serial_map.values())
                voxel_map = torch.ones((max_atoms,max_voxels,3))*float('nan')
                for k, v in sorted(_voxel_map.items(), key=lambda x:x[0]):
                    assert len(v)<=voxel_map.size()[1]
                    voxel_map[k-1, :len(v)] = torch.Tensor(sorted(v))



                #assert voxel_map is not None

        else:
            indices, data, truth, voxel_map, serial, b_factors = voxelizer.map_atoms_to_voxel_space(
                autoencoder=autoencoder,
                truth_residues=truth_residues,
                use_deepsite_features=True,
                only_surface=False,
                return_voxel_map=True,
                return_b=True,
                return_serial=True,
                nClasses=self.nClasses
                )

        if self.test:
            n_truth = len(truth) if truth is not None else len(data)
            domain_col = np.tile(self.cath_domain_embedding.transform([cath_domain]), (n_truth,1))
            truth = np.append(truth if truth is not None else data, domain_col, axis=1)

        del truth_residues

        return voxelizer, indices, data, truth, voxel_map, serial, b_factors
