import os
import torch
import numpy as np
from scipy.stats import special_ortho_group

from DeepUrfold.Datasets.DistributedDataset import DistributedDataset
from molmimic.common.DistributedVoxelizedStructure import DistributedVoxelizedStructure

class DistributedDomainStructureDataset(DistributedDataset):
    hierarchy = ["C", "A", "T", "H", "S35", "S60", "S95", "S100"]
    rvs = np.eye(3)

    def __init__(self, path, key, use_features=None, predict_features=None,
      truth_key=None, cluster_level="S35", volume=256, nClasses=1, rotate=True,
      test=False, validation=False, representatives=False, domains=None, all_domains=False,
      file_mode="r", expand_surface_data_loader=True, compare_kdtree=False,
      space_fill_algorithm="kdtree", dataset_group_name=None, use_keys=None,
      ignore_keys=None, remove_loops=False, return_structure=False, label_encoder_classes=None):
        assert [validation, test].count(True)<2, "Can only select none or one at a time"
        self.use_features = use_features
        self.cluster_level = cluster_level
        self.use_features = use_features
        self.predict_features = predict_features
        self.truth_key = truth_key
        self.volume = volume
        self.nClasses = nClasses
        self.rotate = rotate
        self.test = test
        self.validation = validation
        self.domains = domains
        self.all_domains = all_domains
        self.representatives = representatives
        self.expand_surface_data_loader = expand_surface_data_loader
        self.compare_kdtree = compare_kdtree
        self.space_fill_algorithm = space_fill_algorithm
        self.remove_loops = remove_loops
        self.return_structure = return_structure

        if truth_key is not None:
            if isinstance(predict_features, (list, tuple)) and len(predict_features)>0:
                print("ignoring predict_features")
        elif isinstance(predict_features, (list, tuple)) and len(predict_features)>0:
            assert isinstance(use_features, (list, tuple)) and len(use_features) > 0, \
                "Cannot train on all features and predict a feature in that set"
            assert len(set(predict_features).intersection(set(use_features)))==0, \
                "Cannot train on and predict the same features"

        assert [isinstance(domains,(str,list,tuple)), all_domains, representatives].count(True)<2

        if dataset_group_name is None:
            if all_domains or isinstance(domains, (list,tuple)):
                dataset_group_name = "domains"
            elif representatives:
                dataset_group_name = "representatives"
            elif cluster_level is not None:
                dataset_group_name = f"data_splits/{cluster_level}/"
                if self.validation:
                    dataset_group_name += "validation"
                elif self.test:
                    dataset_group_name += "test"
                else:
                    dataset_group_name += "train"
            else:
                dataset_group_name = "domains"
        self.dataset_group_name = dataset_group_name

        if key is None:
            key = "/"
            self.dataset_group_name = ""

        super().__init__(path, key, test=self.test, dataset_group_name=dataset_group_name,
            use_keys=self.domains, file_mode=file_mode, label_encoder_classes=label_encoder_classes)

    def reset_rotation_matrix(self):
        self.rvs = special_ortho_group.rvs(3)
        return self.rvs

    def set_rotation_matrix(self, rvs):
        self.rvs = rvs

    @staticmethod
    def encode_labels(labels, from_fitted=False):
        from sklearn.preprocessing import LabelEncoder
        if not from_fitted:
            embedding = LabelEncoder().fit([l.split("/")[-1] for l in labels])
        else:
            embedding = LabelEncoder()
            embedding.classes_ = labels
        return embedding

    def __getitem__(self, index, voxel_map=False):
        voxelizer, indices, data, truth, _voxel_map, serial, b_factors = [None]*7
        try:
            if self.return_structure:
                return self.get_structure_and_voxels(index)

            voxelizer, indices, data, truth, _voxel_map, serial, b_factors = self.get_structure_and_voxels(index)
            #data = np.nan_to_num(data)

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
        cath_domain_dataset = super().__getitem__(index)
        key = self.order[index]

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

        voxelizer = DistributedVoxelizedStructure(
            self.path, key, cath_domain_dataset, volume=self.volume, rotate=rotate,
            use_features=self.use_features, predict_features=self.predict_features,
            replace_na=True)

        if self.return_structure:
            return voxelizer

        if self.remove_loops:
            voxelizer.remove_loops()

        if truth_residues is None and self.truth_key is not None:
            truth_residues = cath_domain_dataset.attrs[self.truth_key].split(",")
            autoencoder = False
        else:
            autoencoder = True

        if not self.expand_surface_data_loader:
            indices, data, truth, voxel_map, serial, b_factors = voxelizer.map_atoms_to_voxel_space(
                truth_residues=truth_residues,
                autoencoder=autoencoder,
                return_voxel_map=True,
                return_serial=True,
                return_b=True,
                nClasses=self.nClasses,
                use_raw_atom_coords=True)
            voxel_map, serial, b_factors = None, None, None

            if self.compare_kdtree:
                indices, data, truth, voxel_map, serial, b_factors = voxelizer.map_atoms_to_voxel_space(
                    truth_residues=truth_residues,
                    autoencoder=autoencoder,
                    return_voxel_map=True,
                    return_serial=True,
                    return_b=True,
                    nClasses=self.nClasses)

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
                truth_residues=truth_residues,
                autoencoder=autoencoder,
                return_voxel_map=True,
                return_serial=True,
                return_b=True,
                nClasses=self.nClasses)

        if self.test:
            n_truth = len(truth) if truth is not None else len(data)
            domain_col = np.tile(self.embedding.transform([key.split("/")[-1]]), (n_truth,1))
            truth = np.append(truth if truth is not None else data, domain_col, axis=1)

        del truth_residues

        return voxelizer, indices, data, truth, voxel_map, serial, b_factors
