import os, sys

import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import Subset

from imblearn.over_sampling import SMOTEN
from sklearn.neighbors import NearestNeighbors

class ImmutableNearestNeighbors(NearestNeighbors):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.data = None

    def first_fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), "X must be a DataFrame"
        self.data = X
        return super().fit(X)

    def fit(self, X, y=None):
        assert self.data is not None, "Must run first_fit before fit"
        return super().fit(self.data[X, X])

class ImbalancedDataset(Dataset):
    def correct_imbalance(self, combined_structures, separate_sfams=True):
        results_file = os.path.join(self.work_dir, "all_distances.csv")
        if not os.path.isfile(results_file):
            combined_structures_paths = combined_structures["structure_file"].tolist()
            distances = mc.all_vs_all(combined_structures_paths, table_out_file=results_file, distributed=96)
        else:
            distances = pd.read_csv(results_file)

        k_neighbors = ImmutableNearestNeighbors(metric="precomputed", n_jobs=96)
        k_neighbors.first_fit(distances)

        sampler = SMOTE(random_state=0, k_neighbors=k_neighbors, n_jobs=96)
        X_res, y_res = sampler.fit_resample(combined_structures.cathDomain, combined_structures.true_sfam)

        df_res = pd.DataFrame({"cathDomain":X_res, "true_sfam":y_res})

        if not separate_sfams:
            yield df_res
        else:
            for superfamily, sfam_group in df_res.groupby("true_sfam"):
                df_sfam_res = combined_structures[combined_structures.cathDomain.isin(sfam_group.cathDomain)]
                yield df_sfam_res
