import os, sys
from copy import deepcopy

import pandas as pd
from torch.utils.data import Dataset as _Dataset
from torch.utils.data import Subset

class Dataset(_Dataset):
    def __init__(self, data_file, data_key="table"):
        if isinstance(data_file, str) and os.path.isfile(data_file) and data_key is not None:
            self.data = pd.read_hdf(data_file, data_key)
        elif isinstance(data_file, pd.DataFrame):
            self.data = data_file
        else:
            raise RuntimeError("Incorrect data_file. Must be a valid path or Pandas DataFrame, not {}".format(data_file))

    def split(self, split_size, cluster_key, return_type="dataset"):
        if self.data is None:
            return None, None

        if cluster_key is not None:
            clusters = self.data.groupby(cluster_key)
            sorted_cluster_indices = list(sorted(clusters.indices.keys()))

            ideal_set1_size = int(clusters.ngroups*split_size)

            last_size = [None, None]
            while True:
                set1_clusters = sorted_cluster_indices[:ideal_set1_size]
                set1 = [idx for cluster in set1_clusters for idx in clusters.get_group(cluster).index] #.indices[cluster]]
                size_pct = len(set1)/(len(self.data))
                print("size", len(set1), len(self.data), size_pct, ideal_set1_size)
                if size_pct in last_size:
                    break
                if size_pct > split_size+.01:
                    ideal_set1_size -= 1
                elif size_pct < split_size-.01:
                    ideal_set1_size += 1
                else:
                    break

                last_size[0] = last_size[1]
                last_size[1] = size_pct

            set1 = list(sorted(set1))
            set1 = self.data.index.isin(set1)
            set2 = ~set1

            if return_type.lower() == "indices":
                return set1, set2.tolist()
            elif return_type == "subset":
                return Subset(self, set1), Subset(self, set2)
            elif return_type.lower() in ["dataset", "dataframe", "df"]:
                df1, df2 = self.data.loc[set1], self.data.loc[set2]
                if return_type in ["dataframe", "df"]:
                    return df1, df2
                else:
                    ds1 = deepcopy(self)
                    ds1.data = df1
                    ds2 = deepcopy(self)
                    ds2.data = df2
                    return ds1, ds2
            else:
                raise ValueError("Incorrect return type")
        else:
            indices = list(range(len(self.data)))
            set1_size = int(split_size * len(self.data))
            set2_size = int((1-split_size) * train_size)
            set3_start = train_size+val_size
            set1 = Subset(self, indices[:set1_size])
            set2 = Subset(self, indices[set1_size:set2_size])
            set3 = Subset(self, indices[set3_start:])
            return set1, (set2, set3)


    def to_hdf(self, path, key="table"):
        print(path, len(self.data))
        self.data.to_hdf(path, key, complevel=9, complib="bzip2", format="table")

    def __getitem__(self, index):
        return self.data.iloc[index]

    def __len__(self):
        return self.data.shape[0]

class ScaledDataset(Dataset):
    def __init__(self, data_file, data_key=None):
        super().__init__(data_file, data_key=data_key)

        from sklearn.preprocessing import StandardScaler
        from joblib import load, dump

        scale_file = "{}.scaler.joblib".format(os.getcwd())
        if os.path.isfile(scale_file):
            self.scaler = load(scale_file)
        else:
            self.scaled = False
            self.scaler = StandardScaler()
            for batch_num, batch in enumerate(self._get_data_loader(as_tensors=False)):
                print(batch.data)
                self.scaler.partial_fit(batch.data.reshape(-1, 1))
            self.scaled = True
            dump(self.scaler, scale_file)

    def __getitem__(self, index):
        return self.scaler.transform(sample["data"])
