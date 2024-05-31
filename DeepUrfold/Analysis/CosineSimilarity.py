from itertools import permutations, combinations
from sklearn.metrics import pairwise_distances

import pandas as pd

import pickle

class CosineSimilarityDistance(object):
    def fit_transform(self, df):
        all_sfam_df = df.reset_index()
        all_dist = pairwise_distances(df.values, metric="cosine")

        all_dist = pd.DataFrame(all_dist, columns=all_sfam_df.cathDomain, index=all_sfam_df.cathDomain)
        
        Sc = {}
        for sfam_name, sfam_df in all_sfam_df.groupby("superfamily"):
            #sfam_dist = pairwise_distances(sfam_df.values, metric="cosine")
            idx = all_sfam_df[all_sfam_df.index.isin(sfam_df.cathDomain)].index

            sfam_dist = all_dist.iloc[idx, idx]

            Sc[sfam_name] = pd.Series(sfam_dist.mean(axis=1), names=sfam_df.cathDomain)

        with open()
        
        
        Sc_all = {}
        for class1, class2 in permutations(["1", "2", "3"], 2):
            Sc_all[(class1, class2)] = {}
            for sfam1, dist1 in Sc.items():
                if sfam1.startswith(class1):
                    for sfam2, dist2 in Sc.items():
                        if sfam2.startswith(class2):
                            sfam1_idx = all_sfam_df[all_sfam_df.cathDomain.isin(dist1.names)].index
                            sfam2_idx = all_sfam_df[all_sfam_df.cathDomain.isin(dist2.names)].index
                            all_vs_all_dist = all_dist.iloc[sfam1_idx, sfam2_idx]
                            Sc_all[(class1, class2)][(sfam1, sfam2)] = pd.Series(sfam_dist.mean(axis=1), names=sfam_df.cathDomain)

        