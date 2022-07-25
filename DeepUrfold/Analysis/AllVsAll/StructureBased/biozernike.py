import os
import json
import argparse

import numpy as np
import pandas as pd
from numba import jit
from joblib import Parallel, delayed
from Bio.PDB.Polypeptide import three_to_one

from DeepUrfold.Analysis.AllVsAll.StructureBased import StructureBasedAllVsAll
from DeepUrfold.DataModules.DomainStructureDataModule import DomainStructureDataModule
from molmimic.parsers.superpose.tmalign import TMAlign

class BioZernikeAllVsAll(StructureBasedAllVsAll):
    NJOBS = 60
    METHOD = "BioZernike"
    DATA_INPUT = "CATH Superfamilies"
    SCORE_INCREASING = False
    MODEL_TYPE = "Structural Alignment"
    SCORE_METRIC = "TM-score"

    def train_all(self, *args, **kwds):
        """No Trianing needed for MaxCluster"""
        return None

    @staticmethod
    def distance(d1, d2):
        global config
        Ng = len(config["coefficients.geometry"])
        Nm = len(d1)-Ng
        Dg = sum([wgi*2*np.abs(g1i-g2i)/(1+np.abs(g1i)+np.abs(g2i)) for wgi, g1i, g2i in \
            zip(config["coefficients.geometry"], d1[:Ng], d2[:Ng])])

        m_coeffs = [c for i in range(5) for c in config[f"selected.coefficients.zernike.{i}"]]
        Dm = sum([wmi*np.abs(m1i-m2i) for wmi, m1i, m2i in zip(m_coeffs, d1[Ng:], d2[Ng:])])

        return Dg + Dm

    def infer_all(self, model, combined_structures):
        print(combined_structures)

        results_file = os.path.join(self.work_dir, "allpdbs.txt.biozernike_descriptors")
        if not os.path.isfile(results_file):
            assert 0
        else:
            descriptors = pd.read_csv(results_file, header=None)

        import pdb; pdb.set_trace()

        from sklearn.metrics import pairwise_distances

        distance_file = os.path.join(self.work_dir, "allpdbs.txt.biozernike_distances")
        if not os.path.isfile(distance_file):
            distances = pairwise_distances(descriptors.values, metric=biozernike_distance, n_jobs=128)
            np.save(distances, distance_file)
        else:
            distances = np.load(distance_file)

        descriptors = pd.concat((descriptors, self.representative_domains))

        import pdb; pdb.set_trace()
        distances = pd.DataFrame(distances, index=descriptors["cathDomain"], columns=descriptors["cathDomain"])

        sfam_groups = descriptors.groupby("superfamily")

        ordered_superfamilies = self.representative_domains["superfamily"].drop_duplicates().sort_values().tolist()

        results_df = pd.DataFrame(np.nan, index=descriptors["cathDomain"], columns=ordered_superfamilies)
        results_df = results_df.assign(cathDomain=descriptors["cathDomain"], true_sfam=descriptors["superfamily"])
        for sfam1_name in ordered_superfamilies:
            sfam1_domains = sfam_groups.get_group(sfam1_name)
            for sfam2_name in ordered_superfamilies:
                sfam2_domains = sfam_groups.get_group(sfam2_name)
                sfam1_domains_to_sfam2 = distances.loc[list(sfam1_domains["cathDomain"]), list(sfam2_domains["cathDomain"])].median(axis=1)
                results_df.loc[sfam1_domains["cathDomain"], sfam2_name] = sfam1_domains_to_sfam2

        results_df = results_df.reset_index(drop=True).set_index(["cathDomain", "true_sfam"])
        return results_df

        # rev = distances.copy(deep=True)
        # rev = rev.rename(columns={"PDB1":"_PDB2"}).rename(columns={"PDB2":"PDB1"}).rename(columns={"_PDB2":"PDB2"})
        # distances = pd.concat((distances, rev))


        #import pdb; pdb.set_trace()

        results = pd.DataFrame(np.nan, index=self.representative_domains["cathDomain"],
            columns=sorted(self.superfamily_datasets.keys()))
        results = pd.merge(results, self.representative_domains, left_index=True, right_on="cathDomain")
        results = results.rename(columns={"superfamily":"true_sfam"})
        results = results.set_index(["cathDomain", "true_sfam"])

        distances = pd.merge(distances, self.representative_domains, how="outer", left_on="chain1", right_on="cathDomain")
        distances = distances.rename(columns={"cathDomain":"cathDomain1", "superfamily":"superfamily1"})
        distances = pd.merge(distances, self.representative_domains, how="outer", left_on="chain2", right_on="cathDomain")
        distances = distances.rename(columns={"cathDomain":"cathDomain2", "superfamily":"superfamily2"})
        print(distances)
        domain_groups = distances.groupby(["cathDomain1", "superfamily1", "superfamily2"])

        score_type = "rmsd" if self.SCORE_METRIC=="RMSD" else "TM-Score"
        for (cathDomain, true_sfam, test_superfam), group in domain_groups:
            distance = group[score_type].median()
            results.loc[(cathDomain, true_sfam), test_superfam] = distance

        results = results.fillna(0.0)
        results = results.replace([np.inf, -np.inf], [1, 0])
        results = results.astype(np.float64)

        import pdb; pdb.set_trace()

        return results

config = {
  "max.order.zernike": 20,
  "max.order.zernike.align": 6,
  "coefficients.geometry": \
  [0.0, 5.51565593524, 9.66525104849, 4.93442839597, 2.50750917658, 1.55847471001, 1.9140171088, 0.582210891599, \
  3.10681711804, 1.38327861056, 0.0, 0.624017911283, 1.36067403053, 0.796482380717],
  "reference.radius": 60,
  "weight.geometry": 1.06512524918,
  "norm.orders.zernike ":  [0, 2, 3, 4, 5],
  "selected.indices.zernike.0": \
  [3, 6, 7, 8, 13, 14, 17, 22, 25, 26, 31, 32, 34, 38, 39, 43, 44, 45, 49, 50, 53, 60, 67, 71, 75, 80, 81, \
  87, 99, 106, 107, 109, 110, 112, 118, 120],
  "selected.coefficients.zernike.0": \
  [0.0485305205154, 0.129656911791, 0.0544215438012, 0.0904627554539, 0.138467259391, 0.0127114224533, 0.0744285053605, \
  0.0347966435104, 0.0292699548426, 0.0209688044696, 0.00576000917886, 0.0528232341656, 0.0041827217108, \
  0.121101476455, 0.0333995810977, 0.00825465197945, 0.0485726376168, 0.0319589200315, 0.0383237676073, \
  0.0145310417377, 0.0518025566985, 0.0826998498743, 0.00940666975537, 0.00269768189411, 0.000792392043519, \
  0.0146556219858, 0.00428441261224, 0.0319705762709, 0.0420487906384, 0.0423906555454, 0.0148767030518, \
  0.0455207501348, 0.0122999071965, 0.025959284531, 0.0203992549054, 0.139822226096],
  "selected.indices.zernike.1": \
  [7, 22, 35, 50, 51, 73, 95, 128, 150, 164, 181, 192, 228, 229, 239, 252, 296, 344, 345, 357, 430, 431, \
  494, 525, 599, 600, 700, 715, 806, 929, 930],
  "selected.coefficients.zernike.1": \
  [0.0896749555311, 0.0595701477555, 0.0290237258573, 0.0407762545118, 0.00697654885436, 0.0579447190062, \
  0.0473607662244, 0.038873665831, 0.0750085525205, 0.0202382877236, 0.0130166345472, 0.0305636067663, \
  0.0389196511771, 0.0409495289101, 0.0472516250436, 0.00222957364753, 0.00466859428376, 0.0282377352163, \
  0.068932166836, 0.0296967279378, 0.0179772856293, 0.00638452036181, 0.0295276044996, 0.0337430227512, \
  0.00407470558177, 0.0183763686585, 0.00120181125358, 0.0360522052161, 0.00536507695917, 0.0141921115845, \
  0.0295736292843],
  "selected.indices.zernike.2": \
  [22, 23, 50, 74, 192, 252, 295, 296, 297, 358, 415, 509, 510, 598, 599, 697, 698, 805, 806, 807, 925, \
  926, 927],
  "selected.coefficients.zernike.2": \
  [0.00624975693875, 0.00141147558631, 0.0263974859787, 0.0210416989132, 0.0176472096266, 0.030548870599, \
  0.11368614885, 0.0418389460803, 0.00252849062644, 0.0905180998502, 0.0441438929571, 0.0130318995468, \
  0.0265489568535, 0.0186450206393, 0.0430238863548, 0.0606296259822, 0.0179091588367, 0.0171391115986, \
  0.0336794573152, 0.0553883321747, 0.0220033085932, 0.0522521596253, 0.042683840434],
  "selected.indices.zernike.3": \
  [7, 22, 23, 35, 50, 71, 80, 96, 97, 115, 128, 151, 161, 163, 204, 229, 239, 295, 344, 357, 414, 415, 428, \
  447, 508, 525, 581, 597, 599, 680, 696, 698, 789, 805, 826, 927, 931, 932],
  "selected.coefficients.zernike.3": \
  [0.0392352388099, 0.033359373425, 0.0366907412906, 0.0130836937677, 0.0417873703434, 0.0562068893191, \
  0.00691486197401, 0.0971542873675, 0.0475746051222, 0.0142688790489, 0.0303360113657, 0.00627021566501, \
  0.0306308534106, 0.0398639076972, 0.0233423456892, 0.0467231227522, 0.00883569909909, 0.0291797514177, \
  0.0825369705239, 0.128674264594, 0.0242345003843, 0.0153059692094, 0.0564215894476, 0.00162234903812, \
  0.00432142431926, 0.0120460789175, 0.0595479639338, 0.0131824732908, 0.0463241426111, 0.00734691475906, \
  0.011282255628, 0.0240810581759, 0.00811955424876, 0.0491992903494, 1.10445158978e-05, 0.00520428262921, \
  0.00378974442419, 0.0215478368717],
  "selected.indices.zernike.4": \
  [7, 21, 37, 49, 51, 65, 78, 85, 119, 133, 162, 196, 244, 252, 253, 370, 373, 432, 506, 510, 526, 588, \
  618, 698, 704, 713, 715, 776, 820, 896, 923, 938, 942, 944],
  "selected.coefficients.zernike.4": \
  [0.0192989238354, 0.0154513810932, 0.011124166567, 0.0134660376157, 0.0879905088541, 0.00288220049881, \
  0.0210213602734, 0.0224982241052, 0.00375282211972, 0.0191605255891, 0.0218754450245, 0.041321105889, \
  0.0445119195272, 0.0311700477466, 0.0219529704878, 0.0182515051694, 0.0371986499609, 0.00733144975479, \
  0.0233777737361, 0.00252644009968, 0.0122538214664, 0.0111206899803, 0.0483388254258, 0.0367987397017, \
  0.00259695799344, 0.0100739934332, 0.00845897027182, 0.00220571492155, 2.13277674485e-06, 0.00419638821352, \
  0.0439447552148, 0.0215500965333, 0.0111575243991, 0.0279042810281],
  "num.threshold.sets": 2,
  "threshold.geometry.0": 6.6,
  "threshold.zernike.0": 9,
  "threshold.geometry.1": 11.6,
  "threshold.zernike.1": 14.6
}
m_coeffs = np.array([c for i in range(5) for c in config[f"selected.coefficients.zernike.{i}"]])
g_coeffs = np.array(config["coefficients.geometry"])

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def biozernike_distance(d1, d2, g_weights=g_coeffs, m_weights=m_coeffs): # Function is compiled to machine code when called the first time
    Ng = g_weights.shape[0]
    Dg = np.sum(g_weights*np.abs(d1[:Ng]-d2[:Ng])/(1+np.abs(d1[:Ng])+np.abs(d2[:Ng])))
    Dm = np.sum(m_weights*np.abs(d1[Ng:]-d2[Ng:]))
    return Dg + Dm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Superfamily models and perform an all vs all comparison between a set of domains")
    #parser.add_argument("-d", "--data_dir", default="/home/bournelab/data-eppic-cath-features/", required=False)
    parser.add_argument("-w", "--work_dir", default=os.getcwd(), required=False)
    parser.add_argument("-p", "--permutation_dir", default="/home/bournelab/urfold_runs/multiple_loop_permutations/sh3_3", required=False)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("ava_superfamily", nargs="+")
    #parser = DomainStructureDataModule.add_data_specific_args(parser)
    parser = DomainStructureDataModule.add_data_specific_args(parser, eval=True)
    parser.set_defaults(
        data_dir="/home/bournelab/data-eppic-cath-features/",
        all_domains=True)
    args = parser.parse_args()
    runner = BioZernikeAllVsAll(args.ava_superfamily, args.data_dir, args, permutation_dir=args.permutation_dir,
        work_dir=args.work_dir, force=args.force)
    runner.run()
