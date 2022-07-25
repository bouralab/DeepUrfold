import os, sys
import re
from glob import glob, iglob

import numpy as np
from Bio import SeqIO

from molmimic.generate_data import data_stores
from molmimic.util.pdb import get_pdb_residues
from molmimic.common.ProteinTables import three_to_one
from DeepUrfold.Datasets.DomainStructureDataset import DomainStructureDataset, merge_superfamilies, get_superfamilies, download_file
from molmimic.util.toil import map_job

from toil.realtimeLogger import RealtimeLogger

import pandas as pd
from pandarallel import pandarallel

N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))

def run_superfamily(job, sfam, data_dir, cores=8, memory="72G", **kwds):
    return DDIBindingSiteDataset.from_superfamily(sfam, data_dir, **kwds)

def merge_superfamilies_ddi(job, data_dir, merge_type="merge", cores=8, memory="72G"):
    assert merge_type in ["merge", "mask"]

    import dask
    import dask.dataframe as dd
    from multiprocessing.pool import Pool

    dask.config.set(scheduler="processes")
    dask.config.set(pool=Pool(cores-1))

    if merge_type == "merge":
        dataset = "DDIBindingSiteDataset-full-train.h5"
    else:
        dataset = "DDIBindingSiteDataset-full-train-mask.h5"

    all_rows = glob(str(os.path.join(data_dir, "train_files", "*", "*",
        "*", "*", dataset)))

    ddf = dd.read_hdf(all_rows, "table")
    ddf = ddf.repartition(npartitions=cores-1)

    out_file = str(os.path.join(data_dir, "train_files", dataset))

    data_columns = ["cathDomain"] if merge_type == "merge" else ["cathDomain", "secondCathCode"]

    ddf.to_hdf(out_file, "table", format="table", table=True, complevel=9,
            data_columns=data_columns, complib="bzip2", min_itemsize=4096)

class DDIBindingSiteDataset(DomainStructureDataset):
    def __init__(self, data_file, data_key="table", use_features=None, split_level="H", volume=256, nClasses=1):
        super().__init__(data_file, data_key=data_key, use_features=use_features,
            split_level=split_level, use_domain_index=True,
            structure_key="structure_file", feature_key="feature_file",
            truth_key="pdbBindingSite", volume=volume, nClasses=nClasses)

    # @classmethod
    # def from_superfamilies_directory(cls, data_dir, structure_dir=None, feature_dir=None, interaction_dir=None, split_level="H", with_sequences=False, job=None):
    #     sfams = get_superfamilies()
    #     full_data = sfams.parallel_apply(lambda sfam: cls.from_superfamily(
    #         sfam, data_dir, structure_dir=structure_dir, feature_dir=feature_dir,
    #         interaction_dir=interaction_dir, with_sequences=with_sequences, return_df=True, n_jobs=1))
    #     return cls(full_data)

    @staticmethod
    def from_superfamilies_directory(data_dir, structure_dir=None, feature_dir=None, interaction_dir=None, split_level="H", with_sequences=False, merge_type="merge", job=None, include_sfam=None, exclude_sfam=None, force=False):
        if include_sfam is None:
            sfams = get_superfamilies()
        else:
            sfams = include_sfams

        if exclude_sfam is not None:
            sfams = list(set(sfams)-set(exclude_sfam))

        if not force:
            sfams = [sfam for sfam in sfams if not os.path.isfile(os.path.join(data_dir,
            "train_files", *sfam.replace("/", ".").split("."),
            "DDIBindingSiteDataset-full-train.h5" if merge_type == "merge" else \
            "DomainStructureDataset-full-train-mask.h5"))]
        if len(sfams) == 0:
            return

        RealtimeLogger.info("N SFAMS: {} {}".format(len(sfams), sfams[:5]))

        if job is not None:
            #Run all in parallel
            map_job(job, run_superfamily, sfams, data_dir,
                structure_dir=structure_dir, feature_dir=feature_dir,
                interaction_dir=interaction_dir, split_level=split_level,
                with_sequences=with_sequences, return_df=True)

            job.addFollowOnJobFn(merge_superfamilies, data_dir)
            job.addFollowOnJobFn(merge_superfamilies_ddi, data_dir)

            #Return "return value promise" from job, which is a pandas dataframe. Since
            #It was run with toil, a regular Dataset will not be returned
            return

        else:
            pandarallel.initialize(nb_workers=n_jobs)
            full_data = sfams.parallel_apply(lambda sfam: DDIBindingSiteDataset.from_superfamily(
                sfam, data_dir, structure_dir=structure_dir, feature_dir=feature_dir,
                interaction_dir=interaction_dir, split_level=split_level,
                with_sequences=with_sequences, return_df=True, n_jobs=1))
            return DDIBindingSiteDataset(full_data)


    @staticmethod
    def from_superfamily(superfamily, data_dir, structure_dir=None, feature_dir=None, interaction_dir=None, split_level="H", with_sequences=False, use_features=None, return_df=False, n_jobs=-1):
        if isinstance(superfamily, str):
            superfamily = superfamily.replace("/", ".").split(".")
        superfamily = list(map(str, superfamily))

        if n_jobs == -1:
            n_jobs = N_JOBS

        #Get superfamily hierarchy and path to train files
        structures_and_features = DomainStructureDataset.from_superfamily(superfamily, data_dir,
            structure_dir=structure_dir, feature_dir=feature_dir,
            split_level=split_level,
            with_sequences=with_sequences, return_df=True, n_jobs=n_jobs)

        interaction_dir = interaction_dir if interaction_dir is not None else os.path.join(data_dir, "cath_interfaces")

        train_dir = os.path.join(data_dir, "train_files", *superfamily)
        full_train_file = os.path.join(train_dir, "DDIBindingSiteDataset-full-train.h5")
        # if os.path.isfile(full_train_file):
        #     out = DDIBindingSiteDataset(full_train_file, split_level=split_level, use_features=use_features)
        #     if return_df:
        #         return out.data
        #     return out

        if not os.path.isdir(train_dir):
            os.makedirs(train_dir)

        train_file = os.path.join(train_dir, "DDIBindingSiteDataset-train.h5")
        try:
            data = pd.read_hdf(train_file, "table")
        except (FileNotFoundError, KeyError):
            data = None
            for eppic_ddi_file in iglob(os.path.join(interaction_dir, *superfamily, "*_*ddi.h5")):
                df = pd.read_hdf(eppic_ddi_file, "table")
                df = df[df["reverse"]==False]
                if data is None:
                    data = df
                else:
                    data = pd.concat((data, df), axis=0)

            if data is None:
                if return_df:
                    return pd.DataFrame()
                return cls(pd.DataFrame(), split_level=split_level, use_features=use_features)

            data.to_hdf(train_file, "table", format="table")

        data = data[["firstCathDomain", "firstResi", "secondCathCode"]]
        data = data.rename(columns={"firstCathDomain":"cathDomain"})

        #Add in CATH hierarchy and paths to train files for the cath domains with binding site information
        full_data = pd.merge(data, structures_and_features, on="cathDomain")

        groups = full_data.groupby("cathDomain", as_index=False)
        ngroups = groups.ngroups

        if ngroups>0:
            #If multiple binidng sites on one domain, merge them
            if n_jobs != 1:
                pandarallel.initialize(nb_workers=n_jobs)
                full_data = groups.parallel_apply(
                    lambda df: merge_residues(df, with_sequences=with_sequences)
                ).set_index("cathDomain")
            else:
                full_data = groups.apply(
                    lambda df: merge_residues(df, with_sequences=with_sequences)
                ).set_index("cathDomain")

            # #Add in CATH hierarchy and paths to train files for the cath domains with binding site information
            # full_data = pd.merge(full_data, structures_and_features, left_index=True, right_on="cathDomain")
            #
            full_data.to_hdf(full_train_file, "table", format="table")

            if return_df:
                return full_data

            return cls(full_data, split_level=split_level, use_features=use_features)

        if return_df:
            return pd.DataFrame()

        return cls(pd.DataFrame(), split_level=split_level, use_features=use_features)

def merge_residues(df, with_sequences=None, include_pdb_residues=True):
    cath_domain = df.iloc[0]["cathDomain"]
    structure_file = df.iloc[0]["structure_file"]

    annotated_binding_site = df.iloc[0][['cathDomain', 'C', 'A', 'T', 'H', 'S35', 'S60', 'S95', 'S100',
       'structure_file', 'feature_file']]

    range_re = re.compile("(\-{0,1}[0-9]+[A-Z]{0,1})\-(\-{0,1}[0-9]+[A-Z]{0,1})")

    #Get all annotated binding sites for domain from EPPIC flattened into a single list
    pdbBindingSite = np.unique(df["firstResi"].str.split(",", expand=True).dropna().values.flatten())
    pdbBindingSite_merged = ",".join(pdbBindingSite.tolist())

    if include_pdb_residues and not with_sequences:
        #Return only pdb binding site info
        annotated_binding_site.at["pdbBindingSite"] = pdbBindingSite_merged
        return annotated_binding_site

    resn, resi = zip(*get_pdb_residues(structure_file, include_resn=True, use_raw_resi=True))
    resn = "".join(map(three_to_one, resn))

    true_resi = np.zeros(len(resn)).astype(int)
    for r in pdbBindingSite:
        try:
            index = resi.index(r)
            true_resi[index] = 1
        except (KeyError, IndexError, ValueError):
            print("{}({}) not in {}: {}".format(r.__class__.__name__, r, cath_domain, resi))

    if with_sequences:
        annotated_binding_site.at["sequence"] = resn
        annotated_binding_site.at["seqBindingSite"] = "".join(true_resi.astype(str).tolist())

    if include_pdb_residues:
        annotated_binding_site.at["pdbBindingSite"] = pdbBindingSite_merged

    # if sequence is not None:
    #     try:
    #         assert str(sequence.seq) == resn, "Domain sequence does not match input sequence: {} =/= {}".format(str(sequence.seq), resn)
    #
    #         #Initialize EPPIC API to read in mappings
    #         from molmimic.parsers.eppic import EPPICApi
    #         eppic_api = EPPICApi(cath_domain[:4].lower(), data_stores.eppic_store,
    #             data_stores.pdbe_store, use_representative_chains=False,
    #             work_dir=os.getcwd())
    #
    #         #Read in full protein mappign of pdbResidueNumber (str, resi and insertion code) -> residueNumber (0-based indexing) from EPPIC
    #         pdb_residues = eppic_api.get_residue_info(cath_domain[4])
    #         _pdb_residues = pdb_residues[["pdbResidueNumber", "residueNumber", "chain"]].drop_duplicates().dropna()
    #         pdb_residues = _pdb_residues.drop(columns=["chain"])
    #         pdb_residues = pdb_residues.set_index("pdbResidueNumber")
    #
    #         #CATH Domains can start in middle of full protein so index in domain sequence
    #         #is different than full length protein. Find the correct mapping from full
    #         #protein sequence to smaller domain sequence eg: >cath|domain/46-100, where 46->0
    #         resi_map = {}
    #         start = 0
    #         for srange in sequence.description.split("/")[-1].split("_"):
    #             m = range_re.match(srange.replace("(", "").replace(")", ""))
    #             if not m:
    #                 raise RuntimeError("{} {}".format(cath_domain, srange))
    #             lo, hi = m.group(1, 2)
    #
    #             try:
    #                 res = pdb_residues.loc[lo:hi]
    #             except KeyError as e:
    #                 raise RuntimeError("Invlaid key {} in {} {}".format(str(e), cath_domain, pdb_residues))
    #
    #             res = res.assign(domainNumbering=range(start, len(res)+start)).drop(columns=["residueNumber"])
    #             res = res.T.to_dict("records")[0]
    #             resi_map.update(res)
    #             start += len(res)
    #
    #         resi = pd.DataFrame({"binding_site": pdbBindingSite})
    #         resi = pd.merge(resi, pdb_residues, how="left", left_on="binding_site", right_index=True)
    #         resi = resi.sort_values("residueNumber")["binding_site"]
    #
    #         true_resi_from_seq = np.zeros(len(sequence.seq)).astype(int)
    #         for r in resi:
    #             try:
    #                 true_resi_from_seq[resi_map[r]] = 1
    #             except (KeyError, IndexError):
    #                 pass
    #
    #         annotated_binding_site.at["sequence_approach_same"] = str(true_resi_from_seq == true_resi)
    #
    #     except (SystemExit, KeyboardInterrupt):
    #         raise
    #     except Exception as e:
    #         annotated_binding_site.at["sequence_approach_same"] = repr(e)

    return annotated_binding_site

def mask_binding_site(structure, bindingSite, otherBindgingSites, num_samples=10000):
    surface = [r.id for r in structure.get_surface()]
    all_binding_sites = bindingSite+otherBindgingSites

    nis = [r for r in surface if r not in all_binding_sites]

    def calcScoreForResidueSet(residues):
        """COnverted fromn EPPIC"""
        totalScore = 0.0
        totalWeight = 0.0
        conservScores = structureresidue_features.loc[residues].residue_features["eppic_entropy"]
        for entropy in conservScores:
            if entropy > 0:
                weight = 1.0
                totalScore += weight*entropy
                totalWeight += weight
                return totalScore/totalWeight

    def calcZScore(bindingSiteScore, nisScore, nisStd):
        if nisStd != 0:
            zScore = (bindingSiteScore-nisScore)/std;
        else:
            if (bindingSiteScore-nisScore)>0:
                zScore = np.inf
            elif (coreScore-nisScore)<0:
                zScore = -np.inf
            else:
                zScore = np.nan
        return zScore

    bindingSite_masks = {}
    for otherBindgingSite in otherBindgingSites:
        bindingSiteScore = calcScoreForResidueSet(otherBindgingSite)
        mask_scores = {}
        for i in range(num_samples):
            mask = None
            while mask in mask_scores:
                mask = np.random.choice(nis, len(otherBindgingSite))

            mask_score = calcScoreForResidueSet(mask)
            mask_scores[mask] = mask_score

        mean = np.mean(mask_scores.values())
        std = np.std(mask_scores.values())
        zScore = calcZScore(bindingSiteScore, mean, std)

        best_mask = max(mask_scores.items(), key=lambda x: calcZScore(x[1]))
        bindingSite_masks[otherBindgingSite] = best_mask[1]

        pass

    modeller = MODELLER(work_dir=work_dir, job=job)
    modeller.mutate_structure(pdb_file, mutants, chain=cath_domain[4])
