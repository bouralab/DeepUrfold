import os
import sys
import shutil
import tempfile
import contextlib
import subprocess
from pathlib import Path

import click
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm


import joblib
from joblib import Parallel, delayed

from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import PDBxFile, set_structure, get_structure
from sklearn.preprocessing import MinMaxScaler
from biotite.structure import apply_residue_wise, get_residue_starts

from Prop3D.parsers.superpose.tmalign import TMAlign

@contextlib.contextmanager
def tqdm_joblib(tqdm_object=None, desc=None, total=None):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    if tqdm_object is None and total is None:
        raise RuntimeError("Must input either tqdm_object or total")
    elif tqdm_object is None:
        tqdm_object = tqdm(desc=desc, total=total)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def process_lrp(superfamily, data_path):
    """Now built into LRPSaver"""
    superfamily_path = Path(data_path) / superfamily.replace('/', '.') / "lrp"
    for pdb_file in superfamily_path.rglob("*-v_agg=arithmetic_mean__f_agg=npsum-total_relevance.pdb"):
        #Read in protein, set new field of resi+ins_code
        protein = PDBFile.read(str(pdb_file)).get_structure(model=1, extra_fields=["b_factor"], altloc='first')
        all_res_ids = protein.res_id.astype(str)+protein.ins_code
        res_ids = np.unique(protein.res_id.astype(str)+protein.ins_code)
        protein.add_annotation("res_name_index", all_res_ids)

        #Sum up relevance values for all atoms in each residue, extract residues with a score > 0.75 of quantile
        total_per_residue_sum = apply_residue_wise(protein, protein.b_factor, np.nansum)
        threshold = np.quantile(total_per_residue, 0.75)
        best_res_ids = res_ids[total_per_residue_sum >= np.quantile(total_per_residue, 0.75)]
        best_structure = protein[np.isin(protein.res_name_index, best_res_ids)]

        pdb_file = PDBFile()
        pdb_file.set_structure(best_structure)
        pdb_file.lines.insert(0, f"REMARK Scores={';'.join([f'{i:0.4f}' for i in total_per_residue_sum])}")
        pdb_file.lines.insert(0, f"REMARK Threshold={threshold:0.4f} (75% quantile)")
        pdb_file.lines.insert(0, f"REMARK Residues={';'.join(best_res_ids)}")
        pdb_file.write(str(pdb_file.with_suffix(".75pctquntile.pdb")))

def cluster_lrp_full(data_path, distributed):
    data_path = Path(data_path)
    print(data_path)
    def get_subregions(ff):
        return [str(f) for f in (ff / "lrp").glob("*/*.75pctquntile.pdb")]
    all_subregions = Parallel(n_jobs=-1)(delayed(get_subregions)(f) for i, f in enumerate(data_path.glob("*")) if f.is_dir())
    all_subregions = [f for dir in all_subregions for f in dir][:10]
    #all_subregions = [str(f) for f in data_path.glob("*/lrp/*/*.75pctquntile.pdb")]
    print("# Subregions", len(all_subregions), "=?=", 3674*20)
    tmalign = TMAlign(work_dir=str(data_path))
    df = tmalign.cluster(all_subregions, distributed=distributed)
    df.to_csv("all_clusters.csv")

def save_as_bfactors(data_path):
    data_path = Path(data_path)
    def get_h5(ff):
        return [f for f in (ff / "lrp").glob("*.h5")]
    all_h5 = Parallel(n_jobs=-1)(delayed(get_h5)(f) for i, f in enumerate(data_path.glob("*")) if f.is_dir())
    all_h5 = [f for ff in all_h5 for f in ff]

    def process_pdb(h5_file):
        pdb_name = h5_file.stem.split("-")[0]

        df = pd.read_hdf(h5_file, pdb_name)


        pdb_dir = h5_file.parent / pdb_name
        print(pdb_dir)
        print(list(pdb_dir.glob("*")))
        pdb_file = [f for f  in pdb_dir.glob("*.pdb") if "75pct" not in f.name][0]
        protein = PDBFile.read(str(pdb_file)).get_structure(model=1, extra_fields=["b_factor", "atom_id"], altloc='first')
        all_res_ids = np.core.defchararray.add(protein.res_id.astype(str), protein.ins_code)
        print(all_res_ids)
        res_ids = np.unique(all_res_ids)
        #protein.add_annotation("res_name_index", all_res_ids.tolist())
        protein.set_annotation("b_factor", df.total_relevance.values.tolist())

        pdb_file.rename(pdb_file.with_suffix('.pdb.old'))
        [f.unlink() for f in pdb_dir.glob("*.pdb") if "75pct" in f.name]

        #Save full relevances
        pdbx_file = PDBxFile()
        set_structure(pdbx_file, protein, data_block="structure")
        pdbx_file.write(str(pdb_file.with_suffix(".cif")))

        #Sum up relevance values for all atoms in each residue, extract residues with a score > 0.75 of quantile
        total_per_residue_sum = apply_residue_wise(protein, protein.b_factor, np.nansum)
        threshold = np.quantile(total_per_residue_sum, 0.80)
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
        small_pdb_file.write(str(pdb_file.with_suffix(".75percentile.scaled.pdb")))

        custom_params = {"axes.spines.right": False, "axes.spines.top": False}
        sns.set_theme(style="ticks", rc=custom_params)

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        sns.kdeplot(df.total_relevance.values)
        atom_threshold = np.quantile(df.total_relevance.values, 0.80)
        ax.axvspan(atom_threshold, df.total_relevance.values.max(), facecolor='gray', alpha=0.5)
        ax.plot([], [], ' ', label=f"Min: {df.total_relevance.values.min()}")
        ax.plot([], [], ' ', label=f"Max: {df.total_relevance.values.max()}")
        ax.plot([], [], ' ', label=f"80th percentile: {atom_threshold}")
        ax.set_title(f"{pdb_name} through {h5_file.parent.parent.stem} model (Atom wise)")
        plt.legend()
        plt.savefig(str(pdb_file.with_suffix(".atom_wise.pdf")), dpi=300)
        plt.clf()

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        sns.kdeplot(total_per_residue_sum)
        ax.axvspan(threshold, total_per_residue_sum.max(), facecolor='gray', alpha=0.5)
        ax.plot([], [], ' ', label=f"Min: {total_per_residue_sum.min()}")
        ax.plot([], [], ' ', label=f"Max: {total_per_residue_sum.max()}")
        ax.plot([], [], ' ', label=f"80th percentile: {threshold}")
        ax.set_title(f"{pdb_name} through {h5_file.absolute().parent.parent.stem} model (Residue wise)")
        plt.legend()
        plt.savefig(str(pdb_file.with_suffix(".residue_wise.pdf")), dpi=300)
        plt.clf()

    # all_h5 = [Path("/media/smb-rivanna/ed4bu/UrfoldServer/urfold_runs/superfamilies_for_paper_latest/lrp/3.40.50.720/lrp/3tjrA00-v_agg=arithmetic_mean__f_agg=npsum.h5")]
    # print(all_h5)
    with tqdm_joblib(total=len(all_h5)):
        Parallel(n_jobs=-1)(delayed(process_pdb)(f) for f in all_h5)

def copy_files(data_path, superfamilies, domains):
    data_path = Path(data_path)

    for model_sfam in superfamilies:
        sfam_dir = data_path / model_sfam / "lrp"
        sfam_out_dir = Path.cwd() / model_sfam
        sfam_out_dir.mkdir(exist_ok=True)
        for domain in domains:
            domain_dir = sfam_dir / domain
            print()
            print()
            print(domain_dir)
            print(list(domain_dir.glob("*.cif")))

            try:
                full_lrp_file = [d for d in domain_dir.glob("*.cif") if "75percentile" not in d.name and "scaled" not in d.name][0]
            except IndexError:
                continue

            print("Using full lrp file:", full_lrp_file)
            fragment_lrp = full_lrp_file.with_suffix(".75percentile.scaled.pdb")
            print("Using fragment lrp file:", fragment_lrp)

            domain_sfam = full_lrp_file.name.split("-")[0].replace("-", ".")
            domain_sfam_out_dir = sfam_out_dir / domain_sfam
            domain_sfam_out_dir.mkdir(exist_ok=True)

            shutil.copyfile(str(full_lrp_file), str(domain_sfam_out_dir/f"{domain}.full_lrp.cif"))
            shutil.copyfile(str(fragment_lrp), str(domain_sfam_out_dir/f"{domain}.fragemnt_lrp.pdb"))

def run_geomtricus(data_path, resolution=8):
    import umap
    import prody
    from geometricus import MomentInvariants, SplitType, GeometricusEmbedding
    data_path = Path(data_path)
    def get_superfamily_reps(ff):
        invariants = []
        sfam = ff.name.replace(".", "_")
        with open(f"{sfam}_invariants.csv", "w") as inv_file:
            print("pdb_file,invariant,type", file=inv_file)
            for f in (ff / "lrp").rglob("*.cif"):
                if f.name.startswith(sfam) and not "75" in f.name:
                    print("Running CIF", f)
                    atom_array = get_structure(PDBxFile.read(str(f)))
                    invariants_kmer = MomentInvariants.from_coordinates(f.stem, atom_array.coord[0], split_type=SplitType.KMER, split_size=16)
                    invariants.append(invariants_kmer)
                    kmer_moments = (np.log1p(invariants_kmer.moments) * resolution).astype(int)
                    for x in kmer_moments:
                        print(f, f"k{x[0]}i{x[1]}i{x[2]}i{x[3]}", "kmer", sep=",", file=inv_file)

                    print(atom_array.coord[0])
                    invariants_radius = MomentInvariants.from_coordinates(f.stem, atom_array.coord[0], split_type=SplitType.RADIUS, split_size=10)
                    invariants.append(invariants_radius)
                    kmer_moments = (np.log1p(invariants_radius.moments) * resolution).astype(int)
                    for x in kmer_moments:
                        print(f, f"k{x[0]}i{x[1]}i{x[2]}i{x[3]}", "radius", sep=",", file=inv_file)

        return invariants

    invariants = Parallel(n_jobs=-1, backend="threading")(delayed(get_superfamily_reps)(f) for i, f in enumerate(data_path.glob("*")) if f.is_dir())
    all_invariants = [f for ff in invariants for f in ff]

    embedder = GeometricusEmbedding.from_invariants(all_invariants, resolution=resolution)
    reducer = umap.UMAP(metric="cosine", n_components=2)
    reduced = reducer.fit_transform(embedder.embedding)

    np.save("reduced_geometricus_embedding.npy", reduced)

def read_clusters_and_process(cluster_file, process_node=None, process_leaf=None, n_jobs=None):
    with open(cluster_file) as f:
        header = next(f).rstrip().split(",")
        header[0] = "id"
        data = [l.rstrip().split(",") for l in f if l.count(",")+1==len(header)]

    df = pd.DataFrame(data, columns=header)
    df = df.rename(columns={"sfam":"true_sfam"})
    levels = df["id"].str.split(".", expand=True)
    levels = levels.drop(columns=[len(levels.columns)-1])
    levels = levels.rename(columns={l:f"Level {l}" for l in levels.columns}).set_index(df.index)
    df = pd.concat((levels, df), axis=1)
    df = df.set_index(["cathDomain", "true_sfam"])
    print(df)

    if process_node is None and process_leaf is None:
        return df

    if process_node is None:
        process_node = lambda c,n: None

    if process_leaf is None:
        process_leaf = lambda c,n: None

    def process_hierarchy(h, level=None, name=None):
        levels = sorted([c for c in h.columns if "Level" in c], key=lambda n: int(n.split()[1]), reverse=False)+["END"]
        assert level is None or level in levels
        if level is None:
            print("Running start with", levels[0])
            return process_hierarchy(h, levels[0], "root")
        elif level == levels[-1]:
            print("Running last level", level)
            return process_leaf(h, name)

        print("Running level", level)

        hgroups = h.groupby(levels[:levels.index(level)+1], as_index=False)
        next_level = levels[levels.index(level)+1]

        if n_jobs is None:
            children = [process_hierarchy(hierarchy, next_level, level_name) \
                for level_name, hierarchy in hgroups]
        else:
            print("Running", len(hgroups), "with", n_jobs, "cores")
            children = Parallel(n_jobs=n_jobs)(delayed(process_hierarchy)( \
                hierarchy, next_level, level_name) for level_name, hierarchy in hgroups)

        return process_node(children, name)

    return process_hierarchy(df)

def align_lrp_from_community(cluster_file, data_dir):
    from Prop3D.parsers.superpose.tmalign import TMAlign
    from Prop3D.parsers.mmseqs import MMSeqs
    data_dir = Path(data_dir)
    def process_leaf(h, level_name):
        print("Level", level_name)
        all_lrp_files = []
        sfams_in_group = h.index.get_level_values(1).drop_duplicates()
        print(sfams_in_group)
        for sfam_group in sfams_in_group:
            sfam_group_dir = data_dir / sfam_group / "lrp"
            for cathDomain, sfam in h.index:
                lrp_file = next((sfam_group_dir / cathDomain).glob("*75percentile.scaled.pdb"))
                assert lrp_file.is_file()
                all_lrp_files.append(str(lrp_file))

        print("Aligning", len(all_lrp_files), "fragments from", len(h.index), "domains and", len(sfams_in_group), "superfamilies")
        all_lrp_files = all_lrp_files[:10]
        tmalign = TMAlign(work_dir=str(data_dir))
        similairties, clusters = tmalign.cluster(all_lrp_files, table_out_file=f"LRP_cluster_{'.'.join(level_name)}", distributed=36, force=True)

        results_file = f"LRP_cluster_{'.'.join(level_name)}_clusters.csv"
        clusters.to_csv(results_file)

    def get_domain_in_community(h, level_name):
        print("Level", level_name)
        all_lrp_files = []
        sfams_in_group = h.index.get_level_values(1).drop_duplicates()
        print(sfams_in_group)

        cluster_dir = Path.cwd() / f"LRP_cluster_{'.'.join(level_name)}"
        cluster_dir.mkdir(exist_ok=True)

        for sfam_group in sfams_in_group:
            sfam_group_dir = data_dir / sfam_group / "lrp"
            for cathDomain, sfam in h.index:
                lrp_file = next((sfam_group_dir / cathDomain).glob("*75percentile.scaled.pdb"))
                assert lrp_file.is_file()
                print("Saving to",  cluster_dir / f'{sfam_group}_{cathDomain}.pdb')
                shutil.copyfile(str(lrp_file), str( cluster_dir / f'{sfam_group}_{cathDomain}.pdb' ))
                #all_lrp_files.append(str(lrp_file))


        return f"LRP_cluster_{'.'.join(level_name)}", all_lrp_files

    df = read_clusters_and_process(cluster_file, process_leaf=None, process_node=None, n_jobs=16)
    groups = df.groupby([c for c in df.columns if "Level" in c])
    #Parallel(n_jobs=1)(delayed(process_leaf)(group, name) for name, group in groups)
    # #groups = [next(iter(groups))]
    #domains = list(Parallel(n_jobs=16)(delayed(get_domain_in_community)(group, name) for name, group in groups))

    for i, (level_name, _) in enumerate(groups):
        level_name = f"LRP_cluster_{'.'.join(level_name)}"

        level_name_dir = Path.cwd() / f"{level_name}_foldseek" / f"{level_name}"
        level_name_dir.parent.mkdir(exist_ok=True)

        #if not level_name == "LRP_cluster_0.0.1.5.34": continue

        createdb_args = [
            "/home/bournelab/foldseek/build/bin/foldseek",
            "createdb",
            level_name,
            f"{level_name_dir}_db"
        ]
        subprocess.run(createdb_args)

        # for index_name in ('', '_ca', '_h', '_ss'):
        #     MMSeqs.fake_pref(f"{level_name_dir}_db{index_name}", f"{level_name_dir}_db{index_name}", f"{level_name_dir}_db_all{index_name}")
        #     shutil.copy("{level_name_dir}_db{index_name}.lookup", f"{level_name_dir}_db_all{index_name}).lookup")
        #     shutil.copy("{level_name_dir}_db{index_name}.source", f"{level_name_dir}_db_all{index_name}).source")
        #     if index_name != "":
        #         shutil.copy("{level_name_dir}_db{index_name}.dbtype", f"{level_name_dir}_db_all{index_name}).dbtype")

        search_args = [
            "/home/bournelab/foldseek/build/bin/foldseek",
            "search",
            f"{level_name_dir}_db",
            f"{level_name_dir}_db",
            #f"{level_name_dir}_db",
            #f"{level_name_dir}_db",
            f"{level_name_dir}_aln",
            f"/home/bournelab/foldseqtmp_{level_name}",
            "--alignment-type", "1",
            "-c", "0",
            "-e", "1000",
            "--cov-mode", "0",
            "-s", "100",
            "-a", "1"
        ]
        subprocess.run(search_args)

        cluster_args = [
            "/home/bournelab/foldseek/build/bin/foldseek",
            "clust",
            f"{level_name_dir}_db",
            #f"{level_name_dir}_db",
            f"{level_name_dir}_aln",
            f"{level_name_dir}_clu"
        ]
        subprocess.run(cluster_args)

        cluster_tsv_args = [
            "/home/bournelab/foldseek/build/bin/foldseek",
            "createtsv",
            f"{level_name_dir}_db",
            f"{level_name_dir}_db",
            #f"{level_name_dir}_db",
            #f"{level_name_dir}_db",
            f"{level_name_dir}_clu",
            f"{level_name_dir}_clu.tsv"
        ]
        subprocess.run(cluster_tsv_args)

        ###
        # Get rotation matrices
        ###
        aln2tmscore_args = [
            "/home/bournelab/foldseek/build/bin/foldseek",
            "aln2tmscore",
            f"{level_name_dir}_db",
            f"{level_name_dir}_db",
            #f"{level_name_dir}_db",
            #f"{level_name_dir}_db",
            f"{level_name_dir}_aln",
            f"{level_name_dir}_aln_tmscore"
        ]
        subprocess.run(aln2tmscore_args)

        tmscore_tsv_args  = [
            "/home/bournelab/foldseek/build/bin/foldseek",
            "createtsv",
            f"{level_name_dir}_db",
            f"{level_name_dir}_db",
            #f"{level_name_dir}_db",
            #f"{level_name_dir}_db",
            f"{level_name_dir}_aln_tmscore",
            f"{level_name_dir}_aln_tmscore.tsv"
        ]
        subprocess.run(tmscore_tsv_args)

        # search_cmd.communicate()
        # search_cmd.wait()

        #cluster_cmd = subprocess.run(cluster_args)
        # cluster_cmd.communicate()
        # cluster_cmd.wait()

        #tsv_cmd = subprocess.run(tsv_args)
        # tsv_cmd.communicate()
        # tsv_cmd.wait()
    # # group_names, domains_by_group = zip(*domains)

    # tmalign = TMAlign(work_dir=str(data_dir))
    # # similairties_clusters = tmalign.all_vs_all(domains_by_group, table_out_file=group_names,
    # #     cluster=0.2, distributed=36, force=True, many_independent_jobs=True)

    # for i, (name, group) in enumerate(groups):
    #     if i==0: continue
    #     group_name, domains_by_group = get_domain_in_community(group, name)
    #     tmalign.all_vs_all(domains_by_group[:50], table_out_file=group_name,
    #         cluster=0.2, distributed=36, force=True)

def analyze_cluster(cluster_name, radius=2.0, pct_frags_in_urfold=0):
    from biotite.structure import CellList
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure.util import matrix_rotate

    cluster_name = '.'.join(cluster_name)

    clusters = pd.read_csv(f"LRP_cluster_{cluster_name}_foldseek_clu.tsv", header=None, names=["cluster_rep", "lrp_domain"], sep="\t")
    best_cluster = clusters.groupby("cluster_rep").apply(len).argmax()

    alignments = pd.read_csv(f"LRP_cluster_{cluster_name}_foldseek_aln_tmscore.tsv", delim_whitespace=True, header=None,
        names=["query", "target", "tm", "t1", "t2", "t3", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"])
    alignments = alignments.set_index(["query", "target"])

    print(clusters)

    for cluster_rep, fragments in clusters.groupby("cluster_rep"):
        cluster_rep_array = PDBFile.read(os.path.join(f"LRP_cluster_{cluster_name}", cluster_rep)).get_structure(extra_fields=["atom_id"], model=1)
        cluster_rep_array_cell_list = CellList(cluster_rep_array, cell_size=5)
        all_nearby = []
        print("Cluster rep", cluster_rep)
        if len(fragments) == 1:
            urfold_file = PDBFile()
            urfold_file.set_structure(cluster_rep_array)
            urfold_file.write(f"LRP_cluster_{cluster_name}_foldseek_{Path(cluster_rep).stem}_singleton.pdb")
            continue

        for fragment in fragments.lrp_domain:
            if cluster_rep == fragment: continue
            fragment_array = PDBFile.read(os.path.join(f"LRP_cluster_{cluster_name}", fragment)).get_structure(model=1)
            try:
                translation = alignments.loc[(cluster_rep, fragment)][["t1", "t2", "t3"]].values
                rotation = alignments.loc[(cluster_rep, fragment)][["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"]].values.reshape((3,3))
                aligned_coords = matrix_rotate(fragment_array.coord, rotation)+translation
                near_atoms = set(a for aa in cluster_rep_array[cluster_rep_array_cell_list.get_atoms(aligned_coords, radius=radius)].atom_id.tolist() for a in aa)
            except KeyError:
                print((cluster_name, cluster_rep, fragment))
                try:
                    translation = alignments.loc[(fragment, cluster_rep)][["t1", "t2", "t3"]].values
                    rotation = alignments.loc[(fragment, cluster_rep)][["r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"]].values.reshape((3,3))
                    aligned_coords = matrix_rotate(cluster_rep_array.coord, rotation)+translation
                    cluster_rep_array2 = cluster_rep_array.copy()
                    cluster_rep_array2.coords = aligned_coords
                    target_cell_list2 = CellList(cluster_rep_array2, cell_size=5)
                    near_atoms = set(a for aa in cluster_rep_array2[target_cell_list2.get_atoms(fragment_array.coord, radius=radius)].atom_id.tolist() for a in aa)
                except KeyError:
                    continue

            if len(near_atoms) == 0:
                print((cluster_rep, fragment))
                continue
            print(near_atoms)
            all_nearby.append(near_atoms)

        if len(all_nearby) == 0:
            with open(f"LRP_cluster_{cluster_name}_foldseek_{Path(cluster_rep).stem}_members{len(fragments)}.pdb"):
                pass

        urfold_atoms = []
        all_nearby_atoms = set([atom for atoms in all_nearby for atom in atoms]) #set.union(*all_nearby)
        for i, nearby_atom in enumerate(all_nearby_atoms):
            num_with_atom = 0
            for j, nearby_frag in enumerate(all_nearby):
                num_with_atom += bool(nearby_atom in nearby_frag)

            if num_with_atom/len(all_nearby) >= pct_frags_in_urfold:
                urfold_atoms.append(nearby_atom)

        urfold = cluster_rep_array[np.isin(cluster_rep_array.atom_id, urfold_atoms)]
        urfold_file = PDBFile()
        urfold_file.set_structure(urfold)
        urfold_file.write(f"LRP_cluster_{cluster_name}_foldseek_urfold_{Path(cluster_rep).stem}_members{len(fragments)}.pdb")
        print(f"Wrote LRP_cluster_{cluster_name}_foldseek_urfold_{Path(cluster_rep).stem}_members{len(fragments)}.pdb")


def analyze_all_clusters(cluster_file, data_dir):
    df = read_clusters_and_process(cluster_file, process_leaf=None, process_node=None, n_jobs=16)
    groups = df.groupby([c for c in df.columns if "Level" in c])

    with tqdm_joblib(total=len(groups)):
        Parallel(n_jobs=32)(delayed(analyze_cluster)(name) for name, group in groups)


import open3d as o3d
PointCloud = o3d.geometry.PointCloud
Vector3dVector = o3d.utility.Vector3dVector
Feature = o3d.pipelines.registration.Feature
read_point_cloud = o3d.io.read_point_cloud
registration_ransac_based_on_feature_matching = o3d.pipelines.registration.registration_ransac_based_on_feature_matching
registration_icp = o3d.pipelines.registration.registration_icp
TransformationEstimationPointToPoint = o3d.pipelines.registration.TransformationEstimationPointToPoint
CorrespondenceCheckerBasedOnEdgeLength = o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength
CorrespondenceCheckerBasedOnDistance = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance
CorrespondenceCheckerBasedOnNormal = o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal
TransformationEstimationPointToPlane = o3d.pipelines.registration.TransformationEstimationPointToPlane
RANSACConvergenceCriteria = o3d.pipelines.registration.RANSACConvergenceCriteria
KDTreeFlann = o3d.geometry.KDTreeFlann

def get_point_cloud(f):
    points = PointCloud()
    features = Feature()
    with h5py.File(f, "r") as f:
        points.points = Vector3dVector(f["best_indices"][:])
        features.data = f["best_relevance_scaled"][:].T # need transpose? descriptors[idx, :].T
        #print(features.data)

    points.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    
    return points, features

def ransac_align(pc1, pc2, feats1, feats2, ransac_radius=5.0, random_seed=42, ransac_iter=2000):
    # pc1, feats1 = d1 #get_point_cloud(d1)
    # pc2, feats2 = d2 #get_point_cloud(d2)

    print("start")

    print(pc1)
    print(feats1)
    print(np.asarray(pc1.points))
    print(np.asarray(feats1.data))
    print(np.asarray(feats1.data).min())

    result = registration_ransac_based_on_feature_matching(
        pc1,
        pc2,
        feats1,
        feats2,
        True,
        ransac_radius,
        TransformationEstimationPointToPoint(False),
        3,
        [
            CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(2.0),
            #CorrespondenceCheckerBasedOnNormal(np.pi / 2),
        ],
        RANSACConvergenceCriteria(100000, 0.999), #ransac_iter, 500), 
    )

    print("result")

    result = registration_icp(
        pc1, 
        pc2, 
        1.0, 
        result.transformation, 
        TransformationEstimationPointToPlane()
    )

    return result.inlier_rmse, result.transformation
    
    print(result, result.transformation)
    
#     source, target, threshold, trans_init,
#     o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    return result

def align_community_ransac(level_name, group, data_dir):
    from sklearn.cluster import AgglomerativeClustering
    level_name = f"LRP_cluster_{'.'.join(level_name)}"
    level_name_dir = Path.cwd() / f"{level_name}_registration" / f"{level_name}"
    level_name_dir.parent.mkdir(exist_ok=True)

    sfams_in_group = group.index.get_level_values(1).drop_duplicates()

    data_dir = Path(data_dir)

    lrp_structures = []
    for sfam_group in sfams_in_group:
        for cathDomain, sfam in group.index:
            sfam_group_dir = data_dir / sfam_group / "lrp"
            lrp_file = next((sfam_group_dir / cathDomain).glob("*.h5"))
            lrp_structures.append(lrp_file)

    lrp_structures = lrp_structures[:2]

    distances, transformations = align_structures_ransac(lrp_structures)
    np.save(distances, f"{level_name_dir}_distances.npy")
    np.save(distances, f"{level_name_dir}_transformations.npy")

    cluster_assignments = AgglomerativeClustering(linkage="single", affinity='precomputed').fit_predict(distances)
    df = pd.DataFrame({"structure":lrp_structures, "cluster":cluster_assignments, "centroid":np.nan})

    for cluster in range(cluster_assignments.n_clusters_):
        domain_idx = np.where(cluster_assignments.labels_==cluster)[0]
        cluster_distances = distances[domain_idx,domain_idx]
        centroid = np.linalg.norm(cluster_distances-cluster_distances.mean(), axis=1).argmin(axis=1)
        df.loc[df[df.cluster==cluster].index, "centroid"] = centroid
    
    cluster_assignments.to_csv(f"{level_name_dir}_clusters.npy")

def align_structures_ransac(lrp_structures):
    point_clouds = []
    features = []
    for s in lrp_structures:
        pc, feats = get_point_cloud(s)
        point_clouds.append(pc)
        features.append(feats)
    
    distances = np.empty((len(point_clouds),len(point_clouds)))
    transformations = np.empty((len(point_clouds)*len(point_clouds), 16))
    for s1 in range(len(point_clouds)):
        for s2 in range(len(point_clouds)):
            if s1==s2: continue
            rmse, transformation = ransac_align(point_clouds[0], point_clouds[1], features[0], features[1])
            distances[s1,s2] = rmse
            transformations[s1*len(point_clouds)+s2] = transformation.flatten()
    
    return distances, transformations

def analyze_all_clusters_ransac(cluster_file, data_dir):
    df = read_clusters_and_process(cluster_file, process_leaf=None, process_node=None, n_jobs=16)
    groups = df.groupby([c for c in df.columns if "Level" in c])

    # with tqdm_joblib(total=len(groups)):
    #     Parallel(n_jobs=1)(delayed(align_community_ransac)(name, group, data_dir) for name, group in groups)

    for name, group in groups:
        align_community_ransac(name, group, data_dir)
        break

if __name__ == "__main__":
    #Always data_dir, cluster_file
    if sys.argv[1] == "community_lrp":
        align_lrp_from_community(sys.argv[3], sys.argv[2])
    if sys.argv[1] == "analyze_community_lrp":
        analyze_all_clusters(sys.argv[3], sys.argv[2])
    elif sys.argv[1] == "geometricus":
        run_geomtricus(sys.argv[2])
    elif sys.argv[1] == "analyze_all_clusters_ransac":
        analyze_all_clusters_ransac(sys.argv[3], sys.argv[2])

    #copy_files(sys.argv[1], ["3.40.50.720", "3.40.50.300", "2.30.30.100", "2.40.50.140", "2.60.40.10"], ["3tjrA00", "1ko7A02", "1kq2A00", "4y91L00", "1kq1H00", "1uebA03", "2dgyA01", "4unuA00"])
    #save_as_bfactors(sys.argv[1])
    #cluster_lrp_full(sys.argv[1], int(sys.argv[2]) if len(sys.argv)>2 else False)
