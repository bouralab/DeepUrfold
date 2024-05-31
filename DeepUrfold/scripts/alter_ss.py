import os
import sys
import shutil
import random
from itertools import combinations, tee, combinations_with_replacement

import numpy as np
import pandas as pd
from Bio.PDB import PDBIO
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
from Bio.PDB import Selection
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.vectors import calc_dihedral

from Prop3D.common.Structure import Structure
from Prop3D.parsers.MODELLER import MODELLER
from Prop3D.generate_data.create_input_files import create_input_files

"""MLP: Multiple Loop Permutations

Modified by Eli Draizen

Dai L, Zhou Y. Characterizing the existing and potential structural space of
proteins by large-scale multiple loop permutations. J Mol Biol. 2011 May 6;
408(3):585-95. doi: 10.1016/j.jmb.2011.02.056. Epub 2011 Mar 2.
PMID: 21376059; PMCID: PMC3075335.
"""

DATA_DIR = os.environ.get("DATA_DIR", "/home/bournelab/data-eppic-cath-features")

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def triplewise(iterable):
    "s -> (s0,s1,s2), (s1,s2,s3), (s2,s3,s4), ..."
    a, b = tee(iterable)
    next(b, None)
    bb, c = tee(b)
    next(c, None)
    return zip(a, bb, c)

def nwise(iterable, n=2):
    "s -> (s0,s1,s2), (s1,s2,s3), (s2,s3,s4), ..."
    assert n>1
    it = []
    for n in range(n-1):
        a, b = tee(iterable)
        next(b, None)
        it += [a,b]
        iterable = b
    return zip(*it)

def unfold_entities(entities, level):
    return sorted(Selection.unfold_entities(entities, level), key= lambda x:x.get_id())

writer = PDBIO()
def write_atom(fp, atom, atom_number, resseq, chain):
    hetfield, _, _ = atom.get_parent().get_id()
    resname = atom.get_parent().get_resname()
    s = writer._get_atom_line(
                    atom,
                    hetfield,
                    " ",
                    atom_number,
                    resname,
                    resseq,
                    " ",
                    chain,
                )
    fp.write(s)

class Alanine(Residue):
    def __init__(self):
        Residue.__init__(self, (" ", 1, " "), "ALA", 0)
        for atom in (" N  ", " H  ", " CA ", " HA ", " CB ", " HB1", " HB2", " HB3"):
            self.add(Atom(atom.strip(), np.random.rand(3), 20., 1., " ", atom, 1))

class MLP(object):
    def __init__(self, cath_domain, cathcode, cutoff=15, exact_size=False):
        self.cath_domain = cath_domain
        self.cathcode = cathcode
        self.cutoff = cutoff
        self.exact_size = exact_size

        self.pdb_file = os.path.join(DATA_DIR, "prepared-cath-structures", *cathcode.split("."), f"{cath_domain}.pdb")
        self.features_path = os.path.join(DATA_DIR, "cath_features", *cathcode.split("."))
        self.structure = Structure(self.pdb_file, self.cath_domain, features_path=self.features_path)

        self.ss_groups = []
        self.loop_for_ss = {}
        self.original_order = {}
        self.ss_type = {}
        self.loop_distance = {}
        self.number_ss = 0

        self.allowed_ss_combos = []
        self.prohibited_ss_combos = []

        self.leading_trailing_residues = {}

        self.rearrange_ss()

    def rearrange_ss(self):
        #Step 1: Get Secondary Structure
        self.get_secondary_structures()

        #Step 3: Permute strands
        input_file = None
        for perm_pdb_name, loop_data, ss_data, sequences in self.get_possible_ss_combinations():
            #Step 4: Model loops correctly with MODELLER
            files = os.listdir(os.getcwd())
            for f in files:
                if os.path.splitext(os.path.basename(perm_pdb_name))[0] in f and ".BL" in f and f.endswith(".pdb"):
                    permuted_pdb = f
                    break
            else:
                if self.exact_size or "-" in sequences[0]:
                    permuted_pdb = self.model_ss(perm_pdb_name, loop_data,
                        ss_restraints=ss_data, sequences=sequences)
                else:
                    permuted_pdb = self.model_ss(perm_pdb_name, loop_data,
                        ss_restraints=ss_data)
            #input_file = create_input_files(permuted_pdb, input_file=input_file)

        #return input_file

    def get_secondary_structures(self):
        """1. Secondary structure for each domain was assigned by the program DSSP.
        Short helical and strand segments (<4 residues) were treated as coils to
        decrease the number of loops for a given protein by reducing the number of
        secondary structure segments (SSSs).
        """
        ss_type = self.structure.atom_features[["is_helix", "is_sheet", "Unk_SS"]]

        ss_type = ss_type.rename(columns={"is_helix":"H", "is_sheet":"E", "Unk_SS":"X"})
        ss_type = ss_type.idxmax(axis=1)

        ss_groups = ss_type.groupby([(ss_type != ss_type.shift()).cumsum()-1])

        #Merge group shorter than 4 residues
        for i, ss_group in ss_groups:
            if 0<i<ss_groups.ngroups-1:
                this_group = ss_group.iloc[0]
                prev_group = ss_groups.get_group(i-1).iloc[0]
                next_group = ss_groups.get_group(i+1).iloc[0]
                this_group_atoms = tuple(self.structure.get_atoms(include_atoms=ss_group.index))
                this_group_residues = tuple(unfold_entities(this_group_atoms, "R"))

                print(this_group_residues[0].get_id(), this_group_residues[-1].get_id(), this_group)

                # if len(this_group_residues)<4 and this_group != "X":
                #     ss_type.loc[ss_group.index] = "X"
                if len(this_group_residues)<3 and prev_group == next_group:
                    ss_type.loc[ss_group.index] = prev_group
                elif len(this_group_residues)<3 and next_group == "X" and this_group != prev_group:
                    ss_type.loc[ss_group.index] = "X"

                if len(this_group_residues)<3:
                    if prev_group == next_group:
                        ss_type.loc[ss_group.index] = prev_group
                    else:
                        pass

                    if this_group=="H" and prev_group=="E" and next_group=="E":
                        ss_type.loc[ss_group.index] = "X"
                    elif this_group=="E" and prev_group=="H" and next_group=="H":
                        ss_type.loc[ss_group.index] = "X"

                if len(this_group_residues)<4 and this_group=="H" and prev_group=="E" and next_group=="E":
                    ss_type.loc[ss_group.index] = "X"

        #Regroup with correct SS
        ss_atom_groups = ss_type.groupby([(ss_type != ss_type.shift()).cumsum()-1])

        self.ss_groups = []
        self.loop_for_ss = {}
        self.original_order = {}
        self.ss_type = {}

        for i, ss_group in ss_atom_groups:
            #Get all atoms from SS and loops
            ss_atoms = tuple(self.structure.get_atoms(include_atoms=ss_group.index))
            ss_residues = tuple(unfold_entities(ss_atoms, "R"))

            if ss_group.iloc[0] != "X":
                self.ss_groups.append(ss_residues)
                self.original_order[ss_residues] = len(self.ss_groups)
                self.ss_type[ss_residues] = ss_group.iloc[0]
            elif len(self.ss_groups)>0 and ss_group.iloc[0] == "X":
                self.loop_for_ss[self.ss_groups[-1]] = ss_residues

        first_group = ss_atom_groups.get_group(0)
        if first_group.iloc[0] == "X":
            loop_atoms = tuple(self.structure.get_atoms(include_atoms=first_group.index))
            self.leading_trailing_residues[1] = tuple(unfold_entities(loop_atoms, "R"))

        last_group = ss_atom_groups.get_group(ss_atom_groups.ngroups-1)
        if last_group.iloc[0] == "X":
            loop_atoms = tuple(self.structure.get_atoms(include_atoms=last_group.index))
            self.leading_trailing_residues[len(self.ss_groups)] = tuple(unfold_entities(loop_atoms, "R"))

        self.number_ss = len(self.ss_groups)
        print()
        for ss in self.ss_groups:
            print(ss[0].get_id(), ss[-1].get_id(), self.ss_type[ss])

    def combinations_with_replacement(self):
        yield from combinations_with_replacement("HS", self.number_ss)

    def get_possible_ss_combinations(self):
        original_ss = [self.ss_type[ss] for ss in self.ss_groups]
        original_ss_name = "".join(original_ss)
        for ss_content in self.combinations_with_replacement():
            ss_content = list(ss_content)
            ss_content_name = "".join(ss_content)

            print(ss_content_name)

            if ss_content_name == original_ss_name:
                #Same as orignal PDB
                print("Skipped -- same as original")
                continue

            n_diff = sum(1 for a, b in zip(original_ss_name, ss_content_name) if a != b)

            ss_data = []
            loop_data = []
            atom_number = 1
            resseq = 1
            perm_pdb_name = f"{self.cath_domain}_{n_diff}_{ss_content_name}.pdb"
            target_sequence = ""
            original_sequence = ""
            with open(perm_pdb_name, "w") as perm_pdb:
                resseq, atom_number, target_sequence, original_sequence = self.write_residue_group(
                    perm_pdb, self.leading_trailing_residues[1], resseq, atom_number,
                    target_sequence, original_sequence)

                for i, (ss, ss_type) in enumerate(zip(self.ss_groups, ss_content)):
                    ss_start = (resseq, self.structure.chain)
                    ss_end = (resseq+len(ss)-1, self.structure.chain)
                    ss_data.append((ss_type, ss_start, ss_end))

                    if self.exact_size and ss_type != original_ss[i]:
                        new_ss = []
                        if ss_type=="E":
                            #H->S, Keep every fourth resiue and i+1 or i+2 sincde rise 4
                            for j in range(0, len(ss), 4):
                                new_ss.append(original_sequence[j])
                                new_ss.append(original_sequence[j+1+int(i%2==0)])
                        else:
                            #S->H
                            for j, s in enumerate(ss):
                                new_ss.append(s)
                                new_ss += [Alanine(), Alanine()]

                    resseq, atom_number, target_sequence, original_sequence = self.write_residue_group(
                        perm_pdb, ss, resseq, atom_number, target_sequence, original_sequence, mark_unknown=ss_type != original_ss[i])

                    loop_aa = self.loop_for_ss.get(ss)
                    if loop_aa is not None:
                        loop_start = (resseq, self.structure.chain)
                        loop_end = (resseq+len(loop_aa)-1, self.structure.chain)
                        loop_data.append((loop_start, loop_end))

                        resseq, atom_number, target_sequence, original_sequence = self.write_residue_group(
                            perm_pdb, loop_aa, resseq, atom_number, target_sequence, original_sequence)


                resseq, atom_number, target_sequence, original_sequence = self.write_residue_group(
                    perm_pdb, self.leading_trailing_residues[len(self.ss_groups)], resseq, atom_number,
                    target_sequence, original_sequence)

            yield perm_pdb_name, loop_data, ss_data, (target_sequence, original_sequence)

    modeller = None
    def model_ss(self, perm_pdb_name, loop_data, ss_restraints=None, sequences=None):
        """4. All new loops were built by the program Modloop47. We estimated the
        number of residues for a new loop by dividing the end-to-end distance with
        2.5Å. This approximate formula was obtained from a statistical analysis of
        the end-to-end distances of short loops. Because the maximum end-to-end
        distance for a loop to be permutated is 15Å, the maximum number of residues
        for a rebuilt loop is 6. That is, we have avoided building potentially
        unrealistic long loops (>6) 47. All loops were built with alanine residues
        for computational efficiency."""
        if self.modeller is None:
            modeller = MODELLER()
        permuted_pdb = modeller.model_loops(perm_pdb_name, loop_data,
            ss_restraints=ss_restraints, sequences=sequences)
        return permuted_pdb

    def write_residue_group(self, perm_pdb, residue_group, start_resseq, start_atom_number, no_loop_sequence, loop_sequence, mark_unknown=False):
        resseq = start_resseq
        atom_number = start_atom_number

        if mark_unknown:
            no_loop_sequence += "-"*len(residue_group)

        for residue in residue_group:
            if not mark_unknown:
                for atom in residue:
                    write_atom(perm_pdb, atom, atom_number, resseq, self.structure.chain)
                    atom_number += 1
            resseq += 1
            if not mark_unknown:
                no_loop_sequence += three_to_one(residue.get_resname())
            loop_sequence += three_to_one(residue.get_resname())

        return resseq, atom_number, no_loop_sequence, loop_sequence

    def create_dataset(self):
        data = {"cathDomain":[], "structure_file":[], "feature_file":[]}
        files = sorted([f for f in os.listdir() if f.endswith("_target.pdb")])
        for f in files:
            if f.endswith(".h5"): continue
            perm = f.rsplit("_", 1)[0]
            if not os.path.isfile(f"{perm}_target_atom.h5"): continue
            data["cathDomain"].append(perm)
            data["structure_file"].append(os.path.abspath(f))
            data["feature_file"].append(os.path.abspath(f"{perm}_target_atom.h5"))

    def design(self, ss_restraints):
        from trDesign.design import mk_design_model
        ffrom trDesign.util import N_to_AA

        def get_pdb(design, pdb_filename="out.pdb", mds="classic"):
            '''given features, return approx. 3D structure'''
            if "I" in design: seq = design["I"]
            if "pssm" in design: seq = design["pssm"]
            seq = N_to_AA(np.squeeze(seq).argmax(-1))[0]
            xyz, dm = feat_to_xyz(np.squeeze(design["feat"]), mds=mds)
            save_PDB(pdb_filename, xyz, dm, seq)

        model = mk_design_model(add_pdb=True, n_models=5)
        design = model.design(inputs={"pdb":_feat[None]}, return_traj=True)
        N_to_AA(design["I"].argmax(-1))
        get_pdb(design,"6MRR_redesign.pdb",mds="metric")


    def design_prep(self, ss_restraints, mask_gaps=False):
        '''Parse PDB file and return features compatible with TrRosetta'''
        ncac, seq = parse_PDB(pdb,["N","CA","C"], chain=chain)

        # mask gap regions
        if mask_gaps:
            mask = seq != 20
            ncac, seq = ncac[mask], seq[mask]

        N,CA,C = ncac[:,0], ncac[:,1], ncac[:,2]
        CB = extend(C, N, CA, 1.522, 1.927, -2.143)

        dist_ref  = to_len(CB[:,None], CB[None,:])
        omega_ref = to_dih(CA[:,None], CB[:,None], CB[None,:], CA[None,:])
        theta_ref = to_dih( N[:,None], CA[:,None], CB[:,None], CB[None,:])
        phi_ref   = to_ang(CA[:,None], CB[:,None], CB[None,:])

        for ss_type, ss_start, ss_end in ss_restraints:
            ss_start = ss_start[0]-1
            ss_end = ss_end[0]-1
            if ss_type == "H":
                for i in range(ss_start, ss_end+1):
                    theta_ref[i] = -60
                    phi_ref[i] = -50
            else:
                for i in range(ss_start, ss_end+1):
                    theta_ref[i] = -140
                    phi_ref[i] = 130

        def mtx2bins(x_ref, start, end, nbins, mask):
            bins = np.linspace(start, end, nbins)
            x_true = np.digitize(x_ref, bins).astype(np.uint8)
            x_true[mask] = 0
            return np.eye(nbins+1)[x_true][...,:-1]

        p_dist  = mtx2bins(dist_ref,     2.0,  20.0, 37, mask=(dist_ref > 20))
        p_omega = mtx2bins(omega_ref, -np.pi, np.pi, 25, mask=(p_dist[...,0]==1))
        p_theta = mtx2bins(theta_ref, -np.pi, np.pi, 25, mask=(p_dist[...,0]==1))
        p_phi   = mtx2bins(phi_ref,      0.0, np.pi, 13, mask=(p_dist[...,0]==1))
        feat    = np.concatenate([p_theta, p_phi, p_dist, p_omega],-1)
        return {"seq":N_to_AA(seq), "feat":feat, "dist_ref":dist_ref}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cath_domain")
    parser.add_argument("cathcode")
    parser.add_argument("-c", "--cutoff", default=15, type=int)
    parser.add_argument("--exact_size", default=False, action="store_true")

    args = parser.parse_args()

    MLP(args.cath_domain, args.cathcode, cutoff=args.cutoff, exact_size=args.exact_size)
