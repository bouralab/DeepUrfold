import os
import sys
import json
import glob
import shutil
import random
from itertools import combinations, permutations, tee

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from Bio.PDB import PDBIO
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
#from Bio.PDB.vectors import Vector
from Bio.PDB import Selection
#from Bio.PDB.Polypeptide import three_to_one
#from Bio.PDB.vectors import calc_dihedral


from Prop3D.common.DistributedStructure import DistributedStructure
from Prop3D.parsers.MODELLER import MODELLER
#from Prop3D.generate_data.create_input_files import create_input_files
from Prop3D.common.ProteinTables import atoms_to_aa, three_to_one

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

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def calc_dihedral(p0, p1, p2, p3):
    """Praxeolitic formula
    1 sqrt, 1 cross product
        https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python

    """

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))

def calc_angle(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793

    """
    v1_u = v1 #normalized(v1)[0]
    v2_u = v2 #normalized(v2)[0]
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def _model_permutation(obj, perm):
    obj.model_loop_permutation(perm)

writer = PDBIO()
def write_atom(fp, atom, atom_number, resname, resseq, chain):
    if not isinstance(atom, Atom):
        atom = Atom(
            name=atom["atom_name"].decode("utf-8").strip(), 
            coord=tuple(atom[["X", "Y", "Z"]]),
            bfactor=atom["bfactor"], occupancy=1.0, altloc=" ",
            fullname=atom["atom_name"].decode("utf-8"),
            serial_number=atom["serial_number"]
        )
    try:
        s = writer._get_atom_line(
            atom,
            " ",
            " ",
            atom_number,
            resname,
            resseq,
            " ",
            str(chain),
        )
    except TypeError:
        import pdb; pdb.set_trace()
    fp.write(s)

class Alanine(Residue):
    def __init__(self):
        Residue.__init__(self, (" ", 1, " "), "ALA", 0)
        for atom in (" N  ", " H  ", " CA ", " HA ", " CB ", " HB1", " HB2", " HB3"):
            self.add(Atom(atom.strip(), np.random.rand(3), 20., 1., " ", atom, 1))

class MLP(object):
    def __init__(self, h5_file, cath_domain, cathcode, cutoff=15, short_loops=False, random_loops=False, only_show_count=False, n_jobs=1, min_ss_len=3):
        self.h5_file = h5_file
        self.cath_domain = cath_domain
        self.cathcode = cathcode
        self.cutoff = cutoff
        self.short_loops=short_loops
        self.random_loops=random_loops
        self.only_show_count = only_show_count
        self.min_ss_len = min_ss_len

        #self.pdb_file = os.path.join(DATA_DIR, "prepared-cath-structures", *cathcode.split("."), f"{cath_domain}.pdb")
        #self.features_path = os.path.join(DATA_DIR, "cath_features", *cathcode.split("."))
        self.structure = DistributedStructure(self.h5_file, cathcode.replace(".", "/"), self.cath_domain)

        self.ss_groups = []
        self.loop_for_ss = {}
        self.original_order = {}
        self.ss_type = {}
        self.loop_distance = {}
        self.number_ss = 0

        self.allowed_ss_combos = []
        self.prohibited_ss_combos = []

        self.leading_trailing_residues = {}

        self.rearrange_ss(n_jobs)

    def rearrange_ss(self, n_jobs=1):
        #Step 1: Get Secondary Structure
        print("Is OB?", self.cathcode=="2.40.50.140")
        self.ss_groups, self.loop_for_ss, self.original_order, self.ss_type, \
            self.leading_trailing_residues, self.number_ss = \
            self.structure.get_secondary_structures_groups(verbose=True, ss_min_len=self.min_ss_len,
                is_ob=self.cathcode=="2.40.50.140")

        #Step 2: Find allowable combinations
        self.get_allowable_loop_combinations()

        if self.only_show_count:
            permutations = list(self._permutations())
            print(f"Creating {len(permutations)} permutations...")
            return

        #Step 3: Permute strands
        if n_jobs == 1:
            for perm_pdb_name, loop_data, ss_data, sequences in self.get_possible_loop_permutations():
                #Step 4: Model loops correctly with MODELLER
                files = os.listdir(os.getcwd())
                for f in files:
                    if os.path.splitext(os.path.basename(perm_pdb_name))[0] in f and ".BL" in f and f.endswith(".pdb"):
                        permuted_pdb = f
                        break
                else:
                    if self.short_loops or "-" in sequences[0]:
                        permuted_pdb = self.model_loops(perm_pdb_name, loop_data,
                            ss_restraints=ss_data, sequences=sequences)
                    else:
                        permuted_pdb = self.model_loops(perm_pdb_name, loop_data,
                            ss_restraints=ss_data)
        else:
            permutations = list(self._permutations())
            print(f"Creating {len(permutations)} permutations...")
            Parallel(n_jobs=1)(delayed(_model_permutation)(self, perm) for perm in permutations)

        #self.create_dataset(f"{self.cath_domain}_{self.cathcode}_MLP_files.txt")

        #return input_file

    # def get_secondary_structures(self):
    #     """1. Secondary structure for each domain was assigned by the program DSSP.
    #     Short helical and strand segments (<4 residues) were treated as coils to
    #     decrease the number of loops for a given protein by reducing the number of
    #     secondary structure segments (SSSs).
    #     """
    #     ss_type = pd.DataFrame.from_records(self.structure.atom_features[["serial_number", "is_helix", "is_sheet", "Unk_SS"]],
    #         index="serial_number")
    #
    #     ss_type = ss_type.rename(columns={"is_helix":"H", "is_sheet":"E", "Unk_SS":"X"})
    #     ss_type = ss_type.idxmax(axis=1)
    #
    #     ss_groups = ss_type.groupby([(ss_type != ss_type.shift()).cumsum()-1])
    #
    #     #Merge group shorter than 4 residues
    #     for i, ss_group in ss_groups:
    #         if 0<i<ss_groups.ngroups-1:
    #             this_group = ss_group.iloc[0]
    #             prev_group = ss_groups.get_group(i-1).iloc[0]
    #             next_group = ss_groups.get_group(i+1).iloc[0]
    #             this_group_atoms = tuple(self.structure.get_atoms(include_atoms=ss_group.index))
    #             this_group_residues = tuple(unfold_entities(this_group_atoms, "R"))
    #
    #             print(this_group_residues[0].get_id(), this_group_residues[-1].get_id(), this_group)
    #
    #             # if len(this_group_residues)<4 and this_group != "X":
    #             #     ss_type.loc[ss_group.index] = "X"
    #             if len(this_group_residues)<3 and prev_group == next_group:
    #                 ss_type.loc[ss_group.index] = prev_group
    #             elif len(this_group_residues)<3 and next_group == "X" and this_group != prev_group:
    #                 ss_type.loc[ss_group.index] = "X"
    #
    #             if len(this_group_residues)<5:
    #                 if prev_group == next_group:
    #                     ss_type.loc[ss_group.index] = prev_group
    #                 else:
    #                     pass
    #
    #                 if this_group=="H" and prev_group=="E" and next_group=="E":
    #                     ss_type.loc[ss_group.index] = "X"
    #                 elif this_group=="E" and prev_group=="H" and next_group=="H":
    #                     ss_type.loc[ss_group.index] = "X"
    #
    #             if len(this_group_residues)>10 and this_group=="X":
    #                 ss_type.loc[ss_group.index] = "H"
    #
    #     #Regroup with correct SS
    #     ss_atom_groups = ss_type.groupby([(ss_type != ss_type.shift()).cumsum()-1])
    #
    #     self.ss_groups = []
    #     self.loop_for_ss = {}
    #     self.original_order = {}
    #     self.ss_type = {}
    #
    #     for i, ss_group in ss_atom_groups:
    #         #Get all atoms from SS and loops
    #         ss_atoms = tuple(self.structure.get_atoms(include_atoms=ss_group.index))
    #         ss_residues = tuple(unfold_entities(ss_atoms, "R"))
    #
    #         if ss_group.iloc[0] != "X":
    #             self.ss_groups.append(ss_residues)
    #             self.original_order[ss_residues] = len(self.ss_groups)
    #             self.ss_type[ss_residues] = ss_group.iloc[0]
    #         elif len(self.ss_groups)>0 and ss_group.iloc[0] == "X":
    #             self.loop_for_ss[self.ss_groups[-1]] = ss_residues
    #
    #     first_group = ss_atom_groups.get_group(0)
    #     if first_group.iloc[0] == "X":
    #         loop_atoms = tuple(self.structure.get_atoms(include_atoms=first_group.index))
    #         self.leading_trailing_residues[1] = tuple(unfold_entities(loop_atoms, "R"))
    #
    #     last_group = ss_atom_groups.get_group(ss_atom_groups.ngroups-1)
    #     if last_group.iloc[0] == "X":
    #         loop_atoms = tuple(self.structure.get_atoms(include_atoms=last_group.index))
    #         self.leading_trailing_residues[len(self.ss_groups)] = tuple(unfold_entities(loop_atoms, "R"))
    #
    #     self.number_ss = len(self.ss_groups)
    #     print()
    #     for ss in self.ss_groups:
    #         print(ss[0].get_id(), ss[-1].get_id(), self.ss_type[ss])

    def get_allowable_loop_combinations(self):
        """2. The distances between the N-terminus of one SSS and the C-terminus of
        another SSS were calculated for all SSS pairs. The N and C termini of
        two SSSs were allowed to connect by building a new loop between them if
        their distance is less than a cutoff distance (15 angstroms initially).
        The connection between two N (or C) termini of two SSSs was not allowed
        in order to maintain the original N to C direction. The original loops
        longer than 15 angstroms were unchanged."""

        self.allowed_ss_combos = []
        self.prohibited_ss_combos = []

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)

        for ss1, ss2 in permutations(self.ss_groups, 2):
                #
            atom1 = self.structure._to_unstructured(ss1[-1][ss1[-1]["atom_name"]==b' C  '][["X", "Y", "Z"]])
            atom2 = self.structure._to_unstructured(ss2[0][ss2[0]["atom_name"]==b' N  '][["X", "Y", "Z"]])

            dist = np.linalg.norm(atom1-atom2)

            ss1_id = tuple(np.unique(r["residue_id"])[0] for r in ss1)
            ss2_id = tuple(np.unique(r["residue_id"])[0] for r in ss2)

            if self.ss_type[ss1_id] == "E":
                #Look at last 2 residues
                atom1a = self.structure._to_unstructured(ss1[-2][ss1[-2]["atom_name"]==b' N  '][["X", "Y", "Z"]])
                atom1b = self.structure._to_unstructured(ss1[-1][ss1[-1]["atom_name"]==b' C  '][["X", "Y", "Z"]])
            else:
                #Helix, look at last 4
                atom1a = self.structure._to_unstructured(ss1[-4][ss1[-4]["atom_name"]==b' N  '][["X", "Y", "Z"]])
                atom1b = self.structure._to_unstructured(ss1[-1][ss1[-1]["atom_name"]==b' C  '][["X", "Y", "Z"]])

            if self.ss_type[ss2_id] == "E":
                #Look at last 2 residues
                atom2a = self.structure._to_unstructured(ss2[0][ss2[0]["atom_name"]==b' N  '][["X", "Y", "Z"]])
                atom2b = self.structure._to_unstructured(ss2[1][ss2[1]["atom_name"]==b' C  '][["X", "Y", "Z"]])
            else:
                #Helix, look at last 4
                atom2a = self.structure._to_unstructured(ss2[0][ss2[0]["atom_name"]==b' N  '][["X", "Y", "Z"]])
                atom2b = self.structure._to_unstructured(ss2[3][ss2[3]["atom_name"]==b' C  '][["X", "Y", "Z"]])

            dihedral = np.abs(calc_dihedral(
                normalized(atom1a[0])[0], normalized(atom1b[0])[0],
                normalized(atom2a[0])[0], normalized(atom2b[0])[0]
            ))

            ss_vec1 = normalized(atom1b[0])-normalized(atom1a[0])
            ss_vec2 = normalized(atom2b[0])-normalized(atom2a[0])
            angle = calc_angle(ss_vec1.reshape(-1), ss_vec2.reshape(-1))

            points = pca.fit_transform(np.concatenate([atom1a, atom1b,
                                                       atom2a, atom2b]))
            a = points[1]-points[0]
            b = points[3]-points[2]

            a1 = a/np.linalg.norm(a)
            b1 = b/np.linalg.norm(b)
            angle2 = np.arccos(np.dot(a1,b1))

            ss1_id = tuple(np.unique(r["residue_id"])[0] for r in ss1)
            ss2_id = tuple(np.unique(r["residue_id"])[0] for r in ss2)

            if str(type(ss1_id[0])) != "<class 'numpy.bytes_'>":
                import pdb; pdb.set_trace()

            if dist < self.cutoff:
                #print("allowed", self.original_order[ss1], self.ss_type[ss1], f"({ss1[-1].get_id()}, {atom1.coord})", "->", self.original_order[ss2], self.ss_type[ss2], f"({ss2[0].get_id()}, {atom2.coord})", dist)
                self.allowed_ss_combos.append((ss1_id, ss2_id))
                self.loop_distance[(ss1_id, ss2_id)] = dist
            else:
                if angle2 > np.pi/2:
                    #If anti-parellel
                    overlap = 0
                    for other_ss in self.ss_groups:
                        other_ss_id = tuple(np.unique(r["residue_id"])[0] for r in other_ss)
                        if other_ss_id==ss1_id or other_ss_id==ss2_id:
                            continue
                        for other_residue in other_ss:
                            bad_resi = False
                            for other_atom in other_residue:
                                other_atom = self.structure._to_unstructured(other_atom[["X", "Y", "Z"]])
                                d = np.linalg.norm(np.cross(
                                        atom2a-atom1b,
                                        atom1b-other_atom))
                                d /= np.linalg.norm(atom2a-atom1b)
                                if d<1:
                                    bad_resi = True
                                    break
                            if bad_resi:
                                overlap += 1

                    if overlap<2:
                        self.allowed_ss_combos.append((ss1_id, ss2_id))
                        self.loop_distance[(ss1_id, ss2_id)] = dist
                    else:
                        # print("prohibited", self.original_order[ss1], self.ss_type[ss1],
                        #     f"({ss1[-1].get_id()}, {atom1.coord})", "->", self.original_order[ss2], self.ss_type[ss2],
                        #     f"({ss2[0].get_id()}, {atom2.coord})", dist, np.rad2deg(dihedral), dihedral>np.pi/2, np.rad2deg(angle),a, b, angle2,
                        #     parallel[(self.original_order[ss1], self.original_order[ss2])], "overlap=", overlap)
                        self.prohibited_ss_combos.append((ss1_id, ss2_id))
                else:
                    # print("prohibited", self.original_order[ss1], self.ss_type[ss1],
                    #     f"({ss1[-1].get_id()}, {atom1.coord})", "->", self.original_order[ss2], self.ss_type[ss2],
                    #     f"({ss2[0].get_id()}, {atom2.coord})", dist, np.rad2deg(dihedral), dihedral>np.pi/2, np.rad2deg(angle),a, b, angle2,
                    #     parallel[(self.original_order[ss1], self.original_order[ss2])])
                    self.prohibited_ss_combos.append((ss1_id, ss2_id))

    def _permutations(self):
        yield from permutations(self.ss_groups, self.number_ss)

    def get_possible_loop_permutations(self):
        for perm in self._permutations():
            rc = self.get_possible_loop_permutation(perm)
            if rc:
                yield rc

    def get_possible_loop_permutation(self, perm):
        """3. A combinatorial search was made for all possible loop permutations
        allowed. If two SSSs change from sequence neighbor to non-neighbor after
        rearrangement, their connection loop will be removed. Meanwhile, new loops
        will be built to connect two SSSs that become sequence neighbors after
        rearrangement. For example, a protein with 6 SSSs is arranged in a native
        structure as 1-2-3-4-5-6. One possible rearrangement of this sequence is
        6-5-2-3-4-1. This rearrangement requires retaining two native loops for
        unchanged neighboring SSSs between 2-3 and 3-4, removing three native loops
        (1-2, 4-5 and 5-6) because they are no longer sequence neighbors (5-6 is not
        same as 6-5 because of the N to C direction), and building three new loops
        between 6-5, 5-2, and 4-1. In this study, we limited ourselves to generate
        100 MLP structures and a maximum of five permutated loops per proteins. If
        the number of permutations is greater than 100, we decreased the cutoff
        distance with a step size of 0.5Å to reduce the number of loops allowed to
        permute until the number of permutations is less than or equal to 100."""
        order = list(perm)

        perm_name = []
        for ss in order:
            ss_id = tuple(np.unique(r["residue_id"])[0] for r in ss)
            perm_name.append(self.original_order[ss_id])

        #perm_name = [self.original_order[tuple(np.unique(r["residue_id"])[0] for r in ss)] for ss in order]

        if not self.short_loops and perm_name == list(range(1,self.number_ss+1)):
            #Same as orignal PDB
            print("Skipped -- same as original")
            return

        skip = False
        for ss1, ss2 in pairwise(order):
            ss1_id = tuple(np.unique(r["residue_id"])[0] for r in ss1)
            ss2_id = tuple(np.unique(r["residue_id"])[0] for r in ss2)
            
            if (ss1_id, ss2_id) not in self.allowed_ss_combos:
                skip = True
                break

        if skip:
            return

        perm_name_str = "-".join(map(str, perm_name))
        print(perm_name_str)
        loop_data = []
        ss_data = []
        atom_number = 1
        resseq = 1
        perm_pdb_name = f"{self.cath_domain}_{perm_name_str}.pdb"
        no_loop_sequence = ""
        loop_sequence = ""
        with open(perm_pdb_name, "w") as perm_pdb:
            if perm_name[0] in self.leading_trailing_residues: #[1, len(self.ss_groups)]:
                # for residue in self.loop_for_ss[perm_name[0]]:
                #     for atom in residue:
                #         write_atom(perm_pdb, atom, atom_number, resseq, chain)
                #         atom_number += 1
                #     resseq += 1
                #     no_loop_sequence += three_to_one(residue.get_resname())
                #     loop_sequence += three_to_one(residue.get_resname())
                resseq, atom_number, no_loop_sequence, loop_sequence = self.write_residue_group(
                    perm_pdb, self.leading_trailing_residues[perm_name[0]], resseq, atom_number,
                    no_loop_sequence, loop_sequence)

            for i, ss in enumerate(order):
                ss_id = tuple(np.unique(r["residue_id"])[0] for r in ss)
                ss_start = (resseq, self.structure.chain)
                ss_end = (resseq+len(ss)-1, self.structure.chain)
                ss_data.append((self.ss_type[ss_id], ss_start, ss_end))

                resseq, atom_number, no_loop_sequence, loop_sequence = self.write_residue_group(
                    perm_pdb, ss, resseq, atom_number, no_loop_sequence, loop_sequence)

                # for residue in ss:
                #     for atom in residue:
                #         write_atom(perm_pdb, atom, atom_number, resseq, chain)
                #         atom_number += 1
                #     resseq += 1
                #     no_loop_sequence += three_to_one(residue.get_resname())
                #     loop_sequence += three_to_one(residue.get_resname())

                loop_aa = self.loop_for_ss.get(ss_id)
                if loop_aa is not None:
                    loop_start = (resseq, self.structure.chain)
                    loop_end = (resseq+len(loop_aa)-1, self.structure.chain)
                    loop_data.append((loop_start, loop_end))

                    if i<len(order)-1:
                        next_ss = order[i+1]
                        next_ss_id = tuple(np.unique(r["residue_id"])[0] for r in next_ss)
                        dist = self.loop_distance[(ss_id,next_ss_id)]
                        n_res = min(int(dist/2.5), 6)
                    else:
                        n_res = len(loop_aa)

                    if (self.short_loops and i<len(order)-1) or len(loop_aa)<n_res:
                        #Create shorter loops
                        if self.random_loops:
                            if len(loop_aa)<n_res:
                                loop_aa = list(loop_aa)+[Alanine() for i in range(n_res-len(loop_aa))]
                            loop_aa = random.choices(loop_aa, k=n_res) #, replace=False)
                        else:
                            loop_aa = [Alanine() for i in range(n_res)]

                        resseq, atom_number, no_loop_sequence, loop_sequence = self.write_residue_group(
                            perm_pdb, loop_aa, resseq, atom_number, no_loop_sequence, loop_sequence, mark_unknown=True)

                        # no_loop_sequence += "-"*len(loop_aa)
                        # for residue in loop_aa:
                        #     loop_sequence += three_to_one(residue.get_resname())
                        #     atom_number += len(residue)
                        # resseq += len(loop_aa)

                    else:
                        resseq, atom_number, no_loop_sequence, loop_sequence = self.write_residue_group(
                            perm_pdb, loop_aa, resseq, atom_number, no_loop_sequence, loop_sequence)
                        # for residue in loop_aa:
                        #     for atom in residue:
                        #         write_atom(perm_pdb, atom, atom_number, resseq, chain)
                        #         atom_number += 1
                        #     resseq += 1
                        #     no_loop_sequence += three_to_one(residue.get_resname())
                        #     loop_sequence += three_to_one(residue.get_resname())

            if perm_name[-1] in self.leading_trailing_residues : #[1, len(self.ss_groups)]:
                # for residue in self.loop_for_ss[perm_name[-1]]:
                #     for atom in residue:
                #         write_atom(perm_pdb, atom, atom_number, resseq, chain)
                #         atom_number += 1
                #     resseq += 1
                #     no_loop_sequence += three_to_one(residue.get_resname())
                #     loop_sequence += three_to_one(residue.get_resname())
                resseq, atom_number, no_loop_sequence, loop_sequence = self.write_residue_group(
                    perm_pdb, self.leading_trailing_residues[perm_name[-1]], resseq, atom_number,
                    no_loop_sequence, loop_sequence)
        
        return perm_pdb_name, loop_data, ss_data, (no_loop_sequence, loop_sequence)

    def model_loop_permutation(self, perm):
        rc = self.get_possible_loop_permutation(perm)
        if rc is None:
            return
        perm_pdb_name, loop_data, ss_data, sequences = rc

        #Step 4: Model loops correctly with MODELLER
        files = os.listdir(os.getcwd())
        for f in files:
            if os.path.splitext(os.path.basename(perm_pdb_name))[0] in f and ".BL" in f and f.endswith(".pdb"):
                permuted_pdb = f
                break
        else:
            if self.short_loops or "-" in sequences[0]:
                permuted_pdb = self.model_loops(perm_pdb_name, loop_data,
                    ss_restraints=ss_data, sequences=sequences)
            else:
                permuted_pdb = self.model_loops(perm_pdb_name, loop_data,
                    ss_restraints=ss_data)

    modeller = None
    def model_loops(self, perm_pdb_name, loop_data, ss_restraints=None, sequences=None):
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
            if isinstance(residue, Alanine):
                resname = "ALA"
            elif not "residue_name" in residue.dtype.names:
                atom_names = frozenset([a.decode("utf-8").strip() for a in residue["atom_name"]])
                resname = atoms_to_aa(atom_names, raise_unknown=False)
                print(resname, atom_names)
            else:
                resname = residue["residue_name"][0].decode("utf-8").strip()
            if not mark_unknown:
                for atom in residue:
                    write_atom(perm_pdb, atom, atom_number, resname, resseq, self.structure.chain)
                    atom_number += 1
            resseq += 1
            if not mark_unknown:
                no_loop_sequence += three_to_one(resname)
            loop_sequence += three_to_one(resname)

        return resseq, atom_number, no_loop_sequence, loop_sequence

    @staticmethod
    def create_dataset(mlp_dir, out=None):
        best_pdbs = []
        for f in glob.glob(os.path.join(mlp_dir, "*.dope_scores")):
            with open(f) as fh:
                dope_scores = json.load(fh)

            best_pdb = os.path.abspath(os.path.join(mlp_dir, min(dope_scores["scores"], key=lambda x: x["score"])["name"]))
            best_pdbs.append(best_pdb)
            if out is None:
                out = best_pdb.rsplit("_", 1)[0]+"_MLP_files.txt"

        with open(out, "w") as fh:
            for f in best_pdbs:
                print(f, file=fh)

        return out

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("h5_file")
    parser.add_argument("cath_domain")
    parser.add_argument("cathcode")
    parser.add_argument("-c", "--cutoff", default=15, type=int)
    parser.add_argument("--short_loops", default=False, action="store_true")
    parser.add_argument("--random_loops", default=False, action="store_true")
    parser.add_argument("--only_show_count", default=False, action="store_true")
    parser.add_argument("-j", "--cpus", default=1, type=int)
    parser.add_argument("--min_ss_len", default=3, type=int)
    parser.add_argument("--create-dataset", default=None)

    args = parser.parse_args()

    if args.create_dataset is not None:
        MLP.create_dataset(args.create_dataset, f"{args.cath_domain}_{args.cathcode}_MLP_files.txt")
    else:
        MLP(args.h5_file, args.cath_domain, args.cathcode, cutoff=args.cutoff,
            short_loops=args.short_loops, random_loops=args.random_loops,
            only_show_count=args.only_show_count, n_jobs=args.cpus,
            min_ss_len=args.min_ss_len)
