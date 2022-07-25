from itertools import combinations, permutations, tee, product
import numpy as np
from sklearn.decomposition import PCA
from DeepUrfold.scripts.rearrange_ss import MLP, pairwise

#for ss1, ss2 in self.allowed_ss_combos: combos.append((self.original_order[ss1], self.original_order[ss2]))

class StrandReversalMLP(MLP):
    def __init__(self, cath_domain, cathcode, cutoff=15, short_loops=False, random_loops=False):
        self.reverse_strands = {}
        super().__init__(cath_domain, cathcode, cutoff=cutoff, short_loops=short_loops, random_loops=random_loops)

    def get_secondary_structures(self):
        super().get_secondary_structures()
        self.leading_trailing_residues["^1"] = tuple(reversed(self.loop_for_ss[self.ss_groups[0]]))
        self.leading_trailing_residues[f"^{len(self.ss_groups)}"] = tuple(
            reversed(self.leading_trailing_residues[len(self.ss_groups)]))
        self.reverse_strands["^1"] = self.leading_trailing_residues["^1"]
        self.reverse_strands[f"^{len(self.ss_groups)}"] = self.leading_trailing_residues[f"^{len(self.ss_groups)}"]

        for ss in self.ss_groups:
            newSS = tuple(reversed(ss))
            self.loop_for_ss[newSS] = self.loop_for_ss[ss]
            self.original_order[newSS] = f"^{self.original_order[ss]}"
            self.ss_type[newSS] = self.ss_type[ss]
            self.reverse_strands[ss] = newSS

    def get_ss_terminal_residues(self, ss1, ss2, shortened=False):
        strand1_end = ss1[-1]["C"]
        if shortened:
            if self.ss_type[ss1] == "E":
                #Look at last 2 residues, -1 saved above
                strand1_start = ss1[-2]["N"]
            else:
                #Helix, look at last 4, -1 saved above
                strand1_start = ss1[-4]["N"]
        else:
            strand1_start = ss1[0]["N"]

        strand2_start = ss2[0]["N"]
        if shortened:
            if self.ss_type[ss2] == "E":
                #Look at first 2 residues
                strand2_end = ss2[1]["C"] #atom2b
            else:
                #Helix, look at first 4
                strand2_end = ss2[3]["C"] #atom2b
        else:
            strand2_end = ss1[-1]["C"]

        return strand1_start, strand1_end, strand2_start, strand2_end

    def angle_between_ss(self, ss1, ss2):
        strand1_start, strand1_end, strand2_start, strand2_end = self.get_ss_terminal_residues(ss1, ss2, shortened=True)
        pca = PCA(n_components=2)
        points2D = pca.fit_transform((strand1_start.coord, strand1_end.coord, strand2_start.coord, strand2_end.coord))
        strand1_vec = points2D[1]-points2D[0]
        strand2_vec = points2D[3]-points2D[2]
        strand1_norm = strand1_vec/np.linalg.norm(strand1_vec)
        strand2_norm = strand2_vec/np.linalg.norm(strand2_vec)
        angle = np.arccos(np.dot(strand1_norm,strand2_norm))

        return angle

    def strands_parallel(self, ss1, ss2):
        #If strands are parallel, the C terminals should be closer and the angle should < 90 deg
        angle = self.angle_between_ss(ss1, ss2)
        strand1_last_C = ss1[-1]["C"]
        strand2_last_C = ss2[-1]["C"]
        parallel_dist = strand1_last_C-strand2_last_C
        return angle < np.pi/2 and parallel_dist<self.cutoff, parallel_dist, angle

    def strands_anti_parallel(self, ss1, ss2):
        #If strands are anti parallel, the N term of ss1 and C term should be close and the angle should < 90 deg
        angle = self.angle_between_ss(ss1, ss2)
        strand1_last_C = ss1[-1]["C"]
        strand2_first_N = ss2[0]["N"]


        anti_parallel_dist = strand1_last_C-strand2_first_N
        return angle > np.pi/2 and anti_parallel_dist<self.cutoff, anti_parallel_dist, angle

    def will_new_loop_intersect_other_ss(self, ss1, ss2, tolerate_n_atoms=3):
        strand1_start, strand1_end, strand2_start, strand2_end = self.get_ss_terminal_residues(ss1, ss2, shortened=True)
        overlap = []
        for other_ss in self.ss_groups:
            if other_ss in (ss1, ss2):
                continue
            for other_residue in other_ss:
                bad_resi = False
                for other_atom in other_residue:
                    d = np.linalg.norm(np.cross(strand2_start.coord-strand1_end.coord,
                        strand1_end.coord-other_atom.coord))/np.linalg.norm(strand2_start.coord-strand1_end.coord)
                    if d<1:
                        bad_resi = True
                        break
                if bad_resi:
                    overlap.append(other_residue)

        return len(overlap)>tolerate_n_atoms, overlap

        if overlap<tolerate_n_atoms:
            newSS1 = tuple(reversed(ss2))
            if newSS1 not in self.reverse_strands:
                self.loop_for_ss[newSS1] = self.loop_for_ss[ss1]
                self.original_order[newSS1] = f"^{self.original_order[ss1]}"
                self.ss_type[newSS1] = self.ss_type[ss1]
                self.reverse_strands.append(newSS1)

            self.allowed_ss_combos.append((newSS1, ss2))

    def get_allowable_loop_combinations(self):
        """2. The distances between the N-terminus of one SSS and the C-terminus of
        another SSS were calculated for all SSS pairs. The N and C termini of
        two SSSs were allowed to connect by building a new loop between them if
        their distance is less than a cutoff distance (15 angstroms initially).
        The connection between two N (or C) termini of two SSSs was not allowed
        in order to maintain the original N to C direction. The original loops
        longer than 15 angstroms were unchanged."""

        #Get all allowable combos without reversing strands
        super().get_allowable_loop_combinations()

        for ss1, ss2 in permutations(self.ss_groups, 2):
            strands_parallel, parallel_dist, angle = self.strands_parallel(ss1, ss2)
            strands_anti_parallel, anti_parallel_dist, _ = self.strands_anti_parallel(ss1, ss2)

            print(self.original_order[ss1], ss1[-1].get_id(), "->", self.original_order[ss2], ss2[-1].get_id(), "parallel?", parallel_dist, np.rad2deg(angle))
            print(self.original_order[ss1], ss1[-1].get_id(), "->", self.original_order[ss2], ss2[0].get_id(), "anti parallel?", anti_parallel_dist, np.rad2deg(angle))

            rev_ss1 = self.reverse_strands[ss1]
            rev_ss2 = self.reverse_strands[ss2]

            if parallel_dist < self.cutoff and parallel_dist<anti_parallel_dist:
                print("Saved parallel")
                self.allowed_ss_combos.append((ss1, rev_ss2))
                self.allowed_ss_combos.append((rev_ss1, ss2))
                self.loop_distance[(ss1, rev_ss2)] = parallel_dist
                self.loop_distance[(rev_ss1, ss2)] = parallel_dist
            elif anti_parallel_dist < self.cutoff and anti_parallel_dist<parallel_dist:
                print("Saved antiparallel")
                #Both flip
                self.allowed_ss_combos.append((rev_ss1, rev_ss2))
                self.loop_distance[(rev_ss1, rev_ss2)] = anti_parallel_dist
            else:
                might_overlap, overlap = self.will_new_loop_intersect_other_ss(ss1, ss2)
                print("   might_overlap?", might_overlap, overlap)
                if angle < np.pi/2 and not might_overlap:
                    self.allowed_ss_combos.append((ss1, rev_ss2))
                    self.allowed_ss_combos.append((rev_ss1, ss2))
                    self.loop_distance[(ss1, rev_ss2)] = parallel_dist
                    self.loop_distance[(rev_ss1, ss2)] = parallel_dist

                elif angle > np.pi/2 and not might_overlap:
                    #If antiparallel, flip both
                    self.allowed_ss_combos.append((rev_ss1, rev_ss2))
                    self.loop_distance[(rev_ss1, rev_ss2)] = anti_parallel_dist

    def _permutations(self):
        #Get reverse strands without leading/traling residues
        rev_strands = sorted([rev_ss for orig_ss, rev_ss in self.reverse_strands.items() \
            if orig_ss not in self.leading_trailing_residues], key=lambda x: self.original_order[x])

        #Make a group to choose from which SS. eg [(ss1, rev_ss2), (ss2, rev_ss2)]
        #So only one of those can be chosen
        groups = list(zip(self.ss_groups, rev_strands))
        groups = [[*g, None] for g in groups]

        for which_ss in product(*groups):
            if None in which_ss: continue

            for i, ss in enumerate(which_ss):
                name = self.original_order[ss]
                if 0<i<len(which_ss)-1 and isinstance(name, str) and name.startswith("^"):
                    break
            else:
                continue

            for ss_perm in permutations(which_ss, self.number_ss):
                order = [str(self.original_order[ss]) for ss in ss_perm]
                if order[0]=="1" or "^6" in order or "^7" in order or order[0].startswith("^") or order[-1].startswith("^"):
                    continue
                yield ss_perm

    def write_residue_group(self, perm_pdb, residue_group, start_resseq, start_atom_number, no_loop_sequence, loop_sequence, mark_unknown=False):
        mark_unknown = mark_unknown or residue_group in self.reverse_strands.values()
        return super().write_residue_group(perm_pdb, residue_group, start_resseq, start_atom_number, no_loop_sequence, loop_sequence, mark_unknown=mark_unknown)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("cath_domain")
    parser.add_argument("cathcode")
    parser.add_argument("-c", "--cutoff", default=15, type=int)
    parser.add_argument("--short_loops", default=False, action="store_true")
    parser.add_argument("--random_loops", default=False, action="store_true")

    args = parser.parse_args()

    StrandReversalMLP(args.cath_domain, args.cathcode, cutoff=args.cutoff,
        short_loops=args.short_loops, random_loops=args.random_loops)
