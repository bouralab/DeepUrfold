#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import glob
import pickle
from itertools import groupby, cycle, islice

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram as dendrogram2
from sklearn.cluster import AgglomerativeClustering as AgglomerativeClustering2
from scipy.stats import norm
from scipy.optimize import brentq
from sklearn.neighbors import KernelDensity

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerPatch
import seaborn as sns

#Must be before pytorch
import graph_tool.all as gt
import graph_tool as gta

import torch



# In[2]:


def get_distances(compare_dir, sample=False):
#     models = {}
#     distances = None
#     for f in glob.glob(os.path.join(compare_dir, "*.csv")):
#         df = pd.read_csv(f)
#         df = df[["cathDomain", "ELBO"]]
#         sources = dict(kv.split("=") for kv in f.split("-")[-2].split("_"))
#         #df = df.rename(columns={"ELBO":sources["model"]+"-ELBO"})
#         df = df.assign(true_sfam=sources["input"], model=sources["model"])
#         if distances is not None: #sources["model"] in models:
#             distances = pd.concat((distances, df))
#         else:
#             distances = df
#
#     distances = distances.pivot_table(index=['cathDomain', 'true_sfam'], columns='model')
#
#     return distances
#
# def load_distances(compare_dir):
    distances = None
    for f in glob.glob(os.path.join(compare_dir, "*", "*", "elbo", "*.pt")):
        fields = f.split("/")
        model_sfam = fields[-4]
        input_sfam = fields[-3]
        representatives = f"/home/bournelab/data-eppic-cath-features/train_files/{input_sfam.replace('.', '/')}/DomainStructureDataset-representatives.h5"
        df = pd.read_hdf(representatives, "table")[["cathDomain"]]
        df = df.assign(true_sfam=input_sfam, model=model_sfam)
        elbo_scores = pd.Series(torch.load(f).numpy(), name="ELBO", index=df.index)
        df = pd.concat((df, elbo_scores), axis=1)
        if distances is not None: #sources["model"] in models:
            distances = pd.concat((distances, df))
        else:
            distances = df

    #distances = distances.set_index(["cathDomain", "true_sfam"])
    distances = distances.pivot_table(index=['cathDomain', 'true_sfam'], columns='model')
    return distances


class AllVsAll(object):
    def __init__(self, compare_dir, sample=50):
        self.distances = get_distances(compare_dir)
        self.sfams = self.distances.index.get_level_values('true_sfam').drop_duplicates()
        self.create_discriminators()
        if sample:
            if isinstance(sample, int):
                self.sample(n=sample)
            else:
                self.sample()

    def sample(self, n=50):
        def _sample(sfam):
            domain_scores = self.distances.loc[(slice(None), sfam), :]
            try:
                domain_scores = domain_scores.sample(n=n)
            except ValueError:
                pass
            return domain_scores
        self.original_distances = self.distances.copy()
        self.distances = pd.concat(_sample(sfam) for sfam in self.sfams)
        print(f"Size {len(self.original_distances)} -> {len(self.distances)}")

    def create_discriminators(self):
        self.kde_descriminators = {}
        self.log_odds_descriminator = {}
        self.log_odds = {}
        for sfam in self.sfams:
            try:
                self.get_descriminator(sfam)
            except KeyError:
                #Skip key
                continue

    def find_communities_vary_paramaters(self):
         for nested in (True, False):
            kwds = {"nested":nested}
            for overlap in (True, False):
                kwds["overlap"] = overlap
                for deg_corr in  (True, False):
                    kwds["deg_corr"] = deg_corr
                    for weighted in (True, False):
                        kwds["weighted"] = weighted
                        for score_type in ("elbo", "log_odds"):
                            kwds["score_type"] = score_type
                            for descriminator in ("kde", "log_odds", None):
                                if descriminator == "kde":
                                    kwds["kde_descriminator"] = True
                                    kwds["log_odds_descriminator"] = False
                                elif descriminator == "log_odds":
                                    kwds["kde_descriminator"] = False
                                    kwds["log_odds_descriminator"] = True
                                else:
                                    kwds["kde_descriminator"] = False
                                    kwds["log_odds_descriminator"] = False

                                self.find_communities(**kwds)
                                # for i in range(2):
                                #     try:
                                #         self.find_communities(**kwds)
                                #         break
                                #     except AttributeError:
                                #         continue
                                # else:
                                #     #Failed
                                #     with open("failed.txt", "a") as f:
                                #         print(kwds, file=f)
    def find_communities(self, nested=True, overlap=True, deg_corr=True, kde_descriminator=False, log_odds_descriminator=False, weighted=True, score_type="elbo"):
        self.g = gt.Graph()
        #v_domain = g.new_vertex_property("str")
        #v_sfam = g.new_vertex_property("str")

        pefix_data = {"nested":nested, "overlap":overlap, "deg_corr":deg_corr,
            "weighted":weighted, "kde_descriminator":kde_descriminator, "log_odds_descriminator":log_odds_descriminator, "score_type":score_type}
        prefix = "urfold-sbm-"
        prefix += "_".join([f"{k}={v}" for k,v in pefix_data.items()])

        if not os.path.isdir(prefix):
            os.makedirs(prefix)

        print("-"*20)
        print("Running", prefix)
        print("-"*20)

        self.v_label = self.g.new_vertex_property("string")
        self.e_elbo = self.g.new_edge_property("float")
        self.domain_vertices = {}
        self.sfam_vertices = {}
        for sfam in self.sfams:
            if kde_descriminator and sfam not in self.kde_descriminators:
                #Skip key
                continue
            if log_odds_descriminator and sfam not in self.log_odds_descriminator:
                #Skip key
                continue
            v = self.g.add_vertex()
            self.sfam_vertices[sfam] = v
            self.v_label[v] = f"SUPERFAMILY={sfam}"


        for (cathDomain, true_sfam), row in self.distances.iterrows():
            v = self.g.add_vertex()
            self.domain_vertices[cathDomain] = v
            self.v_label[v] = f"({cathDomain}, {true_sfam})"

            log_odd_thru_true_sfam = self.log_odds_descriminator.get(true_sfam)

            skipped = 0
            for (_, model), elbo in row.iteritems():
                try:
                    sfam_vertex = self.sfam_vertices[model]
                except KeyError:
                    continue

                if kde_descriminator and self.kde_descriminators[model] is not None and elbo > self.kde_descriminators[model]:
                    print("KDE skipped")
                    skipped += 1
                    continue
                elif log_odds_descriminator and log_odd_thru_true_sfam is not None:
                    print("Log skipped")
                    try:
                        if not self.log_odds_descriminator[model][cathDomain]:
                            skipped += 1
                            continue
                    except KeyError:
                        skipped += 1
                        continue

                e = self.g.add_edge(v, sfam_vertex)

                if score_type == "elbo":
                    self.e_elbo[e] = -np.log(elbo)
                elif score_type == "log_odds":
                    self.e_elbo[e] = self.log_odds[model][cathDomain]

            print(cathDomain, true_sfam, "SKIPPED", skipped, "sfams")

        del_list = [sfam_vertex for sfam, sfam_vertex in self.sfam_vertices.items() if \
            len(self.g.get_all_edges(sfam_vertex))==0]
        for v in reversed(sorted(del_list)):
            self.g.remove_vertex(v)

        print("n=", self.g.num_vertices(), "e=", self.g.num_edges(), "<?", self.g.num_vertices()*(len(self.sfams)-len(del_list)))

        block_model = gta.inference.minimize_nested_blockmodel_dl if nested else \
            gta.inference.minimize.minimize_blockmodel_dl

        state_args = dict(overlap=overlap, deg_corr=deg_corr)
        if weighted:# not kde_descriminator and not log_odds_descriminator:
            state_args.update(dict(recs=[self.e_elbo], rec_types=["real-normal"]))

        print(state_args)

        print("Creating block model...")

        state_cls = gta.inference.overlap_blockmodel.OverlapBlockState if overlap else \
            gta.inference.blockmodel.BlockState

        self.state = block_model(self.g, state_args=state_args)

        print("Finished creating block model")

        # improve solution with merge-split
        self.state = self.state.copy(bs=self.state.get_bs() + [np.zeros(1)] * 4, sampling=True)

        print("Improving block model", end="")

        for i in range(100):
            if i%10==0: print(".", end="")
            ret = self.state.multiflip_mcmc_sweep(niter=10, beta=np.inf)

        print("\nFinished improving block model")

        with open(f"{prefix}/{prefix}.pickle", "wb") as f:
            pickle.dump(self.state, f)

        self.get_group_memberships(self.state, prefix)
        self.draw(prefix)
        import pdb; pdb.set_trace()

    def get_group_memberships(self, state, prefix):
        # b = gt.contiguous_map(state.get_blocks())
        # state = state.copy(b=b)

        levels = state.get_levels()


        for n_levels, s in enumerate(levels):
            # e = s.get_matrix()
            # B = s.get_nonempty_B()
            # plt.matshow(e.todense()[:B, :B])
            # plt.savefig(f"{prefix}/{prefix}-{n_levels}-edge-counts.pdf")
            if s.get_N() == 1:
                break


        self.domain_group_membership = self.distances.assign(**{f"Level {l}":np.nan for l in range(n_levels+1)})

        for cathDomain, domain_vertex in self.domain_vertices.items():
            r = domain_vertex
            for i, s in enumerate(levels):
                r = s.get_blocks()[r]
                self.domain_group_membership.loc[cathDomain, f"Level {i}"] = r

                if s.get_N() == 1:
                    break

        self.domain_group_membership.to_hdf(f"{prefix}/{prefix}-domain-group-membership.h5", "table")

        self.sfam_group_membership = pd.DataFrame(
            {f"Level {l}":np.nan for l in range(n_levels+1)},
            index=self.sfams)

        for sfam, sfam_vertex in self.sfam_vertices.items():
            r = sfam_vertex
            for i, s in enumerate(levels):
                r = s.get_blocks()[r]
                self.sfam_group_membership.loc[sfam, f"Level {i}"] = r

                if s.get_N() == 1:
                    break

        self.sfam_group_membership.to_hdf(f"{prefix}/{prefix}-sfam-group-membership.h5", "table")

        vorder = self.g.degree_property_map("total")

        import pdb; pdb.set_trace()

        return

        groups = group_membership.groupby([f"Level {l}" for l in range(len(levels+1))])
        for n, g in groups:
            print(n, len(g), g.index.to_frame().reset_index(drop=True)["true_sfam"].drop_duplicates())


    def draw(self, prefix="urfold-sbm"):

        #self.mplfig = plt.figure(figsize=[56,56], frameon=False)

        self.state.draw(
            # vertex_text_position="centered",
            # vertex_text=self.v_label,
            # vertex_size=10,
            # vertex_font_size=9,
            vertex_size=1,
            edge_color=gt.prop_to_size(all_vs_all.e_elbo, power=1, log=True),
            ecmap=(matplotlib.cm.viridis, .6),
            eorder=self.e_elbo,
            edge_pen_width=gt.prop_to_size(all_vs_all.e_elbo, 1, 4, power=1, log=True),
            edge_gradient=[],
            hedge_color="#555555",
            hvertex_fill_color="#555555",
            output=f"{prefix}/{prefix}.pdf",
            output_size=[4024,4024],
            #mplfig=self.mplfig
        )



        t = gt.get_hierarchy_tree(self.state)[0]
        tpos = pos = gt.radial_tree_layout(t, t.vertex(t.num_vertices() - 1), weighted=True)
        cts = gt.get_hierarchy_control_points(self.g, t, tpos)
        self.pos = self.g.own_property(tpos)
        b = self.state.levels[0].b

        import pdb; pdb.set_trace()

        from itertools import chain

        verts = {vert:name for name,vert in chain(self.domain_vertices.items(), self.sfam_vertices.items()}
        vertex_positions = {v:vpos for vpos, v in zip(self.pos, self.g.vertices())}

        group_domain_positions = {}

        domain_groups = self.domain_group_membership.groupby([f"Level {l}" for l in range(len(levels+1))])
        sfam_groups = self.sfam_group_membership.groupby([f"Level {l}" for l in range(len(levels+1))])
        all_groups = set(domain_groups.groups.keys()).intersection(set(sfam_groups.groups.keys()))
        for group_name in all_groups:
            try:
                domains = domain_groups.get(group_name)
                group_vertices = {cathDomain:self.domain_vertices[cathDomain] for cathDomain in domains.index \
                    if cathDomain in self.domain_vertices}
            except KeyError:
                vertices = []

            try:
                sfams = sfam_groups.get(group_name)
                group_vertices += {sfam:self.sfam_vertices[sfam] for sfam in sfams.index \
                    if sfam in self.domain_vertices}
            except KeyError:
                pass

            group_positions = {label:vertex_positions[vert] for label, vert in group_vertices}
            domain_order = list(sorted(group_positions.items(), key=lambda g,p:np.arctan(p), reversed=True))
            group_domain_positions[group_name] = domain_order

        group_order = sorted(group_domain_positions.items(), key=\
            lambda name,domains: np.arctan(domains[-1][1], reversed=True))

        mapping = {
            'N': np.pi * 0.5,
            'NW': np.pi * 0.75,
            'W': np.pi,
            'SW': np.pi * 1.25,
            'S': np.pi * 1.5,
            'SE': np.pi * 1.75,
            'E': 0,
            'NE': np.pi * 0.25}

        start_pos = np.arctan(group_order[0][0][0])

        best_dist = 100
        best_dir = None
        for direction, rot in mapping.items():
            dist = np.linalg.norm(start_pos-rot)
            if dist<best_dist:
                best_dist = dist
                best_dir = direction

        if start_pos < mapping[best_dir]:
            pass

        offset = np.arctan(group_order[0][0])
        if offset>np.pi/2:
            offset = (offset-np.pi/2)*-1




        return



        bstack = self.state.get_bstack()
        t = gt.get_hierarchy_tree(self.state)[0]
        tpos = pos = gt.radial_tree_layout(t, t.vertex(t.num_vertices() - 1), weighted=True)
        cts = gt.get_hierarchy_control_points(self.g, t, tpos)
        pos = self.g.own_property(tpos)
        b = bstack[0].vp["b"]

        text_rot = self.g.new_vertex_property('double')
        self.g.vertex_properties['text_rot'] = text_rot
        for v in self.g.vertices():
            if pos[v][0] >0:
                text_rot[v] = math.atan(pos[v][1]/pos[v][0])
            else:
                text_rot[v] = math.pi + math.atan(pos[v][1]/pos[v][0])

#         bv, bc_in, bc_out, bc_total = self.state.get_overlap_blocks()
#         if deg_corr:
#             pie_fractions = bc_total.copy("vector<double>")
#         else:
#             pie_fractions = self.g.new_vp("vector<double>",
#                                           vals=[ones(len(bv[v])) for v \
#                                                 in self.g.vertices()])

        pickle.dump(self, "urfold-sbm.pickle")

        return self.state.draw(
            pos=pos,
            vertex_text_rotation=self.g.vertex_properties['text_rot'],
            vertex_text=self.v_label,
            vertex_size=10,
            vertex_font_size=9,
            vertex_text_position=1,
            vertex_anchor=0,
            edge_control_points=cts,
            #vertex_text_offset=1,
            edge_color=gt.prop_to_size(self.e_elbo, power=1), #, log=True),
            ecmap=(matplotlib.cm.inferno, .6),
            eorder=self.e_elbo,
            edge_pen_width=gt.prop_to_size(self.e_elbo, 1, 4, power=1, log=True),
            edge_gradient=[],
            output="urfold-sbm.pdf",
            output_size=[4024,4024]
        )





        print("Drawing block model...")

        return self.state.draw(
            vertex_text=v_label,
            vertex_size=5,
            vertex_text_position=4,
            #vertex_text_offset=1,

            edge_color=gt.prop_to_size(e_elbo, power=1, log=True),
            ecmap=(matplotlib.cm.inferno, .6),
            eorder=e_elbo, edge_pen_width=gt.prop_to_size(e_elbo, 1, 4, power=1, log=True),
            edge_gradient=[],
            output="urfold-sbm.pdf")





#         gt.graph_draw(
#             g,
#             #pos=pos,
#             #vertex_fill_color=g.vertex_properties['plot_color'],
#             #vertex_color=g.vertex_properties['plot_color'],
#             #edge_control_points=cts,
#             vertex_size=10,
#             vertex_text=g.vertex_properties['label'],
#             vertex_text_rotation=g.vertex_properties['text_rot'],
#             vertex_text_position=1,
#             vertex_font_size=9,
#             vertex_shape="pie",
#             vertex_pie_colors=bv,
#             vertex_pie_fractions=pie_fractions,
#             edge_color=gt.prop_to_size(e_elbo, power=1, log=True),
#             #edge_color=g.edge_properties['edge_color'],
#             ecmap=(matplotlib.cm.inferno, .6),
#             eorder=e_elbo, edge_pen_width=gt.prop_to_size(e_elbo, 1, 4, power=1, log=True),
#             edge_gradient=[],
#             vertex_anchor=0,
#             #bg_color=[0,0,0,1],
#             output_size=[4024,4024],
#             output='urfold-sbm-labels.pdf')

        print("Finished rawing block model")


#         from graph_tool.draw import graph_draw
#         return graph_draw(g,
#                           vertex_shape=kwargs.get("vertex_shape", "pie"),
#                           vertex_pie_colors=kwargs.get("vertex_pie_colors", bv),
#                           vertex_pie_fractions=kwargs.get("vertex_pie_fractions",
#                                                           pie_fractions),
#                           edge_gradient=gradient,
#                           **dmask(kwargs, ["vertex_shape", "vertex_pie_colors",
#                                            "vertex_pie_fractions",
#                                            "edge_gradient"]))

    # Fit KDE
    def _kde_sklearn(self, x, x_grid, **kwargs):
        """Kernel Density Estimation with Scikit-learn"""
        from seaborn.external.kde import gaussian_kde
        bw = gaussian_kde(x).scotts_factor() * x.std(ddof=1)
        kde_skl = KernelDensity(bandwidth=bw, kernel="gaussian", **kwargs)
        kde_skl.fit(x[:, np.newaxis])
        # score_samples() returns the log-likelihood of the samples
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
        return kde_skl, np.exp(log_pdf)

    # Find intersection
    def _findIntersection(self, fun1, fun2, lower, upper):
        return brentq(lambda x : fun1(x) - fun2(x), lower, upper)

    def get_descriminator(self, sfam_name, plot=True):
        self.get_descriminator_logodds(sfam_name)
        return self.get_descriminator_kde(sfam_name, plot=plot)

    def _plot(self, sfam, other, sfam_name, prefix="", vertical_lines={}): #descriminator=None, max_sfam=None, max_other=None):
        fig = plt.figure(figsize=(10, 6), dpi=300)
        ax = fig.subplots(1, 1)
        if sfam is not None:
            sns.kdeplot(sfam, label=f"True {sfam_name} (SNS)", ax=ax)
        sns.kdeplot(other, label="Other Superfamilies (SNS)", ax=ax)

        # if sklearn_kde is not None:
        #     sns.scatterplot(x_axis, pdfSfam, label=f"True {sfam_name} (sklearn)", ax=ax)
        #     sns.scatterplot(x_axis, pdfOther, label=f"Other Superfamilies (sklearn)", ax=ax)
        # if peaks is not None:
        #     plt.plot(peaks, sfam.values[peaks])
        colors = sns.color_palette("hls", 8)
        for i, (name, value) in enumerate(vertical_lines.items()):
            if value is None: continue
            ax.axvline(value, label="{} ({:.4f})".format(name, value), color=colors[i])
        # if descriminator is not None:
        #     ax.axvline(descriminator, label=f"Discriminator ({descriminator})", color='red')
        # if max_sfam is not None:
        #     ax.axvline(max_sfam, label=f"Max {sfam_name} ({max_sfam})", color='purple')
        # if max_other is not None:
        #     ax.axvline(max_other, label=f"Max Other ({max_other})", color='pink')

        plt.legend()
        plt.savefig(f"{prefix}_kde.png")

    def find_descriminator(self, sfam, other, sfam_name):
        x_axis = np.linspace(0, .5, 500)
        from scipy.signal import find_peaks
        kdeSfam, pdfSfam = self._kde_sklearn(sfam, x_axis)
        kdeOther, pdfOther = self._kde_sklearn(other, x_axis)
        funcA = lambda x: kdeSfam.score_samples([[[x]]][0])
        funcB = lambda x: kdeOther.score_samples([[[x]]][0])

        sfamPeaks = find_peaks(pdfSfam)[0]
        otherPeaks = find_peaks(pdfOther)[0]

        try:
            max_sfam = x_axis[ sfamPeaks[pdfSfam[sfamPeaks].argmax()] ]
            max_other = x_axis[ otherPeaks[pdfOther[otherPeaks].argmax()] ]

            no_max=False
        except ValueError:
            no_max=True
            max_sfam = sfam.mean()
            max_other = other.mean()

        search_factor = 1
        if max_other <= max_sfam:
            #Something is wierd, but ok
            search_factor = 3

        try:
            descriminator = self._findIntersection(funcA, funcB,
                max_sfam, #sfam.mean(), #-sfam.std()*1,
                max_other+other.std()*search_factor #.3
                )
        except ValueError as e:
            print("Failed", sfam_name)
            descriminator = None

        if no_max:
            max_sfam = None
            max_other = None

        return descriminator, max_sfam, max_other

    def get_descriminator_kde(self, sfam_name, plot=True):
        x_axis = np.linspace(0, .5, 500)
        one_sfam = self.distances[('ELBO', sfam_name)].to_frame().reset_index()
        one_sfam.columns = [x[0] for x in one_sfam.columns.tolist()]
        one_sfam = one_sfam.drop(columns=["cathDomain"])
        sfam = one_sfam[one_sfam["true_sfam"]==sfam_name]["ELBO"]
        other = one_sfam[one_sfam["true_sfam"]!=sfam_name]["ELBO"]

        # from scipy.signal import find_peaks
        # #
        #
        # kdeSfam, pdfSfam = self._kde_sklearn(sfam, x_axis)
        # kdeOther, pdfOther = self._kde_sklearn(other, x_axis)
        # funcA = lambda x: kdeSfam.score_samples([[[x]]][0])
        # funcB = lambda x: kdeOther.score_samples([[[x]]][0])
        #
        # sfamPeaks = find_peaks(pdfSfam)[0]
        # otherPeaks = find_peaks(pdfOther)[0]
        #
        # max_sfam = x_axis[ sfamPeaks[pdfSfam[sfamPeaks].argmax()] ]
        # max_other = x_axis[ otherPeaks[pdfOther[otherPeaks].argmax()] ]
        #
        # search_factor = 1
        # if max_other <= max_sfam:
        #     #Something is wierd, but ok
        #     search_factor = 3
        #
        #
        #
        # try:
        #     descriminator = self._findIntersection(funcA, funcB,
        #         max_sfam, #sfam.mean(), #-sfam.std()*1,
        #         max_other+other.std()*search_factor #.3
        #         )
        # except ValueError as e:
        #     print("Failed", sfam_name)
        #     descriminator = None
        #
        # # peaks, _ = find_peaks(sfam.values, height=0)
        # # peaks = kdeSfam.score_samples([[[peaks]]][0])
        # # print(peaks)

        descriminator, max_sfam, max_other = self.find_descriminator(sfam, other, sfam_name)
        prefix=f"{sfam_name}-elbo"
        self._plot(sfam, other, sfam_name, prefix, vertical_lines=
            {"Descriminator":descriminator,
            f"Max Sfam {sfam_name}":max_sfam,
            f"Max Other {sfam_name}":max_other})

        # if plot:
        #     _plot(d)

        self.kde_descriminators[sfam_name] = descriminator

    def get_descriminator_logodds(self, sfam_name, plot=True):
        one_sfam = self.distances[('ELBO', sfam_name)].to_frame().reset_index("true_sfam")
        one_sfam.columns = [x[0] for x in one_sfam.columns.tolist()]
        sfam = one_sfam[one_sfam["true_sfam"]==sfam_name]["ELBO"]
        other = one_sfam[one_sfam["true_sfam"]!=sfam_name]["ELBO"]

        sfam_log_median_elbo = np.log(sfam.median())
        sfam_log_elbo = sfam.apply(np.log).rename("log(ELBO)")
        other_log_elbo = other.apply(np.log).rename("log(ELBO)")
        log_odds = other_log_elbo-sfam_log_median_elbo
        sfam_log_odds = sfam_log_elbo-sfam_log_median_elbo
        log_odds = pd.concat((log_odds, sfam_log_elbo), axis=0)
        log_odds = log_odds.rename("Log-odds Score")
        self.log_odds[sfam_name] = log_odds
        self.log_odds_descriminator[sfam_name] = log_odds<0.05

        descriminator, max_sfam, max_other = self.find_descriminator(sfam_log_elbo, other_log_elbo, sfam_name)

        self._plot(sfam_log_elbo, other_log_elbo, sfam_name, f"{sfam_name}-log_elbo",
            vertical_lines=
                {"Descriminator":descriminator,
                f"Max Sfam {sfam_name}":max_sfam,
                f"Max Other {sfam_name}":max_other})

        self._plot(sfam_log_odds, log_odds, sfam_name, f"{sfam_name}-log_odds", vertical_lines=
            {f"Median log(ELBO) ({sfam_name})":sfam_log_median_elbo})

if __name__ == "__main__":
    all_vs_all = AllVsAll("/home/bournelab/urfold_runs/superfamilies_for_paper")

    if len(sys.argv)>1 and sys.argv[1] == "--vary-parameters":
        all_vs_all.find_communities_vary_paramaters()
    else:
        all_vs_all.find_communities()
    # except:
    #     raise
    #     import traceback as tb
    #     print(tb.format_exc())
    #     import pdb; pdb.set_trace()

# import math
# bstack = all_vs_all.state.get_bstack()
# t = gt.get_hierarchy_tree(all_vs_all.state)[0]
# tpos = pos = gt.radial_tree_layout(t, t.vertex(t.num_vertices() - 1), weighted=True)
# cts = gt.get_hierarchy_control_points(all_vs_all.g, t, tpos)
# pos = all_vs_all.g.own_property(tpos)
# b = bstack[0].vp["b"]
#
#
# d = get_distances("/project/ppi_workspace/ed4bu/urfold_compare/compare/")
#
#
#
# dist = d.pivot_table(index=['cathDomain', 'true_sfam'], columns='model')
#
# # In[14]:
#
#
# one_sfam = all_vs_all.distances[('ELBO', '2.30.30.100')].to_frame().reset_index()
# one_sfam.columns = [x[0] for x in one_sfam.columns.tolist()]
# one_sfam = one_sfam.drop(columns=["cathDomain"])
# sns.kdeplot(one_sfam[one_sfam["true_sfam"]=="2.60.40.10"]["ELBO"])
# sns.kdeplot(one_sfam[one_sfam["true_sfam"]=="2.30.30.100"]["ELBO"])
# sns.kdeplot(one_sfam[one_sfam["true_sfam"]=="2.40.50.140"]["ELBO"])
#
#
# fig = plt.figure(figsize=(10, 6), dpi=300)
# ax = fig.add_subplot(1, 1, 1)
# ig1 = sns.kdeplot(one_sfam[one_sfam["true_sfam"]=="2.60.40.10"]["ELBO"], label="2.60.40.10", ax=ax)
# sh3 = sns.kdeplot(one_sfam[one_sfam["true_sfam"]=="2.30.30.100"]["ELBO"], label="2.30.30.100", ax=ax)
# sns.kdeplot(one_sfam[one_sfam["true_sfam"]=="2.40.50.140"]["ELBO"], label="2.40.50.140", ax=ax)
# plt.legend()
#
# fig = plt.figure(figsize=(10, 6), dpi=300)
# ax = fig.add_subplot(1, 1, 1)
# ig1 = sns.kdeplot(one_sfam[one_sfam["true_sfam"]=="2.60.40.10"]["ELBO"], label="2.60.40.10", ax=ax)
# other = sns.kdeplot(one_sfam[(one_sfam["true_sfam"]=="2.30.30.100")|(one_sfam["true_sfam"]=="2.40.50.140")]["ELBO"], label="Other Superfamilies", ax=ax)
# ax.axvline(result, color='red')
# #sns.kdeplot(one_sfam[one_sfam["true_sfam"]=="2.40.50.140"]["ELBO"], label="2.40.50.140", ax=ax)
# plt.legend()
#
#
# # In[88]:
#
#
# from scipy.stats import norm
# from scipy.optimize import brentq
# from sklearn.neighbors.kde import KernelDensity
#
#
#
# has_common_substrucutre("2.30.30.100")
#
#
# # In[115]:
#
#
# result
#
#
# # In[116]:
#
#
# # Plot
# f, ax = plt.subplots(1, 1)
# ax.plot(x_axis, pdfIg, color='green')
# ax.plot(x_axis, pdfOther, color='blue')
# ax.set_title('KDEs of subsampled Gaussians')
# ax.axvline(result, color='red')
# plt.show()
#
#
# # In[107]:
#
#
# from seaborn import KDE
#
#
# # In[9]:
#
#
# import seaborn as sns
#
#
# # In[34]:
#
#
# sns.set_theme(style="ticks")
#
#
# # In[35]:
#
#
# df = all_vs_all.distances.reset_index()
# df.columns = [c[0] if len(c[1])==0 else c[1] for c in df.columns]
# sns.pairplot(data=df, hue="true_sfam")
# all_vs_all.distances.reset_index()
#
#
# # In[54]:
#
#
#
#
#
# sns.kdeplot(data=one_sfam, x="ELBO", hue="true_sfam", log_scale=True)
#
#
# # In[57]:
#
#
# one_sfam
#
#
# # In[ ]:
