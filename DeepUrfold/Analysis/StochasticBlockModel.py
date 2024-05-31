from operator import concat
import sys
import os
import glob
import pickle
import argparse
import subprocess
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import h5pyd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#Must be before pytorch
try:
    import graph_tool.all as gt
    import graph_tool as gta
except (ImportError, ModuleNotFoundError):
    print("Unable to run StochasticBlockModel without graph_tool")
    gt = gta = None

from DeepUrfold.Analysis.Clustering import Clustering

from Prop3D.parsers.cath import CATHApi

from pandarallel import pandarallel

pandarallel.initialize(nb_workers=16, progress_bar=True)

#Note: Must be import before PyTorch to aboid GOMP5 issues

bs = []

def calculate_enrichment(feats, group_key=None, prefix=None):
    from goatools import obo_parser
    from sklearn.metrics import pairwise_distances
    from goatools.semantic import semantic_similarity
    from goatools.go_enrichment import GOEnrichmentStudy
    from goatools.mapslim import mapslim
    from itertools import groupby
    from collections import Counter

    if group_key in [None, "deepurfold"]:
        group_key = [c for c in feats.columns if c.startswith("l_")]
        if prefix is None:
            prefix = group_key
    elif group_key == "cath":
        group_key = "sfam"
        if prefix is None:
            prefix = group_key
    else:
        if isinstance(group_key, (list, tuple)) and set(feats.columns).issubset(group_key):
            pass
        elif isinstance(group_key, str) and group_key in feats.columns:
            pass
        else:
            raise RuntimeError(f"Invalid group_key: {group_key}")

    if prefix is None:
        prefix = ""

    go_dag_file = Path('go-basic.obo')
    slim_file = Path("goslim_agr.obo")

    if not go_dag_file.exists():
        urllib.request.urlretrieve("http://geneontology.org/ontology/go-basic.obo", "go-basic.obo")

    if not slim_file.exists():
        urllib.request.urlretrieve("http://current.geneontology.org/ontology/subsets/goslim_agr.obo", "goslim_agr.obo")

    go_dag = obo_parser.GODag('go-basic.obo')
    obodolete_dag = obo_parser.GODag('go-basic.obo', optional_attrs={'consider', 'replaced_by'}, load_obsolete=True, prt=None)
    slim = obo_parser.GODag("goslim_agr.obo")
    print(feats)
    population_ids = set(feats.index)
    id2gos = {n:set(go) if len(go)>1 or (len(go)==1 and go[0]!="") else set() \
        for n, go in feats["go_acc"].str.split("+").to_dict().items()}
    id2gos = {}
    id2go_slims = {}
    for n, go in feats["go_acc"].str.split("+").to_dict().items():
        _go_acc = set()
        slim_go_terms = set()
        for go_t in go:
            if go_t == "": continue
            node = obodolete_dag[go_t]
            if not node.is_obsolete:
                _go_acc.add(go_t)
            else:
                if node.replaced_by[:3] == "GO:":
                    _go_acc.add(node.replaced_by)
                elif node.consider:
                    for new in node.consider:
                        if go_dag[new].namespace == "molecular_function":
                            _go_acc.add(new)
                            break
                    else:
                        for new in node.consider:
                            if go_dag[new].namespace == "biological_process":
                                _go_acc.add(new)
                                break
                        else:
                            for new in node.consider:
                                if go_dag[new].namespace == "cellular_component":
                                    _go_acc.add(new)
                                    break


        for got in _go_acc:
            slim_go_terms = slim_go_terms.union(mapslim(got, go_dag, slim)[0])
        id2go_slims[n] = slim_go_terms
        id2gos[n] = _go_acc
    print(population_ids)
    print(id2gos)
    goeaobj = GOEnrichmentStudy(
        population_ids,
        id2go_slims,
        slim, #go_dag,
        methods=['bonferroni', 'fdr_bh'],
        pvalcalc='fisher_scipy_stats')
    n_go = set()
    cc_go = set()
    enriched_go = []
    print(group_key)
    print(feats)
    for block, domains in feats.groupby(group_key):
        # go_acc = domains["go_acc"]
        # go_acc = go_acc[go_acc!=""].dropna().str.split("+", expand=True).values.flatten()
        # go_acc = go_acc[go_acc!=None]
        # _go_acc = []

        go_acc = [go_t for domain in domains.index for go_t in id2go_slims[domain]]

        #go_acc = _go_acc
        #go_acc = [obodolete_dag[go_t].replaced_by if obodolete_dag[go_t].is_obsolete else go_t for go_t in go_acc if go_t != ""]
        go_acc_c = Counter(go_acc)
        go_names = dict([(go_dag[go].name,c) for go, c in go_acc_c.most_common(10) if go_dag[go].namespace in ["molecular_function", "biological_process"]])
        cc_names = dict([(go_dag[go].name,(go, c)) for go, c in go_acc_c.most_common(10) if go_dag[go].namespace in ["cellular_component"]])
        n_go = n_go.union(set(go_names.keys()))
        cc_go = cc_go.union(set(cc_names.keys()))
        print(block, len(domains), go_names)

        results = goeaobj.run_study_nts(set(domains.index))
        enriched_go_group = [(r.GO, r.p_fdr_bh) for r in results if r.p_fdr_bh < 0.05 and r.enrichment=="e"]
        enriched_go += enriched_go_group

    enriched_go_grps = {n:list(g) for n, g in groupby(enriched_go, key=lambda x: x[0])}
    go_terms = list(enriched_go_grps.keys())

    go_bp = [term for term in go_terms if go_dag[term].namespace=="biological_process"]
    go_mf = [term for term in go_terms if go_dag[term].namespace=="molecular_function"]
    go_cc = [term for term in go_terms if go_dag[term].namespace=="cellular_component"]
    feats = feats.assign(go_bp="", go_mf="", go_cc="")
    for domain in feats.index:
        d_gos = id2go_slims[domain]
        if len(d_gos) == 0:
            continue


        for ns, terms in [("cc", go_cc), ("mf", go_mf), ("bp", go_bp)]:
            for e_go in terms:
                for d_go in d_gos:
                    if e_go == d_go:
                        feats.loc[domain, f"go_{ns}"] = e_go
                        break
                else:
                    continue
                break

    for ns, terms in [("cc", go_cc), ("mf", go_mf), ("bp", go_bp)]:
        with open(f"{prefix}-{ns}.txt", "w") as f:
            print("code,name", file=f)
            for go in terms:
                print(go, f"\"{go_dag[go].name}\"", sep=",", file=f)

    return feats


class StochasticBlockModel(Clustering):
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

    def find_communities(self, nested=True, overlap=True, deg_corr=True, kde_descriminator=False, log_odds_descriminator=False, weighted=True, score_type="elbo", force=False, prefix=None, old_flare=None, downsample=False):
        global bs
        if gt is None:
            raise RuntimeError("graph_tool must be installed in order to build the sctochastic block model")

        import numpy as np

        self.g = gt.Graph(directed=False)

        if prefix is None:
            pefix_data = {"nested":nested, "overlap":overlap, "deg_corr":deg_corr,
                "weighted":weighted, "kde_descriminator":kde_descriminator,
                "log_odds_descriminator":log_odds_descriminator, "score_type":score_type, "downsample":downsample}
            prefix = "urfold-sbm-"
            prefix += "_".join([f"{k}={v}" for k,v in pefix_data.items()])

            if not os.path.isdir(prefix):
                os.makedirs(prefix)
        else:
            assert Path(prefix).is_dir()

        print("-"*20)
        print("Running", prefix)
        print(os.path.join(self.path_prefix, f"{prefix}.gt"), os.path.isfile(os.path.join(self.path_prefix, f"{prefix}.gt")))
        print("-"*20)

        if not os.path.isfile(os.path.join(self.path_prefix, f"{prefix}.gt")):

            if downsample:
                sampled_df = self.distances.reset_index().groupby("true_sfam").apply(lambda df: df.sample(100) if len(df) >= 100 else None)
                self.distances = sampled_df.set_index(["cathDomain", "true_sfam"])
                print(sampled_df)

            self.v_label = self.g.new_vertex_property("string")
            self.e_score = self.g.new_edge_property("float")
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

                if log_odds_descriminator:
                    log_odd_thru_true_sfam = self.log_odds_descriminator.get(true_sfam)

                skipped = 0
                for model, score in row.items():
                    try:
                        sfam_vertex = self.sfam_vertices[model]
                    except KeyError:
                        print("Skipped sfam", model)
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

                    if score_type == "negative_log":
                        self.e_score[e] = -np.log(score)
                    elif score_type == "log_odds":
                        self.e_score[e] = self.log_odds[model][cathDomain]
                    else:
                        if not self.increasing:
                            self.e_score[e] = -1*score
                        else:
                            self.e_score[e] = score
                            print("reg score")

                print(cathDomain, true_sfam, "SKIPPED", skipped, "sfams")

            del_list = [sfam_vertex for sfam, sfam_vertex in self.sfam_vertices.items() if \
                len(self.g.get_all_edges(sfam_vertex))==0]
            print(f"Removing {len(del_list)} becuase they have 0 edges")

            for v in reversed(sorted(del_list)):
                self.g.remove_vertex(v)

            print("n=", self.g.num_vertices(), "e=", self.g.num_edges(), "<?", self.g.num_vertices()*(len(self.sfams)-len(del_list)))

            _domain_vertices = self.g.new_vertex_property("bool")
            for v in self.domain_vertices.values():
                _domain_vertices[v] = True
            for v in self.sfam_vertices.values():
                _domain_vertices[v] = False

            self.g.vertex_properties["v_label"] = self.v_label
            self.g.vertex_properties["domain_vertices"] = _domain_vertices
            self.g.edge_properties["e_score"] = self.e_score

            self.g.save(os.path.join(self.path_prefix, f"{prefix}.gt"))
        else:
            self.g = gta.load_graph(os.path.join(self.path_prefix, f"{prefix}.gt"))
            self.sfam_veritices = [v for v in self.g.vertices() if "SUPERFAMILY" in self.g.vp["v_label"][v]]
            self.domain_vertices = [v for v in self.g.vertices() if "SUPERFAMILY" not in self.g.vp["v_label"][v]]
            self.distances = pd.DataFrame()
                # np.nan,
                # index=[self.g.vp["v_label"][v] for v in self.domain_vertices],
                # columns=[self.g.vp["v_label"][v].split('=')[1][:-1] for v in self.sfam_vertices]
            for edge in self.g.edges():
                source = self.g.vp["v_label"][edge.source()]
                target = self.g.vp["v_label"][edge.target()].split('=')[1]
                score = self.g.edge_properties["e_score"][edge]

                self.distances.loc[source, target] = score

            self.distances.index = pd.MultiIndex.from_tuples([i[1:-1].split(', ') for i in self.distances.index], names=["cathDomain", "true_sfam"])

            print(self.distances)

            self.sfams = self.distances.columns

        domain_groups_file = os.path.join(self.path_prefix, f"{prefix}-domain-group-membership.h5")
        sfam_groups_file = os.path.join(self.path_prefix, f"{prefix}-sfam-group-membership.h5")
        pickle_file = f"{prefix}/{prefix}.pickle"
        # if False and os.path.isfile(domain_groups_file) and os.path.isfile(sfam_groups_file) and os.path.isfile(pickle_file):
        #     self.domain_group_membership = pd.read_hdf(domain_groups_file, "table")
        #     self.sfam_group_membership = pd.read_hdf(sfam_groups_file, "table")
        #     with open(pickle_file, "rb") as f:
        #         self.state = pickle.load(f)
        #     self.n_clusters = self.domain_group_membership.groupby([c for c in self.domain_group_membership.columns if "Level" in c]).ngroups
        #
        # else:
        #     block_model = gta.inference.minimize_nested_blockmodel_dl if nested else \
        #         gta.inference.minimize.minimize_blockmodel_dl

        """
        mcmc_args = dict(mcmc_args, bundled=True)
        if overlap and not nonoverlap_init:
            _B_max = 2 * g.num_edges()
        State = OverlapBlockState
        max_state = State(g, B=_B_max, deg_corr=deg_corr, clabel=clabel,
                      **dmask(state_args,["B", "b", "deg_corr", "clabel"]))

        state = NestedBlockState(g, bs=bs,
                         base_type=State,
                         deg_corr=deg_corr,
                         sampling=False,
                         **dmask(state_args, ["deg_corr"]))

        bisection_args = dict(dict(mcmc_multilevel_args=mcmc_multilevel_args,
                                   random_bisection=False),
                              **bisection_args)

        hierarchy_minimize(state, B_max=B_max, B_min=B_min, b_max=b_max,
                           b_min=b_min, bisection_args=bisection_args,
                           verbose=verbose,
                           **dmask(hierarchy_minimize_args,
                                   ["B_max", "B_min", "bisection_args", "verbose"]))


        """

        state_args = dict(deg_corr=deg_corr, recs=[self.g.ep["e_score"]], rec_types=["real-normal"])

        #print(state_args)

        if force or not os.path.isfile(pickle_file+".allmodes"):
            print("Creating block model...")
            self.state = gt.NestedBlockState(self.g, **state_args)

            # Equilibration
            gt.mcmc_equilibrate(self.state, force_niter=10000, mcmc_args=dict(niter=10, verbose=True))

            bs = []

            def collect_partitions(s):
               global bs
               bs.append(s.get_bs())

            # We will collect only partitions 1000 partitions. For more accurate
            # results, this number should be increased.
            gt.mcmc_equilibrate(self.state, force_niter=10000, mcmc_args=dict(niter=10, verbose=True),
                                callback=collect_partitions)

            # Infer partition modes
            pmode = gt.ModeClusterState(bs, nested=True)

            # Minimize the mode state itself
            gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf, verbose=True))

            # Get inferred modes
            modes = pmode.get_modes()

            with open(pickle_file+".allmodes", "wb") as f:
                pickle.dump({"pmode":pmode, "modes":modes, "state":self.state, "bs":bs,
                    "state_args":state_args}, f)
        else:
            print("Loading block model...")
            with open(pickle_file+".allmodes", "rb") as f:
                state_info = pickle.load(f)
            pmode = state_info["pmode"]
            modes = state_info["modes"]
            self.state = state_info["state"]
            bs = state_info["bs"]
            state_args = state_info["state_args"]

        for i, mode in enumerate(modes):
            b = mode.get_max_nested()    # mode's maximum
            # pv = mode.get_marginal(self.state.g)    # mode's marginal distribution

            print(f"Mode {i} with size {mode.get_M()/len(bs)}")

            state = self.state.copy(bs=b)

            pv = mode.get_marginal(state.g)    # mode's marginal distribution


            for n_levels, s in enumerate(state.get_levels()):
                if s.get_N() == 1:
                    break

            sbm_data = pd.DataFrame(0, index=pd.MultiIndex.from_tuples(
                [state.g.vp["v_label"][d][1:-1].split(", ") for d in state.g.vertices() if state.g.vp["domain_vertices"][d]],
                names=["cathDomain", "sfam"]), columns=[f"l_{l}" for l in \
                range(n_levels+1)]+[f"pv_{i}" for i in range(52)]).reset_index().set_index("cathDomain")

            sbm_data.to_csv(f"{prefix}/{prefix}.domain_groups.{i}.no_updates.csv")

            print(f"Saved csv file to {os.path.abspath(prefix)}/{prefix}.domain_groups.{i}.no_updates.csv")

            for d in state.g.vertices():
                if "SUPERFAMILY" in state.g.vp["v_label"][d]: continue
                cathDomain = state.g.vp["v_label"][d][1:-1].split(", ")[0]
                last_level = d
                for ii, s in enumerate(state.get_levels()):
                    r = s.get_blocks()[last_level]
                    try:
                        sbm_data.loc[cathDomain, f"l_{ii}"] = r
                    except ValueError:
                        raise
                    last_level = r
                    #print(cathDomain, ii, sbm_data.loc[cathDomain, f"l_{ii}"])
                    if s.get_N() == 1:
                        break
                d_pv = list(pv[d])
                pv_cols = [f"pv_{ii}" for ii in range(len(d_pv))]
                sbm_data.loc[cathDomain, pv_cols] = d_pv

            #Remap marginals index with maximums index.
            pv_cols = [c for c in sbm_data.columns if "pv" in c]
            # reidx_names = sbm_data[pv_cols].T.idxmax().apply(lambda s: int(s.split("_")[-1]))
            # sbm_data["l_0"] = reidx_names

            sbm_data.to_csv(f"{prefix}/{prefix}.domain_groups.{i}.csv")

            #import pdb; pdb.set_trace()

            #b[0] = reidx_names.values #.astype(np.int32)

            #state = state.copy(bs=b)
            # state.draw(vertex_shape="pie", vertex_pie_fractions=pv,
            #            output="lesmis-partition-mode-%i.svg" % i)

            print("Plotting block model")

            if force or not os.path.isfile(f"{prefix}/{prefix}_{i}.pdf"):
                state.draw(
                    vertex_shape="pie",
                    vertex_pie_fractions=pv,
                    edge_color=gt.prop_to_size(state.g.ep["e_score"], power=1),
                    ecmap=(matplotlib.cm.viridis, .6),
                    eorder=state.g.ep["e_score"],
                    hedge_color="#555555",
                    hvertex_fill_color="#555555",
                    vertex_text_position="centered",
                    vertex_text=state.g.vp["v_label"],
                    output_size=[8000,8000],
                    edge_gradient=[],
                    output=f"{prefix}/{prefix}_{i}.pdf",
                    vertex_pie_colors=sns.color_palette('husl', n_colors=52)) #len(reidx_names)))


        #self.state = block_model(self.g, state_args=state_args, multilevel_mcmc_args=dict(verbose=True))

        print("Finished creating block model")


        precalculated_feats = None
        load_flare_file = Path(old_flare) if old_flare is not None else Path("flare.csv")
        if load_flare_file.exists():
            print("READING flare.csv")
            #precalculated_feats = df.read_csv("flare.csv", )
            with load_flare_file.open() as f:
                header = next(f).rstrip().split(",")
                header[0] = "id"
                precalculated_feats = [(l.split(",", 1)[0].split(".")[-1], *l.rstrip().split(",")[1:]) \
                    for l in f if l.count(",")+1==len(header)]
                precalculated_feats = pd.DataFrame(precalculated_feats, columns=header)
                print(precalculated_feats)
                assert len(precalculated_feats)>0, precalculated_feats
                precalculated_feats = precalculated_feats[['cathDomain', 'value', 'ss', 'charge', 'electrostatics', 'go_acc', 'sfam', 'sfam_name']]
            print("Loaded old data from flare", precalculated_feats)
        else:
            print("Cannot find preloaded feats")

        flare_file = Path("flare.csv")

        def get_cc(go_codes):
            go_locs = {'GO:0005886': 'plasma membrane',
                     'GO:0005576': 'extracellular region',
                     'GO:0005887': 'integral component of plasma membrane',
                     'GO:0070062': 'extracellular exosome',
                     'GO:0005829': 'cytosol'}
            for go, loc in go_locs.items():
                if go in go_codes:
                    return loc

            return ""

        def get_feats(cathDomain, sfam):
            try:
                with h5pyd.File("/home/ed4bu/deepurfold-paper-1.h5", use_cache=False) as f:
                    atom_df = f[f"{sfam.replace('.', '/')}/domains/{cathDomain}/atom"][...]
            except KeyError:
                print(f"Cannot open {sfam.replace('.', '/')}/domains/{cathDomain}/atom")
                return None
            size = len(atom_df)
            ss = (atom_df["is_sheet"].sum()-atom_df["is_helix"].sum())/(2*(atom_df["is_sheet"].sum()+atom_df["is_helix"].sum()))+0.5
            charge = atom_df["pos_charge"].sum()/size
            electrostatics = 1-atom_df["is_electronegative"].sum()/size
            conserved = atom_df["is_conserved"].sum()/size
            #go_acc = "+".join(go_codes.get(cathDomain, []).split(" "))
            #print("go_acc", go_acc)
            # if go_codes is None:
            #     assert 0
            cath = CATHApi()
            domain_summary = cath.get_domain_summary(cathDomain)
            try:
                go_acc = pd.DataFrame.from_dict(domain_summary["go_terms"])["go_acc"]
            except KeyError:
                go_acc = pd.Series()

            go_acc = "+".join(go_acc)
            #print("Got go code", go_acc)


            # else:
            #     go_acc = go_codes.get(cathDomain, []).split(" ")
            loc = get_cc(go_acc)



            result = pd.Series({"size":size, "ss":ss, "charge":charge,
                "electrostatics":electrostatics, "conserved":conserved,
                "go_acc":go_acc, "sfam":sfam, "cc":loc}, name=cathDomain)
            #print("result", result)
            return result
        print(sbm_data)


        if precalculated_feats is None:
            feats = sbm_data[~sbm_data.index.str.contains("=")]
            feats = pd.merge(feats, feats.apply(lambda r: get_feats(r.name, r.sfam), axis=1), left_index=True, right_index=True)
        else:
            feats = precalculated_feats.rename(columns={"value":"size"}).set_index("cathDomain")
            missing = sbm_data[(~sbm_data.index.isin(feats.index))&(~sbm_data.index.str.contains("="))]
            if len(missing)>0:
                print("Run subet feats", len(missing))
                feats = pd.concat((feats, missing.apply(lambda r: get_feats(r.name, r.sfam), axis=1)))
            feats = pd.merge(feats, sbm_data, left_index=True, right_index=True)
            print(feats)
        
            # if "cc" not in feats:
            #     feats = feats.assign(cc=feats["go_acc"].apply(get_cc))
            # if "sfam" not in feats:
            #     feats = pd.merge(feats, sbm_data[["sfam"]], left_index=True, right_index=True)
            #if True or "go_cc" not in feats:


                # slim_go_terms = set()
                # for got in go_terms:
                #     slim_go_terms = slim_go_terms.union(mapslim(got, go_dag, slim)[0])

                # slim_go_terms = go_terms
                #
                # go_bp = [term for term in slim_go_terms if go_dag[term].namespace=="biological_process"]
                # go_mf = [term for term in slim_go_terms if go_dag[term].namespace=="molecular_function"]
                # go_cc = [term for term in slim_go_terms if go_dag[term].namespace=="cellular_component"]
                #
                # print("go_bp", len(go_bp))
                # print("go_mf", len(go_mf))
                # print("go_cc", len(go_cc))


                #go_terms = [go_t.repalced_by if go_t.is_obsolete else go_t for go_t in go_terms]

                # import pdb; pdb.set_trace()
                #
                #
                # import numpy as np
                # def ssim(x,y):
                #     print(x,y)
                #     print(go_terms[int(x[0])],go_terms[int(y[0])])
                #     ss = semantic_similarity(go_terms[int(x[0])],go_terms[int(y[0])],p)
                #     if ss is None:
                #         return 0.
                #     print("ss", ss)
                #     return ss
                # distances = pairwise_distances(np.array(list(range(len(go_terms)))).reshape(-1,1),
                #     metric=ssim, n_jobs=20)
                #
                # import pdb; pdb.set_trace()

        def get_sfam_names(sfam):
            cath = CATHApi()
            name = cath.get_superfamily_info(sfam)["data"]["classification_name"]
            if name is None:
                name = cath.list_children_in_heirarchy(sfam.rsplit(".", 1)[0], 4)["name"]
            return sfam, name.replace(",", "-")

        if "sfam_x" in feats:
            feats = feats.rename(columns={"sfam_x":"sfam"})

        print(feats.sfam.drop_duplicates().dropna())

        sfam_full_names = pd.DataFrame([get_sfam_names(sfam) for sfam in feats.sfam.drop_duplicates().dropna()], columns=["sfam", "sfam_name"])

        if "sfam_x" in feats:
            feats = feats.rename(columns={"sfam_x":"sfam"})

        feats_ = pd.merge(feats.reset_index(), sfam_full_names, on="sfam", how="left")
        assert len(feats_) == len(feats), f"{len(feats_)}, {len(feats)}"
        feats = feats_.set_index("cathDomain")

        feats = calculate_enrichment(feats, prefix="sbm")

        if "sfam_x" in feats:
            feats = feats.rename(columns={"sfam_x":"sfam"})

        if "sfam_name_y" in feats:
            feats = feats.drop(columns=["sfam_name_x"]).rename(columns={"sfam_name_y":"sfam_name"})

        print("feats", feats)

        #pv_df2 = sbm_data[[c for c in feats.columns if "pv" in c]]
        #mask = pv_df2.gt(0.0).values

        # print(mask)
        # cols = pv_df2.columns.values
        # out = [cols[x].tolist() for x in mask]
        # #out = [list(pv_df2.iloc[i][["cathDomain"]+x].items()) for i, x in enumerate(mask)]
        # out = [[(int(c[0].split("_")[1]), c[1]) for c in x] for x in out]
        # links = {sbm_data.iloc[i]["cathDomain"]: \
        #     [(to, val) for to, val in x if to!=sbm_data.iloc[i]["l_0"]] \
        #     for i, x in enumerate(out)}

        with open("flare_links.csv", "w") as f:
            pass
            # print("source,target", file=f)
            # for cathDomain, domain_links in links.items():
            #     for l, c in domain_links:
            #         print(f"{level_names.loc[cathDomain].iloc[0]}.{cathDomain}", level_names[level_names.str.endswith(f".{l}")].iloc[0], c, sep=",", file=f)

        levels = feats[[c for c in sbm_data.columns if "l" in c]]
        level_names = levels.astype(str)[sorted(levels.columns.tolist(), reverse=True)].agg('.'.join, axis=1)



        used_names = []
        with open("flare.csv", "w") as f:
            print("id,cathDomain,value,ss,charge,electrostatics,go_acc,sfam,sfam_name,go_bp,go_mf,go_cc", file=f)
            for i, row in feats.iterrows():
                print(row.name)
                name = level_names.loc[row.name]
                name_parts = name.split(".")
                for np in range(len(name_parts)):
                    lev_name = ".".join(name_parts[:np+1])
                    if lev_name not in used_names:
                        print(lev_name+",", file=f)
                        used_names.append(lev_name)
                print(f"{name}.{row.name}", row.name, row["size"], row.ss, row.charge, row.electrostatics, row.go_acc.replace(",","+"), row.sfam, f"\"{row.sfam_name}\"", f"\"{row.go_bp}\"", f"\"{row.go_mf}\"", f"\"{row.go_cc}\"", sep=",", file=f)

        cath_feats = calculate_enrichment(feats.sort_values("sfam").copy(), group_key="cath", prefix="cath")
        used_names = []
        with open("flare-cath.csv", "w") as f:
            print("id,cathDomain,value,ss,charge,electrostatics,go_acc,sfam,sfam_name,go_bp,go_mf,go_cc", file=f)
            print("0,", file=f)
            for sfam, domains in cath_feats.groupby("sfam"):
                name_parts = sfam.split(".")
                for np in range(len(name_parts)):
                    lev_name = "0."+".".join(name_parts[:np+1])
                    if lev_name not in used_names:
                        print(lev_name+",", file=f)
                        used_names.append(lev_name)
                for i, row in domains.iterrows():
                    print(f"0.{sfam}.{row.name}", row.name, row["size"], row.ss, row.charge, row.electrostatics, row.go_acc.replace(",","+"), row.sfam, f"\"{row.sfam_name}\"", f"\"{row.go_bp}\"", f"\"{row.go_mf}\"", f"\"{row.go_cc}\"", sep=",", file=f)


        new_levels = {l:f"Level {l.split('_')[-1]}" for l in levels}
        self.domain_group_membership = feats.rename(columns=new_levels).rename(columns={"sfam":"true_sfam"}).reset_index()[list(new_levels.values())+["cathDomain", "true_sfam"]].set_index(["cathDomain", "true_sfam"])
        self.n_clusters = self.domain_group_membership.groupby(list(new_levels.values())).ngroups

        print( self.domain_group_membership)

        with open("stats.csv", "w") as f:
            print(*self.COMPARISON_ROW_COLUMNS, file=f)
            print(self.make_comparison_table_row(), file=f)

        subprocess.call([sys.executable, "-m", "DeepUrfold.Analysis.Webapp.__init__", "--port", "9999", "--feature", "all", "--save_svg"])


            # improve solution with merge-split
        #     self.state = self.state.copy(bs=self.state.get_bs() + [np.zeros(1)] * 4)
        #
        #     print("Improving block model", end="")
        #
        #     for i in range(100):
        #         if i%10==0: print(".", end="")
        #         ret = self.state.multiflip_mcmc_sweep(niter=10, beta=np.inf, verbose=True)
        #
        #     print("\nFinished improving block model")
        #
        #     with open(f"{prefix}/{prefix}.pickle", "wb") as f:
        #         pickle.dump(self.state, f)
        #
        #     print("Creating overlap block model...")
        #
        #     if overlap:
        #         #state_cls = gta.inference.overlap_blockmodel.OverlapBlockState
        #         #B_max = 2 * self.g.num_edges()
        #         #state_args.update(dict(base_type=state_cls, B=B_max, deg_corr=deg_corr))
        #         self.state_overlap = self.state.copy(overlap=True)
        #         for i in range(100):
        #             if i%10==0: print(".", end="")
        #             ret = self.state_overlap.multiflip_mcmc_sweep(niter=10, beta=np.inf, verbose=True)
        #
        #     with open(f"{prefix}/{prefix}-overlap.pickle", "wb") as f:
        #         pickle.dump(self.state_overlap, f)
        #
        #     import pdb; pdb.set_trace()
        #
        #     self.get_group_memberships(self.state, prefix)
        #
        # self.draw(prefix)

    def get_group_memberships(self, state, prefix):
        levels = state.get_levels()

        for n_levels, s in enumerate(levels):
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

        self.domain_group_membership.to_hdf(os.path.join(self.path_prefix, f"{prefix}-domain-group-membership.h5"), "table")

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

        self.sfam_group_membership.to_hdf(os.path.join(self.path_prefix, f"{prefix}-sfam-group-membership.h5"), "table")

        # vorder = self.g.degree_property_map("total")
        #
        # groups = group_membership.groupby([f"Level {l}" for l in range(len(levels+1))])
        # for n, g in groups:
        #     print(n, len(g), g.index.to_frame().reset_index(drop=True)["true_sfam"].drop_duplicates())
        #
        # self.n_groups = groups.ngroup

        self.n_clusters = self.domain_group_membership.groupby([f"Level {l}" for l in range(n_levels+1)]).ngroups

        return




    def draw(self, prefix="urfold-sbm"):

        #self.mplfig = plt.figure(figsize=[56,56], frameon=False)

        self.state.draw(
            vertex_text_position="centered",
            vertex_text=self.v_label,
            #vertex_size=10,
            vertex_font_size=9,
            vertex_size=1,
            edge_color=gt.prop_to_size(self.e_score, power=1),
            ecmap=(matplotlib.cm.viridis, .6),
            eorder=self.e_score,
            edge_pen_width=gt.prop_to_size(self.e_score, 1, 4, power=1),
            edge_gradient=[],
            hedge_color="#555555",
            hvertex_fill_color="#555555",
            output=os.path.join(self.path_prefix, f"{prefix}.pdf"),
            output_size=[4024,4024],
            #mplfig=self.mplfig
        )

        return

        t = gt.get_hierarchy_tree(self.state)[0]
        tpos = pos = gt.radial_tree_layout(t, t.vertex(t.num_vertices() - 1), weighted=True)
        cts = gt.get_hierarchy_control_points(self.g, t, tpos)
        self.pos = self.g.own_property(tpos)
        b = self.state.levels[0].b

        import pdb; pdb.set_trace()

        from itertools import chain

        verts = {vert:name for name,vert in chain(self.domain_vertices.items(), self.sfam_vertices.items())}
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

def main(old_flare=None, force=False):
    possible_dirs = [d for d in Path.cwd().glob("*_plots") if d.is_dir()]
    assert len(possible_dirs)==1 and possible_dirs[0].stem.count("-")==1, "You must have run an AllVsAll algorithm once before"
    output_dir = possible_dirs[0]
    tool_name, model_input = output_dir.stem[:-6].split("-")

    graph_files = list(sorted(output_dir.glob("*.gt"), key=lambda p: p.stat().st_mtime, reverse=True))
    score_type = None
    sbm_info = {}
    if len(graph_files)>0:
        graph_file = graph_files[0]
        print(graph_file)
        #print(graph_file[11:-3].split("_"))
        #sbm_info = dict([kv.split("=") for kv in graph_file[11:-3].split("_")])
        #val_map = {"True":True, "False":False, "None":None}
        #sbm_info = {k:val_map.get(v, v) for k, v in sbm_info.items()}
        #score_type = "elbo" #sbm_info["score_type"]
    else:
        assert 0, "You should call the AllVsAll alorithm"

    sbm = StochasticBlockModel(None, tool_name, None, model_input, score_type, None, None)

    sbm.find_communities(prefix=graph_file.stem, force=force, old_flare=old_flare)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-flare", default=None)
    parser.add_argument("--force", action="store_true", default=False)
    args = parser.parse_args()
    main(args.old_flare, args.force)



#DeepUrfold-CATH_S35_plots/urfold-sbm-nested=True_overlap=True_deg_corr=True_weighted=True_kde_descriminator=False_log_odds_descriminator=False_score_type=None.gt'
#DeepUrfold-CATH_S35_plots/urfold-sbm-nested=True_overlap=True_deg_corr=True_weighted=True_kde_descriminator=False_log_odds_descriminator=False_score_type=elbo.gt
