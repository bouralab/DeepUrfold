import os
import subprocess
from pathlib import Path
from zipfile import ZipFile
from DeepUrfold.Analysis import StochasticBlockModel

def process_sbm_images(results_dir):
    results_dir = Path(results_dir)
    replicas = list(sorted((results_dir / "replicas").glob("test*")))

    results_latex = results_dir / "results.tex"

    with results_latex.open("w"):
        pass

    image_zip = ZipFile(results_dir / "supp_images_sbm.zip", 'w')
    radial_image_zip = ZipFile(results_dir / "supp_images_sbm_radial.zip", 'w')

    cwd = os.getcwd()
    for replica, dir in enumerate([results_dir] + replicas):
        print("Running replica", replica,  "@", dir, "...")
        sbm_figs = dir / "sbm_figures"
        stats_file = dir / "stats.csv"
        if not sbm_figs.is_dir():
            os.chdir(str(dir))
            try:
                StochasticBlockModel.main(old_flare="flare.csv")
            except ValueError:
                continue
                os.chdir(cwd)

        radial_tree = next(next(dir.glob("urfold*")).glob("*.pdf"))

        radial_image_zip.write(
            str(radial_tree),
            f"sbm_radial_images/replica{replica}_radial.pdf")

        columns = ["Name", "Method", "Data", "Similairty Function", "\# of Clusters",
                   "Silhouette Score", "Davies-Boundin Score", "", "Overlap Score",
                   "Rand Score", "Rand Score Adjusted", "Adjusted Mutual Information",
                   "Homogeneity Score", "Completeness Score"]

        with stats_file.open() as f:
            next(f)
            data = [f.replace("'", "") for f in next(f)[1:-1].split(", ")]
            print(data)
            assert len(data) >= len(columns)
            data = dict(zip(columns, data[:len(columns)]))
            print(data)

        sbm_features = [("Secondary Structure", 'ss', "We use a the secondary structure score used throughout this paper: {\\tiny{$\\tfrac{\\ttvar{\#}\\beta \enspace atoms \enspace-\enspace \\ttvar{\#}\\alpha \enspace atoms}{2\left(\\ttvar{\#}\\beta \enspace atoms \enspace + \enspace \\ttvar{\#} \ensapce \\alpha \enspace atoms \\right)}+0.5$}}"),
                        ("Charge", 'charge', "Each atom it annotated with the boolean feature \emph{is\_positive}. We sum up all of the postive atoms and take a fraction of the total number of atoms."),
                        ("Average Electrostatics", 'electrostatics', "Each atom it annoated with the boolean feature \emph{is\_electronegtive}. We sum up all of the electronegtive atoms and take a fraction of the total number of atoms."),
                        ("Superfamilies",'sfam', "Annotated CATH Superfamily (H level)"),
                        ("Gene Ontology: Molecular Function", 'go_mf', "We use GOATOOLS to calculate enrichment for each GO term from all domains in the predicted SBM community (Leaf grouping only) using GO Slim terms from AGR. If the domain has a GO term that is enriched in its community (p\_fdr\_bh $\leq$ 0.05), then it is colored for the associated term. If there are multiple enriched terms, only the first is used."),
                        ("Gene Ontology: Biological Process", 'go_bp', "We use GOATOOLS to calculate enrichment for each GO term from all domains in the predicted SBM community (Leaf grouping only) using GO Slim terms from AGR. If the domain has a GO term that is enriched in its community (p\_fdr\_bh $\leq$ 0.05), then it is colored for the associated term. If there are multiple enriched terms, only the first is used."),
                        ("Gene Ontology: Cellular Component", 'go_cc', "We use GOATOOLS to calculate enrichment for each GO term from all domains in the predicted SBM community (Leaf grouping only) using GO Slim terms from AGR. If the domain has a GO term that is enriched in its community (p\_fdr\_bh $\leq$ 0.05), then it is colored for the associated term. If there are multiple enriched terms, only the first is used.")]

        with results_latex.open("a") as f:
            print(f"\subsubsection{{SBM Replica {replica}}}", file=f)
            print("    \\begin{table}[h!]", file=f)
            print("        \\renewcommand{\\arraystretch}{1.4}", file=f)
            print("        \centering", file=f)
            print("        \\begin{tabular}{ r l }", file=f)
            print("            \\toprule", file=f)
            for col in columns[4:]:
                if col == "": continue
                print(f"            \\textbf{{{col}}} & {data[col]} \\\\", file=f)
            print("            \\bottomrule \\\\", file=f)  
            print("         \end{tabular}", file=f)
            print(f"         \caption{{Clustering metrics of DeepUrfold SBM Replica {replica} vs CATH}}", file=f)   
            print("     \end{table}\n", file=f)
            print("\hfill \\break", file=f)

#             print(f"""    \\begin{{figure}}[h]
#     \centering
#     \includegraphics[width=0.6\\textwidth]{{Chapter3/images/SBM/replica{replica}/radial.pdf}}
#     \caption{{\textbf{{Stochastic Block Model Communties of replica {replica} represtend by the original radial tree from \emph{{graph_tool}}.}} Edges are colored by -log(ELBO) scaled by viridis and nodes colored by pie charts denoting which community they could be belong to (e.g. mixed-membership), where the the largest section is the current output of the SBM. Other variations from the mixed-membership is difficult to display in charts like this, so they are not analysed in this paper. Notice all of the nodes from the VAE models cluster together, and are thus removed from the circle packing charts. Due to the complex nature of this visualization, we opt to use the cicle packing charts to speicifcally highlight each comminty.}}
#     \label{{fig:SBM-rep{replica}-radial}}
# \end{{figure}}
# """, file=f)      

            for i, (feature_name, feature_id, feature_desc) in enumerate(sbm_features):
                image_zip.write(
                    str(dir / "sbm_figures" / f"{feature_id}.withLegend.png"),
                    f"sbm_replicas/replica{replica}/{feature_id}.withLegend.png")

                print(f"""    \\begin{{figure}}[h!]
        \centering
        \\vcenter{{\hbox{{\includegraphics[width=0.9\\textwidth]{{Chapter3/images/sbm_replicas/replica{replica}/{feature_id}.withLegend.png}}}}}}
        \caption{{\\textbf{{SBM Communities of replica {replica} represented by a circle packing diagram colored by {feature_name}}}. {feature_desc}}}
        \label{{fig:SBM-rep{replica}-{feature_id}}}
    \end{{figure}}
""", file=f)

                print("\\newpage""", file=f)

    image_zip.close()
    radial_image_zip.close()

if __name__ == "__main__":
    process_sbm_images(".")
