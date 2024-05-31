# DeepUrfold Supplemental Material

## Introduction

Here we provide all of the data produced by the DeepUrfold model for the 20 CATH superfamilies of interest:

- Winged helix-like DNA-binding domain (1.10.10.10)
- EF-hand (1.10.238.10)
- Globins (1.10.490.10)
- Transferase (1.10.510.10)
- Ferritin (1.20.1260.10)
- SH3 (2.30.30.100)
- OB (2.40.50.140)
- Immunoglobulin (2.60.40.10)
- Beta-grasp (3.10.20.30)
- Ribosomal Protein S5 (3.30.230.10)
- KH (3.30.300.20)
- Sm-like (domain 2) (3.30.310.60)
- Gyrase A (3.30.1360.40)
- K Homology domain, type 1 (3.30.1370.10)
- Hedgehog domain (3.30.1380.10)
- P-loop NTPases (3.40.50.300)
- Rossmann-like (3.40.50.720)
- Ribonuclease Inhibitor (3.80.10.10)
- NTPase (3.90.79.10)
- Oxidoreductase (3.90.420.10)

## Files Description

### 1. superfamily_models_checkpoints.tar.gz ###
This file contains all trained models for each CATH Superfamily in the form of PyTorch Lightning checkpoint files.

### 2. superfamily_models_results.tar.gz ###
This file contains ELBO scores and latent space embeddings for all representative domains against each superfamily-specific model.

### 3. superfamily_models_validation_reports ###
We calculate multiple classification metrics (microaveraged) of the reconstructed input for each superfamily using the test dataset from the corresponding superfamily in Prop3D. Each feature group (atom type [21 feaures], secondary structure [3 features], charge [1 feature], electrostatics [1 feature], hydrophobocity [1 feature], and solvent accessibility  [1 feature]) was either split to report independent metrics or combined to produce combined metrics. We provide ROC and PRC curves *and* bar charts of AUROC, AUPRC, and F1 scores for each feature group independently and combined. We also provide the raw predicted values and their target values.

### 4. sbm_results.tar.gz ###
We provide all of the output from the "best" fitted stochastic block model (best in this case means it has largest overlap score to CATH with 20 communtities): 

 - Circle packing diagrams for DeepUrfold colored by secondary structure, charge, electrostatics, CATH Superfamily, Enriched GO Molecular Functions, Enriched GO Biological Processes, and Enriched GO Cellular Components (svg, png, and flare json file to recreate)

- Circle packing diagrams for CATH colored by secondary structure, charge, electrostatics, CATH Superfamily, Enriched GO Molecular Functions, Enriched GO Biological Processes, and Enriched GO Cellular Components (svg, png, and flare json file to recreate)

 - Raw output from graph-tool (urfold* directory)
     - radial tree produced by graph-tool for this stochastic block model (pdf)
     - communites each domain belongs to along with probabilties of being in other communities (csv)
     - graph-tool stochastic block model object (pickle)
     - fully connected biparted graph with -log(ELBO) edge weights (gt graph file)

 - DeepUrfold-CATH_S35_plots: Distributions of ELBO (raw and log) scores from each superfamily vs all 19 other superfamilies to identifier cutoffs

 - DeepUrfold-CATH S35-all_vs_all.hdf and DeepUrfold-results.h5: the ELBO scores for each representive domain form each superfamily subjected to each superfamily-specific model

 - stats.csv: Clustering metrics when compared to CATH

 - sbm-*.txt and cath-*.txt: Unique enriched GO terms for each catagory for DeepUrfold and CATH

### 5. replicas.tar.gz ###
Same output as above but for 5 different replicas.

### 6. comparison.tar.gz ###

We create stochastic block models (regular, not mixed-membership) for several sequence- and structure- based protein similairty tools using their similairty scores as edge weights in the stochastic block model. We provide:

- the cluster comparison metrics vs CATH (tex file)
- radial tree produced by graph-tool for this stochastic block model (pdf)
- communites each domain belongs (*-group-membership.h5)
- similairty scores for each representative domain through (1) superfamily-specifc models; (2) pairwise distance models; or (3) entire potein universe models (e.g. single model) (*-results.h5)
- graph-tool stochastic block model object (pickle)
- Similarity score distribution plots for superfamily vs all 19 other superfamilies to identifier cutoffs 
   



