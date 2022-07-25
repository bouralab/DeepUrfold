# DeepUrfold

**Note: this is all research code and not all components are functional**

This repository contains code for the paper "Exploring Fold Space with Deep Generative Models:
Reconciling the Continuous vs Discrete Dichotomy with an ‘Urfold’ Model of Protein Inter-relationships" by Eli J. Draizen, Stella Veretnik, Cameron Mura, and Philip E. Bourne.

## Goal

We hypothesize that there may be a bona fide level/entity that exists in between the Architecture and Topology entities in hierarchies of protein structure space presented by CATH, SCOP, and ECOD, to represent '3D architectural similarity despite topological variability.' We call this the 'Urfold' [1].

We present DeepUrfold, a new tool to explore the protein structure space in hopes of finding new Urfolds. This includes:

1. A sequence-independent, alignment-free, rotation-invariant similarity metric of proteins that leverages similarities in latent-spaces rather than 3D structures (Superfamily-specific Variational Autoencoders); and

2. A mixed-membership community detection approach to cluster similar protein structures using Stochastic Block Models. This method takes a different approach to clustering, allowing for proteins to span multiple clusters, thereby allowing for the continuous nature of fold space.

To explore the results, please visit https://bournelab.org/research/DeepUrfold/

## Install

Please install [Prop3D](https://github.com/bournelab/Prop3D) and this DeepUrfold repository

```
git clone https://github.com/bournelab/Prop3D.git
cd Prop3D
git submodule init
git submodule update
python setup.py install
```

and then

```
git clone https://github.com/bournelab/DeepUrfold.git
cd DeepUrfold
python setup.py install
```

## Set up HSDS

You can either:

1. Use our HSDS endpoint at `http://hsds.uvarc.io/` with username `protein` and password `protein`.

```bash
$ hsconfigure
Enter new values or accept defaults in brackets with Enter.

Server endpoint [None]: http://hsds.uvarc.io/
Updated endpoint [http://hsds.uvarc.io/]:
Username [None]: protein
Updated username: [protein]
Password [None]: protein
updated password: [protein]
API Key [None]:
Testing connection...
```
 Then set path to the Prop3D dataset:

 ```bash
 export PROP3D_DATA="h5://prop3d.hsds.uvaarc.io"
 ```

2. Set up your own [HSDS](https://github.com/HDFGroup/hsds) endpoint (advanced)

You can set up HSDS on any cloud platform or a single machine using Docker or on a cluster using Kubernetes (or AKS on Microsoft Azure).

For single machine setup, please clone the [HSDS](https://github.com/HDFGroup/hsds) repository and follow the instruction at [https://gitlab.com/uva-arc/hobo-request/-/blob/main/doc/single-node-k3s-hsds-install.md](https://gitlab.com/uva-arc/hobo-request/-/blob/main/doc/single-node-k3s-hsds-install.md).

If you want to download the precalculated Prop3D dataset and use it for inference, run:

```bash
export PROP3D_DATA=/home/$USER/Prop3D.h5 #Change path to any path in HSDS
wget https://zenodo.org/record/6873024/files/Prop3D-20.h5
hsload Prop3D.h5 $PROP3D_DATA
```
Warning: this might take a while

Otherwise, you can recreate the Prop3D dataset by following the [https://github.com/bournelab/Prop3D/README.md](Prop3D) instructions.

## Training a single model

```bash
python -m DeepUrfold.Trainers.DistributedDomainStructureVAE \
  --superfamily 2.60.40.10 \ #Can be any (or multiple) superfamily in the Prop3D dataset,
                          \  #which will use the precalculated train and validation splits.
                          \  #Multiple superfamilies can be included separated by a single space
  --data_dir $PROP3D_DATA \
```

## Evaluating a single model

```bash
python -m DeepUrfold.Evaluators.EvaluateDistrubutedDomainStructureVAE \
  --superfamily 2.60.40.10 \ #Can be any superfamily in the Prop3D dataset,
                           \ #which will use the precalculated test split.
                           \ #Multiple superfamilies can be included separated by a single space
  --data_dir $PROP3D_DATA  \

# --domain d1 d2          \ #Uncomment to  evaluate certain domains in any of
                          #\ #the superfamilies (in either split)
# --return_latent         \ #Uncomment to return latent variable (mean) instead of ELBO loss
# --return_reconstruction \ #Uncomment to return the reconstruction instead of ELBO loss
# --classification        \ #Uncomment to return the reconstruction and perform classification metrics (AUROC, AUPRC, F1)
```

# Running All Vs. All

```bash
python -m DeepUrfold.Analysis.StructureBased.deepurfold_ava \
  --data_dir $PROP3D_DATA \
  --work_dir $PWD \
  [SUPERFAMILY...] #Default is all 20 superfamilies used in paper
```

This will train all superfamily models, evaluate all representative domains against each superfamily model, and create stochastic block model communities for each domain.

We also include sequence- and structure- based models to compare against that can be run in a similar way. Please see their help pages for more info. Some structure-based models haven't been updated to use HSDS so may not work.

## References

[1] Mura, C, Veretnik, S, Bourne, PE. The Urfold: Structural similarity just above the superfold level? Protein Science. 2019; 28: 2119– 2126. https://doi.org/10.1002/pro.3742

## Citation

We will post the bioRxiv link soon
