AG Binding Affinity
==========================

A geometric deep learning framework for predicting binding affinities of antibody-antigen complexes. Interprets protein structures (PDB files) as graphs and predicts their binding affinities by utilizing graph neural networks or transformers.

# Environment

This repo only supports installation through **conda**.

To run this model, use the `diffusion_affinity_combined.yml` environment from the guided-protein-diffusion repository (see subsection below). Instructions on installing that environment are provided in the README of that project.

Data preprocessing code relies on snakemake and comes with their own environments (simply install snakemake anywhere and run `snakemake --use-conda`). Refer to https://snakemake.readthedocs.io/en/stable/executing/cluster.html#cluster-slurm if you want to run snakemake pipelines parallelized with SLURM.

## Parent repository & SLURM training

Although the code in this repository works independently from its parent, this repo is actually a submodule of https://github.com/moritzschaefer/guided-protein-diffusion/

To run training on a SLURM cluster, refer to the `scripts/cluster_affinity** script in guided-protein-diffusion.

# Structure

This repository contains 4 main folders

- data: Datasets that are hard to reproduce
- results: Results from our data-generation/preprocessing as well as from our trainings. Should be reproducible!
- resources: Data downloaded from external sources. Reproducible.
- src: The source code

## src/data_generation

Within src/ the data_generation folder contains all the dataset preprocessing pipelines.

See [Datasets](Datasets section) below for more details.

## src/abag_affinity

This is the python package where all training/inference source code resides.

### src/abag_affinity/main.py

This is the main training script, although actual training code is mostly provided in src/abag_affinity/train/utils.py

# Datasets
- `abag_affinity_dataset`: A combination of [AbDb](http://www.abybank.org/abdb/) non-redundant structures and binding affinity values (Kd) from [SAbDab](http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/).
- `antibody_benchmark`: Benchmark subset collected by (Guest et al. 2021). This is a subset of `abag_affinity_dataset`. Previously this was downloaded from the original publication. For simplicity I am now copying it from `abag_ffinity_dataset`.
- [PDBBind](http://www.pdbbind.org.cn/): Protein-protein complexes with measured binding affinities
- [SKEMPI_v2](https://life.bsc.es/pid/skempi2): Dataset containing measurements about change in affinity upon mutations.
- DMS: A collection of ~20 deep mutational scanning (DMS) high-throughput datasets from various publications

# Usage

The python script `src/abag_affinity/main.py` provides an interface for training specified geometric deep learning models
with different datasets and training modalities.

Call `python main.py --help` to show available options.

Example:

`python -m abag_affinity.main`

`python src/abag_affinity/main.py -t bucket_train -m GraphConv -d BoundComplexGraphs -b 5`

## CLI arguments

One of the most important files is `src/abag_affinity/utils/argparse_utils.py`, indicating all CLI arguments.

You can of course also run `main.py --help`, but I find it easier to check the file to be able to directly code-search the parameter in question.

## Config

There is a config.yaml that provides metadata for all the datasets as well as output paths etc. It should be already configured such that it matches preprocessing outputs from the pipelines in `src/data_generation`.

#### Weights & Biases.

When you are logged in with W&B, everything should work fine.

If this is the first time, you are using W&B, configure your W&B account and create a project called `abag_binding_affinity`. Then add the `-wdb True` argument to the script to log in to your W&B account.

# OLD Installation instructions

**From now on install via guided_protein_diffusion and ignore all this!**

Make sure to link the `resources`, `results` and `data` folders to the correct location (ln -s). Beware that this repository might on its own deliver a `data` folder.

- Downloaded files are to be stored in the `resources` folder
  - E.g., the files from PDBBind should be stored in `resources`
- Converted files are to be stored in the `data` folder

Folder names can be adapted in `src/config`

To install the shared environment, make sure that libxml2 is installed on your machine (via apt-get). Alternatively, just use the libxml from your conda machine:

`find ~/conda | grep libxml2.so` 

`LD_LIBRARY_PATH=~/conda/lib/ mamba env install -f envs/diff_aff_combined_environment.yml`

deep refine is missing too still.. Clone https://github.com/BioinfoMachineLearning/DeepRefine and run `python setup.py install`

Make sure to install the python package by running

`python setup.py develop` (or `install`)

#### For the environment that is specific to this repo, use the code below

Create a conda environment with the `envs/environment.yaml` file.

`mamba env create -f envs/environment.yaml`

As of now, DGL needs to be installed separately:
`mamba install -c dglteam dgl`

`mamba install -c rdkit rdkit`


