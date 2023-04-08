# Description

A geometric deep learning approach for predicting binding affinities of antibody-antigen complexes
This repository provides a pipeline for interpreting protein structures (PDB files) as graphs and predicting their binding affinities by utilizing graph neural networks.

# Datasets
- Dataset_v1: A combination of [AbDb](http://www.abybank.org/abdb/) non-redundant structures and binding affinity values (Kd) from [SAbDab](http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/).
- [PDBBind](http://www.pdbbind.org.cn/): Protein-Ligand complexes with measure binding affinity
- [SKEMPI_v2](https://life.bsc.es/pid/skempi2): Dataset containing measurements about change in affinity upon mutations
- DMS: A collection of ~20 deep mutational scanning (DMS) high-throughput datasets from various publications

### Download

Follow the steps in the respective notebooks found in `src/data_analysis/Datasets** to download and convert the datasets into the necessary format.

# Installation

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


# Usage

### Training

The python script `src/abag_affinity/main.py` provides an interface for training specified geometric deep learning models
with different datasets and training modalities.

Call `python main.py --help` to show available options.

Example:

python -m abag_affinity.main

`python main.py -t bucket_train -m GraphConv -d BoundComplexGraphs -b 5`

#### Weights & Biases
Configure your W&B account and create a project called `abag_binding_affinity`. Then add the `-wdb True` argument to the script to log in to your W&B account.
