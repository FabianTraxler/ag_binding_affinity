# Description

A geometric deep learning approach for predicting binding affinities of antibody-antigen complexes
This repository provides a pipeline for interpreting protein structures (PDB files) as graphs and predicting their binding affinities by utilizing graph neural networks.

# Datasets
- Dataset_v1: A combination of [AbDb](http://www.abybank.org/abdb/) non-redundant structures and binding affinity values (Kd) from [SAbDab](http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/).
- [PDBBind](http://www.pdbbind.org.cn/): Protein-Ligand complexes with measure binding affinity
- [SKEMPI_v2](https://life.bsc.es/pid/skempi2): Dataset containing measurements about change in affinity upon mutations
- DMS: A collection of ~20 deep mutational scanning (DMS) high-throughput datasets from various publications

### Download

Follow the steps in the respective notebooks found in `src/data_analysis/Datasets` to download and convert the datasets into the necessary format.

# Installation
Make sure to link the `resources`, `results` and `data` folders to the correct location (ln -s). Beware that this repository might on its own deliver a `data` folder.

- Downloaded files are to be stored in the `resources` folder
  - E.g., the files from PDBBind should be stored in `resources`
- Converted files are to be stored in the `data` folder

Folder names can be adapted in `src/config`

### Requirements

Create a conda environment with the `envs/environment.yaml` file.

# Usage

### Training

The python script `src/abag_affinity/main.py` provides an interface for training specified geometric deep learning models 
with different datasets and training modalities.

Call `python main.py --help` to show available options.

Example:

`python main.py -t bucket_train -m GraphConv -d BoundComplexGraphs -b 5`


#### Weights & Biases
Configure your W&B account and create a project called `abab_binding_affinity`. Then add the `-wdb True` argument to the script to log in to your W&B account.
