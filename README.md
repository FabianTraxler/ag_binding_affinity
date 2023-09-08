AG Binding Affinity
==========================

A geometric deep learning framework for predicting binding affinities of antibody-antigen complexes. Interprets protein structures (PDB files) as graphs and predicts their binding affinities by utilizing graph neural networks or transformers.

# Environment

This repo only supports installation through **conda**.

To run this model, use the `diffusion_affinity_combined.yml` environment from the guided-protein-diffusion repository (see subsection below). Instructions on installing that environment are provided in the README of that project.

Data preprocessing code relies on snakemake and comes with their own environments (simply install snakemake anywhere and run `snakemake --use-conda`). Refer to https://snakemake.readthedocs.io/en/stable/executing/cluster.html#cluster-slurm if you want to run snakemake pipelines parallelized with SLURM.

Also, you might want to set the --rerun-triggers CLI argument to mtime by default (using a Snakemake profile). See this comment https://github.com/snakemake/snakemake/issues/1694#issuecomment-1709677941

## Parent repository & SLURM training

Although the code in this repository works independently from its parent, this repo is actually a submodule of https://github.com/moritzschaefer/guided-protein-diffusion/

To run training on a SLURM cluster, refer to the `scripts/cluster_affinity** script in guided-protein-diffusion.

# Structure

This repository contains 4 main folders

- `data`: Datasets that are hard to reproduce
- `results`: Results from our data-generation/preprocessing as well as from our trainings. Should be reproducible!
- `resources`: Data downloaded from external sources. Reproducible.
- `src`: The source code

## src/data_generation

Within src/ the data_generation folder contains all the dataset preprocessing pipelines.

See [Datasets](Datasets section) below for more details.

## src/abag_affinity

This is the python package where all training/inference source code resides.

### src/abag_affinity/main.py

This is the main training script, although actual training code is mostly provided in src/abag_affinity/train/utils.py

### src/abag_affinity/train/training_strategies.py

There are multiple training_strategies implemented in this file. `bucket_train` is what is most interesting at the moment. It just combines the selected datasets (from CLI arguments) and trains on this combined dataset in each epoch.

# Datasets
- `abag_affinity_dataset`: A combination of [AbDb](http://www.abybank.org/abdb/) non-redundant structures and binding affinity values (Kd) from [SAbDab](http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/).
- `antibody_benchmark`: Benchmark subset collected by (Guest et al. 2021). This is a subset of `abag_affinity_dataset`. Previously this was downloaded from the original publication. For simplicity I am now copying it from `abag_ffinity_dataset`.
- [PDBBind](http://www.pdbbind.org.cn/): Protein-protein complexes with measured binding affinities
- [SKEMPI_v2](https://life.bsc.es/pid/skempi2): Dataset containing measurements about change in affinity upon mutations.
- DMS: A collection of ~20 deep mutational scanning (DMS) high-throughput datasets from various publications. See subsection below.

## Downloading preprocessed datasets from my server

Although the pipelines to generate these data should be deterministic/reproducible, for now I recommend you to rely on the data that I already generated, instead of running the pipelines yourself.

You can download `abag_affinity_dataset`, `antibody_benchmark` and `SKEMPI_v2` from my server (muwhpc:/msc/home/mschae83/ag_binding_affinity/results) and you should be all set.

For obtaining DMS-related datasets, refer to the DMS-subsection below!

### DMS
The DMS data can be found here:
mschae83@s0-l00.hpc.meduniwien.ac.at:~/guided-protein-diffusion/modules/ag_binding_affinity/results/DMS/
It takes quite a while to copy it with rsync (because it does individual file operations). I therefore zipped it here:
`mschae83@s0-l00.hpc.meduniwien.ac.at:~/guided-protein-diffusion/modules/ag_binding_affinity/results/DMS_2023-09-07.tar.gz`

For local testing, refer to `mschae83@s0-l00.hpc.meduniwien.ac.at:~/guided-protein-diffusion/modules/ag_binding_affinity/results/DMS_reduced_2023-09-08.tar.gz`, which contains no more than 20 files per folder and is therefore much much smaller.

Note 1: These complexes are *not* relaxed. Coming soon
Note 2: Some datapoints failed, such that the CSV files contain more samples than there are PDBs. Run `snakemake --use-conda -j1 trimmed_csvs` as a workaround to trim the CSV files to only contain data points with corresponding PDB files. Note: requires manual replacement of the CSV files after trimming.
Note 3: There is very little you need to program from here to get DMS-training running. Just make sure that the downloaded DMS data folder corresponds to DATASETS.DMS.folder_path

#### Copying the data from CeMM cluster
I ran `rsync -avzh ag_binding_affinity/results/DMS/ mschae83@s0-l00.hpc.meduniwien.ac.at:~/guided-protein-diffusion/modules/ag_binding_affinity/results/DMS/`, which I can simply rerun to update missing files.
## Relaxation

TODO mutated_pdb_path?

All datasets have an associated "pdb_path" field (see config.yaml), which points to the folder containing the processed PDB files that go into the model training.

By convention, this path is appended with a `_relaxed` suffix, when training with relaxed data. I.e. the data_generation pipelines, in accordance, place the relaxed versions of the processed PDBs into the respective `*_relaxed` folders.

# Usage

The python script `src/abag_affinity/main.py` provides an interface for training specified geometric deep learning models
with different datasets and training modalities.

Call `python main.py --help` to show available options.

## Examples

`python -m abag_affinity.main`

`python src/abag_affinity/main.py -t bucket_train -m GraphConv -d BoundComplexGraphs -b 5`

datasets can have a dataset specific loss function:

`--target_dataset=DATASET:MODE#LOSS1-LAMBDA1+LOSS2-LAMBDA2 `

We have `MODE=[absolut,relative]`, `LOSS=[L1,L2,relative_L1,relative_L2,relative_ce]`implemented


From wandb: (note that most of these parameters correspond to the default values!!)


/src/abag_affinity/main.py --target_dataset=abag_affinity:absolute --patience=30 --node_type=residue --batch_size=1 --layer_type=GCN --max_epochs=200 --nonlinearity=leaky --loss_function=L1 --num_fc_layers=10 --num_gnn_layers=5 --validation_set=1 --train_strategy=model_train --channel_halving --no-fc_size_halving --max_edge_distance=3 --aggregation_method=mean -wdb --wandb_name gnn_relaxation_validation_relaxed_both --interface_hull_size none --wandb_mode=online --debug --relaxed_pdbs both

src/abag_affinity/main.py --target_dataset=abag_affinity:absolute --patience=30 --node_type=residue --batch_size=1 --layer_type=GCN --gnn_type=identity --pretrained_model IPA --max_epochs=200 --nonlinearity=leaky --loss_function=L1 --num_fc_layers=10 --num_gnn_layers=5 --validation_set=1 --train_strategy=model_train --channel_halving --no-fc_size_halving --max_edge_distance=3 --aggregation_method=mean -wdb --wandb_name ipa_relaxation_validation_relaxed_both --interface_hull_size none --wandb_mode=online --debug --relaxed_pdbs both

src/abag_affinity/main.py --bucket_size_mode geometric_mean -t bucket_train --target_dataset abag_affinity:absolute --transfer_learning_dataset DMS-madan21_mutat_hiv:relative#relative_L2+L2 --batch_size 10 --learning_rate 0.000005 --num_workers 7 --wandb_mode online --wandb_name madan21_ipa_emb_abagtarget_lr5e-6 --max_epochs 200 -wdb --debug --pretrained_model IPA --gnn_type identity

src/abag_affinity/main.py --bucket_size_mode geometric_mean -t bucket_train --target_dataset DMS-madan21_mutat_hiv:absolute --transfer_learning_dataset DMS-madan21_mutat_hiv:relative#relative_L2+L2 --batch_size 10 --learning_rate 0.000001 --num_workers 7 --wandb_mode online --wandb_name madan21_noemb_new_folder_-6 --seed 7 --no-embeddings_path --max_epochs 200 -wdb --debug

## CLI arguments

One of the most important files is `src/abag_affinity/utils/argparse_utils.py`, indicating all CLI arguments.

You can of course also run `main.py --help`, but I find it easier to check the file to be able to directly code-search the parameter in question.

## Config

There is a config.yaml that provides metadata for all the datasets as well as output paths etc. It should be already configured such that it matches preprocessing outputs from the pipelines in `src/data_generation`.

#### Weights & Biases.

When you are logged in with W&B, everything should work fine.

If this is the first time, you are using W&B, configure your W&B account and create a project called `abag_binding_affinity`. Then add the `-wdb True` argument to the script to log in to your W&B account.

# Validations
## `abag_affinity_dataset` split

This dataset is "split" in 5 parts. Part 0 (I think) is the benchmark dataset. There is a split function, which explains this in its comments.

## Sweeps
Fabian implemented sweeps in his main.py. They can be controlled via CLI args (see his arguparse_tils.py)
## Validation functions

Find validation functions at the bottom of src/abag_affinity/train/utils.py (e.g. get_benchmark_score, ...). They are being called in main.py after training runs.

I used SKEMPI exclusively for testing in my recent runs (but did not pay much attention to its performance)

## Cross-validation
Fabian implemented a cross-validation method in training_strategies. Once we observe stable training on DMS data from different study, we can use this function to do our planned cross-validation.

## Notebooks

Most code should be well in the repository now. However, there are also some notebooks (in the directory `guided-protein-diffusion/notebooks`) that might contain more detailed analyses/illustrations:

- **`affinity_model_benchmark_test.ipynb`** (Comparison of multiple affinity models with respect to most validations we have. Use this for adhoc comparison of models!)
- `test_affinity_model_on_mutations.ipynb` (SKEMPI validation)
- `wwtf_plots.ipynb` (This includes analysis of Rosetta and is a compilation of other notebooks/analyses that Fabian previously ran)

## Debugging

You can debugging (via DAP (e.g. implemented in VSCode)) remotely! using the --debug switch

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


