# Model training on cluster

Example for MUW Cluster:

## Setup

Only needed for the first time

#### 1. Install Miniconda on login node

Alternatively (maybe better), copy the conda environment from a compute node container (see below; in /opt/conda) to your home.

#### 2. Link paths of login node to paths in container

Append to ~/.bashrc file:

- `mkdir -p /msc/home/<username>`
- `ln -s /root/miniconda3 /msc/home/<username>`
- `ln -s /root/projects /msc/home/<username>`

Additionally add paths to PATH environment variable in ~/.bashrc

#### 3. Do `conda init` to use local miniconda as default conda installation

Gets written to ~/.bashrc file


#### 4. Start container (`start_container.sh`) 

Select appropriate Nivida container with PyTorch and Cuda preinstalled (eg. nvcr.io#nvidia/pytorch:22.02-py3)

#### 5. Create an environment using the conda installation in the login node

Most easily done by:
- linking 
