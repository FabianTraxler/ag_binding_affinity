# this script was adapted from https://github.com/YoshitakaMo/localcolabfold to install inside a preinstalled conda
#!/bin/bash

set -e
set -x

. "/msc/home/mschae83/miniconda3/etc/profile.d/conda.sh"  # this is a workaround, because otherwise it does not get conda right -.-
type wget || { echo "wget command is not installed. Please install it at first using apt or yum." ; exit 1 ; }
type curl || { echo "curl command is not installed. Please install it at first using apt or yum. " ; exit 1 ; }

CURRENTPATH=`pwd`
COLABFOLDDIR="${CURRENTPATH}/localcolabfold"

mkdir -p ${COLABFOLDDIR}
cd ${COLABFOLDDIR}

# Assuming conda is already installed and available in PATH
conda create -n colabfold_multimer_patch python=3.10 -y
conda activate colabfold_multimer_patch
conda update -n base conda -y
conda install -c conda-forge python=3.10 cudnn==8.2.1.32 cudatoolkit==11.6.0 openmm==7.7.0 pdbfixer -y
conda install -c nvidia cuda-nvcc=11.4  # depends on installed driver. apparently it's 
# Download the updater
wget -qnc https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/update_linux.sh --no-check-certificate
chmod +x update_linux.sh

# install alignment tools
conda install -c conda-forge -c bioconda kalign2=2.04 hhsuite=3.3.0 mmseqs2=14.7e284 -y

# install ColabFold and Jaxlib
python -m pip install --upgrade pip
python -m pip install --no-warn-conflicts "colabfold[alphafold-minus-jax] @ git+https://github.com/sokrypton/ColabFold@33ae16f350b107cbec3523cdac363ddaae8497ec" tensorflow==2.12.0
python -m pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.3.25+cuda11.cudnn82-cp310-cp310-manylinux2014_x86_64.whl
python -m pip install jax==0.3.25 chex==0.1.6 biopython==1.79 tensorrt=8.6.1


# Use 'Agg' for non-GUI backend
cd ${CONDA_PREFIX}/lib/python3.10/site-packages/colabfold
sed -i -e "s#from matplotlib import pyplot as plt#import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt#g" plot.py
# modify the default params directory
sed -i -e "s#appdirs.user_cache_dir(__package__ or \"colabfold\")#\"${COLABFOLDDIR}/colabfold\"#g" download.py
# Moritz: Generate only 1 template (max_hit=1 instead of 20)
sed -i -e "s#max_hits=20#max_hits=1#g" batch.py

colabfold/batch.py

# remove cache directory
rm -rf __pycache__

# Moritz: Patch colabfold for proper multimer use
cd ${CONDA_PREFIX}/lib/python3.10/site-packages/alphafold
sed -i  '$i \ \ template_mask = jnp.ones_like(template_mask)' model/modules_multimer.py

# start downloading weights
cd ${COLABFOLDDIR}
python -m colabfold.download
cd ${CURRENTPATH}

echo "Download of alphafold2 weights finished."
echo "-----------------------------------------"
echo "Installation of colabfold_multimer_patch finished."
echo "Activate the environment 'colabfold_multimer_patch' to run 'colabfold_multimer_patch'."
echo "i.e. For Bash, conda activate colabfold_multimer_patch"
echo "For more details, please type 'colabfold_multimer_patch --help'."

