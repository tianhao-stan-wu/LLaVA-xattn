#!/bin/bash


# Activate the conda environment
source dataset/vstanwu/miniconda3/etc/profile.d/conda.sh
conda activate llava-xattn || { echo "Failed to activate conda env"; exit 1; }
echo "conda env llava-xattn activated"

# Change directory to project folder
cd dataset/vstanwu/LLaVA-xattn || { echo "Failed to change directory"; exit 1; }
echo "moved to project dir dataset/vstanwu/LLaVA-xattn"

echo "running test.py"

# Execute the python file
python test.py

# Deactivate the conda environment
conda deactivate

# Print message indicating the environment is deactivated
echo "conda env deactivated"


