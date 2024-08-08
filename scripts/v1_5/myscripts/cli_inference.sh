#!/bin/bash

# Activate the conda environment
source dataset/vstanwu/miniconda3/etc/profile.d/conda.sh
conda activate llava-xattn || { echo "Failed to activate conda env"; exit 1; }
echo "conda env llava-xattn activated"


# Export the LD_LIBRARY_PATH and check if it was successful
export LD_LIBRARY_PATH=/dataset/vstanwu/miniconda3/envs/llava-xattn/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
if [ $? -ne 0 ]; then
    echo "Failed to set LD_LIBRARY_PATH"
    exit 1
fi

echo "LD_LIBRARY_PATH set successfully"


# Change directory to project folder
cd dataset/vstanwu/LLaVA-xattn || { echo "Failed to change directory"; exit 1; }
echo "moved to project dir dataset/vstanwu/LLaVA-xattn"
echo " "
echo "***********************************"
echo " running cli inference script... "
echo "***********************************"
echo " "

python -m llava.serve.cli \
    --model-path ./checkpoints/llava-v1.5-7b-pretrain-xattn \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --load-4bit \
    --user-question "What are the things I should be cautious about when I visit this place?"


# "./test_dataset/000000520873.jpg" \
# "Describe the scene in a few sentences. What's the color of the girl's hair?"

echo " "
echo "********************************"
echo " inference finished          "
echo "********************************"
echo " "

# Deactivate the conda environment
conda deactivate

# Print message indicating the environment is deactivated
echo "conda env deactivated"