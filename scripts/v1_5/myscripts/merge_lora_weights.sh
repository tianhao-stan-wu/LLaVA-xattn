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
echo " running merge_lora_weights.py "
echo "***********************************"
echo " "

python scripts/merge_lora_weights.py \
    --model-path checkpoints/test/llava-lora-debug \
    --model-base lmsys/vicuna-7b-v1.5 \
    --save-model-path checkpoints/test/llava-lora-debug-merged

echo " "
echo "********************************"
echo "  finished          "
echo "********************************"
echo " "

# Deactivate the conda environment
conda deactivate

# Print message indicating the environment is deactivated
echo "conda env deactivated"