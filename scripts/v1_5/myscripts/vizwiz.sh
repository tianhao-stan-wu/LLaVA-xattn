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
echo " running vizwiz evaluation... "
echo "***********************************"
echo " "


MODEL_NAME="llava-v1.5-7b-xattn-ft150k-v2-merged"

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/$MODEL_NAME \
    --question-file ../dataset/eval/vizwiz/llava_test.jsonl \
    --image-folder ../dataset/eval/vizwiz/test \
    --answers-file ../dataset/eval/vizwiz/answers/$MODEL_NAME.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ../dataset/eval/vizwiz/llava_test.jsonl \
    --result-file ../dataset/eval/vizwiz/answers/$MODEL_NAME.jsonl \
    --result-upload-file ../dataset/eval/vizwiz/answers_upload/$MODEL_NAME.json



echo " "
echo "********************************"
echo "  finished          "
echo "********************************"
echo " "

# Deactivate the conda environment
conda deactivate

# Print message indicating the environment is deactivated
echo "conda env deactivated"