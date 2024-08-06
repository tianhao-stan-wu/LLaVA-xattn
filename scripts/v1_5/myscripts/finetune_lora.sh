#!/bin/bash

# Activate the conda environment
source dataset/vstanwu/miniconda3/etc/profile.d/conda.sh
conda activate llava-xattn || { echo "Failed to activate conda env"; exit 1; }
echo "conda env llava-xattn activated"

API_KEY="65927b6f8f24622601e9bac3d27a7b08f860dcb5"
wandb login <<EOF
$API_KEY
EOF

echo "Logged into wandb successfully!"

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
echo "********************************"
echo "* running finetuning script... *"
echo "********************************"
echo " "

# --data_path ./test_dataset/llava_instruct_150k.json \
# --image_folder ../../sharedir/research/coco2017/train2017 \

# --data_path ./test_dataset/llava_instruct_10.json \
# --image_folder ./test_dataset \

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./test_dataset/llava_instruct_10.json \
    --image_folder ./test_dataset \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./checkpoints/test/llava-lora-debug \
    --num_train_epochs 100 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --wandb_project_name "test_debug"



echo " "
echo "********************************"
echo "*  finetuning script finished  *"
echo "********************************"
echo " "

# Deactivate the conda environment
conda deactivate

# Print message indicating the environment is deactivated
echo "conda env deactivated"