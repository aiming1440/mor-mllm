#!/bin/bash

# Use the mini dataset paths we created
JSON_FOLDER="mini_train_data/train_json"
IMAGE_FOLDER="mini_train_data"
HF_CACHE_DIR="/data/huggingface_cache"

OUTPUT_DIR="./checkpoints_mini/llavaphi-2.7b-pretrain"

# Enable CUDA memory allocator optimization to reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 deepspeed mor_mllm/train/train.py \
    --deepspeed ./scripts/deepspeed/zero2.json \
    --model_name_or_path microsoft/phi-2 \
    --version plain \
    --data_path ${JSON_FOLDER}/llava_image_.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 20 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory False \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir ${HF_CACHE_DIR}



