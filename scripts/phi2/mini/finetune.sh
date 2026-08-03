#!/bin/bash

# Also uses the mini dataset paths
JSON_FOLDER="mini_train_data/train_json"
IMAGE_FOLDER="mini_train_data"
HF_CACHE_DIR="/data/huggingface_cache"

PRETRAIN_MM_MLP_ADAPTER="./checkpoints_mini/llavaphi-2.7b-pretrain/mm_projector.bin"
OUTPUT_DIR="./checkpoints_mini/llavaphi-2.7b-finetune"

# DATA_PATH="${JSON_FOLDER}/la_tune_256k.json ${JSON_FOLDER}/lrv_tune_331k.json ${JSON_FOLDER}/lvis_tune_220k_.json ${JSON_FOLDER}/svit_tune_157k.json ${JSON_FOLDER}/nlp_tune.json"

DATA_PATH="${JSON_FOLDER}/svit_tune_157k.json"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed mor_mllm/train/train.py \
    --deepspeed ./scripts/deepspeed/zero2.json \
    --model_name_or_path microsoft/phi-2 \
    --version phi \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ${PRETRAIN_MM_MLP_ADAPTER} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 30 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir ${HF_CACHE_DIR}

