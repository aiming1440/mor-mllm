#!/bin/bash

# Also uses the mini dataset paths
JSON_FOLDER="mini_train_data/train_json"
IMAGE_FOLDER="train_data"
HF_CACHE_DIR="/data/huggingface_cache"

# INPUT_MODEL="./checkpoints_mini/llavaphi-2.7b-finetune"
INPUT_MODEL="LanguageBind/MoE-LLaVA-Phi2-Stage2"

OUTPUT_DIR="./experiments/mor_tests/llavaphi-2.7b-finetune-mor"

DATA_PATH="${JSON_FOLDER}/llava_image_tune_small_5p.json ${JSON_FOLDER}/nlp_tune_small_5p.json"

mor_mode="expert"
sharing_strategy="middle_cycle"
num_recursions=3
rand_router=False
router_type="linear"
gating="hard"
z_loss_coeff=0.1
kv_sharing_enable=True
kv_sharing_update_cache=False
expert_capacity="0.5, 0.3, 0.2"
cap_warmup_step=0
aux_loss_coeff=0.01
train_modules="router"

# Waiting on train_modules argument support to be added
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=0 deepspeed mor_mllm/train/train.py \
    --mor_enable True --mor_mode ${mor_mode} --train_modules ${train_modules} \
    --sharing_strategy ${sharing_strategy} --num_recursions ${num_recursions} \
    --rand_router ${rand_router} --router_type ${router_type} --gating ${gating} \
    --use_aux_loss True --aux_loss_coeff ${aux_loss_coeff} --bal_loss_coeff 0.1 --use_z_loss True --z_loss_coeff ${z_loss_coeff} \
    --kv_sharing_enable ${kv_sharing_enable} --kv_sharing_update_cache ${kv_sharing_update_cache} \
    --expert_capacity "${expert_capacity}" --cap_warmup_step ${cap_warmup_step} \
    --deepspeed ./scripts/deepspeed/zero2.json \
    --model_name_or_path ${INPUT_MODEL} \
    --version phi \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --save_strategy "steps" \
    --save_steps 20 \
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
