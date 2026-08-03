#!/bin/bash
# MoR-Tuning Stage II: Router Warm-up
# load from the Stage I checkpoint, call initialize_mor_modules to
# collapse the middle (num_hidden_layers-2)=30 layers into k=6 MoR blocks grouped by
# group_size=5 (middle_cycle, first/last layers stay dense), constructing the shared
# submodule M via mor_block_init=grouped_sharing. 
# Data: 964K mixed instruction data (placeholder path, replace with the actual data).
# Hyperparameters: lr 5e-5, global batch size 64, warmup_ratio 0.05.

JSON_FOLDER="train_data/train_json"
IMAGE_FOLDER="train_data"
HF_CACHE_DIR="/data/huggingface_cache"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Annotation file path for the 964K mixed instruction data
DATA_PATH="${JSON_FOLDER}/mor_instruction_964k.json"

# Projection warm-up checkpoint produced by Stage I.
INPUT_MODEL="./checkpoints/mor_phi2/stage1_projection_warmup"
OUTPUT_DIR="./checkpoints/mor_phi2/stage2_router_warmup"

mor_mode="expert"
sharing_strategy="middle_cycle"
num_recursions=6          # k = (32-2)/group_size = 6
group_size=5
mor_block_init="grouped_sharing"
rand_router=False
router_type="linear"
gating="weighted"
z_loss_coeff=0.1
kv_sharing_enable=True
kv_sharing_update_cache=False
expert_capacity="0.2"     # capacity_factor ~= 1-beta, beta=0.8
cap_warmup_step=0
aux_loss_coeff=0.1
train_modules="router mm_projector"

HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 deepspeed mor_mllm/train/train.py \
    --mor_enable True --mor_mode ${mor_mode} --train_modules ${train_modules} \
    --sharing_strategy ${sharing_strategy} --num_recursions ${num_recursions} --group_size ${group_size} \
    --mor_block_init ${mor_block_init} --freeze_shared_block True \
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
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir ${HF_CACHE_DIR}
