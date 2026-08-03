#!/bin/bash
# MoR-Tuning Stage III: Joint Fine-tuning + LoRA + Entropy Regularization
# load from the Stage II checkpoint (router + MoR structure already
# warmed up), unfreeze everything and add LoRA on top (target_modules automatically covers
# the router linear layers and the shared submodule M's attention/MLP projection layers via
# find_all_linear_names), while enabling the entropy regularization loss to prevent routing collapse.
# Data: LLaVA-Mix-665k.
# Hyperparameters: lr 2e-5, global batch size 32, warmup_ratio 0.10, deepspeed zero2_offload.
#
# Note: in the current sandbox environment, peft cannot be imported due to an
# accelerate/peft version mismatch. The corresponding train.py code path (LoRA+MoR branch,
# see the `elif model_args.mor_enable:` branch in mor_mllm/train/train.py) has been
# implemented and statically checked, but the actual training effect of LoRA cannot be
# verified on this machine -- it needs to be run on a GPU environment where peft is available.

JSON_FOLDER="train_data/train_json"
IMAGE_FOLDER="train_data"
HF_CACHE_DIR="/data/huggingface_cache"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Annotation file path for LLaVA-Mix-665k; replace with the actual data path.
DATA_PATH="${JSON_FOLDER}/llava_mix_665k.json"

# Router warm-up checkpoint produced by Stage II.
INPUT_MODEL="./checkpoints/mor_phi2/stage2_router_warmup"
OUTPUT_DIR="./checkpoints/mor_phi2/stage3_joint_finetune_lora"

mor_mode="expert"
sharing_strategy="middle_cycle"
num_recursions=6
group_size=5
mor_block_init="grouped_sharing"
rand_router=False
router_type="linear"
gating="weighted"
z_loss_coeff=0.1
kv_sharing_enable=True
kv_sharing_update_cache=False
expert_capacity="0.2"
cap_warmup_step=0
aux_loss_coeff=0.1
entropy_reg_coeff=0.1

HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 deepspeed mor_mllm/train/train.py \
    --mor_enable True --mor_mode ${mor_mode} \
    --sharing_strategy ${sharing_strategy} --num_recursions ${num_recursions} --group_size ${group_size} \
    --mor_block_init ${mor_block_init} --entropy_reg_coeff ${entropy_reg_coeff} \
    --rand_router ${rand_router} --router_type ${router_type} --gating ${gating} \
    --use_aux_loss True --aux_loss_coeff ${aux_loss_coeff} --bal_loss_coeff 0.1 --use_z_loss True --z_loss_coeff ${z_loss_coeff} \
    --kv_sharing_enable ${kv_sharing_enable} --kv_sharing_update_cache ${kv_sharing_update_cache} \
    --expert_capacity "${expert_capacity}" --cap_warmup_step ${cap_warmup_step} \
    --lora_enable True --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 \
    --deepspeed ./scripts/deepspeed/zero2_offload.json \
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
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.10 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir ${HF_CACHE_DIR}
