#!/bin/bash
# MoR-Tuning Stage I: Projection Warm-up
# At this stage the model is still an ordinary dense Phi-2 (not yet
# converted to MoR -- MoR conversion starts in Stage II).
# Data: LLaVA-Pretrain (~558K image-caption pairs).
# Hyperparameters: lr 1e-4, global batch size 64, warmup_ratio 0.03.
# global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus


JSON_FOLDER="train_data/train_json"
IMAGE_FOLDER="train_data"
HF_CACHE_DIR="/data/huggingface_cache"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Annotation file path for LLaVA-Pretrain (558K)
DATA_PATH="${JSON_FOLDER}/llava_pretrain_558k.json"

OUTPUT_DIR="./checkpoints/mor_phi2/stage1_projection_warmup"

HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0 deepspeed mor_mllm/train/train.py \
    --deepspeed ./scripts/deepspeed/zero2.json \
    --model_name_or_path microsoft/phi-2 \
    --version plain \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower openai/clip-vit-large-patch14-336 \
    --image_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --freeze_backbone True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
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
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --dataloader_pin_memory False \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir ${HF_CACHE_DIR}
