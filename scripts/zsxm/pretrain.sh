#!/bin/bash

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /c22073/LLM_weights/llava-v1.5-13b \
    --version plain \
    --data_path ./playground/patho_data/combine/quilt_patch-bbox-contour_336_14_train.json \
    --image_folder /c22073/datasets/combination \
    --vision_tower /c22073/LLM_weights/clip-vit-large-patch14-336 \
    --mm_projector_type qformer_128_2 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer="-2,0" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/zcombine/pretrain_patch-bbox-contour_llava-13b_QFormer-128-2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard

# deepspeed --include localhost:0,2 llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path checkpoints/vicuna-7b-v1.5 \
#     --version patho_pretrain \
#     --data_path ./playground/patho_data/pretrain/quilt1m_train.json \
#     --image_folder /medical-data/zsxm/public_dataset/image-caption/Quilt-1M/images \
#     --vision_tower checkpoints/QuiltNet-B-16 \
#     --mm_projector_type mlp2x_gelu \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --bf16 True \
#     --output_dir ./checkpoints/zpatho_0pretrain/llava_QuiltNet-B-16 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 64 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 500 \
#     --save_total_limit 3 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to tensorboard