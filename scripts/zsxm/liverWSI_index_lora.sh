#!/bin/bash
export PATH=/medical-data/zsxm/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/medical-data/zsxm/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/medical-data/zsxm/local/cuda-11.7
export CUDA_ROOT=/medical-data/zsxm/local/cuda-11.7

export CUDA_DEVICE_ORDER=PCI_BUS_ID

deepspeed --include localhost:0,2 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/llava-v1.5-7b \
    --version patho_finetune \
    --data_path ./playground/patho_data/liverWSI/edge04_index_336_14_train.json \
    --image_folder /medical-data/zsxm/datasets/liverWSI/edge04/images \
    --vision_tower ./checkpoints/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/zpatho_0pretrain/llava-patho-a30/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/zpatho_2liverWSI/llava-patho-a30_lora_edge04_index \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
