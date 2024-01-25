#!/bin/bash
export PATH=/medical-data/zsxm/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/medical-data/zsxm/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/medical-data/zsxm/local/cuda-11.7
export CUDA_ROOT=/medical-data/zsxm/local/cuda-11.7

export CUDA_DEVICE_ORDER=PCI_BUS_ID

deepspeed --include localhost:0,1 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path checkpoints/vicuna-7b-v1.5 \
    --version patho_pretrain \
    --data_path ./playground/patho_data/pretrain/quilt1m_train.json \
    --image_folder /medical-data/zsxm/public_dataset/image-caption/Quilt-1M/images \
    --vision_tower ./resnet/yxt_resnet18_hf \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./checkpoints/zpatho_0pretrain/llava_resnet18_yxt \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard