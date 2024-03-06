#!/bin/bash

# 替换为你的存储checkpoint的目录的绝对路径
CHECKPOINT_DIR="/c22073/codes/llava-1.5/checkpoints/zcombine"

# 初始化logdir_spec参数
LOGDIR_SPEC=""

# 遍历checkpoint目录下的所有模型目录
for MODEL_DIR in "$CHECKPOINT_DIR"/*; do
  if [ -d "$MODEL_DIR" ]; then
    # 获取模型名称，即目录名
    MODEL_NAME=$(basename "$MODEL_DIR")
    # 定位到模型的runs目录
    RUNS_DIR="$MODEL_DIR/runs"
    if [ -d "$RUNS_DIR" ]; then
      # 如果LOGDIR_SPEC不为空，添加逗号分隔符
      if [ ! -z "$LOGDIR_SPEC" ]; then
        LOGDIR_SPEC="$LOGDIR_SPEC,"
      fi
      # 添加模型的runs目录到LOGDIR_SPEC
      LOGDIR_SPEC="$LOGDIR_SPEC$MODEL_NAME:$RUNS_DIR"
    fi
  fi
done

# 启动TensorBoard
tensorboard --logdir_spec=$LOGDIR_SPEC

