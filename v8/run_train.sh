#!/usr/bin/env bash
# ============================================================
#  AerialDojo 训练启动脚本（cluster41, v1 venv）
#  用法：
#    tmux new -s train
#    bash run_train.sh
#    Ctrl+B D   # 脱离，断网也不怕
#  日志：/tmp/train_voxel.log 与 /tmp/train_vla.log
# ============================================================
set -u

# —— 环境（cluster41 已知路径；换机器请改这两行）——
source ~/lewm_versions/v1/.venv/bin/activate
cd ~/lewm_versions/lewm_github/v8

# 限制每进程 OpenMP 线程，避免多 worker 时突破 ulimit（与预处理同理）
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "===== [1/2] 基线 voxel（验证整条 pipeline，不依赖语言）====="
python -u train.py --config-name=lewm_voxel_aerialdojo \
  trainer.devices=auto trainer.accelerator=gpu loader.batch_size=32 \
  2>&1 | tee /tmp/train_voxel.log

echo "===== [2/2] VLA（语言条件动作预测，num_actions=6 与数据 6 类对齐）====="
python -u train.py --config-name=lewm_vla_aerialdojo \
  trainer.devices=auto trainer.accelerator=gpu loader.batch_size=32 \
  model.num_actions=6 \
  2>&1 | tee /tmp/train_vla.log

echo "===== 训练结束，日志见 /tmp/train_voxel.log 与 /tmp/train_vla.log ====="
