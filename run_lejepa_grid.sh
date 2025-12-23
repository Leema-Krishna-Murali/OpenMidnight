#!/usr/bin/env bash
set -euo pipefail

# Master node configuration
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29503

export NNODES=1
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NODE_RANK=0

# Base config and grid output
CONFIG_BASE="./dinov2/configs/train/vits14_reg_ablations_lejepa.yaml"
GRID_ROOT="./output_lejepa_grid"
RESUME="False"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# name lambd base_lr weight_decay
LR_GRID=(2e-3 2e-4)
LAMBDA_GRID=(.05)
WD_GRID=(.05)
RUNS=()
for LR in "${LR_GRID[@]}"; do
  for LAMBD in "${LAMBDA_GRID[@]}"; do
    for WD in "${WD_GRID[@]}"; do
      NAME="lr${LR}_lam${LAMBD}_wd${WD}"
      RUNS+=("${NAME} ${LAMBD} ${LR} ${WD}")
    done
  done
done

mkdir -p "${GRID_ROOT}"

for run in "${RUNS[@]}"; do
  read -r NAME LAMBD BASE_LR WEIGHT_DECAY <<< "${run}"
  RUN_DIR="${GRID_ROOT}/${NAME}"
  CONFIG_OUT="${RUN_DIR}/config.yaml"

  if [[ "${RESUME}" == "True" ]]; then
    echo "Resume enabled; preserving ${RUN_DIR}"
  else
    echo "Resume disabled; cleaning ${RUN_DIR}"
    rm -rf "${RUN_DIR}"
  fi
  mkdir -p "${RUN_DIR}"

  uv run python - <<PY
from omegaconf import OmegaConf

cfg = OmegaConf.load("${CONFIG_BASE}")
cfg.train.output_dir = "${RUN_DIR}"
cfg.train.batch_size_per_gpu = 32
cfg.train.resume = "${RESUME}".lower() == "true"
cfg.train.wandb_name = f"${NAME}"
cfg.lejepa.lambd = float("${LAMBD}")
cfg.optim.base_lr = float("${BASE_LR}")
cfg.optim.weight_decay = float("${WEIGHT_DECAY}")
OmegaConf.save(cfg, "${CONFIG_OUT}")
PY

  echo "Launching ${NAME} -> ${CONFIG_OUT}"
  uv run torchrun \
    --nnodes "${NNODES}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    dinov2/train/train_lejepa.py \
    "${CONFIG_OUT}"
done
