#!/usr/bin/env bash
set -euo pipefail

# Master node configuration
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29502

export NNODES=1
export NPROC_PER_NODE=8
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NODE_RANK=0

# Training config (keep OUTPUT_DIR in sync with train.output_dir in the YAML)
CONFIG_FILE="./dinov2/configs/train/vits14_reg_ablations_lejepa.yaml"
OUTPUT_DIR="./output_lejepa"
RESUME=$(uv run python - <<PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("${CONFIG_FILE}")
print(str(cfg.train.resume))
PY
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${RESUME}" == "True" ]]; then
  echo "Resume enabled; preserving ${OUTPUT_DIR}"
else
  echo "Resume disabled; cleaning ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

echo "[Master Node] Starting LeJEPA training..."
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

uv run torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  dinov2/train/train_lejepa.py \
  "${CONFIG_FILE}"
