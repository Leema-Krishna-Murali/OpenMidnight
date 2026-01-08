#!/usr/bin/env bash
#SBATCH --job-name=dinov2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --time=infinite
#SBATCH --mem=0 #all memory on node
#SBATCH --output=slurms/slurm-%j.out
#SBATCH --error=slurms/slurm-%j.err
#SBATCH --partition=main
#SBATCH --nodelist=n-2
#SBATCH --account=training

set -euo pipefail

# ---- Training config ----
CONFIG_FILE="./dinov2/configs/train/vitg14_reg4.yaml"
OUTPUT_DIR="./output_test2"
RESUME="False"   # "True" to keep OUTPUT_DIR and resume if checkpoints exist

# ---- Repo env ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DINOV2_RUN_SCRIPT="${REPO_ROOT}/$(basename "${BASH_SOURCE[0]}")"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Avoid NCCL peer-to-peer issues on nodes with broken/limited GPU P2P.
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1

# Clean output directory only once (rank 0)
if [[ "${RESUME}" == "True" ]]; then
  RESUME_FLAG=""
else
  RESUME_FLAG="--no-resume"
  if [[ "${SLURM_PROCID}" == "0" ]]; then
    rm -rf "${OUTPUT_DIR}"
  fi
fi
mkdir -p "${OUTPUT_DIR}"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

# One task per node; each task runs torchrun and spawns 8 GPU workers locally.
uv run torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  --node_rank 0 \
  dinov2/train/train.py \
  --config-file "${CONFIG_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  ${RESUME_FLAG}
