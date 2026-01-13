#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Two-GPU launch helper for Hi-SAM training.
# Usage:
#   ./run_train_2gpu.sh \
#       --output /path/to/work_dir \
#       --train_datasets crello_train \
#       --val_datasets crello_val
#
# All train.py arguments are forwarded transparently; adjust CUDA_VISIBLE_DEVICES
# if you want to pin different GPUs.
# -----------------------------------------------------------------------------

PROJECT_ROOT="/home/ubuntu/jjseol/data/Hi-SAM"
TRAIN_SCRIPT="${PROJECT_ROOT}/train.py"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Could not locate train.py at ${TRAIN_SCRIPT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}

accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  "${TRAIN_SCRIPT}" \
  "$@"

