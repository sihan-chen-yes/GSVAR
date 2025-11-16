#!/bin/bash

SCENE_ID=${SCENE_ID:-ded5e4b46aedbef4cdb7bd1db7fc4cc5b00a9979ad6464bdadfab052cd64c101}
DATA_BASE_DIR=${DATA_BASE_DIR:-../dataset/DL3DV-10K-Benchmark}
FIXER=${FIXER:-difix}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-../dataset/DL3DV-val/output/${FIXER}/gsplat}
CUDA_DEV=${CUDA_DEV:-0}
EVERY=${EVERY:-50}
DATA_FACTOR=${DATA_FACTOR:-4}
CKPT=${CKPT:-}

# Derived paths using defaults/passed variables
DATA_DIR=${DATA_BASE_DIR}/${SCENE_ID}/gaussian_splat
OUTPUT_DIR=${OUTPUT_BASE_DIR}/${SCENE_ID}/${EVERY}

# 3DGS reconstruction Checkpoint Path (optional)
CKPT_ARG=""
if [[ -n "$CKPT" ]]; then
    # If CKPT is not empty, build the argument.
    CKPT_ARG="--ckpt ${CKPT}"
fi

# --- Execution Command ---

echo "--- Running Difix 3DGS Job ---"
echo "SCENE_ID: ${SCENE_ID}"
echo "EVERY: ${EVERY}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "FIXER: ${FIXER}"
echo "CKPT_PATH: ${CKPT_ARG}"
echo "DATA_DIR: ${DATA_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEV}"
echo "DATA_FACTOR: ${DATA_FACTOR}"
echo "-----------------------------------------"

CUDA_VISIBLE_DEVICES=${CUDA_DEV} python -m examples.gsplat.simple_trainer_difix3d default \
    --data_dir "${DATA_DIR}" \
    --data_factor "${DATA_FACTOR}" \
    --result_dir "${OUTPUT_DIR}" \
    --no-normalize-world-space \
    --test_every "${EVERY}" \
    --fix \
    --fixer "${FIXER}" \
    ${CKPT_ARG} \

