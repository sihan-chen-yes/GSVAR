#!/bin/bash

SCENE_ID=${SCENE_ID:-ded5e4b46aedbef4cdb7bd1db7fc4cc5b00a9979ad6464bdadfab052cd64c101}
DATA_BASE_DIR=${DATA_BASE_DIR:-../dataset/DL3DV-10K-Benchmark}
FIXER=${FIXER:-difix}
OUTPUT_BASE_DIR=${OUTPUT_BASE_DIR:-../dataset/DL3DV-val/output/${FIXER}/gsplat}
CUDA_DEV=${CUDA_DEV:-0}
EVERY=${EVERY:-50}
DATA_FACTOR=${DATA_FACTOR:-4}
CKPT=${CKPT:-}
USE_REF=${USE_REF:-}
USE_BRIDGE=${USE_BRIDGE:-}
TILING=${TILING:-}
VAR_TEST_PATH=${VAR_TEST_PATH:-./results/GS_VARSR_w_ref_1_flex_more/ar-ckpt-last.pth}

# Derived paths using defaults/passed variables
DATA_DIR=${DATA_BASE_DIR}/${SCENE_ID}/gaussian_splat
OUTPUT_DIR=${OUTPUT_BASE_DIR}/${SCENE_ID}/${EVERY}

# 3DGS reconstruction Checkpoint Path (optional)
CKPT_ARG=""
if [[ -n "$CKPT" ]]; then
    # If CKPT is not empty, build the argument.
    CKPT_ARG="--ckpt ${CKPT}"
fi

USE_REF_ARG=""
if [[ -n "$USE_REF" ]]; then
    USE_REF_ARG="--use_ref"
fi

USE_BRIDGE_ARG=""
if [[ -n "$USE_BRIDGE" ]]; then
    USE_BRIDGE_ARG="--use_bridge"
fi

TILING_ARG=""
if [[ -n "$TILING" ]]; then
    TILING_ARG="--tiling"
fi

# --- Execution Command ---

echo "--- Running GSVAR 3DGS Job ---"
echo "SCENE_ID: ${SCENE_ID}"
echo "EVERY: ${EVERY}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "FIXER: ${FIXER}"
echo "VAR_TEST_PATH: ${VAR_TEST_PATH}"
echo "TILING: ${TILING_ARG}"
echo "USE_REF: ${USE_REF_ARG}"
echo "USE_BRIDGE: ${USE_BRIDGE_ARG}"
echo "CKPT_PATH: ${CKPT_ARG}"
echo "DATA_DIR: ${DATA_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEV}"
echo "DATA_FACTOR: ${DATA_FACTOR}"
echo "-----------------------------------------"

CUDA_VISIBLE_DEVICES=${CUDA_DEV} python -m examples.gsplat.simple_trainer_difix3d default \
    --data_dir "${DATA_DIR}" \
    --data_factor "${DATA_FACTOR}" \
    --result_dir "${OUTPUT_DIR}" \
    --var_test_path "${VAR_TEST_PATH}" \
    --no-normalize-world-space \
    --test_every "${EVERY}" \
    --fix \
    --fixer "${FIXER}" \
    ${TILING_ARG} \
    ${CKPT_ARG} \
    ${USE_REF_ARG} \
    ${USE_BRIDGE_ARG}

