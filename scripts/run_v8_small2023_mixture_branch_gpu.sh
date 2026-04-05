#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_PATH="${DATASET_PATH:-/home/suraj/projects/GAIRA/data/processed/embedding_v5_full_true/embedding_dataset.npz}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/home/suraj/projects/GAIRA/data/processed/embedding_v7_anchor_gpu_run1/model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/suraj/projects/GAIRA/data/processed/embedding_v8_small2023_mixture_branch_gpu_run1}"
REPORT_DIR="${REPORT_DIR:-/home/suraj/projects/GAIRA/data/processed/embedding_v8_small2023_mixture_branch_gpu_run1_eval_v2}"
EPOCHS="${EPOCHS:-30}"

cd "$REPO_ROOT"
[[ -f ".venv/bin/activate" ]] && source ".venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src"

python scripts/train_embedding_model.py \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --branch-mode small2023_mixture \
  --preset pass8_small2023_specialized \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --class-positive-weight 0.24 \
  --probe-confusion-weight 0.24 \
  --cross-probe-positive-boost 3.0 \
  --same-probe-positive-weight 0.25 \
  --class-compactness-weight 0.03 \
  --class-compactness-target-radius 0.60 \
  --epochs "$EPOCHS"

python scripts/extract_embeddings.py \
  --dataset-path "$OUTPUT_DIR/branch_dataset.npz" \
  --output-dir "$OUTPUT_DIR"

python scripts/evaluate_embeddings_v2.py \
  --output-dir "$OUTPUT_DIR" \
  --report-dir "$REPORT_DIR" \
  --sample-size-global-metrics 15000 \
  --knn-k 6 \
  --seed 7

python scripts/evaluate_embedding_probes.py \
  --output-dir "$OUTPUT_DIR" \
  --report-dir "$REPORT_DIR"

python scripts/visualize_embeddings_v2.py \
  --output-dir "$OUTPUT_DIR" \
  --report-dir "$REPORT_DIR"
