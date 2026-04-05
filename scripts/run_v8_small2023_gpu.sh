#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/../config/gcp_config.sh"

DATASET_PATH="${DATASET_PATH:-/home/suraj/projects/GAIRA/data/processed/embedding_v5_full_true/embedding_dataset.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/suraj/projects/GAIRA/data/processed/embedding_v8_small2023_gpu_run1}"
REPORT_DIR="${REPORT_DIR:-/home/suraj/projects/GAIRA/data/processed/embedding_v8_small2023_gpu_run1_eval_v2}"
PRESET="${PRESET:-pass7_anchor_invariance}"
EPOCHS="${EPOCHS:-30}"
SEED="${SEED:-7}"

cd "$REPO_ROOT"
[[ -f ".venv/bin/activate" ]] && source ".venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src"

echo "Running v8 small2023 specialized GPU branch"
echo "VM:      $GAIRA_GCP_VM ($GAIRA_GCP_PROJECT / $GAIRA_GCP_ZONE)"
echo "Dataset: $DATASET_PATH"
echo "Output:  $OUTPUT_DIR"

python scripts/train_embedding_model.py \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --preset "$PRESET" \
  --seed "$SEED"

python scripts/extract_embeddings.py \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR"

python scripts/evaluate_embeddings_v2.py \
  --output-dir "$OUTPUT_DIR" \
  --report-dir "$REPORT_DIR" \
  --sample-size-global-metrics 15000 \
  --knn-k 6 \
  --seed "$SEED"

python scripts/visualize_embeddings_v2.py \
  --output-dir "$OUTPUT_DIR" \
  --report-dir "$REPORT_DIR"

python scripts/evaluate_embedding_probes.py \
  --output-dir "$OUTPUT_DIR" \
  --report-dir "$REPORT_DIR"
