#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASET_PATH="${DATASET_PATH:-/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v5_full_true/embedding_dataset.npz}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v7_anchor_gpu_run1/model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/reports/embedding_v8_small2023_cellline_branch_smoke}"
REPORT_DIR="${REPORT_DIR:-$OUTPUT_DIR/eval_v2}"
DIAGNOSTIC_DIR="${DIAGNOSTIC_DIR:-$OUTPUT_DIR/branch_diagnostics}"
COMPOSITION_DIR="${COMPOSITION_DIR:-$OUTPUT_DIR/composition}"
EPOCHS="${EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:-4}"

cd "$REPO_ROOT"
[[ -f ".venv/bin/activate" ]] && source ".venv/bin/activate"
export PYTHONPATH="$REPO_ROOT/src"
export MPLCONFIGDIR="$REPO_ROOT/.matplotlib-cache"

python scripts/train_embedding_model.py \
  --dataset-path "$DATASET_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --branch-mode small2023_cellline \
  --preset pass8_small2023_specialized \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --class-positive-weight 0.24 \
  --probe-confusion-weight 0.0 \
  --class-compactness-weight 0.06 \
  --class-compactness-target-radius 0.55 \
  --epochs "$EPOCHS" \
  --max-steps-per-epoch "$MAX_STEPS"

python scripts/extract_embeddings.py \
  --dataset-path "$OUTPUT_DIR/branch_dataset.npz" \
  --output-dir "$OUTPUT_DIR"

python scripts/evaluate_embeddings_v2.py \
  --output-dir "$OUTPUT_DIR" \
  --report-dir "$REPORT_DIR" \
  --sample-size-global-metrics 3000 \
  --knn-k 6 \
  --seed 7

python scripts/evaluate_embedding_probes.py \
  --output-dir "$OUTPUT_DIR" \
  --report-dir "$REPORT_DIR"

python scripts/visualize_embeddings_v2.py \
  --output-dir "$OUTPUT_DIR" \
  --report-dir "$REPORT_DIR" \
  --projection-backend pca

python scripts/build_v8_small2023_branch_diagnostics.py \
  --mode cellline \
  --branch-run-dir "$OUTPUT_DIR" \
  --output-dir "$DIAGNOSTIC_DIR"

python scripts/build_v8_small2023_composition_analysis.py \
  --mode cellline \
  --branch-run-dir "$OUTPUT_DIR" \
  --output-dir "$COMPOSITION_DIR"

python scripts/build_v8_small2023_branch_report.py \
  --mode cellline \
  --branch-run-dir "$OUTPUT_DIR" \
  --diagnostic-dir "$DIAGNOSTIC_DIR" \
  --composition-dir "$COMPOSITION_DIR" \
  --output-path "$OUTPUT_DIR/branch_report.md"
