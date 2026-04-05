#!/bin/zsh
set -euo pipefail

INSTANCE="${INSTANCE:-gaira-gpu-1}"
ZONE="${ZONE:-us-east4-c}"
REMOTE_REPO="${REMOTE_REPO:-~/projects/GAIRA}"
REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-$REMOTE_REPO/data/processed/embedding_v7_anchor_gpu_run1}"
REMOTE_DATASET_PATH="${REMOTE_DATASET_PATH:-$REMOTE_REPO/data/processed/embedding_v5_full_true/embedding_dataset.npz}"
EPOCHS="${EPOCHS:-30}"
PRESET="${PRESET:-pass7_anchor_invariance}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "
  set -euo pipefail
  cd $REMOTE_REPO
  python3 -m venv .venv
  . .venv/bin/activate
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet torch umap-learn scikit-learn pandas numpy matplotlib seaborn
  mkdir -p $REMOTE_OUTPUT_DIR
  PYTHONPATH=src .venv/bin/python scripts/train_embedding_model.py --dataset-path $REMOTE_DATASET_PATH --output-dir $REMOTE_OUTPUT_DIR --epochs $EPOCHS --preset $PRESET $EXTRA_TRAIN_ARGS
  PYTHONPATH=src .venv/bin/python scripts/extract_embeddings.py --dataset-path $REMOTE_DATASET_PATH --output-dir $REMOTE_OUTPUT_DIR
  PYTHONPATH=src .venv/bin/python scripts/visualize_embeddings.py --output-dir $REMOTE_OUTPUT_DIR
  PYTHONPATH=src .venv/bin/python scripts/evaluate_embeddings.py --output-dir $REMOTE_OUTPUT_DIR
  PYTHONPATH=src .venv/bin/python scripts/evaluate_embedding_probes.py --output-dir $REMOTE_OUTPUT_DIR
  PYTHONPATH=src .venv/bin/python scripts/check_embedding_pipeline.py --dataset-path $REMOTE_DATASET_PATH --output-dir $REMOTE_OUTPUT_DIR
"
