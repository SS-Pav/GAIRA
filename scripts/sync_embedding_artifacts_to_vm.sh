#!/bin/zsh
set -euo pipefail

INSTANCE="${INSTANCE:-gaira-gpu-1}"
ZONE="${ZONE:-us-east4-c}"
LOCAL_DIR="${1:-/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v5_full_true}"
REMOTE_DIR="${2:-~/projects/GAIRA/data/processed/embedding_v5_full_true}"
LOCAL_ANCHOR_DIR="${3:-/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_anchor_audit}"
REMOTE_ANCHOR_DIR="${4:-~/projects/GAIRA/data/processed/embedding_anchor_audit}"

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "mkdir -p $REMOTE_DIR"
gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "mkdir -p $REMOTE_ANCHOR_DIR"
gcloud compute scp --zone "$ZONE" \
  "$LOCAL_DIR/embedding_dataset.npz" \
  "$LOCAL_DIR/dataset_summary.csv" \
  "$INSTANCE:$REMOTE_DIR/"
gcloud compute scp --zone "$ZONE" \
  "$LOCAL_ANCHOR_DIR/embedding_anchor_table_v1.csv" \
  "$INSTANCE:$REMOTE_ANCHOR_DIR/"
