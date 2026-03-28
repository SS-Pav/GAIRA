#!/bin/zsh
set -euo pipefail

INSTANCE="${INSTANCE:-gaira-gpu-1}"
ZONE="${ZONE:-us-east4-c}"
LOCAL_DIR="${1:-/Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v3_pass3}"
REMOTE_DIR="${2:-~/projects/GAIRA/data/processed/embedding_v3_pass3}"

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "mkdir -p $REMOTE_DIR"
gcloud compute scp --zone "$ZONE" \
  "$LOCAL_DIR/embedding_dataset.npz" \
  "$LOCAL_DIR/dataset_summary.csv" \
  "$INSTANCE:$REMOTE_DIR/"
