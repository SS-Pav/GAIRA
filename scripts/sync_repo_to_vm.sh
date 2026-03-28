#!/bin/zsh
set -euo pipefail

INSTANCE="${INSTANCE:-gaira-gpu-1}"
ZONE="${ZONE:-us-east4-c}"
REMOTE_REPO="${REMOTE_REPO:-~/projects/GAIRA}"

gcloud compute ssh "$INSTANCE" --zone "$ZONE" --command "mkdir -p $REMOTE_REPO"
gcloud compute rsync . "$INSTANCE:$REMOTE_REPO" --zone "$ZONE" --recurse --delete --exclude ".git" --exclude ".venv" --exclude "__pycache__"
