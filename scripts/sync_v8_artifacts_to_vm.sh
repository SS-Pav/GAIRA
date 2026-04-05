#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/../config/gcp_config.sh"

VM_ROOT="${VM_ROOT:-~/projects/GAIRA}"

cd "$REPO_ROOT"

echo "Syncing v8 scripts and reports to $GAIRA_GCP_VM:$VM_ROOT"
gcloud compute scp \
  --project="$GAIRA_GCP_PROJECT" \
  --zone="$GAIRA_GCP_ZONE" \
  scripts/build_v8_master_shared_backbone_diagnostics.py \
  scripts/build_v8_ev_stress_prep.py \
  scripts/build_v8_small2023_specialized_prep.py \
  scripts/build_v8_serum_cohort_prep.py \
  scripts/build_v8_master_report.py \
  scripts/run_v8_ev_stress_gpu.sh \
  scripts/run_v8_small2023_gpu.sh \
  reports/v8_run_plan.md \
  "$GAIRA_GCP_VM:$VM_ROOT/scripts/"

gcloud compute scp \
  --project="$GAIRA_GCP_PROJECT" \
  --zone="$GAIRA_GCP_ZONE" \
  src/gaira/demo/v8_master_utils.py \
  src/gaira/demo/v8_theme_utils.py \
  src/gaira/demo/v8_report_layout.py \
  "$GAIRA_GCP_VM:$VM_ROOT/src/gaira/demo/"
