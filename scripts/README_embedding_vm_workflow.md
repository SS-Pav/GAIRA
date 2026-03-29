# GAIRAM Embedding VM Workflow

1. Build the training dataset locally from the SSD-backed DuckDB source of truth:
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/build_embedding_dataset.py --output-dir /Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v3_pass3
   ```

2. Sync repo code to the VM when scripts change:
   ```bash
   scripts/sync_repo_to_vm.sh
   ```

3. Sync training-ready artifacts to the VM:
   ```bash
   scripts/sync_embedding_artifacts_to_vm.sh /Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v3_pass3 ~/projects/GAIRA/data/processed/embedding_v3_pass3
   ```

4. Run the tempered Pass 3 training and evaluation on the VM:
   ```bash
   REMOTE_OUTPUT_DIR=~/projects/GAIRA/data/processed/embedding_v3_pass3_gpu_run1 \
   REMOTE_DATASET_PATH=~/projects/GAIRA/data/processed/embedding_v3_pass3/embedding_dataset.npz \
   PRESET=pass3_tempered \
   EPOCHS=30 \
   scripts/run_embedding_training_vm.sh
   ```

5. Optional: copy results back with `gcloud compute scp` or `gcloud compute rsync` from `~/projects/GAIRA/data/processed/embedding_v3_pass3_gpu_run1`.

Notes:
- Local Mac + SSD remains the preprocessing source of truth.
- The VM only consumes packaged artifacts plus repo code.
- Training/evaluation scripts accept `--dataset-path` and `--output-dir` for overrides.
- Presets available in `scripts/train_embedding_model.py`: `pass2_baseline`, `pass3_aggressive`, `pass3_tempered`.
