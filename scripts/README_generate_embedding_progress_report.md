# Embedding Progress Report

Generate the multi-page GAIRAM embedding evolution PDF from locally discovered run artifacts.

Default usage:

```bash
python scripts/generate_embedding_progress_report.py \
  --output-pdf reports/GAIRAM_embedding_evolution_report.pdf
```

Useful flags:

- `--data-root /Volumes/SSD_Rad/GAIRA_DATA/processed`
- `--previous-report-pdf /Volumes/SSD_Rad/GAIRA_DATA/processed/embedding_v2_gpu_run1_final/GAIRAM_v2_GPU_Run1_Report.pdf`
- `--output-pdf reports/GAIRAM_embedding_evolution_report.pdf`
- `--title "GAIRAM Embedding Evolution Report"`
- `--author "Your Name"`
- `--verbose`

Discovery behavior:

- looks for the expected major run folders under `--data-root`
- gracefully skips missing runs
- uses `embedding_v5_full_true_gpu_run1_eval_v2` when present for full-corpus v5 metrics and sampled UMAPs
- notes staging folders such as `embedding_v4_medium` if they are present but not evaluated

Output:

- multi-page PDF with continuity from the prior report, methods evolution, comparative tables, run-by-run results, visual evolution, interpretation, and next-step recommendation
