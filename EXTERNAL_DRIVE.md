# External Drive Layout

This file documents the expected layout of the external drive mounted at
`/mnt/datasets-big/`. It is used for storing HuggingFace caches, model
checkpoints, eval outputs, and training data locally to avoid streaming
from HuggingFace during measured benchmark runs.

## Directory Structure

```
/mnt/datasets-big/
├── hf-cache/           # HuggingFace cache (datasets + hub)
│   ├── datasets/
│   ├── hub/
│   └── huggingface/
├── models-archive/     # Model checkpoint archives
├── benchmark-runs-archive/  # Archived benchmark run outputs
├── evals/              # Evaluation outputs
├── raw/                # Raw input data
├── tokenized/          # Tokenized datasets
└── training/           # Training data
```

## Setup

```bash
# Mount the external drive
sudo mount /dev/disk/by-id/... /mnt/datasets-big

# Verify the layout
ls -lRt /mnt/datasets-big/
```
