# STORAGE_PLAN — DGX Spark Benchmark / Dataset / Model Storage

## 1. Hardware Overview

### Internal NVMe (hot storage)

```text
/dev/nvme0n1p2  ~3.7T total, ~1.9T used, ~1.7T available
  nvme0n1p1  EFI
  nvme0n1p2  ext4 root mounted at /
```

**Use for:**

- Active/default model weights (`agent-code`, `qwen-dense`, current default coding model)
- Current benchmark harness repo and active runs
- Docker images and active compose stacks
- Latency-sensitive workloads (frequent model loading, always-on services)

### External USB4 NVMe (bulk storage)

```text
/dev/sda  model: ASM246X  ~3.6T
mounted at: /mnt/datasets-big/
filesystem label: datasets-big
```

**Persistent mount entry** (`/etc/fstab`):

```text
LABEL=datasets-big /mnt/datasets-big ext4 defaults,nofail,x-systemd.device-timeout=10 0 2
```

The `nofail` option is intentional so the system can still boot if the external drive is disconnected.

After editing `/etc/fstab`, reload systemd:

```bash
sudo systemctl daemon-reload
sudo umount /mnt/datasets-big
sudo mount -a
df -h /mnt/datasets-big
```

**Verify mount before containers:**

```bash
df -h /mnt/datasets-big
mount | grep datasets-big
```

---

## 2. External Drive Layout

```text
/mnt/datasets-big/
  hf-cache/               Hugging Face cache
    huggingface/           HF_HOME (models, configs, tokenizers)
    datasets/              HF_DATASETS_CACHE (downloaded/processed datasets)
    hub/                   HF_HUB_CACHE (model snapshots, Hub artifacts)
  raw/                    Raw downloaded source data (before cleaning/filtering)
  evals/                  Pinned benchmark/evaluation datasets
  training/               Prepared training datasets (SFT, DPO, LoRA, etc.)
  tokenized/              Pre-tokenized/packed datasets for faster training
  benchmark-runs-archive/ Older benchmark run outputs, logs, reports, artifacts
  models-archive/         Inactive or experimental model checkpoints
```

---

## 3. Directory Details & Access Policies

### 3.1 HF Cache (`hf-cache/`)

Stable host path for Hugging Face cache. Avoids Docker containers inventing root-owned caches.

```bash
export HF_HOME=/mnt/datasets-big/hf-cache/huggingface
export HF_DATASETS_CACHE=/mnt/datasets-big/hf-cache/datasets
export HF_HUB_CACHE=/mnt/datasets-big/hf-cache/hub
```

### 3.2 Raw (`raw/`)

Raw downloaded source data before cleaning, filtering, or tokenization.

Examples: raw HF dataset samples, public corpus slices, unprocessed JSONL/Parquet dumps.

**Do not train directly from this directory** unless intentionally doing raw-data experiments.

### 3.3 Eval Datasets (`evals/`)

Pinned benchmark/evaluation datasets and task packs. Benchmarks must use **local pinned subsets**, never live streaming during measured runs.

**Recommended pattern:**

```text
/mnt/datasets-big/evals/<dataset_id>/
  tasks.jsonl
  MANIFEST.json
```

**Dataset IDs:**

```text
smoke_v1, coding_smoke_v1, local_coding_v1, long_context_v1
mmlu_pro_subset_v1, gpqa_subset_v1, ifeval_v1
```

**Manifest format:**

```json
{
  "name": "fineweb_edu_sample_10k",
  "source": "HuggingFaceFW/fineweb-edu",
  "split": "train",
  "streaming": true,
  "sample_count": 10000,
  "sample_seed": 1234,
  "created_at": "2026-05-04",
  "format": "jsonl",
  "checksum": "..."
}
```

**Pin these fields for reproducibility:** dataset name, source, revision/commit hash, download date, split, sample seed, row IDs, license notes, checksum.

### 3.4 Training Data (`training/`)

Prepared training datasets: SFT, DPO/ORPO preference data, LoRA fine-tuning data, continued-pretraining shards.

**Pattern (flat):**

```text
/mnt/datasets-big/training/local_agent_sft_v1/
  train.jsonl
  validation.jsonl
  MANIFEST.json
```

**Pattern (sharded):**

```text
/mnt/datasets-big/training/openwebmath_sample_v1/
  shard-00000.parquet
  shard-00001.parquet
  MANIFEST.json
```

### 3.5 Tokenized Data (`tokenized/`)

Pre-tokenized and packed datasets ready for faster training runs. Avoids repeated CPU-heavy tokenization.

**Organize by tokenizer/model family:**

```text
/mnt/datasets-big/tokenized/qwen3_tokenizer/local_agent_sft_v1/
/mnt/datasets-big/tokenized/qwen3_tokenizer/openwebmath_sample_v1/
```

**Regenerate when:** tokenizer changes, chat template changes, max sequence length changes, packing strategy changes.

### 3.6 Pipeline (local SSD only)

Intermediate processing pipeline — **not on external drive**. Final output goes to `/mnt/datasets-big/tokenized/`.

```text
~/datasets/pipeline/   (local SSD only)
  raw/
  clean/
  formatted/
  tokenized/           → final output goes to /mnt/datasets-big/tokenized/
  packed/
  manifests/
```

### 3.7 Benchmark Runs Archive (`benchmark-runs-archive/`)

Older benchmark run outputs, logs, reports, raw responses, and artifacts. Use internal NVMe for active/current runs, then archive here.

Examples: raw model responses, judge outputs, SQLite snapshots, HTML/Markdown reports, timing summaries, stderr/stdout logs.

### 3.8 Models Archive (`models-archive/`)

Inactive or experimental model checkpoints — large archives, backup copies, rarely used snapshots.

**Keep on internal NVMe when:** frequent/default use (maxbrain, current coding model).
**Put on external drive when:** occasional use, internal space is tight.

**Impact of external drive on model loading:**

```text
Cold startup:   likely slower (+30–90s depending enclosure/thermal behavior)
Warm inference: effectively unchanged
Decode speed:   effectively unchanged
```

---

## 4. Storage Hierarchy Strategy

```text
1. Small benchmark data
   → vendor into the repo or cache locally as JSONL/Parquet

2. Medium repeated benchmark data
   → download once to /mnt/datasets-big/evals/ or /mnt/datasets-big/training/

3. Huge training/pretraining data
   → stream or pre-shard, then cache only the subset you actually use
```

Avoid repeatedly pulling huge datasets from HuggingFace during every run. That makes benchmark results noisy, creates network dependency, and can pollute timing numbers.

---

## 5. Dataset Access Patterns by Use Case

### A. Public eval benchmark

```text
download once → local JSONL/Parquet → pinned manifest
```

Why: small enough, reproducible, no network variability, easy to rerun.

```bash
python scripts/prepare_eval_dataset.py \
  --dataset mmlu_pro --split test --sample 2000 --seed 1234 \
  --out /mnt/datasets-big/evals/mmlu_pro_v1
```

### B. Long-context benchmark

```text
local source docs/logs/code → generated context packs → cached prompt files
```

```text
/mnt/datasets-big/evals/long_context_v1/
  contexts/
    qwen3_replicate_032k.txt
    qwen3_replicate_064k.txt
    qwen3_replicate_128k.txt
  tasks.jsonl
  MANIFEST.json
```

Why: context packing should be deterministic, token counts known before run, do not rebuild 128k prompts every benchmark.

### C. SFT / LoRA training

```text
local sharded JSONL/Parquet → pre-tokenized cache → packed sequences
```

Keep raw text and formatted chat examples. Also keep tokenized/packed cache. Regenerate only when tokenizer, max length, or formatting changes.

### D. Continued pretraining

```text
stream massive dataset → filter/sample/shard locally → train from local shards
```

Do not mirror all of FineWeb/OpenWebMath/The Stack unless truly needed. Pull a high-quality slice.

### E. One-off exploration

Use HF streaming. This is the right place for streaming:

```python
from datasets import load_dataset

ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    split="train",
    streaming=True,
)

for i, row in enumerate(ds):
    print(row.keys())
    if i == 10:
        break
```

---

## 6. Data Format Guidelines

### Benchmark evals

```text
JSONL    → easiest to inspect, diff, and version
Parquet  → better for larger structured data
Arrow    → good with Hugging Face datasets
SQLite   → good for task registry / metadata, not huge text corpora
```

### Training data

Avoid millions of tiny files. Use sharded formats:

```text
Parquet shards
JSONL.gz shards
WebDataset tar shards
HF Arrow cache
```

WebDataset splits large datasets into many shard files (~1GB each), streamed sequentially. PyTorch describes it as a solution for large datasets and many-file I/O problems.

**Good training layout:**

```text
/mnt/datasets-big/training/code_sft_v1/
  shard-00000.jsonl.gz
  shard-00001.jsonl.gz
  MANIFEST.json
/mnt/datasets-big/training/openwebmath_sample_v1/
  shard-00000.parquet
  shard-00001.parquet
  MANIFEST.json
```

---

## 7. Dataset Registry

The harness should not hardcode dataset paths. Add a registry:

```yaml
datasets:
  local_coding_v1:
    type: jsonl
    path: /mnt/datasets-big/evals/local_coding_v1/tasks.jsonl
    manifest: /mnt/datasets-big/evals/local_coding_v1/MANIFEST.json

  ifeval_v1:
    type: hf_snapshot
    path: /mnt/datasets-big/evals/ifeval_v1/tasks.jsonl
    manifest: /mnt/datasets-big/evals/ifeval_v1/MANIFEST.json

  fineweb_edu_sample_10k:
    type: jsonl
    path: /mnt/datasets-big/evals/fineweb_edu_sample_10k.jsonl
    manifest: /mnt/datasets-big/evals/fineweb_edu_sample_10k/MANIFEST.json
```

Each manifest should include:

```json
{
  "name": "fineweb_edu_sample_10k",
  "source": "HuggingFaceFW/fineweb-edu",
  "split": "train",
  "streaming": true,
  "sample_count": 10000,
  "sample_seed": 1234,
  "created_at": "2026-05-04",
  "format": "jsonl",
  "checksum": "..."
}
```

Benchmark runs store: `dataset_id`, `dataset_manifest_hash`, `task_id`, `task_version`. This makes runs reproducible.

---

## 8. Environment Variables

```bash
# External dataset/cache drive
export DATASETS_BIG=/mnt/datasets-big
export HF_HOME=/mnt/datasets-big/hf-cache/huggingface
export HF_DATASETS_CACHE=/mnt/datasets-big/hf-cache/datasets
export HF_HUB_CACHE=/mnt/datasets-big/hf-cache/hub
```

Add to `~/.bashrc` and run `source ~/.bashrc`. Verify with:

```bash
echo $DATASETS_BIG
echo $HF_HOME
echo $HF_DATASETS_CACHE
echo $HF_HUB_CACHE
```

### Symlink convenience (optional)

```bash
mkdir -p ~/datasets
ln -sfn /mnt/datasets-big/evals ~/datasets/evals
ln -sfn /mnt/datasets-big/training ~/datasets/training
ln -sfn /mnt/datasets-big/hf-cache ~/datasets/cache
ln -sfn /mnt/datasets-big/raw ~/datasets/raw
ln -sfn /mnt/datasets-big/tokenized ~/datasets/tokenized
```

---

## 9. Docker Compose Patterns

### Data-prep / benchmark containers

```yaml
services:
  benchmark-harness:
    image: local/benchmark-harness:latest
    container_name: benchmark-harness
    volumes:
      - /path/to/benchmark-harness:/workspace
      - /mnt/datasets-big:/datasets-big
      - /mnt/datasets-big/hf-cache/huggingface:/root/.cache/huggingface
    environment:
      DATASETS_BIG: /datasets-big
      HF_HOME: /datasets-big/hf-cache/huggingface
      HF_DATASETS_CACHE: /datasets-big/hf-cache/datasets
      HF_HUB_CACHE: /datasets-big/hf-cache/hub
    working_dir: /workspace
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

Point at LiteLLM from inside the container:

```text
http://host.docker.internal:4000/v1
```

or from the host directly:

```text
http://localhost:4000/v1
```

### Training containers (read-only dataset mounts)

```yaml
volumes:
  - /mnt/datasets-big/evals:/datasets/evals:ro
  - /mnt/datasets-big/training:/datasets/training:ro
  - /mnt/datasets-big/hf-cache:/datasets/cache
```

### HF cache with internal path mapping

For containers that expect Hugging Face cache at `/root/.cache/huggingface`:

```yaml
volumes:
  - /mnt/datasets-big/hf-cache/huggingface:/root/.cache/huggingface
```

---

## 10. Model Weight Placement Policy

### Internal NVMe (hot models)

Keep frequently used or startup-sensitive models on internal NVMe:

- `agent-code`, `qwen-dense`, current default coding model
- Frequently used max-brain if startup time matters

Benefits: faster cold load, less USB/mount risk, better reliability for always-on services, simpler Docker startup.

### External drive (cold/experimental models)

Use for: inactive snapshots, experimental checkpoints, rarely loaded variants, large archives, backup copies of HF snapshots.

### Max-brain / Qwen3.5 122B guidance

```text
If maxbrain is frequent/default:     keep it internal.
If maxbrain is occasional + internal space tight: external is acceptable.
```

---

## 11. Safety & Operational Notes

**Never run `mkfs` against:**

```text
/dev/nvme0n1
/dev/nvme0n1p1
/dev/nvme0n1p2
```

Always inspect before formatting:

```bash
lsblk -o NAME,SIZE,TYPE,FSTYPE,LABEL,MOUNTPOINTS,MODEL,SERIAL
df -h
```

**Use label-based mounting** rather than raw device names. `LABEL=datasets-big` is safer because `/dev/sda` can change depending on boot order.

---

## 12. Storage Hygiene

### Git tracking

**Commit:**

```text
dataset registry YAML
manifests
small smoke tasks
scripts
checksums
```

**Do NOT commit:**

```text
large JSONL
Parquet shards
HF cache
tokenized cache
benchmark run artifacts
```

### Retention policy

- Keep active runs on internal NVMe while in use.
- Archive completed runs to `/mnt/datasets-big/benchmark-runs-archive`.
- Compress old raw responses if space grows too large.
- Keep manifests/checksums even if deleting bulky raw artifacts.
- Do not let HF cache silently consume the root filesystem.

### Future CLI: `storage report`

```bash
python -m bench_harness storage report
```

Should report: dataset sizes, HF cache size, tokenized cache size, benchmark run artifact size, largest runs/datasets, available space on internal and external drives.

---

## 13. Performance Expectations

External USB4 NVMe practical throughput is lower than internal NVMe but fast enough for dataset/cache use:

```text
benchmark dataset reads:     excellent
Parquet/JSONL shard reads:   excellent
HF dataset cache:            excellent
model cold load:             slower than internal, acceptable for occasional models
warm model inference:        no meaningful difference once loaded
heavy repeated tokenization: acceptable on TLC drives; avoid QLC if possible
```

---

## 14. Summary: Storage Recommendation

**External drive (`/mnt/datasets-big/`) — primary use:**

```text
large dataset store
HF dataset/model cache overflow
training shard store
tokenized dataset cache
benchmark archive
inactive model archive
```

**Internal NVMe — primary use:**

```text
hot model serving storage
active coding/model repos
active benchmark runs
latency-sensitive model weights
Docker runtime and current compose stacks
```

**Rule of thumb:**

```text
Benchmarking:        never depend on live streaming during measured runs
Fine-tuning:         train from local shards, preferably pre-tokenized
Huge corpus explore: use HF streaming, then materialize only the slice you want
Repeated experiments: pin everything with manifests and checksums
```

---

## 15. Next Steps

### Step 1

```bash
# External drive is already structured at /mnt/datasets-big/
# Optional symlink convenience:
mkdir -p ~/datasets
ln -sfn /mnt/datasets-big/evals ~/datasets/evals
ln -sfn /mnt/datasets-big/training ~/datasets/training
ln -sfn /mnt/datasets-big/hf-cache ~/datasets/cache
ln -sfn /mnt/datasets-big/raw ~/datasets/raw
ln -sfn /mnt/datasets-big/tokenized ~/datasets/tokenized
```

Add HF env vars to `~/.bashrc`:

```bash
export HF_HOME=/mnt/datasets-big/hf-cache/huggingface
export HF_DATASETS_CACHE=/mnt/datasets-big/hf-cache/datasets
export HF_HUB_CACHE=/mnt/datasets-big/hf-cache/hub
```

### Step 2

Build a small local eval dataset:

```text
/mnt/datasets-big/evals/coding_smoke_v1/tasks.jsonl
/mnt/datasets-big/evals/coding_smoke_v1/MANIFEST.json
```

### Step 3

Add `configs/datasets.yaml` to the harness.

### Step 4

Write `prepare_dataset.py` that can:

```text
download/stream → sample → write JSONL/Parquet → write manifest → compute checksum
```

### Step 5

Make benchmark runs use only registered local datasets.
