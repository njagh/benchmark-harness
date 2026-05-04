For the DGX Spark, I’d use **three different access patterns**, depending on whether you are doing quick evals, repeated benchmarks, or training.

## Best default strategy

Use this hierarchy:

```text
1. Small benchmark data
   → vendor into the repo or cache locally as JSONL/Parquet

2. Medium repeated benchmark data
   → download once to a user-owned local dataset cache

3. Huge training/pretraining data
   → stream or pre-shard, then cache only the subset you actually use
```

The main thing I would avoid is repeatedly pulling huge datasets from Hugging Face during every run. That makes benchmark results noisy, creates network dependency, and can pollute your timing numbers.

---

# 1. For benchmark runs: local, pinned, reproducible subsets

For quality benchmarking, you usually do **not** need the whole dataset. You need a fixed, representative, reproducible slice.

For example:

```text
MMLU-Pro       → fixed 500–2,000 question subset
GPQA           → full or fixed subset
IFEval         → full, small enough
HumanEval      → full
MBPP           → full or subset
LiveCodeBench  → pinned release/date range
Local tasks    → full curated set
```

Recommended layout:

```text
/mnt/datasets-big/
  evals/
    mmlu_pro/
      mmlu_pro_2026-05-04_subset.jsonl
      MANIFEST.json
    gpqa/
    ifeval/
    humaneval/
    mbpp/
    local_coding/
  training/
    fineweb_edu/
    openwebmath/
    stack_v2/
  cache/
    hf/
    tokenized/
```

For evals, pin:

```text
dataset name
source
revision / commit hash if available
download date
split
sample seed
row ids
license notes
checksum
```

That lets you say:

```text
agent-code scored 72.4 on local_coding_v1
```

and actually know what that means six weeks later.

---

# 2. For large Hugging Face datasets: use streaming for exploration, local cache for real runs

Hugging Face `datasets` supports streaming so you can iterate over examples without downloading the full dataset first. That is good for exploration or one-off sampling, especially when the dataset exceeds local disk. ([Hugging Face][1])

Example:

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

But for **benchmarking**, I would not stream live from the internet during the measured run. Instead:

```text
stream once
→ sample deterministic subset
→ write local JSONL/Parquet
→ benchmark from local disk
```

Example sampler:

```python
from datasets import load_dataset
import json
from itertools import islice

OUT = "/mnt/datasets-big/evals/fineweb_edu_sample_10k.jsonl"

ds = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    split="train",
    streaming=True,
)

with open(OUT, "w", encoding="utf-8") as f:
    for row in islice(ds, 10_000):
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
```

Then your benchmark harness reads:

```text
/mnt/datasets-big/evals/fineweb_edu_sample_10k.jsonl
```

not the remote dataset.

---

# 3. Put Hugging Face cache somewhere explicit and user-owned

Given your prior Docker/HF cache permission issues, I’d avoid letting each container invent its own root-owned cache.

The external drive at `/mnt/datasets-big/hf-cache/` already has the correct subdirectories. Set:

```bash
export HF_HOME=/mnt/datasets-big/hf-cache/huggingface
export HF_DATASETS_CACHE=/mnt/datasets-big/hf-cache/datasets
export HF_HUB_CACHE=/mnt/datasets-big/hf-cache/hub

For Docker Compose, mount it read/write for data-prep containers:

```yaml
volumes:
  - /mnt/datasets-big:/datasets
  - /mnt/datasets-big/hf-cache/huggingface:/root/.cache/huggingface

environment:
  HF_HOME: /root/.cache/huggingface
  HF_DATASETS_CACHE: /datasets/cache/hf_datasets
  HF_HUB_CACHE: /datasets/cache/hf_hub
```

For training containers, I’d often mount prepared eval datasets read-only:

```yaml
volumes:
  - /mnt/datasets-big/evals:/datasets/evals:ro
  - /mnt/datasets-big/training:/datasets/training:ro
  - /mnt/datasets-big/hf-cache:/datasets/cache
```

---

# 4. Use Parquet/Arrow/JSONL for evals; use shards for training

For benchmark evals, keep it simple:

```text
JSONL    → easiest to inspect, diff, and version
Parquet  → better for larger structured data
Arrow    → good with Hugging Face datasets
SQLite   → good for task registry / metadata, not huge text corpora
```

For training, avoid millions of tiny files. Use sharded formats:

```text
Parquet shards
JSONL.gz shards
WebDataset tar shards
HF Arrow cache
```

WebDataset is specifically built around TAR shards; large datasets are split into many shard files, often around ~1GB each, and streamed sequentially. ([GitHub][3]) PyTorch’s own blog also describes WebDataset as a solution for large datasets and many-file I/O problems in PyTorch. ([PyTorch][4])

Good training layout:

```text
/mnt/datasets-big/training/
  code_sft_v1/
    shard-00000.jsonl.gz
    shard-00001.jsonl.gz
    manifest.json
  openwebmath_sample_v1/
    shard-00000.parquet
    shard-00001.parquet
    manifest.json
  local_agent_traces_v1/
    train.jsonl
    validation.jsonl
    manifest.json
```

---

# 5. Separate “raw”, “clean”, “tokenized”, and “packed”

For training, especially if you fine-tune, use a pipeline like:

```text
raw downloaded data
→ cleaned / filtered text
→ formatted training examples
→ tokenized examples
→ packed sequences
```

Suggested layout:

```text
~/datasets/pipeline/   (local SSD only — intermediate processing, not on external drive)
  raw/
  clean/
  formatted/
  tokenized/           → final output goes to /mnt/datasets-big/tokenized/
  packed/
  manifests/
```

Do **not** tokenize repeatedly during every training run. Tokenization is CPU-heavy and can create noise. Tokenize once per tokenizer/model family and reuse.

Example:

```text
/mnt/datasets-big/tokenized/
  qwen3_tokenizer/
    local_agent_sft_v1/
    openwebmath_sample_v1/
```

For Qwen-family experiments, tokenized caches are especially useful because you may compare several Qwen variants with the same tokenizer or closely related tokenizers.

---

# 6. For your benchmark harness, add a dataset registry

Your harness should not hardcode dataset paths. Add something like:

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
    manifest: /mnt/datasets-big/evals/fineweb_edu_sample_10k.MANIFEST.json
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

Then benchmark runs store:

```text
dataset_id
dataset_manifest_hash
task_id
task_version
```

That makes runs reproducible.

---

# 7. Recommended DGX Spark access pattern by use case

## A. Public eval benchmark

Use:

```text
download once → local JSONL/Parquet → pinned manifest
```

Why:

```text
small enough
reproducible
no network variability
easy to rerun
```

Command pattern:

```bash
python scripts/prepare_eval_dataset.py \
  --dataset mmlu_pro \
  --split test \
  --sample 2000 \
  --seed 1234 \
  --out /mnt/datasets-big/evals/mmlu_pro_v1
```

Then:

```bash
python -m bench_harness run \
  --suite public_baseline \
  --dataset mmlu_pro_v1 \
  --models agent-code,qwen-dense,max-brain
```

## B. Long-context benchmark

Use:

```text
local source docs/logs/code
→ generated context packs
→ cached prompt files
```

Layout:

```text
/mnt/datasets-big/evals/long_context_v1/
  contexts/
    qwen3_replicate_032k.txt
    qwen3_replicate_064k.txt
    qwen3_replicate_128k.txt
  tasks.jsonl
  manifest.json
```

Why:

```text
context packing should be deterministic
token counts should be known before the run
you do not want to rebuild 128k prompts every benchmark
```

## C. SFT / LoRA training

Use:

```text
local sharded JSONL/Parquet
→ pre-tokenized cache
→ packed sequences
```

Recommended:

```text
Keep raw text and formatted chat examples.
Also keep tokenized/packed cache.
Regenerate tokenized cache only when tokenizer, max length, or formatting changes.
```

## D. Continued pretraining

Use:

```text
stream massive dataset
→ filter/sample/shard locally
→ train from local shards
```

Do not attempt to mirror all of FineWeb/OpenWebMath/The Stack unless you truly need it. Pull a high-quality slice.

## E. One-off exploration

Use:

```text
HF streaming
```

This is the right place for streaming.

---

# 8. Practical Docker Compose pattern

For a benchmark/data-prep container:

```yaml
services:
  benchmark-harness:
    image: local/benchmark-harness:latest
    container_name: benchmark-harness
    volumes:
      - /home/njalbicelli/research/benchmark-harness:/workspace
      - /mnt/datasets-big:/datasets
      - /mnt/datasets-big/hf-cache/huggingface:/root/.cache/huggingface
    environment:
      HF_HOME: /root/.cache/huggingface
      HF_DATASETS_CACHE: /datasets/cache/hf_datasets
      HF_HUB_CACHE: /datasets/cache/hf_hub
    working_dir: /workspace
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

Then point at LiteLLM:

```text
http://host.docker.internal:4000/v1
```

or from the host directly:

```text
http://spark-e287.local:4000/v1
```

---

# 9. Storage recommendation

I’d use something like this:

```text
/mnt/datasets-big
  /hf-cache
    /huggingface
    /datasets
    /hub
  /tokenized
  /evals
    /smoke_v1
    /coding_smoke_v1
    /local_coding_v1
    /long_context_v1
    /public_baseline_v1
  /training
    /local_agent_sft_v1
    /local_agent_dpo_v1
    /openwebmath_sample_v1
    /fineweb_edu_sample_v1
    /stack_v2_code_sample_v1
  /raw
    /hf_downloads
  /models-archive
  /benchmark-runs-archive
```

And in git, commit only:

```text
dataset registry YAML
manifests
small smoke tasks
scripts
checksums
```

Do **not** commit:

```text
large JSONL
Parquet shards
HF cache
tokenized cache
benchmark run artifacts
```

---

# 10. My concrete recommendation for your next step

For your benchmark harness project, implement dataset access in this order:

## Step 1

Create:

```bash
# External drive is already structured at /mnt/datasets-big/
# Local symlink or mount for convenience:
mkdir -p ~/datasets
ln -sfn /mnt/datasets-big/evals ~/datasets/evals
ln -sfn /mnt/datasets-big/training ~/datasets/training
ln -sfn /mnt/datasets-big/hf-cache ~/datasets/cache
ln -sfn /mnt/datasets-big/raw ~/datasets/raw
ln -sfn /mnt/datasets-big/tokenized ~/datasets/tokenized
```

Add to shell profile:

```bash
export HF_HOME=/mnt/datasets-big/hf-cache/huggingface
export HF_DATASETS_CACHE=/mnt/datasets-big/hf-cache/datasets
export HF_HUB_CACHE=/mnt/datasets-big/hf-cache/hub
```

## Step 2

Build a small local eval dataset:

```text
/mnt/datasets-big/evals/coding_smoke_v1/tasks.jsonl
/mnt/datasets-big/evals/coding_smoke_v1/MANIFEST.json
```

## Step 3

Add `configs/datasets.yaml` to the harness.

## Step 4

Write `prepare_dataset.py` that can:

```text
download/stream
sample
write JSONL/Parquet
write manifest
compute checksum
```

## Step 5

Make benchmark runs use only registered local datasets.

---

## Rule of thumb

For the Spark:

```text
Benchmarking:
  never depend on live streaming during measured runs

Fine-tuning:
  train from local shards, preferably pre-tokenized

Huge corpus exploration:
  use HF streaming, then materialize only the slice you want

Repeated experiments:
  pin everything with manifests and checksums
```

That gives you clean timing numbers, reproducible quality comparisons, and a sane path from eval failures into training data.

[1]: https://huggingface.co/docs/datasets/stream?utm_source=chatgpt.com "Stream"
[2]: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables?utm_source=chatgpt.com "Environment variables"
[3]: https://github.com/webdataset/webdataset?utm_source=chatgpt.com "webdataset/webdataset: A high-performance Python- ..."
[4]: https://pytorch.org/blog/efficient-pytorch-io-library-for-large-datasets-many-files-many-gpus/?utm_source=chatgpt.com "Efficient PyTorch I/O library for Large Datasets, Many Files, ..."
