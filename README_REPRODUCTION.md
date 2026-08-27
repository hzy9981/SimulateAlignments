# BetaAlign Reproduction Guide

[中文版本](README_REPRODUCTION_ZH.md)

This guide records the reproducible workflow verified in this repository on Windows with WSL2 (Ubuntu). It covers local dependency installation, data generation, Fairseq preprocessing, and a CPU training smoke test.

Run the commands below from the repository root in WSL Bash. The workflow uses repository-local environments and does not require administrator privileges.

## Prerequisites

- Windows with WSL2 and Ubuntu
- WSL packages: `curl`, `bzip2`, `gcc`, and Python 3
- At least 5 GB of free disk space for local Python environments and the 1,000-sample smoke-test dataset
- Internet access to download Python packages and Miniforge

The full Transformer-Big training workflow additionally requires an NVIDIA GPU exposed to WSL. The verified smoke test uses CPU only.

## 1. Install Local Dependencies

### 1.1 Data-generation environment

```bash
python3 -m venv .venv
.venv/bin/pip install biopython
```

### 1.2 INDELible

Download and unpack the Ubuntu `indelible` package inside the repository:

```bash
cd /tmp
apt download indelible
cd -
mkdir -p .local
dpkg-deb -x /tmp/indelible_*.deb .local
```

INDELible is available at `.local/usr/bin/indelible`. Use an absolute repository path when exporting `PATH`, because the generator launches INDELible inside per-sample temporary directories:

```bash
export PATH="$PWD/.local/usr/bin:/usr/bin:/bin"
```

### 1.3 Fairseq environment

Fairseq 0.12.2 does not run correctly with Python 3.12 because of legacy Hydra and dataclass compatibility issues. The verified environment uses Python 3.10 installed locally with Miniforge.

```bash
curl -L --fail --retry 3 \
  -o /tmp/Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/Miniforge3-Linux-x86_64.sh -b -p "$PWD/.miniforge"

.miniforge/bin/mamba create -y -p "$PWD/.fairseq-py310" python=3.10 pip
.fairseq-py310/bin/pip install 'pip<24.1'
.fairseq-py310/bin/pip install torch==2.5.1 \
  --index-url https://download.pytorch.org/whl/cpu
.fairseq-py310/bin/pip install fairseq==0.12.2
```

`pip<24.1` is required because Fairseq 0.12.2 depends on OmegaConf packages whose metadata is rejected by newer pip versions.

Verify the CLI:

```bash
PATH="$PWD/.fairseq-py310/bin:/usr/bin:/bin" fairseq-preprocess --help
```

## 2. Generate Training Data

`reproduce_generate_data.py` generates unaligned source sequences and column-interleaved MSA targets. It invokes INDELible once per sample and stores temporary simulator output under `reproduce_data/tmp_<n>/`.

```bash
rm -rf reproduce_data
PATH="$PWD/.local/usr/bin:/usr/bin:/bin" \
  .venv/bin/python reproduce_generate_data.py 1000
```

### Verified result

The command completed in approximately 116 seconds and generated:

```text
reproduce_data/train.source  1,000 lines, 5,441,483 bytes
reproduce_data/train.target  1,000 lines, 6,656,260 bytes
```

Validate the sample counts:

```bash
wc -l reproduce_data/train.source reproduce_data/train.target
```

### Generator fixes included in this repository

- Writes valid INDELible `MODEL`, `submodel`, `indelmodel`, `indelrate`, and `PARTITIONS` blocks.
- Writes insertion/deletion rates as fixed-point decimals because INDELible rejects scientific notation for small values.
- Reads `output_TRUE.phy`, the true aligned PHYLIP output, for targets. `output.fas` contains unaligned sequences and cannot be column-interleaved when gaps are present.

## 3. Preprocess Data for Fairseq

Ensure the shell scripts use LF line endings when running them from WSL:

```bash
sed -i 's/\r$//' reproduce_preprocess.sh reproduce_train.sh reproduce_workflow.sh
```

Run preprocessing:

```bash
PATH="$PWD/.fairseq-py310/bin:/usr/bin:/bin" \
  bash reproduce_preprocess.sh
```

### Verified result

Fairseq created a shared dictionary with 32 tokens and wrote `reproduce_data_bin/`:

```text
dict.source.txt
dict.target.txt
train.source-target.source.bin
train.source-target.source.idx
train.source-target.target.bin
train.source-target.target.idx
```

For the 1,000 generated pairs, Fairseq reported 2,721,737 source tokens and 3,329,130 target tokens, with `0.0%` unknown-token replacement.

## 4. CPU Training Smoke Test

The supplied `reproduce_train.sh` targets Transformer-Big and uses `--fp16`; it is for GPU training and is not suitable for the CPU-only verification environment. The following command validates the full Fairseq training path using a compact model and one parameter update:

```bash
rm -rf checkpoints-smoke
PATH="$PWD/.fairseq-py310/bin:/usr/bin:/bin" fairseq-train reproduce_data_bin \
  --save-dir checkpoints-smoke \
  --arch transformer \
  --encoder-layers 1 --decoder-layers 1 \
  --encoder-embed-dim 64 --decoder-embed-dim 64 \
  --encoder-ffn-embed-dim 128 --decoder-ffn-embed-dim 128 \
  --encoder-attention-heads 2 --decoder-attention-heads 2 \
  --share-decoder-input-output-embed \
  --optimizer adam --lr 0.0001 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 8192 \
  --max-source-positions 8192 --max-target-positions 8192 \
  --max-update 1 --num-workers 0 --cpu \
  --no-epoch-checkpoints --valid-subset train --log-interval 1
```

### Verified result

The smoke test completed one update in approximately 388 seconds and wrote:

```text
checkpoints-smoke/checkpoint_best.pt
checkpoints-smoke/checkpoint_last.pt
```

Both checkpoint files were approximately 1.25 MB.

## 5. Full GPU Training

For the paper-scale model, move the repository to a WSL environment with an NVIDIA GPU, install the matching CUDA-enabled PyTorch build, and then run:

```bash
PATH="$PWD/.fairseq-py310/bin:$PATH" bash reproduce_train.sh
```

The 1,000-sample smoke dataset has sequences whose token counts exceed the original `--max-tokens 2048` batch budget. Use a larger budget, such as `--max-tokens 4096` or higher, or filter long samples before full training. Do not use `--fp16` unless CUDA is available.

## 6. Inference

After producing a trained checkpoint, invoke permutation-consensus inference as follows:

```bash
.fairseq-py310/bin/python reproduce_inference.py \
  <model_dir> <checkpoint_file> reproduce_data_bin <input_fasta>
```

Example for the CPU smoke checkpoint:

```bash
.fairseq-py310/bin/python reproduce_inference.py \
  checkpoints-smoke checkpoint_best.pt reproduce_data_bin input.fasta
```

The smoke checkpoint only verifies the execution path and is not a biologically meaningful trained model.

## Generated Files and Git

The following directories are local environments or generated artifacts and should normally remain untracked:

```text
.local/
.venv/
.miniforge/
.fairseq-py310/
reproduce_data/
reproduce_data_bin/
checkpoints-smoke/
```

Run `git status --short` before committing source or documentation changes to avoid committing these large local artifacts.