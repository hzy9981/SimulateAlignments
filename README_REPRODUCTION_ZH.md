# BetaAlign 复现指南

[English version](README_REPRODUCTION.md)

本文记录了在 Windows + WSL2（Ubuntu）环境中实际验证过的 BetaAlign 复现流程，包括本地依赖安装、训练数据生成、Fairseq 预处理和 CPU 训练冒烟测试。

以下命令均应在仓库根目录的 WSL Bash 中执行。所有环境均安装在仓库目录内，无需管理员权限。

## 前置条件

- Windows、WSL2 和 Ubuntu 发行版
- WSL 工具：`curl`、`bzip2`、`gcc`、Python 3
- 至少 5 GB 可用磁盘空间，用于本地 Python 环境和 1,000 条样本的冒烟数据集
- 可访问网络，以下载 Python 包和 Miniforge

论文规模的 Transformer-Big 训练还需要 WSL 能访问 NVIDIA GPU。本指南中验证的冒烟测试仅使用 CPU。

## 1. 安装本地依赖

### 1.1 数据生成环境

```bash
python3 -m venv .venv
.venv/bin/pip install biopython
```

### 1.2 安装 INDELible

下载 Ubuntu 的 `indelible` 软件包，并解压到仓库目录内：

```bash
cd /tmp
apt download indelible
cd -
mkdir -p .local
dpkg-deb -x /tmp/indelible_*.deb .local
```

INDELible 的路径为 `.local/usr/bin/indelible`。数据生成器会在每个样本的临时目录中启动 INDELible，因此设置 `PATH` 时应使用仓库的绝对路径：

```bash
export PATH="$PWD/.local/usr/bin:/usr/bin:/bin"
```

### 1.3 Fairseq 环境

Fairseq 0.12.2 与 Python 3.12 的旧版 Hydra/dataclass 组合不兼容。已验证的方案是用 Miniforge 在仓库内安装 Python 3.10：

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

必须使用 `pip<24.1`，因为 Fairseq 0.12.2 所依赖的 OmegaConf 包元数据会被新版 pip 拒绝。

验证 Fairseq 命令：

```bash
PATH="$PWD/.fairseq-py310/bin:/usr/bin:/bin" fairseq-preprocess --help
```

## 2. 生成训练数据

`reproduce_generate_data.py` 生成未比对的 source 序列和按 MSA 列交错编码的 target。每条样本都会调用一次 INDELible，并将模拟器的临时输出保留在 `reproduce_data/tmp_<n>/`。

```bash
rm -rf reproduce_data
PATH="$PWD/.local/usr/bin:/usr/bin:/bin" \
  .venv/bin/python reproduce_generate_data.py 1000
```

### 已验证结果

该命令约耗时 116 秒，生成：

```text
reproduce_data/train.source  1,000 行，5,441,483 字节
reproduce_data/train.target  1,000 行，6,656,260 字节
```

验证 source/target 样本数一致：

```bash
wc -l reproduce_data/train.source reproduce_data/train.target
```

### 已包含的数据生成修复

- 按有效语法写入 INDELible 的 `MODEL`、`submodel`、`indelmodel`、`indelrate` 和 `PARTITIONS` 块。
- 将插入/缺失率固定为十进制格式。INDELible 不接受极小数值的科学计数法。
- 使用真实的已比对 PHYLIP 输出 `output_TRUE.phy` 生成 target。`output.fas` 是未比对序列，在存在 gap 时不能用于按列编码。

## 3. 用 Fairseq 预处理数据

在 WSL 中运行脚本前，确保 Shell 脚本为 LF 行尾：

```bash
sed -i 's/\r$//' reproduce_preprocess.sh reproduce_train.sh reproduce_workflow.sh
```

执行预处理：

```bash
PATH="$PWD/.fairseq-py310/bin:/usr/bin:/bin" \
  bash reproduce_preprocess.sh
```

### 已验证结果

Fairseq 创建了包含 32 个 token 的共享词典，并写入 `reproduce_data_bin/`：

```text
dict.source.txt
dict.target.txt
train.source-target.source.bin
train.source-target.source.idx
train.source-target.target.bin
train.source-target.target.idx
```

对 1,000 对生成样本，Fairseq 统计到 2,721,737 个 source token 和 3,329,130 个 target token，未知 token 替换率为 `0.0%`。

## 4. CPU 训练冒烟测试

仓库提供的 `reproduce_train.sh` 使用 Transformer-Big 和 `--fp16`，面向 GPU 训练，不适用于当前仅 CPU 的验证环境。以下命令使用小型模型执行一次参数更新，验证完整 Fairseq 训练链路：

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

### 已验证结果

冒烟测试完成一次更新，耗时约 388 秒，生成：

```text
checkpoints-smoke/checkpoint_best.pt
checkpoints-smoke/checkpoint_last.pt
```

两个 checkpoint 文件均约 1.25 MB。

## 5. 完整 GPU 训练

要进行论文规模训练，应将仓库迁移到 WSL 可访问 NVIDIA GPU 的环境，安装匹配的 CUDA PyTorch，然后运行：

```bash
PATH="$PWD/.fairseq-py310/bin:$PATH" bash reproduce_train.sh
```

当前 1,000 条冒烟数据包含 token 数超过原脚本 `--max-tokens 2048` 批预算的样本。完整训练前可将该参数提高到 `4096` 或更高，或过滤过长样本。仅在 CUDA 可用时使用 `--fp16`。

## 6. 推理

生成训练 checkpoint 后，可按以下方式运行排列共识推理：

```bash
.fairseq-py310/bin/python reproduce_inference.py \
  <model_dir> <checkpoint_file> reproduce_data_bin <input_fasta>
```

CPU 冒烟 checkpoint 的示例：

```bash
.fairseq-py310/bin/python reproduce_inference.py \
  checkpoints-smoke checkpoint_best.pt reproduce_data_bin input.fasta
```

该冒烟 checkpoint 仅证明推理链路可以运行，不具备生物学上有意义的训练质量。

## 7. 生成文件与 Git

下列目录是本地环境或运行产物，通常不应提交：

```text
.local/
.venv/
.miniforge/
.fairseq-py310/
reproduce_data/
reproduce_data_bin/
checkpoints-smoke/
```

提交源码或文档前，请运行 `git status --short`，避免将这些大体积本地文件加入 Git。