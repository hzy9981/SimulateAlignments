# BetaAlign Reproduction & Codebase Improvements

This document summarizes the technical modifications made to the repository to enable full reproduction of the BetaAlign methodology as described in "Dotan et al. (2024/2025)".

---

## 1. Codebase Bug Fixes & Portability Improvements

The following fixes were applied to ensure the pipeline runs correctly in new environments:

- **Path Portability**: Replaced all hardcoded absolute paths (e.g., `/home/elyawy/`, `/groups/pupko/`) with dynamic paths using `os.getcwd()`. Affected files:
    - `configuration.py`
    - `pipeline_click.py`
    - `run_sparta_abc_single_folder_pipeline.py`
    - `msa_to_summary_statistics.py`
    - `for_edo_amino_10MSA.py`
    - `for_edo_nuc_10MSA.py`
- **Missing Dependencies**: Added missing `Input`, `Dense`, and `Model` imports from `tensorflow.keras` in `infer_abc_params_single_folder_pipeline.py`.
- **Logic Correction**: Fixed a premature `return` statement in `msa_bias_corrector.py` that caused the bias correction process to stall after the first iteration.
- **Binary Permissions**: Ensured `SpartaABC` executable has execution permissions.

---

## 2. Paper Reproduction Workflow

New scripts were added to implement the methodology from the BetaAlign paper.

### Phase 1: Data Generation
Run `reproduce_generate_data.py` to simulate training samples.
- **Encoding**: Implements the column-interleaved target encoding.
- **Indel Model**: Uses the Power-law (POW) model (shape 1.7, max 500) as specified in the paper.
- **Command**: `python reproduce_generate_data.py <num_samples>`

### Phase 2: Training (Fairseq)
1.  **Preprocessing**: Use `reproduce_preprocess.sh` to binarize data.
2.  **Training**: Use `reproduce_train.sh` to train the `transformer_vaswani_wmt_en_de_big` model.
    - Includes Adam optimizer, LR 5e-5, and specific warmup/dropout settings from the paper.

### Phase 3: Inference with Permutation Consensus
The "new technique" mentioned in the paper to solve positional bias is implemented in `reproduce_inference.py`.
- **Method**: It generates multiple permutations of the input sequences, predicts alignments for each, and reorders them back to a consistent consensus.
- **Command**: `python reproduce_inference.py <model_dir> <checkpoint> <data_bin> <input_fasta>`

---

## 3. Quick Start
To see the full list of commands and the order of execution, refer to:
```bash
bash reproduce_workflow.sh
```

---
**Author Note**: The codebase is now verified to interface correctly with the `SpartaABC` tool and follows the exact specifications of the 2024/2025 BetaAlign publication.
