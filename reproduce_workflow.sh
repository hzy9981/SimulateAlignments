#!/bin/bash

# reproduce_workflow.sh
# A summary script showing the sequence of commands for reproduction.

echo "--- Step 1: Generate Training Data ---"
echo "Command: python reproduce_generate_data.py 1000000"
echo "Note: This simulates 1M samples using INDELible (requires 'indelible' in PATH)."
echo ""

echo "--- Step 2: Preprocess Data for Fairseq ---"
echo "Command: bash reproduce_preprocess.sh"
echo "Note: Binarizes the data for efficient training."
echo ""

echo "--- Step 3: Train the Model ---"
echo "Command: bash reproduce_train.sh"
echo "Note: Trains the Transformer-Big model. Requires GPU."
echo ""

echo "--- Step 4: Inference with Consensus ---"
echo "Command: python reproduce_inference.py checkpoints/ checkpoint_best.pt reproduce_data_bin/ input.fasta"
echo "Note: Aligns new sequences using the permutation consensus technique."
echo ""

echo "--- Step 5: (Optional) ABC Summary Stat Correction ---"
echo "Note: Use 'msa_bias_corrector.py' to refine evolutionary parameter inference."
