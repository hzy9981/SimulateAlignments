#!/bin/bash

# reproduce_preprocess.sh
# Preprocess the simulated data for Fairseq

DATA_DIR="reproduce_data"
DEST_DIR="reproduce_data_bin"

# Make sure source and target files exist
if [ ! -f "$DATA_DIR/train.source" ] || [ ! -f "$DATA_DIR/train.target" ]; then
    echo "Error: train.source or train.target not found in $DATA_DIR"
    exit 1
fi

# We use the same vocabulary for source and target since they share the same characters (nucleotides/amino acids)
# --joined-dictionary ensures this.
fairseq-preprocess \
    --source-lang source --target-lang target \
    --trainpref "$DATA_DIR/train" \
    --destdir "$DEST_DIR" \
    --workers 4 \
    --joined-dictionary
