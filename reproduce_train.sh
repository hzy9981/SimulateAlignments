#!/bin/bash

# reproduce_train.sh
# Train the BetaAlign Transformer model

DATA_BIN="reproduce_data_bin"
SAVE_DIR="checkpoints"

mkdir -p $SAVE_DIR

fairseq-train $DATA_BIN \
    --save-dir $SAVE_DIR \
    --arch transformer_vaswani_wmt_en_de_big \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 3000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 2048 \
    --max-source-positions 2048 --max-target-positions 2048 \
    --validate-interval 1 \
    --save-interval-updates 10000 \
    --best-checkpoint-metric loss \
    --fp16 # Use half-precision if supported by GPU
