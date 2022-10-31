#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export CUDA_VISIBLE_DEVICES=0,2,3
DATAPATH="/data2/zrx/Multimodal_Retrieval"

python -u src/training/main.py \
    --save-frequency 1 \
    --train-data="${DATAPATH}/MR_train_queries.jsonl"  \
    --train-img="${DATAPATH}/MR_train_imgs.224.npz"  \
    --val-data="${DATAPATH}/MR_valid_queries.jsonl"  \
    --val-img="${DATAPATH}/MR_valid_imgs.224.npz"  \
    --clip-weight-path="${DATAPATH}/ViT-B-16.state_dict.pt" \
    --bert-weight-path="${DATAPATH}/pytorch_model.bin" \
    --warmup 500 \
    --batch-size=64 \
    --lr=8e-5 \
    --wd=0.001 \
    --epochs=10 \
    --model ViT-B-16