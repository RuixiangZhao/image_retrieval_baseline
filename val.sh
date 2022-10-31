#!/bin/bash
# only supports single-GPU inference
export PYTHONPATH="$PYTHONPATH:$PWD/src"
export CUDA_VISIBLE_DEVICES=0
epoch_num=9
DATAPATH="/data2/zrx/Multimodal_Retrieval"
experiment_name="lr=8e-05_wd=0.001_agg=True_model=ViT-B-16_batchsize=64_date=2022-10-31-06-28-47"

# python -u src/eval/extract_features.py \
#     --extract-image-feats \
#     --extract-text-feats \
#     --image-data="${DATAPATH}/MR_valid_imgs.224.npz" \
#     --text-data="${DATAPATH}/MR_valid_queries.jsonl" \
#     --img-batch-size=32 \
#     --text-batch-size=32 \
#     --resume="logs/${experiment_name}/checkpoints/epoch_${epoch_num}.pt" \
#     --model ViT-B-16

# python -u src/eval/make_topk_predictions.py \
#     --image-feats="${DATAPATH}/MR_valid_imgs.224.img_feat.epoch${epoch_num}.jsonl" \
#     --text-feats="${DATAPATH}/MR_valid_queries.txt_feat.epoch${epoch_num}.jsonl" \
#     --top-k=10 \
#     --eval-batch-size=32768 \
#     --output="${DATAPATH}/MR_valid_predictions.epoch${epoch_num}.jsonl"

python src/eval/evaluation.py \
    ${DATAPATH}/MR_valid_queries.jsonl \
    ${DATAPATH}/MR_valid_predictions.epoch${epoch_num}.jsonl \
    ${DATAPATH}/MR_valid_output.epoch${epoch_num}.jsonl