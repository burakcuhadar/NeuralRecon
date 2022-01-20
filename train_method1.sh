#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python main.py --cfg ./config/train.yaml --use_sparse --sparse_depth_rate 0.1 \
    MODEL.FUSION.FUSION_ON False MODEL.FUSION.FULL False TRAIN.EPOCHS 20
