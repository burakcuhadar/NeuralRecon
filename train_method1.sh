#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python main.py --cfg ./config/train.yaml --use_sparse --use_sparse_method1 --sparse_depth_rate 0.01 \
    MODEL.FUSION.FUSION_ON False MODEL.FUSION.FULL False TRAIN.EPOCHS 20
