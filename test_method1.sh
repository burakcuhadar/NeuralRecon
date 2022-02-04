#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python main.py --use_sparse --use_sparse_method1 --sparse_depth_rate 0.01 --cfg ./config/test.yaml MODEL.FUSION.FUSION_ON False MODEL.FUSION.FULL False TRAIN.EPOCHS 10
