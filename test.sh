#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
python main.py --cfg ./config/test.yaml MODEL.FUSION.FUSION_ON False MODEL.FUSION.FULL False TRAIN.EPOCHS 20
