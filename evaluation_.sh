#!/usr/bin/env bash
PYOPENGL_PLATFORM=osmesa python tools/evaluation.py \
	--model ./results/scene_scannet_checkpoints_fusion_eval_19 \
	--n_proc 2 --n_gpu 1 --num_workers=1 --loader_num_workers=2 \
	--data_path=/mnt/raid/dareth/ScanNet/scannet/scans_test \
	--gt_path=/mnt/ScanNet/public/v2/scans
