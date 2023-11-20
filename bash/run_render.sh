#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
nvidia-smi

exp_name="hypernerf"
python ../render.py --model_path ./output/$exp_name/split-cookie --configs ../arguments/$exp_name/default.py --skip_train