#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
nvidia-smi

exp_name="hypernerf"
python ../train.py -s /home/whwjdqls99/data/$exp_name/split-cookie --port 6017 --expname "$exp_name/split-cookie" --configs ../arguments/$exp_name/default.py 