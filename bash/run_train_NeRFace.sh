#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
nvidia-smi

exp_name="NeRFace"
python ../train.py -s /home/whwjdqls99/data/$exp_name/person_1 --port 6017 --expname "$exp_name/person_1" --configs ../arguments/$exp_name/default.py 