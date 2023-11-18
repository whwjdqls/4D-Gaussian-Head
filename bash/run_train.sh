#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
nvidia-smi
python ../train.py -s /home/whwjdqls99/data/hypernerf/aleks-teapot --port 6017 --expname "hypernerf/teapot" --configs ../arguments/hypernerf/default.py 