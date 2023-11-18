#!/bin/bash
echo $CUDA_VISIBLE_DEVICES
echo $SLURM_NODELIST
echo $SLURM_NODEID
# sbatch --qos=cpu_qos --partition=cpu --job-name=test .sh
ml purge
# wget https://www.robots.ox.ac.uk/~wenjing/Tanks.zip
python train.py -s /home/qkrwlgh0314/datasets/mono-video --port 6017 --expname "imavatar" --configs arguments/hypernerf/default.py 
# python train.py -s /home/qkrwlgh0314/datasets/mono-video --ip=127.0.0.2 
# sbatch --qos=base_qos --partition=base --gres=gpu:1 --job-name=vanila train_vanila.sh

# python train.py -s /home/qkrwlgh0314/datasets/tanks_temple/tandt/train --ip=127.0.0.2 