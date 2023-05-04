#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --job-name=MAEProb_Loss
#SBATCH --output=MAEProb_Loss.out
#SBATCH --error=MAEProb_Loss.err

python -m scripts.eval.snapshot /mundus/aaslam308/DriftModelling daily2016_15 pretrained --channels 5