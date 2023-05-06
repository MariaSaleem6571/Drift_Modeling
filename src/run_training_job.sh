#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --partition=mundus
#SBATCH --gres=gpu:8
#SBATCH --job-name=MAEProb_Training
#SBATCH --output=MAEProb_Training.out
#SBATCH --error=MAEProb_Training.err

conda activate drift
python -m scripts.train.snapshot /mundus/folanrewa525/workspace/DriftModelling daily2016_15 MAEProb_Loss --channels 5