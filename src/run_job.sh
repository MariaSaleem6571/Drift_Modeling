#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --job-name=MAEProb_Loss
#SBATCH --output=MAEProb_Loss.out
#SBATCH --error=MAEProb_Loss.err

python -m scripts.train.snapshot /mundus/folanrewa525/workspace/DriftModelling daily2016_15 MAEProb_Loss --channels 5 --loss MAEProbDistrLoss
python -m scripts.eval.snapshot /mundus/folanrewa525/workspace/DriftModelling daily2016_15 MAEProb_Loss --channels 5