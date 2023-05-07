#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --partition=mundus
#SBATCH --gres=gpu:2
#SBATCH --job-name=MAEProb_0.2_Training
#SBATCH --output=MAEProb_0.2_Training.out
#SBATCH --error=MAEProb_0.2_Training.err

python -m scripts.train.snapshot /mundus/isarac488/workspace/DriftModelling daily2018_15 MAEProb_Loss --channels 5 --loss MSEProbDistrLoss --lossalpha 0.2 --residual
