#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --job-name=MAEProb_Training
#SBATCH --output=MAEProb_Training.out
#SBATCH --error=MAEProb_Training.err

python -m scripts.eval.snapshot /mundus/folanrewa525/workspace/DriftModelling daily2018_15 MSEProbDistrLoss_0.2_residual --channels 5 --saving_name daily2018_15_test_1_day