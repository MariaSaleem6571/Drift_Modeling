#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --job-name=MAEProb_Training
#SBATCH --output=MAEProb_Training.out
#SBATCH --error=MAEProb_Training.err

~/miniconda3/envs/drift/bin/python -m scripts.eval.snapshot /mundus/folanrewa525/workspace/DriftModelling daily2018_15 MSEProbDistrLoss_0.2_residual --channels 5 --args.no_of_days 1