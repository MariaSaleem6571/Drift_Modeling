#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=testing_drift_model
#SBATCH --output=testing_drift_model.out
#SBATCH --error=testing_drift_model.err

python -m scripts.eval.snapshot /mundus/folanrewa525/workspace/DriftModelling daily2018_15 pretrained --channels 5

