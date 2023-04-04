#!/bin/bash

#SBATCH --time=00:30:00
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_all_models_04_04
#SBATCH --output=run_all_models_04_04.out
#SBATCH --error=run_all_models_04_04.err

python -m scripts.eval.snapshot /mundus/folanrewa525/workspace/DriftModelling daily2016_15 pretrained --channels 5

