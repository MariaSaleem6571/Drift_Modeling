#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --partition=mundus
#SBATCH --job-name=DayShift2_pretrained
#SBATCH --output=DayShift2_pretrained.out
#SBATCH --error=DayShift2_pretrained.err

python -m scripts.eval.snapshot /mundus/aaslam308/DriftModelling daily2016_15 pretrained --channels 5