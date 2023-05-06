#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --job-name=MAEProb_Training
#SBATCH --output=MAEProb_Training.out
#SBATCH --error=MAEProb_Training.err

<<<<<<< HEAD
python -m scripts.train.snapshot /mundus/folanrewa525/workspace/DriftModelling daily2018_15 MAEProb_Loss --channels 5 --loss MSEProbDistrLoss --lossalpha 0.2
=======
python -m scripts.train.snapshot /mundus/folanrewa525/workspace/DriftModelling daily2018_15 MAEProb_Loss --channels 5 --loss MAE --lossalpha 0.4
>>>>>>> main
