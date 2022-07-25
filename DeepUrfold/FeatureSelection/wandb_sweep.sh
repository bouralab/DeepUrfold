#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --account=muragroup
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=28
#SBATCH --mem="36G"
#SBATCH --time=3-00:00:00
#SBATCH --array=1-20

wandb agent -e edraizen -p UrfoldSweeper xag2hmw5
