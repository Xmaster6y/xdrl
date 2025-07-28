#!/bin/bash

#SBATCH --job-name=demo-cv
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=a100_7g.80gb:1
#SBATCH --cpus-per-task=7
##SBATCH --hint=nomultithread
#SBATCH --mem=100G
#SBATCH --time=00:10:00
#SBATCH --mail-type=FAIL
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err

module purge
uv run --no-sync -m scripts.demo
