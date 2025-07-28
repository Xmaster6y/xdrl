#!/bin/bash

#SBATCH --job-name=demo-jz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
##SBATCH -C v100-16g
#SBATCH --cpus-per-task=10
##SBATCH --hint=nomultithread
##SBATCH --partition=gpu_p2l
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=00:10:00
#SBATCH --mail-type=FAIL
#SBATCH --output=results/slurm/%x-%j.out
#SBATCH --error=results/slurm/%x-%j.err
#SBATCH --account=nwq@v100

module purge
uv run --no-sync -m scripts.demo
