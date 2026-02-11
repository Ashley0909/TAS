#!/bin/bash
#SBATCH -c 8
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --job-name=test_unl
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --time=11:59:00

source /nfs-share/ahta3/workspace/LUNAR/.venv/bin/activate

export HF_TOKEN=$(cat /nfs-share/ahta3/.huggingface_token)
export NETRC=/nfs-share/ahta3/.wandb_token

cd /nfs-share/ahta3/workspace/LUNAR/
srun python run_lunar.py