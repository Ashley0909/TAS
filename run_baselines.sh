#!/bin/bash
#SBATCH -c 8
#SBATCH -p hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --job-name=test_baselines
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --time=11:59:00

source /nfs-share/ahta3/workspace/LUNAR/.venv/bin/activate

export HF_TOKEN=$(cat /nfs-share/ahta3/.huggingface_token)
export NETRC=/nfs-share/ahta3/.wandb_token

cd /nfs-share/ahta3/workspace/LUNAR/
srun python run_baselines.py method=rmu save_folder=rmu


# python run_baselines.py method=ga  save_folder=ga
# python run_baselines.py method=gd  save_folder=gd
# python run_baselines.py method=ukl save_folder=ukl
# python run_baselines.py method=dpo save_folder=dpo
# python run_baselines.py method=npo save_folder=npo
# python run_baselines.py method=rmu save_folder=rmu