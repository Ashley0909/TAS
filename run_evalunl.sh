#!/bin/bash
#SBATCH -c 8
#SBATCH -p hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --job-name=test_eval
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --time=11:59:00

source /nfs-share/ahta3/workspace/LUNAR/.venv/bin/activate

export HF_TOKEN=$(cat /nfs-share/ahta3/.huggingface_token)
export NETRC=/nfs-share/ahta3/.wandb_token

cd /nfs-share/ahta3/workspace/LUNAR/
srun python eval_unlearned.py \
  --model_path /nfs-share/ahta3/workspace/LUNAR/unlearn_results/completions/lunar/gemma-7b-it/tofu/model \
  --save_file unlearn_results/completions/lunar/gemma-7b-it/tofu/forget_dpo_eval_16.json

  # reminder: change model_family in forget.yaml before evaluating