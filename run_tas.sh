#!/bin/bash
#SBATCH -c 8
#SBATCH -p hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --job-name=test_tas
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --time=11:59:00
#SBATCH --array=0-10

source /nfs-share/ahta3/workspace/LUNAR/.venv/bin/activate

export HF_TOKEN=$(cat /nfs-share/ahta3/.huggingface_token)
export NETRC=/nfs-share/ahta3/.wandb_token

cd /nfs-share/ahta3/workspace/LUNAR/

# One entry per array task — each is a space-separated list of key=value overrides.
RUNS=(
  "output_dir=debug_search/random_search/LUNAR/llama3-8b-instruct/1000_iteration_#1 unlearned_model.model_family=llama3-8b-instruct unlearned_model.model_path=unlearn_results/completions/lunar/llama3-8b-instruct/pistol_sample1/model search_mode=random"
  "output_dir=debug_search/brute_force_search/DPO/llama3-8b-instruct unlearned_model.model_family=llama3-8b-instruct unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/llama3-8b-instruct_forget_AB/dpo_40epochs_LoRA16_lr1.25e-05 search_mode=brute"
  "output_dir=debug_search/brute_force_search/NPO/llama3-8b-instruct unlearned_model.model_family=llama3-8b-instruct unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/llama3-8b-instruct_forget_AB/npo_20epochs_LoRA32_lr5e-05 search_mode=brute"
  "output_dir=debug_search/brute_force_search/LUNAR/llama3-8b-instruct unlearned_model.model_family=llama3-8b-instruct unlearned_model.model_path=unlearn_results/completions/lunar/llama3-8b-instruct/pistol_sample1/model search_mode=brute"
  "output_dir=debug_search/random_search/DPO/gemma-7b-it/1000_iteration_#1 unlearned_model.model_family=gemma-7b-it unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/gemma-7b-it_forget_AB/dpo_40epochs_LoRA16_lr1.25e-05 search_mode=random"
  "output_dir=debug_search/brute_force_search/DPO/gemma-7b-it unlearned_model.model_family=gemma-7b-it unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/gemma-7b-it_forget_AB/dpo_40epochs_LoRA16_lr1.25e-05 search_mode=brute"
  "output_dir=debug_search/random_search/NPO/gemma-7b-it/1000_iteration_#1 unlearned_model.model_family=gemma-7b-it unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/gemma-7b-it_forget_AB/npo_20epochs_LoRA32_lr5e-05 search_mode=random"
  "output_dir=debug_search/brute_force_search/NPO/gemma-7b-it unlearned_model.model_family=gemma-7b-it unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/gemma-7b-it_forget_AB/npo_20epochs_LoRA32_lr5e-05 search_mode=brute"
  "output_dir=debug_search/random_search/LUNAR/gemma-7b-it/1000_iteration_#1 unlearned_model.model_family=gemma-7b-it unlearned_model.model_path=unlearn_results/completions/lunar/gemma-7b-it/pistol_sample1/model search_mode=random"
  "output_dir=debug_search/brute_force_search/LUNAR/gemma-7b-it unlearned_model.model_family=gemma-7b-it unlearned_model.model_path=unlearn_results/completions/lunar/gemma-7b-it/pistol_sample1/model search_mode=brute"
)

srun python -u -m run_attack --config config/tas.yaml ${RUNS[$SLURM_ARRAY_TASK_ID]}