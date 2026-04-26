#!/bin/bash
#SBATCH -c 8
#SBATCH -p hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --job-name=test_tas
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --time=11:59:00
#SBATCH --array=0-17

source /nfs-share/ahta3/workspace/LUNAR/.venv/bin/activate

export HF_TOKEN=$(cat /nfs-share/ahta3/.huggingface_token)
export NETRC=/nfs-share/ahta3/.wandb_token

cd /nfs-share/ahta3/workspace/LUNAR/

# One entry per array task — each is a space-separated list of key=value overrides.
RUNS=(
  "output_dir=debug_search/brute_force_search/DPO/dusk/llama2-7b-chat unlearned_model.model_family=llama2-7b-chat unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/llama2-7b-chat_forget_dusk/dpo_20epochs_LoRA32_lr5e-05 search_mode=brute"
  "output_dir=debug_search/random_search/DPO/dusk/llama2-7b-chat unlearned_model.model_family=llama2-7b-chat unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/llama2-7b-chat_forget_dusk/dpo_20epochs_LoRA32_lr5e-05 search_mode=random"
  "output_dir=debug_search/brute_force_search/NPO/dusk/llama2-7b-chat unlearned_model.model_family=llama2-7b-chat unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/llama2-7b-chat_forget_dusk/npo_20epochs_LoRA32_lr5e-05 search_mode=brute"
  "output_dir=debug_search/random_search/NPO/dusk/llama2-7b-chat unlearned_model.model_family=llama2-7b-chat unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/llama2-7b-chat_forget_dusk/npo_20epochs_LoRA32_lr5e-05 search_mode=random"
  "output_dir=debug_search/brute_force_search/LUNAR/dusk/llama2-7b-chat unlearned_model.model_family=llama2-7b-chat unlearned_model.model_path=/nfs-share/ahta3/workspace/LUNAR/unlearn_results/completions/lunar/llama2-7b-chat/dusk/model search_mode=brute"
  "output_dir=debug_search/random_search/LUNAR/dusk/llama2-7b-chat unlearned_model.model_family=llama2-7b-chat unlearned_model.model_path=/nfs-share/ahta3/workspace/LUNAR/unlearn_results/completions/lunar/llama2-7b-chat/dusk/model search_mode=random"
  "output_dir=debug_search/brute_force_search/DPO/dusk/llama3-8b-instruct unlearned_model.model_family=llama3-8b-instruct unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/llama3-8b-instruct_forget_dusk/dpo_20epochs_LoRA32_lr5e-05 search_mode=brute"
  "output_dir=debug_search/random_search/DPO/dusk/llama3-8b-instruct unlearned_model.model_family=llama3-8b-instruct unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/llama3-8b-instruct_forget_dusk/dpo_20epochs_LoRA32_lr5e-05 search_mode=random"
  "output_dir=debug_search/brute_force_search/NPO/dusk/llama3-8b-instruct unlearned_model.model_family=llama3-8b-instruct unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/llama3-8b-instruct_forget_dusk/npo_20epochs_LoRA32_lr5e-05 search_mode=brute"
  "output_dir=debug_search/random_search/NPO/dusk/llama3-8b-instruct unlearned_model.model_family=llama3-8b-instruct unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/llama3-8b-instruct_forget_dusk/npo_20epochs_LoRA32_lr5e-05 search_mode=random"
  "output_dir=debug_search/brute_force_search/LUNAR/dusk/llama3-8b-instruct unlearned_model.model_family=llama3-8b-instruct unlearned_model.model_path=/nfs-share/ahta3/workspace/LUNAR/unlearn_results/completions/lunar/llama3-8b-instruct/dusk/model search_mode=brute"
  "output_dir=debug_search/random_search/LUNAR/dusk/llama3-8b-instruct unlearned_model.model_family=llama3-8b-instruct unlearned_model.model_path=/nfs-share/ahta3/workspace/LUNAR/unlearn_results/completions/lunar/llama3-8b-instruct/dusk/model search_mode=random"
  "output_dir=debug_search/brute_force_search/DPO/dusk/gemma-7b-it unlearned_model.model_family=gemma-7b-it unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/gemma-7b-it_forget_dusk/dpo_20epochs_LoRA32_lr5e-05 search_mode=brute"
  "output_dir=debug_search/random_search/DPO/dusk/gemma-7b-it unlearned_model.model_family=gemma-7b-it unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/gemma-7b-it_forget_dusk/dpo_20epochs_LoRA32_lr5e-05 search_mode=random"
  "output_dir=debug_search/brute_force_search/NPO/dusk/gemma-7b-it unlearned_model.model_family=gemma-7b-it unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/gemma-7b-it_forget_dusk/npo_20epochs_LoRA32_lr5e-05 search_mode=brute"
  "output_dir=debug_search/random_search/NPO/dusk/gemma-7b-it unlearned_model.model_family=gemma-7b-it unlearned_model.model_path=/nfs-share/ahta3/workspace/PISTOL/models_forget/gemma-7b-it_forget_dusk/npo_20epochs_LoRA32_lr5e-05 search_mode=random"
  "output_dir=debug_search/brute_force_search/LUNAR/dusk/gemma-7b-it unlearned_model.model_family=gemma-7b-it unlearned_model.model_path=/nfs-share/ahta3/workspace/LUNAR/unlearn_results/completions/lunar/gemma-7b-it/dusk/model search_mode=brute"
  "output_dir=debug_search/random_search/LUNAR/dusk/gemma-7b-it unlearned_model.model_family=gemma-7b-it unlearned_model.model_path=/nfs-share/ahta3/workspace/LUNAR/unlearn_results/completions/lunar/gemma-7b-it/dusk/model search_mode=random"
)

srun python -u -m run_attack --config config/tas.yaml ${RUNS[$SLURM_ARRAY_TASK_ID]}