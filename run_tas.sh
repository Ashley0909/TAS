#!/bin/bash
#SBATCH -c 8
#SBATCH -p hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:1
#SBATCH --job-name=tas_search
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --time=11:59:00
#SBATCH --array=0-2

#
# TAS search experiment — one structured grid over five search modes:
#   random / brute / smart / smart_fast (smart + early stop) / greedy
#   (refusal-score baseline, merged in from the old run_greedy_experiment.sh).
# Full grid: 5 modes x 2 datasets x 3 unlearnings x 3 models x 3 seeds = 270 runs.
# Results land in debug_search/<mode_root>/seed<S>/<unl>/<dataset>/<model>/.
#
# To size the array, count the cells your filters select:
#   python scripts/tas_experiment_runs.py --modes smart --datasets pistol --unlearnings NPO,DPO --models llama2-7b-chat --count
# then set #SBATCH --array=0-<count-1> above (defaults below select 18 cells).
#
# After tasks finish, aggregate with scripts/eval_summary_table.py (reads
# eval_summary.csv produced by eval_pipeline.ipynb).

# Config rule: Can use comma to separate values (e.g. NPO,DPO to run both).
: "${EXP_MODES:=smart}"                 # random / brute / smart / smart_fast / greedy / blank = all 5
: "${EXP_DATASETS:=tofu}"               # pistol / dusk / tofu / blank = both
: "${EXP_UNLEARNINGS:=NPO}"             # NPO / DPO / LUNAR / blank = all
: "${EXP_MODELS:=llama3-8b-instruct}"       # llama2-7b-chat / llama3-8b-instruct / gemma-7b-it / blank = all
: "${EXP_SEEDS:=}"                      # 0 / 1 / 2 / blank = all (brute is deterministic: use 0)

source /nfs-share/ahta3/workspace/LUNAR/.venv/bin/activate

export HF_TOKEN=$(cat /nfs-share/ahta3/.huggingface_token)
export NETRC=/nfs-share/ahta3/.wandb_token

cd /nfs-share/ahta3/workspace/LUNAR/

GEN_ARGS=""
[ -n "$EXP_MODES" ]       && GEN_ARGS="$GEN_ARGS --modes $EXP_MODES"
[ -n "$EXP_DATASETS" ]    && GEN_ARGS="$GEN_ARGS --datasets $EXP_DATASETS"
[ -n "$EXP_UNLEARNINGS" ] && GEN_ARGS="$GEN_ARGS --unlearnings $EXP_UNLEARNINGS"
[ -n "$EXP_MODELS" ]      && GEN_ARGS="$GEN_ARGS --models $EXP_MODELS"
[ -n "$EXP_SEEDS" ]       && GEN_ARGS="$GEN_ARGS --seeds $EXP_SEEDS"

mapfile -t RUNS < <(python scripts/tas_experiment_runs.py $GEN_ARGS)

echo "Task ${SLURM_ARRAY_TASK_ID}/${#RUNS[@]}: ${RUNS[$SLURM_ARRAY_TASK_ID]}"
srun python -u -m run_attack --config config/tas.yaml ${RUNS[$SLURM_ARRAY_TASK_ID]}

# Greedy baseline: regenerate the aggregate metrics table.
# Array tasks finish out of order, so we rebuild after every greedy task from
# whatever greedy runs have landed so far — partial in-flight, complete once the
# last greedy cell finishes. No-op for the other (non-greedy) modes.
if grep -q 'search_mode=greedy' <<<"${RUNS[$SLURM_ARRAY_TASK_ID]}"; then
    python scripts/greedy_metrics_table.py \
        --root debug_search/greedy_search --out greedy_metrics_table.md || true
fi

# Copy this task's SLURM log into its per-run output_dir (parsed from the run args).
# Note: the .out is still open, so the very last lines written after this copy
# (final flushes) won't be captured — fine for inspecting logs alongside results.
DEST=$(grep -oP 'output_dir=\K[^ ]+' <<<"${RUNS[$SLURM_ARRAY_TASK_ID]}")
OUT_FILE="${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
if [ -n "$DEST" ] && [ -f "$OUT_FILE" ]; then
    mkdir -p "$DEST"
    cp "$OUT_FILE" "$DEST/"
    echo "Copied $OUT_FILE -> $DEST/"
fi

# Refresh grand table
srun python scripts/search_metrics_table.py

# Check unlearning model quality
# srun python scripts/refusal_neighborhood_probe.py \
#   --model_family gemma-7b-it \
#   --model_path /nfs-share/ahta3/workspace/PISTOL/models_forget/gemma-7b-it_forget_DUSK/dpo_20epochs_LoRA32_lr5e-05 \
#   --dataset dusk --forget_edge Roland_Lancaster_personal --num_target_entities 1 \
#   --templates 12 --out debug_search/refusal_probe/dpo_gemma_dusk.csv

# Example: run every mode on dusk for all unlearnings/models/seeds (216/2 = 108 cells):
#   EXP_MODES= EXP_DATASETS=dusk EXP_UNLEARNINGS= python scripts/tas_experiment_runs.py --count
