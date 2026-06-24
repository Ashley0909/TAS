#!/usr/bin/env python3
"""Generate the override matrix for the entropy-vs-refusal search experiment.

Prints one line per run (a space-separated list of `key=value` overrides for
run_attack.py), covering:

    dataset      x  {dusk, pistol}
    unlearning   x  {LUNAR, NPO, DPO}
    model        x  {llama2-7b-chat, llama3-8b-instruct, gemma-7b-it}
    signal       x  {refusal, entropy, combined}

= 2 * 3 * 3 * 3 = 54 runs. run_entropy_experiment.sh indexes into this list with
$SLURM_ARRAY_TASK_ID. Edit MODEL_PATHS below if your checkpoints moved.

Usage:
    python scripts/entropy_experiment_runs.py            # all lines
    python scripts/entropy_experiment_runs.py --count    # just the count
"""
from __future__ import annotations

import argparse

PISTOL = "/nfs-share/ahta3/workspace/PISTOL/models_forget"
LUNAR = "/nfs-share/ahta3/workspace/LUNAR/unlearn_results/completions/lunar"

MODELS = ["llama2-7b-chat", "llama3-8b-instruct", "gemma-7b-it"]
UNLEARNINGS = ["LUNAR", "NPO", "DPO"]
DATASETS = ["dusk", "pistol"]
SIGNALS = ["refusal", "entropy", "combined"]

# Per-dataset search settings (mirrors the comments in config/tas.yaml).
DATASET_CFG = {
    "dusk": {
        "dataset_name": "dusk",
        "num_target_entities": 1,
        "forget_edge": "['Roland_Lancaster_personal']",
        "warmup_per_pair": 1,
        "use_embeddings": "false",
    },
    "pistol": {
        "dataset_name": "pistol_sample1",
        "num_target_entities": 2,
        "forget_edge": "['A_B']",
        "warmup_per_pair": 2,
        "use_embeddings": "false",
        "cosample_prob": 0.5,  # fix 4: partner-discovery for the pair task
        # The forget signal is directional + template-sparse, so Thompson often
        # finds nothing (all posteriors flat) and ranks noise. Fall back to an
        # exhaustive anchor x partner sweep in confirm. Scoped to pistol; dusk
        # (single-entity) doesn't need it.
        "confirm_exhaustive": True,
    },
}
# NOTE: budget is intentionally NOT set here. run_attack.py resolves it from
# config/tas.yaml `smart_search.budget_by_dataset[dataset_name]`, so the budget
# lives in exactly one place. To sweep it, override the map entry, e.g.
#   smart_search.budget_by_dataset.dusk=50

# model_path per (unlearning, dataset, model). LUNAR is templated; the
# DPO/NPO PISTOL checkpoints use irregular epoch/LoRA tags so they are explicit.
def model_path(unl, ds, model):
    if unl == "LUNAR":
        ds_folder = "dusk" if ds == "dusk" else "pistol_sample1"
        return f"{LUNAR}/{model}/{ds_folder}/model"
    if ds == "dusk":
        method = unl.lower()  # dpo | npo
        return f"{PISTOL}/{model}_forget_DUSK/{method}_20epochs_LoRA32_lr5e-05"
    # pistol DPO/NPO — explicit (epochs/LoRA differ per model)
    return {
        ("DPO", "llama2-7b-chat"): f"{PISTOL}/llama2-7b-chat_forget_AB/dpo_40epochs_LoRA32_lr5e-05",
        ("NPO", "llama2-7b-chat"): f"{PISTOL}/llama2-7b-chat_forget_AB/npo_20epochs_LoRA32_lr5e-05",
        ("DPO", "llama3-8b-instruct"): f"{PISTOL}/llama3-8b-instruct_forget_AB/dpo_80epochs_LoRA16_lr1.25e-05",
        ("NPO", "llama3-8b-instruct"): f"{PISTOL}/llama3-8b-instruct_forget_AB/npo_20epochs_LoRA32_lr5e-05",
        ("DPO", "gemma-7b-it"): f"{PISTOL}/gemma-7b-it_forget_AB/dpo_40epochs_LoRA16_lr1.25e-05",
        ("NPO", "gemma-7b-it"): f"{PISTOL}/gemma-7b-it_forget_AB/npo_20epochs_LoRA32_lr5e-05",
    }[(unl, model)]


def build_runs(datasets=None, unlearnings=None, models=None, signals=None):
    datasets = datasets or DATASETS
    unlearnings = unlearnings or UNLEARNINGS
    models = models or MODELS
    signals = signals or SIGNALS
    runs = []
    for ds in datasets:
        dc = DATASET_CFG[ds]
        for unl in unlearnings:
            for model in models:
                mp = model_path(unl, ds, model)
                for signal in signals:
                    out = f"debug_search/entropy_exp/{signal}/{unl}/{ds}/{model}"
                    ov = [
                        f"output_dir={out}",
                        f"unlearned_model.model_family={model}",
                        f"unlearned_model.model_path={mp}",
                        "search_mode=smart",
                        f"smart_search.signal={signal}",
                        f"smart_search.warmup_per_pair={dc['warmup_per_pair']}",
                        f"smart_search.cosample_prob={dc.get('cosample_prob', 0.0)}",
                        f"prompts.dataset_name={dc['dataset_name']}",
                        f"prompts.num_target_entities={dc['num_target_entities']}",
                        f"prompts.forget_edge={dc['forget_edge']}",
                        f"refusal.use_embeddings={dc['use_embeddings']}",
                        "save_csvs=true",
                    ]
                    runs.append(" ".join(ov))
    return runs


def _csv_filter(val, allowed, axis):
    if not val:
        return allowed
    picked = [x.strip() for x in val.split(",") if x.strip()]
    bad = [x for x in picked if x not in allowed]
    if bad:
        raise SystemExit(f"--{axis}: unknown {bad}; choose from {allowed}")
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--datasets", default="", help="comma list subset of " + ",".join(DATASETS))
    ap.add_argument("--unlearnings", default="", help="comma list subset of " + ",".join(UNLEARNINGS))
    ap.add_argument("--models", default="", help="comma list subset of " + ",".join(MODELS))
    ap.add_argument("--signals", default="", help="comma list subset of " + ",".join(SIGNALS))
    args = ap.parse_args()

    runs = build_runs(
        datasets=_csv_filter(args.datasets, DATASETS, "datasets"),
        unlearnings=_csv_filter(args.unlearnings, UNLEARNINGS, "unlearnings"),
        models=_csv_filter(args.models, MODELS, "models"),
        signals=_csv_filter(args.signals, SIGNALS, "signals"),
    )
    if args.count:
        print(len(runs))
    else:
        for r in runs:
            print(r)


if __name__ == "__main__":
    main()
