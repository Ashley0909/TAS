#!/usr/bin/env python3
"""Generate the override matrix for the TAS search experiment.

Companion to entropy_experiment_runs.py — same cell grid, plus a search-mode
axis (this also subsumes the old greedy_experiment_runs.py: greedy is now just
another mode below):

    mode         x  {random, brute, smart, smart_fast, greedy}
    dataset      x  {dusk, pistol}
    unlearning   x  {LUNAR, NPO, DPO}
    model        x  {llama2-7b-chat, llama3-8b-instruct, gemma-7b-it}
    seed         x  {0, 1, 2}

= 5 * 2 * 3 * 3 * 3 = 270 runs. Each line is a space-separated list of
`key=value` overrides for run_attack.py. The query budget is NOT set here:
run_attack.py resolves it from config/tas.yaml
`smart_search.budget_by_dataset[dataset_name]` (dusk=100, pistol=1000), so every
mode spends the same per-dataset budget.

Search modes
    random      pure random probing               (search_mode=random)
    brute       exhaustive enumeration            (search_mode=brute)
    smart       Thompson-sampling active search   (search_mode=smart)
    smart_fast  smart + early stopping            (search_mode=smart,
                                                    smart_search.early_stop.enabled=true)
    greedy      refusal-score baseline            (search_mode=greedy)

Results land in
    debug_search/<root>/seed<S>/<unlearning>/<dataset>/<model>/
with <root> one of random_search / brute_force_search / smart_search /
smart_fast_search / greedy_search — one seed<S>/<unl>/<dataset>/<model> layout
shared across modes, so the analysis scripts can glob every mode the same way.

NB: brute is deterministic (full enumeration), so its 3 seeds are exact
replicas — kept only for grid uniformity. Use `EXP_SEEDS=0` to run it once.

Usage:
    python scripts/tas_experiment_runs.py                       # all 216 lines
    python scripts/tas_experiment_runs.py --count               # just the count
    python scripts/tas_experiment_runs.py --modes smart --count # subset count
    EXP_SEEDS=0 python scripts/tas_experiment_runs.py           # one seed
"""
from __future__ import annotations

import argparse

from entropy_experiment_runs import (
    DATASETS,
    UNLEARNINGS,
    MODELS,
    DATASET_CFG,
    model_path,
    _csv_filter,
)

SEEDS = ["0", "1", "2"]
MODES = ["random", "brute", "smart", "smart_fast", "greedy"]

# Per-mode: results root, the run_attack search_mode, whether it uses the smart
# (Thompson) machinery, and whether early stopping is on. greedy is the plain
# refusal-score baseline (no smart machinery), merged in from the old
# greedy_experiment_runs.py.
MODE_CFG = {
    "random":     {"root": "random_search",     "search_mode": "random", "smart": False, "early_stop": False},
    "brute":      {"root": "brute_force_search", "search_mode": "brute",  "smart": False, "early_stop": False},
    "smart":      {"root": "smart_search",       "search_mode": "smart",  "smart": True,  "early_stop": False},
    "smart_fast": {"root": "smart_fast_search",  "search_mode": "smart",  "smart": True,  "early_stop": True},
    "greedy":     {"root": "greedy_search",      "search_mode": "greedy", "smart": False, "early_stop": False},
}


def build_runs(modes=None, datasets=None, unlearnings=None, models=None, seeds=None):
    modes = modes or MODES
    datasets = datasets or DATASETS
    unlearnings = unlearnings or UNLEARNINGS
    models = models or MODELS
    seeds = seeds or SEEDS
    runs = []
    for mode in modes:
        mc = MODE_CFG[mode]
        for ds in datasets:
            dc = DATASET_CFG[ds]
            for unl in unlearnings:
                for model in models:
                    mp = model_path(unl, ds, model)
                    for seed in seeds:
                        out = f"debug_search/{mc['root']}/seed{seed}/{unl}/{ds}/{model}"
                        ov = [
                            f"output_dir={out}",
                            f"seed={seed}",
                            f"unlearned_model.model_family={model}",
                            f"unlearned_model.model_path={mp}",
                            f"search_mode={mc['search_mode']}",
                            f"prompts.dataset_name={dc['dataset_name']}",
                            f"prompts.num_target_entities={dc['num_target_entities']}",
                            f"prompts.forget_edge={dc['forget_edge']}",
                            f"refusal.use_embeddings={dc['use_embeddings']}",
                            "save_csvs=true",
                        ]
                        if mc["smart"]:
                            ov += [
                                f"smart_search.warmup_per_pair={dc['warmup_per_pair']}",
                                f"smart_search.cosample_prob={dc.get('cosample_prob', 0.0)}",
                            ]
                        if mc["early_stop"]:
                            ov.append("smart_search.early_stop.enabled=true")
                        runs.append(" ".join(ov))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--modes", default="", help="comma list subset of " + ",".join(MODES))
    ap.add_argument("--datasets", default="", help="comma list subset of " + ",".join(DATASETS))
    ap.add_argument("--unlearnings", default="", help="comma list subset of " + ",".join(UNLEARNINGS))
    ap.add_argument("--models", default="", help="comma list subset of " + ",".join(MODELS))
    ap.add_argument("--seeds", default="", help="comma list subset of " + ",".join(SEEDS))
    args = ap.parse_args()

    runs = build_runs(
        modes=_csv_filter(args.modes, MODES, "modes"),
        datasets=_csv_filter(args.datasets, DATASETS, "datasets"),
        unlearnings=_csv_filter(args.unlearnings, UNLEARNINGS, "unlearnings"),
        models=_csv_filter(args.models, MODELS, "models"),
        seeds=_csv_filter(args.seeds, SEEDS, "seeds"),
    )
    if args.count:
        print(len(runs))
    else:
        for r in runs:
            print(r)


if __name__ == "__main__":
    main()
