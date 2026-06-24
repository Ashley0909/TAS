#!/usr/bin/env python3
"""Aggregate the greedy refusal-score baseline over its 3 seeds.

Reads the live greedy-search results laid out as

    <root>/seed<S>/<unlearning>/<dataset>/<model>/ent_slot*.csv

where each ent_slot CSV ranks candidates by the greedy signal (max observed
refusal, written to the `Mean` column by Akinator._RankArm). For every
(dataset x unlearning x model) cell it reports the 1-based rank of the true
forget target per seed (best across slots), then aggregates the seeds into a
mean rank and top-1/top-3 recovery — the same target-rank metric used by the
entropy-vs-refusal study, so the two tables sit side by side.

No GPU / model required — operates purely on the saved CSVs.

Usage:
    python scripts/greedy_baseline_analysis.py \
        --root debug_search/greedy_search --out greedy_baseline_experiment.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from statistics import mean

# Ground-truth forget target(s) per dataset. Pistol's forget edge A_B is a
# *pair*; recovering either member counts as locating the edge.
TARGETS = {
    "dusk": ["Roland Lancaster"],
    "pistol": ["Wnzatj SAS", "Jzrcws SA"],
}


def target_rank(ranking, targets):
    """1-based rank of the best-placed target entity in `ranking`."""
    positions = [ranking.index(t) + 1 for t in targets if t in ranking]
    return min(positions) if positions else None


def cell_rank(cell_dir, targets):
    """Best target rank across the slot files in one seed's cell dir."""
    best = None
    n = None
    for sf in sorted(glob.glob(os.path.join(cell_dir, "ent_slot*.csv"))):
        rows = list(csv.DictReader(open(sf)))
        if not rows:
            continue
        n = len(rows)
        ranking = [r["Entity"] for r in sorted(rows, key=lambda r: float(r["Mean"]), reverse=True)]
        r = target_rank(ranking, targets)
        if r is not None:
            best = r if best is None else min(best, r)
    return best, n


def fmt_rank(r):
    if r is None:
        return "—"
    return f"{r} ✅" if r == 1 else (f"{r} ⚠️" if r <= 3 else str(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="debug_search/greedy_search")
    ap.add_argument("--out", default="greedy_baseline_experiment.md")
    args = ap.parse_args()

    # (unl, ds, model) -> {seed: (rank, n)}
    cells = defaultdict(dict)
    pat = os.path.join(args.root, "seed*", "*", "*", "*", "ent_slot0.csv")
    for f0 in sorted(glob.glob(pat)):
        d = os.path.dirname(f0)
        parts = d.split(os.sep)
        seed, unl, ds, model = parts[-4], parts[-3], parts[-2], parts[-1]
        if ds not in TARGETS:
            continue
        seed_idx = seed.replace("seed", "")
        cells[(unl, ds, model)][seed_idx] = cell_rank(d, TARGETS[ds])

    seeds = sorted({s for v in cells.values() for s in v})

    L = ["# Greedy refusal-score baseline: recovering the hidden forget target\n",
         "Baseline = rank candidates by **max** observed refusal score and "
         "repeatedly query the current best (pure exploitation; budget matched "
         "to the smart search). Each cell reports the 1-based rank of the true "
         "forget target in the final ranking (`ent_slot*.csv`, best across "
         "slots), per seed and aggregated. `1` = the baseline's top guess was "
         "correct.\n",
         "- **dusk** target: `Roland Lancaster` (single entity).",
         "- **pistol** target: forget edge `A_B` = pair (`Wnzatj SAS`, "
         "`Jzrcws SA`); rank is the best-placed member.\n"]

    seed_cols = " | ".join(f"seed {s}" for s in seeds)
    seed_sep = "|".join("--:" for _ in seeds)
    L.append("## Per-cell target rank (3 seeds)\n")
    L.append(f"| Dataset | Unlearning | Model | n | {seed_cols} | mean rank | top-1 |")
    L.append(f"|---|---|---|--:|{seed_sep}|--:|--:|")

    all_ranks = []  # (cell, seed) ranks present
    for (unl, ds, model) in sorted(cells, key=lambda k: (k[1], k[0], k[2])):
        per_seed = cells[(unl, ds, model)]
        ns = [n for (_, n) in per_seed.values() if n is not None]
        n = ns[0] if ns else "—"
        ranks = []
        seed_cells = []
        for s in seeds:
            r = per_seed.get(s, (None, None))[0]
            seed_cells.append(fmt_rank(r))
            if r is not None:
                ranks.append(r)
                all_ranks.append(r)
        mr = f"{mean(ranks):.2f}" if ranks else "—"
        top1 = f"{sum(1 for r in ranks if r == 1)}/{len(ranks)}" if ranks else "—"
        L.append(
            f"| {ds} | {unl} | {model} | {n} | "
            + " | ".join(seed_cells)
            + f" | {mr} | {top1} |"
        )
    L.append("")

    # ---- aggregate over all (cell, seed) runs, split by dataset ----
    def agg(scope_ds):
        rs = []
        for (unl, ds, model), per_seed in cells.items():
            if scope_ds is not None and ds != scope_ds:
                continue
            rs += [r for (r, _) in per_seed.values() if r is not None]
        if not rs:
            return None
        t1 = sum(1 for r in rs if r == 1)
        t3 = sum(1 for r in rs if r <= 3)
        return t1, t3, len(rs), mean(rs)

    L.append("## Aggregate recovery rates (over cell × seed runs)\n")
    L.append("| Scope | top-1 | top-3 | mean rank |")
    L.append("|---|--:|--:|--:|")
    for scope, ds in [("all", None), ("dusk", "dusk"), ("pistol", "pistol")]:
        a = agg(ds)
        if a is None:
            continue
        t1, t3, tot, mr = a
        L.append(f"| {scope} | {t1}/{tot} | {t3}/{tot} | {mr:.2f} |")
    L.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote {args.out} ({len(cells)} cells, seeds={seeds}).")


if __name__ == "__main__":
    main()
