#!/usr/bin/env python3
"""Render debug_search/eval_summary.csv into a markdown report.

eval_summary.csv is produced by eval_pipeline.ipynb and holds one row per
search run (search_mode × method × dataset × model × run) with all the
pre-computed metrics. This script presents it two ways in a single .md:

  1. Per-mode aggregated tables — same columns/format as
     greedy_metrics_table.md (Per model + Averaged over all models), one
     section per search mode (smart / random / brute). For each
     (mode, dataset, model, method) cell a single *canonical* run is chosen:
     the full-budget headline run (max queries among non-`early_stop`
     `*_iteration` runs), averaging seed replicates at that budget.

  2. Faithful row dump — every row of eval_summary.csv (key columns), sorted,
     no aggregation.

Metric columns mirror the greedy report:
    Exact ↑ MRR ↑ Prompt recall ↑ Prompt precision ↑ Queries ↓ Cost (%) ↓
    Hit rate ↑ First hit ↓

  * Exact / MRR / recall / precision / queries come straight from the CSV.
  * Cost (%) = queries / brute-force enumeration cost(dataset) × 100, so
    `brute` = 100% and the other modes show the fraction of an exhaustive
    search they spend. (NB: the greedy report anchors Cost on the per-dataset
    *budget* instead — different denominator, stated in each file.)
  * Hit rate = fraction of runs that ever probe the target
    (`target_first_occurrence >= 0`; -1 = never found in eval_summary).
  * First hit ↓ = mean `target_first_occurrence`, budget-censored: a miss
    counts as that run's own `queries` so every run is included; `(Nc)` flags
    how many were censored.

Usage:
    python scripts/eval_summary_table.py \
        --csv debug_search/eval_summary.csv --out eval_summary_table.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODE_ORDER = ["smart", "random", "brute"]
HEAD_BASE = ["Exact ↑", "MRR ↑", "Prompt recall ↑", "Prompt precision ↑",
             "Queries ↓", "Cost (%) ↓", "Hit rate ↑", "First hit ↓"]
RAW_COLS = ["search_mode", "method", "dataset", "model", "run",
            "exact_match", "avg_rank", "mrr", "discovered", "recall",
            "precision", "queries", "target_first_occurrence"]


def brute_cost_by_dataset(df: pd.DataFrame) -> dict:
    """Full-enumeration query cost per dataset, taken from the brute rows."""
    b = df[df["search_mode"] == "brute"]
    return {ds: float(g["queries"].max()) for ds, g in b.groupby("dataset")}


def select_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """One canonical full-budget run set per (mode, dataset, model, method):
    the max-queries level among non-`early_stop` `*_iteration`/`single` runs,
    keeping all seed replicates at that level so they can be averaged."""
    keep = []
    grp = df.groupby(["search_mode", "dataset", "model", "method"])
    for _, g in grp:
        pool = g[~g["run"].astype(str).str.startswith("early_stop")]
        if pool.empty:
            pool = g
        maxq = pool["queries"].max()
        keep.append(pool[pool["queries"] == maxq])
    return pd.concat(keep, ignore_index=True)


def _agg(sub: pd.DataFrame, brute_cost: dict) -> dict:
    total = len(sub)
    found = sub["target_first_occurrence"] >= 0
    n_found = int(found.sum())

    ds = sub["dataset"].iloc[0]
    bcost = brute_cost.get(ds)
    cost = (sub["queries"] / bcost * 100).mean() if bcost else np.nan

    hit_rate = f"{n_found / total:.3f} ({n_found}/{total})" if total else "—"

    # First hit, censored at each run's own query budget for the misses.
    tfo = sub["target_first_occurrence"].where(found, sub["queries"])
    n_cens = total - n_found
    suffix = f" ({n_cens}c)" if n_cens else ""
    fh = f"{tfo.mean():.1f}{suffix}" if total else "—"

    return {
        "Exact ↑": f"{sub['exact_match'].mean():.3f}",
        "MRR ↑": f"{sub['mrr'].mean():.3f}",
        "Prompt recall ↑": f"{sub['recall'].mean():.3f}",
        "Prompt precision ↑": f"{sub['precision'].mean():.3f}",
        "Queries ↓": f"{sub['queries'].mean():.0f}",
        "Cost (%) ↓": "—" if np.isnan(cost) else f"{cost:.1f}",
        "Hit rate ↑": hit_rate,
        "First hit ↓": fh,
    }


def _table(rows_iter, lead_cols, lead_seps):
    out = ["| " + " | ".join(lead_cols + HEAD_BASE) + " |",
           "|" + "|".join(lead_seps + ["--:"] * len(HEAD_BASE)) + "|"]
    for lead, a in rows_iter:
        out.append("| " + " | ".join(lead + [a[h] for h in HEAD_BASE]) + " |")
    return out


def render(df_all: pd.DataFrame) -> str:
    brute_cost = brute_cost_by_dataset(df_all)
    canon = select_canonical(df_all)

    bc = ", ".join(f"{k}={int(v)}" for k, v in sorted(brute_cost.items()))
    out = ["# Search-mode metrics — from `eval_summary.csv`\n",
           "Aggregated tables average over unlearning method (NPO/DPO/LUNAR) "
           "and seed replicates, using the **canonical full-budget run** per "
           "cell (max queries among non-`early_stop` runs).\n",
           "## Metric definitions\n",
           "- **Exact ↑** — fraction of runs whose **rank-1** predicted entity "
           "tuple exactly equals the ground-truth forget target (all-or-nothing "
           "on the top guess).",
           "- **MRR ↑** — mean reciprocal rank of the true target.",
           "- **Prompt recall / precision ↑** — over discovered forget prompts "
           "(refusal ≥ 0.5) vs the ground-truth forget set.",
           "- **Queries ↓** — model queries spent (mean over the cell).",
           f"- **Cost (%) ↓** — queries / brute-force enumeration cost × 100 "
           f"(brute = 100%; enumeration cost per dataset: {bc}). Lets every "
           "mode be compared as a fraction of exhaustive search.",
           "- **Hit rate ↑** — fraction of runs that ever **probe** the true "
           "target (`target_first_occurrence ≥ 0`; coverage, `k/n` shown).",
           "- **First hit ↓** — mean query index of the first target probe, "
           "**censored**: a run that never finds the target counts as its own "
           "query budget, so every run is included; `(Nc)` flags censored runs.",
           "- **Exact ≤ Hit rate** always: a run can probe the target yet rank a "
           "confuser #1; Exact = 1 implies it was probed.\n"]

    # ---- Section 1: per-mode aggregated tables ----
    for mode in MODE_ORDER:
        sub = canon[canon["search_mode"] == mode]
        if sub.empty:
            continue
        out.append(f"## {mode}\n")
        out.append("### Per model\n")
        rows = (([ds, model, mode], _agg(g, brute_cost))
                for (ds, model), g in sub.groupby(["dataset", "model"]))
        out += _table(rows, ["Dataset", "Model", "Search mode"],
                      ["---", "---", "---"])
        out.append("")
        out.append("### Averaged over all models\n")
        rows = (([ds, mode], _agg(g, brute_cost))
                for ds, g in sub.groupby("dataset"))
        out += _table(rows, ["Dataset", "Search mode"], ["---", "---"])
        out.append("")

    # ---- Section 2: faithful row dump ----
    out.append("## Raw rows (faithful dump of eval_summary.csv)\n")
    out.append(f"All {len(df_all)} runs, key columns "
               "(`target_first_occurrence = -1` ⇒ target never probed; "
               "see the CSV for the remaining columns).\n")
    cols = [c for c in RAW_COLS if c in df_all.columns]
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join("---" for _ in cols) + "|")
    dump = df_all.sort_values(["search_mode", "dataset", "model", "method", "run"])
    for _, r in dump.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if c not in ("queries", "target_first_occurrence")
                             else f"{int(v)}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="debug_search/eval_summary.csv")
    ap.add_argument("--out", default="eval_summary_table.md")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    md = render(df)
    Path(args.out).write_text(md)
    print(f"[wrote {args.out}: {len(df)} rows, "
          f"modes={sorted(df['search_mode'].unique())}]")


if __name__ == "__main__":
    main()
