#!/usr/bin/env python3
"""Build the headline metrics tables for the greedy search baseline.

Walks the greedy results tree

    <root>/seed<S>/<unlearning>/<dataset>/<model>/{raw_result.json,forget_all_qs.csv}

scores every run with the *same* metric definitions used by the smart-search
eval pipeline (eval_pipeline.ipynb, cells 3 & 7), then emits two markdown
tables:

  1. Per model   — one row per (dataset, model), averaged over unlearning
                   method and seed.
  2. Per dataset — one row per dataset, averaged over every model/method/seed.

Columns (matching the paper's table):
    Dataset | [Model] | Search mode | Exact ↑ | MRR ↑ | Prompt recall ↑ |
    Prompt precision ↑ | Queries ↓ | Cost (%) ↓ | First hit

  * Exact            = top-1 predicted entity tuple == true forget target.
  * MRR              = mean reciprocal rank of the true target across slots.
  * Prompt recall    = |discovered ∩ forget| / |forget prompts|.
  * Prompt precision = |discovered ∩ forget| / |discovered prompts|.
  * Queries          = number of model queries (len of search history).
  * Cost (%)         = queries / budget(dataset) * 100, where budget(dataset)
                       is the per-dataset search budget (dusk=100, pistol=1000,
                       tofu=30000) from config/tas.yaml smart_search.budget_by_dataset.
  * First hit        = query index at which the true target was first probed,
                       averaged over runs that located it (count shown as k/n).

No GPU / model required — pure post-hoc scoring of saved CSV/JSON.

Usage:
    python scripts/greedy_metrics_table.py \
        --root debug_search/greedy_search \
        --out greedy_metrics_table.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config — mirrors eval_pipeline.ipynb so the greedy numbers line up with the
# smart-search eval_summary.csv.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
REFUSAL_THRESHOLD = 0.5      # cliff.refusal_threshold from config/tas.yaml
# Per-dataset search budget (config/tas.yaml smart_search.budget_by_dataset),
# used as the denominator for Cost (%).
BUDGET_BY_DATASET = {"dusk": 100, "pistol": 1000, "tofu": 30000}
# A (dataset, model) cell is "Finished" once all 9 runs are present:
# 3 unlearning methods (LUNAR/NPO/DPO) × 3 seeds.
EXPECTED_RUNS_PER_MODEL = 9

DATASET_FILES = {
    "dusk":   ROOT / "dataset" / "unlearning" / "dusk.json",
    "pistol": ROOT / "dataset" / "unlearning" / "pistol_sample1.json",
    "tofu":   ROOT / "dataset" / "unlearning" / "tofu.json",
}
FORGET_EDGE = {"dusk": "Roland_Lancaster_personal", "pistol": "A_B", "tofu": "author_1_personal"}
TRUE_TARGET = {
    "dusk":   ("Roland Lancaster",),
    "pistol": ("Wnzatj SAS", "Jzrcws SA"),
    "tofu":   ("Jaime Vasquez",),
}
ENTITY_TOKENS = {
    "dusk":   ["Roland Lancaster"],
    "pistol": ["Wnzatj SAS", "Jzrcws SA"],
    "tofu":   ["Jaime Vasquez"],
}
SUFFIX_ARTIFACTS = ["cedillo"]
SYNONYMS_DUSK = {"personal": "hobby"}
DATE_RE = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}\b")
CONTRACT_PHRASES = [
    r"\bbased on the contract between\b", r"\bin the contract between\b",
    r"\bunder the contract between\b", r"\bof the contract between\b",
    r"\bthe contract between\b", r"\bin the contract with\b",
    r"\bcontract with\b", r"\bbetween\b",
]
ENT_TOK, DATE_TOK = "entxx", "datexx"

BETA_RE = re.compile(r"Beta\(a=([0-9.]+),\s*b=([0-9.]+)\)")


# ---------------------------------------------------------------------------
# Prompt normalisation + dataset splits (from eval_pipeline cell 3)
# ---------------------------------------------------------------------------
def normalize(text, dataset):
    s = text
    for e in ENTITY_TOKENS[dataset]:
        s = re.sub(re.escape(e), f" {ENT_TOK} ", s)
    s = s.lower().replace("'s", "")
    for art in SUFFIX_ARTIFACTS:
        s = re.sub(rf"{ENT_TOK}\s*\.?\s*{art}", ENT_TOK, s)
    s = DATE_RE.sub(f" {DATE_TOK} ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    if dataset == "dusk":
        for a, b in SYNONYMS_DUSK.items():
            s = re.sub(rf"\b{a}\b", b, s)
    s = re.sub(rf"({ENT_TOK}\s+and\s+{ENT_TOK})", ENT_TOK, s)
    for p in CONTRACT_PHRASES:
        s = re.sub(p, " ", s)
    s = re.sub(rf"({ENT_TOK}\s+){{2,}}", f"{ENT_TOK} ", s)
    return re.sub(r"\s+", " ", s).strip()


def is_date_template(norm: str) -> bool:
    return DATE_TOK in norm.split()


def load_dataset_split(name: str):
    items = json.load(open(DATASET_FILES[name]))
    edge = FORGET_EDGE[name]
    forget = {normalize(x["question"], name) for x in items if x["edge"] == edge}
    forget = {q for q in forget if not is_date_template(q)}
    return forget


def parse_beta(s: str):
    m = BETA_RE.search(s)
    return (float(m.group(1)), float(m.group(2))) if m else (1.0, 1.0)


def slot_posterior(slot_dict):
    rows = [(ent, a / (a + b)) for ent, (a, b) in
            ((ent, parse_beta(bs)) for ent, bs in slot_dict.items())]
    return sorted(rows, key=lambda r: r[1], reverse=True)


def parse_history_item(h):
    # new schema: [template, [ent1, ...], score]; old: [template, ent1, ..., score]
    if len(h) >= 2 and isinstance(h[1], list):
        return tuple(h[1]), float(h[2])
    return tuple(h[1:-1]), float(h[-1])


# ---------------------------------------------------------------------------
# Per-run scoring (from eval_pipeline cell 7, trimmed to the reported columns)
# ---------------------------------------------------------------------------
def metrics_for_run(path: Path, dataset: str):
    raw = json.load(open(path / "raw_result.json"))
    forget_set = FORGET_SETS[dataset]
    true_target = TRUE_TARGET[dataset]
    n_slots = len(true_target)

    if "ranked_slots" in raw:
        ranked_slots = raw["ranked_slots"]
    else:
        ranked_slots = [raw.get(f"ranked_slot{s}", []) for s in range(1, n_slots + 1)]

    top1_pred = tuple(rs[0] if rs else "" for rs in ranked_slots)
    exact_match = int(top1_pred == true_target)

    slot_ranks = []
    for s, rs in enumerate(ranked_slots):
        try:
            slot_ranks.append(rs.index(true_target[s]) + 1)
        except ValueError:
            slot_ranks.append(len(rs) + 1)
    mrr = float(np.mean([1.0 / r for r in slot_ranks]))

    # ---- prompt-level recall / precision ----
    fq_path = path / "forget_all_qs.csv"
    if fq_path.exists():
        fq = pd.read_csv(fq_path)
        norm = (fq.loc[fq["refusal_score"] >= REFUSAL_THRESHOLD, "edited_prompt"]
                  .map(lambda t: normalize(t, dataset)))
        discovered = {q for q in norm if not is_date_template(q)}
    else:
        discovered = set()
    tp = len(discovered & forget_set)
    n_disc = len(discovered)
    recall = tp / len(forget_set) if forget_set else 0.0
    precision = tp / n_disc if n_disc else 0.0

    # ---- query-level ----
    parsed = [parse_history_item(h) for h in raw.get("history", [])]
    queries = len(parsed)
    target_first = None
    for i, (ents, _) in enumerate(parsed):
        if ents == true_target:
            target_first = i
            break
    cost_pct = queries / BUDGET_BY_DATASET[dataset] * 100

    return {
        "exact_match": exact_match,
        "mrr": mrr,
        "recall": recall,
        "precision": precision,
        "queries": queries,
        "cost_pct": cost_pct,
        "first_hit": target_first,   # None if never probed
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_runs(root: Path):
    rows = []
    for raw_path in sorted(root.rglob("raw_result.json")):
        d = raw_path.parent
        # layout: <root>/seed<S>/<unlearning>/<dataset>/<model>/
        rel = d.relative_to(root).parts
        if len(rel) != 4:
            continue
        seed, unl, dataset, model = rel
        if dataset not in TRUE_TARGET:
            continue
        m = metrics_for_run(d, dataset)
        m.update(seed=seed, method=unl, dataset=dataset, model=model)
        rows.append(m)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
HEAD_BASE = ["Exact ↑", "MRR ↑", "Prompt recall ↑", "Prompt precision ↑",
             "Queries ↓", "Cost (%) ↓", "Hit rate ↑", "First hit ↓"]


def _agg(sub: pd.DataFrame) -> dict:
    total = len(sub)
    found = sub["first_hit"].dropna()
    n_found = len(found)
    budget = BUDGET_BY_DATASET[sub["dataset"].iloc[0]]

    # Coverage: fraction of runs that ever probe the true target.
    hit_rate = f"{n_found / total:.3f} ({n_found}/{total})" if total else "—"

    # First hit, budget-censored: misses count as `budget` so every run is
    # included (conservative — true value is >= budget for a miss). The (c)
    # tag flags how many runs were censored at the budget.
    if total:
        censored = sub["first_hit"].fillna(budget)
        n_cens = total - n_found
        suffix = f" ({n_cens}c)" if n_cens else ""
        fh = f"{censored.mean():.1f}{suffix}"
    else:
        fh = "—"

    return {
        "Exact ↑": f"{sub['exact_match'].mean():.3f}",
        "MRR ↑": f"{sub['mrr'].mean():.3f}",
        "Prompt recall ↑": f"{sub['recall'].mean():.3f}",
        "Prompt precision ↑": f"{sub['precision'].mean():.3f}",
        "Queries ↓": f"{sub['queries'].mean():.0f}",
        "Cost (%) ↓": f"{sub['cost_pct'].mean():.1f}",
        "Hit rate ↑": hit_rate,
        "First hit ↓": fh,
    }


def render(df: pd.DataFrame, mode_label: str) -> str:
    out = ["# Greedy search — headline metrics\n",
           f"Search mode = **{mode_label}**. Averages taken over the unlearning "
           f"method (NPO/DPO/LUNAR) and seeds. Metric definitions match "
           f"`eval_pipeline.ipynb` (smart-search eval), so figures are directly "
           f"comparable. Cost (%) = queries / per-dataset budget × 100 "
           f"({', '.join(f'{k}={v}' for k, v in BUDGET_BY_DATASET.items() if k != 'tofu')}). "
           f"Hit rate = fraction of runs that ever probe the true target "
           f"(coverage; k/n shown). First hit ↓ = mean query index of the first "
           f"target probe, **budget-censored**: runs that never find the target "
           f"count as `budget`, so every run is included (conservative). `(Nc)` "
           f"flags how many runs were censored at the budget.\n",
           "## Metric definitions\n",
           "- **Exact ↑** — exact-match accuracy of the search's single top "
           "guess. Per run it is `1` iff the **rank-1** entity in every slot "
           "(`ranked_slots[s][0]`) equals the ground-truth forget target tuple "
           "(dusk = `(Roland Lancaster,)`; pistol = `(Wnzatj SAS, Jzrcws SA)`, "
           "matched positionally), else `0`. The reported value is the mean "
           "over runs = the fraction whose #1 prediction was exactly correct. "
           "Strictest entity metric: all-or-nothing on the final top guess.",
           "- **Hit rate ↑** — coverage of the search *process*. Per run it is "
           "`1` iff the search ever **probed** the true target tuple at any "
           "point in its query history (i.e. `target_first_occurrence` exists), "
           "else `0`, regardless of where the target ended up ranked. The "
           "reported value is the mean over runs (`k/n` shown).",
           "- **Exact vs Hit rate.** They are nested, not equal: a run can probe "
           "the target (Hit rate = 1) yet rank a confuser #1 (Exact = 0), and "
           "Exact = 1 implies the target was probed — so **Exact ≤ Hit rate** "
           "always. The gap is the runs that *found* the target but did not "
           "*rank it first*.\n"]

    # ---- Table 1: per model ----
    # "Finished" = all 9 runs present (3 unlearning methods × 3 seeds).
    cols = ["Dataset", "Model", "Search mode"] + HEAD_BASE + ["Finished"]
    out.append("## Per model\n")
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join(["---", "---", "---"] + ["--:"] * len(HEAD_BASE) + ["--:"]) + "|")
    for (ds, model), sub in df.groupby(["dataset", "model"]):
        a = _agg(sub)
        finished = "Yes" if len(sub) >= EXPECTED_RUNS_PER_MODEL else "No"
        out.append("| " + " | ".join([ds, model, mode_label]
                                      + [a[h] for h in HEAD_BASE] + [finished]) + " |")
    out.append("")

    # ---- Table 2: averaged over all models ----
    cols2 = ["Dataset", "Search mode"] + HEAD_BASE
    out.append("## Averaged over all models\n")
    out.append("| " + " | ".join(cols2) + " |")
    out.append("|" + "|".join(["---", "---"] + ["--:"] * len(HEAD_BASE)) + "|")
    for ds, sub in df.groupby("dataset"):
        a = _agg(sub)
        out.append("| " + " | ".join([ds, mode_label] + [a[h] for h in HEAD_BASE]) + " |")
    out.append("")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="debug_search/greedy_search")
    ap.add_argument("--out", default="greedy_metrics_table.md")
    ap.add_argument("--mode-label", default="greedy")
    args = ap.parse_args()

    global FORGET_SETS
    FORGET_SETS = {n: load_dataset_split(n) for n in DATASET_FILES}

    root = Path(args.root)
    df = discover_runs(root)
    if df.empty:
        raise SystemExit(f"No runs found under {root}")

    md = render(df, args.mode_label)
    Path(args.out).write_text(md)
    print(md)
    print(f"\n[wrote {args.out}: {len(df)} runs over "
          f"{df['dataset'].nunique()} datasets × {df['model'].nunique()} models]")


if __name__ == "__main__":
    main()
