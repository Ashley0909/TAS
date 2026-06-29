#!/usr/bin/env python3
"""Build the headline metrics table for the seeded smart/random search runs.

This is the file-tree analogue of `scripts/eval_summary_table.py`: instead of
reading a pre-computed `eval_summary.csv`, it walks the seeded results trees

    <root>/seed<S>/<unlearning>/<dataset>/<model>/{raw_result.json,forget_all_qs.csv}

— the same layout `scripts/greedy_metrics_table.py` consumes — and scores every
run on the fly with the *identical* metric definitions (it imports the scoring
functions from `greedy_metrics_table`, so the numbers stay in lockstep with the
greedy report and `eval_pipeline.ipynb`).

Output mirrors `eval_summary_table.md`: one section per search mode (smart /
smart_fast / random), each with a "Per model" table and an "Averaged over all
models" table, plus a faithful per-run dump at the end.

Columns:
    Dataset | [Model] | Search mode | Exact ↑ | MRR ↑ | Prompt recall ↑ |
    Prompt precision ↑ | Queries ↓ | Cost (%) ↓ | Hit rate ↑ | First hit ↓

  * Cost (%) = queries / brute-force enumeration cost(dataset) × 100, taken from
    the brute_force_search tree (dusk≈1988, pistol≈18216) so figures line up
    with eval_summary_table.md (brute = 100%). Pass --brute-root to relocate.
    TOFU's brute run is too expensive to enumerate, so its cost falls back to
    the theoretical THEORETICAL_BRUTE_COST (tofu=656958).
  * First hit ↓ = mean query index of the first true-target probe, censored at
    each run's own query count for misses; `(Nc)` flags censored runs.

No GPU / model required — pure post-hoc scoring of saved CSV/JSON.

Usage:
    python scripts/search_metrics_table.py \
        --roots debug_search/smart_search debug_search/smart_fast_search debug_search/random_search \
        --out debug_search/search_metrics_table.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the greedy report's scoring so metric definitions stay identical.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import greedy_metrics_table as g  # noqa: E402

HEAD_BASE = ["Exact ↑", "MRR ↑", "Prompt recall ↑", "Prompt precision ↑",
             "Queries ↓", "Cost (%) ↓", "Hit rate ↑", "First hit ↓"]
RAW_COLS = ["search_mode", "method", "dataset", "model", "seed",
            "exact_match", "mrr", "recall", "precision", "queries", "first_hit"]
# How many runs make a (dataset, model) cell complete. Seeded modes run
# 3 methods × 3 seeds; brute force is single-seed (3 methods × 1 seed).
EXPECTED_RUNS_PER_MODEL = 9
EXPECTED_RUNS_BRUTE = 3

# Theoretical full-enumeration cost for datasets whose brute_force_search was
# never run (too expensive). Used as a fallback denominator for Cost (%) when
# the brute tree has no entry for the dataset. TOFU = |entities| × |templates|.
THEORETICAL_BRUTE_COST = {"tofu": 656958}


def mode_label(root: Path) -> str:
    """debug_search/smart_search -> 'smart', random_search -> 'random',
    brute_force_search -> 'brute_force'."""
    return root.name.replace("_search", "")


def expected_runs(mode: str) -> int:
    """Per (dataset, model) run count that marks a cell 'Finished'."""
    return EXPECTED_RUNS_BRUTE if mode == "brute_force" else EXPECTED_RUNS_PER_MODEL


def discover_any(root: Path) -> pd.DataFrame:
    """Seeded trees (seed<S>/...) via greedy's discover_runs; the flat,
    single-seed brute tree (<method>/<dataset>/<model>/) via _discover_flat."""
    return g.discover_runs(root) if _is_seeded(root) else _discover_flat(root)


def brute_cost_by_dataset(brute_root: Path) -> dict:
    """Full-enumeration query cost per dataset = max history length over the
    brute_force_search runs. Empty dict if the tree is absent."""
    cost: dict = {}
    if not brute_root.exists():
        return cost
    df = g.discover_runs(brute_root) if _is_seeded(brute_root) else _discover_flat(brute_root)
    for ds, sub in df.groupby("dataset"):
        cost[ds] = float(sub["queries"].max())
    return cost


def _is_seeded(root: Path) -> bool:
    return any(p.name.startswith("seed") for p in root.iterdir() if p.is_dir())


def _discover_flat(root: Path) -> pd.DataFrame:
    """Brute tree is <root>/<unlearning>/<dataset>/<model>/ (no seed level)."""
    rows = []
    for raw_path in sorted(root.rglob("raw_result.json")):
        d = raw_path.parent
        rel = d.relative_to(root).parts
        if len(rel) != 3:
            continue
        unl, dataset, model = rel
        if dataset not in g.TRUE_TARGET:
            continue
        m = g.metrics_for_run(d, dataset)
        m.update(seed="single", method=unl, dataset=dataset, model=model)
        rows.append(m)
    return pd.DataFrame(rows)


def _agg(sub: pd.DataFrame, brute_cost: dict) -> dict:
    total = len(sub)
    found = sub["first_hit"].notna()
    n_found = int(found.sum())

    ds = sub["dataset"].iloc[0]
    bcost = brute_cost.get(ds)
    cost = (sub["queries"] / bcost * 100).mean() if bcost else np.nan

    hit_rate = f"{n_found / total:.3f} ({n_found}/{total})" if total else "—"

    # First hit, censored at each run's own query budget for the misses.
    tfo = sub["first_hit"].where(found, sub["queries"])
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


def _table(rows_iter, lead_cols, lead_seps, extra_cols=None, extra_seps=None):
    extra_cols = extra_cols or []
    extra_seps = extra_seps or []
    out = ["| " + " | ".join(lead_cols + HEAD_BASE + extra_cols) + " |",
           "|" + "|".join(lead_seps + ["--:"] * len(HEAD_BASE) + extra_seps) + "|"]
    for lead, a, extra in rows_iter:
        out.append("| " + " | ".join(lead + [a[h] for h in HEAD_BASE] + extra) + " |")
    return out


def render(df_all: pd.DataFrame, brute_cost: dict) -> str:
    bc = ", ".join(f"{k}={int(v)}" for k, v in sorted(brute_cost.items())) or "n/a"
    modes = [m for m in ["smart", "smart_fast", "random", "brute_force"]
             if m in df_all["search_mode"].unique()]
    modes += [m for m in sorted(df_all["search_mode"].unique()) if m not in modes]

    out = ["# Seeded search-mode metrics — from the results tree\n",
           "Scored directly off the seeded `smart_search` / `smart_fast_search` / "
           "`random_search` trees (`seed<S>/<method>/<dataset>/<model>/`) and the "
           "single-seed `brute_force_search` tree "
           "(`<method>/<dataset>/<model>/`) with the **same metric "
           "definitions as the greedy report** (`scripts/greedy_metrics_table.py`, "
           "matching `eval_pipeline.ipynb`). Aggregated tables average over the "
           "unlearning method (NPO/DPO/LUNAR) and seeds 0/1/2; brute force is "
           "exhaustive so it is run once per (method, dataset, model).\n",
           "## Metric definitions\n",
           "- **Exact ↑** — fraction of runs whose **rank-1** predicted entity "
           "tuple exactly equals the ground-truth forget target (all-or-nothing "
           "on the top guess).",
           "- **MRR ↑** — mean reciprocal rank of the true target across slots.",
           "- **Prompt recall / precision ↑** — over discovered forget prompts "
           "(refusal ≥ 0.5) vs the ground-truth forget set.",
           "- **Queries ↓** — model queries spent (mean over the cell).",
           f"- **Cost (%) ↓** — queries / brute-force enumeration cost × 100 "
           f"(brute = 100%; enumeration cost per dataset: {bc}). Lets every mode "
           "be compared as a fraction of exhaustive search.",
           "- **Hit rate ↑** — fraction of runs that ever **probe** the true "
           "target (coverage, `k/n` shown).",
           "- **First hit ↓** — mean query index of the first target probe, "
           "**censored**: a run that never finds the target counts as its own "
           "query budget, so every run is included; `(Nc)` flags censored runs.",
           "- **Exact ≤ Hit rate** always: a run can probe the target yet rank a "
           "confuser #1; Exact = 1 implies it was probed.\n"]

    # ---- Section 1: per-mode aggregated tables ----
    for mode in modes:
        sub = df_all[df_all["search_mode"] == mode]
        if sub.empty:
            continue
        out.append(f"## {mode}\n")
        out.append("### Per model\n")
        exp = expected_runs(mode)
        rows = (([ds, model, mode], _agg(grp, brute_cost),
                 ["Yes" if len(grp) >= exp else "No"])
                for (ds, model), grp in sub.groupby(["dataset", "model"]))
        out += _table(rows, ["Dataset", "Model", "Search mode"],
                      ["---", "---", "---"], ["Finished"], ["--:"])
        out.append("")
        out.append("### Averaged over all models\n")
        rows = (([ds, mode], _agg(grp, brute_cost), [])
                for ds, grp in sub.groupby("dataset"))
        out += _table(rows, ["Dataset", "Search mode"], ["---", "---"])
        out.append("")

    # ---- Section 2: faithful per-run dump ----
    out.append("## Raw rows (faithful dump of every scored run)\n")
    out.append(f"All {len(df_all)} runs, key columns "
               "(`first_hit` blank ⇒ target never probed).\n")
    cols = [c for c in RAW_COLS if c in df_all.columns]
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join("---" for _ in cols) + "|")
    dump = df_all.sort_values(["search_mode", "dataset", "model", "method", "seed"])
    for _, r in dump.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "first_hit":
                cells.append("" if pd.isna(v) else f"{int(v)}")
            elif c in ("queries", "exact_match"):
                cells.append(f"{int(v)}")
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        # Flag smart / smart_fast runs that missed the top-1 target
        # (exact_match != 1): <mark> every non-empty cell so the whole row reads
        # as highlighted.
        if (str(r.get("search_mode")) in ("smart", "smart_fast")
                and not pd.isna(r.get("exact_match"))
                and int(r["exact_match"]) != 1):
            cells = [f"<mark>{c}</mark>" if c != "" else c for c in cells]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=["debug_search/smart_search",
                             "debug_search/smart_fast_search",
                             "debug_search/random_search"])
    ap.add_argument("--brute-root", default="debug_search/brute_force_search")
    ap.add_argument("--out", default="debug_search/search_metrics_table.md")
    args = ap.parse_args()

    # Build the forget-prompt sets the greedy scorer expects.
    g.FORGET_SETS = {n: g.load_dataset_split(n) for n in g.DATASET_FILES}

    # Present every seeded root plus the (single-seed) brute tree as its own
    # mode; the brute tree doubles as the cost denominator below.
    roots = list(dict.fromkeys(args.roots + [args.brute_root]))
    frames = []
    for root_str in roots:
        root = Path(root_str)
        if not root.exists():
            print(f"[skip] {root} not found")
            continue
        df = discover_any(root)
        if df.empty:
            print(f"[skip] no runs under {root}")
            continue
        df["search_mode"] = mode_label(root)
        frames.append(df)
    if not frames:
        raise SystemExit("No runs found under any --roots")

    df_all = pd.concat(frames, ignore_index=True)
    brute_cost = brute_cost_by_dataset(Path(args.brute_root))
    # Fall back to the theoretical enumeration cost for datasets with no brute run.
    for ds, c in THEORETICAL_BRUTE_COST.items():
        brute_cost.setdefault(ds, float(c))

    md = render(df_all, brute_cost)
    Path(args.out).write_text(md)
    print(f"[wrote {args.out}: {len(df_all)} runs, "
          f"modes={sorted(df_all['search_mode'].unique())}, "
          f"datasets={sorted(df_all['dataset'].unique())}]")


if __name__ == "__main__":
    main()
