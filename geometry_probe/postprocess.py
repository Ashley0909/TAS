from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_grouped_csvs(run_dir: Path) -> None:
    probe_path = run_dir / "probe_details.csv"
    prompt_path = run_dir / "prompt_features.csv"
    if not probe_path.exists():
        raise FileNotFoundError(f"Missing file: {probe_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Missing file: {prompt_path}")

    probe = pd.read_csv(probe_path)
    prompt = pd.read_csv(prompt_path)

    for col in ["base_entropy", "edit_entropy", "base_gap", "edit_gap", "base_refusal_score", "edit_refusal_score"]:
        if col in probe.columns:
            probe[col] = pd.to_numeric(probe[col], errors="coerce")

    probe["delta_entropy"] = probe["edit_entropy"] - probe["base_entropy"]
    probe["delta_gap"] = probe["edit_gap"] - probe["base_gap"]
    probe["delta_refusal"] = probe["edit_refusal_score"] - probe["base_refusal_score"]
    probe["abs_delta_abnormality"] = probe["delta_abnormality"].abs()

    # 1) Per-edit rows with explicit deltas for each prompt/edit pair.
    per_edit_cols = [
        "seed_prompt",
        "edge",
        "set_label",
        "edit_type",
        "edit_prompt",
        "base_abnormality",
        "edit_abnormality",
        "delta_abnormality",
        "abs_delta_abnormality",
        "base_entropy",
        "edit_entropy",
        "delta_entropy",
        "base_gap",
        "edit_gap",
        "delta_gap",
        "base_refusal_score",
        "edit_refusal_score",
        "delta_refusal",
    ]
    per_edit_cols = [c for c in per_edit_cols if c in probe.columns]
    per_edit = probe[per_edit_cols].copy()
    per_edit.to_csv(run_dir / "forget_retain_per_edit_deltas.csv", index=False)

    # 2) Per-prompt x edit_type summary (mean/std across repeated edits).
    prompt_edit_summary = (
        per_edit.groupby(["seed_prompt", "edge", "set_label", "edit_type"], dropna=False)
        .agg(
            n_edits=("delta_abnormality", "size"),
            mean_delta_abnormality=("delta_abnormality", "mean"),
            std_delta_abnormality=("delta_abnormality", "std"),
            mean_abs_delta_abnormality=("abs_delta_abnormality", "mean"),
            mean_delta_entropy=("delta_entropy", "mean"),
            mean_delta_gap=("delta_gap", "mean"),
            mean_delta_refusal=("delta_refusal", "mean"),
        )
        .reset_index()
    )
    prompt_edit_summary.to_csv(run_dir / "forget_retain_prompt_edit_summary.csv", index=False)

    # 3) Set-level x edit_type summary (forget vs retain comparison).
    set_edit_summary = (
        per_edit.groupby(["set_label", "edit_type"], dropna=False)
        .agg(
            n_rows=("delta_abnormality", "size"),
            mean_delta_abnormality=("delta_abnormality", "mean"),
            mean_abs_delta_abnormality=("abs_delta_abnormality", "mean"),
            median_abs_delta_abnormality=("abs_delta_abnormality", "median"),
            mean_delta_entropy=("delta_entropy", "mean"),
            mean_delta_gap=("delta_gap", "mean"),
            mean_delta_refusal=("delta_refusal", "mean"),
        )
        .reset_index()
    )
    set_edit_summary.to_csv(run_dir / "forget_retain_set_edit_summary.csv", index=False)

    # 4) Prompt-level metric summary by set.
    metric_cols = ["abnormality", "cliff_rate", "max_jump", "avg_jump", "instability", "anisotropy_ratio", "entropy", "gap", "refusal_score"]
    metric_cols = [c for c in metric_cols if c in prompt.columns]
    set_prompt_summary = (
        prompt.groupby("set_label", dropna=False)[metric_cols]
        .agg(["mean", "median", "std"])
        .reset_index()
    )
    set_prompt_summary.columns = [
        "_".join([str(x) for x in col if x]).strip("_") for col in set_prompt_summary.columns.to_flat_index()
    ]
    set_prompt_summary.to_csv(run_dir / "forget_retain_prompt_metric_summary.csv", index=False)

    # 5) Boundary-focused table requested for manual inspection.
    # Use prompt-level refusal crossing metric already produced in prompt_rows.
    crossing_col = "refusal_crossing_rate" if "refusal_crossing_rate" in prompt.columns else "refusal_crossings"
    prompt_metrics = prompt[["prompt", "cliff_rate", "max_jump", crossing_col]].copy()
    boundary = probe[
        ["seed_prompt", "edit_type", "set_label", "edit_prompt"]
    ].copy()
    boundary = boundary.merge(
        prompt_metrics,
        left_on="seed_prompt",
        right_on="prompt",
        how="left",
    )
    if "prompt" in boundary.columns:
        boundary = boundary.drop(columns=["prompt"])
    boundary = boundary.rename(
        columns={
            "seed_prompt": "prompt",
            "edit_type": "type of edit",
            "set_label": "forget or retain",
            "edit_prompt": "edited_prompt",
            crossing_col: "refusal_crossing_rate",
        }
    )[
        [
            "prompt",
            "type of edit",
            "forget or retain",
            "cliff_rate",
            "max_jump",
            "refusal_crossing_rate",
            "edited_prompt",
        ]
    ]
    boundary.to_csv(run_dir / "boundary_analysis.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create forget vs retain grouped analysis CSVs for geometry_probe outputs.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default="unlearn_results/geometry_probe/run_default/all_prompts",
        help="Directory containing probe_details.csv and prompt_features.csv",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    build_grouped_csvs(run_dir)
    print(f"Wrote grouped CSVs to: {run_dir}")


if __name__ == "__main__":
    main()
