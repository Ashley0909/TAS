from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None

from geometry_probe.analysis import cluster_abnormality, detect_cliffs, estimate_direction_sensitivity
from geometry_probe.entity_generator import (
    EntityGenerator,
    EntityGeneratorConfig,
    build_entity_candidates_from_vocab,
)
from geometry_probe.io_utils import ensure_dir, timestamp, write_csv, write_json, write_jsonl
from geometry_probe.metrics import RefusalScorer, abnormality_score, paraphrase_instability
from geometry_probe.model_interface import (
    ProbeModel,
    cosine_distance_to_direction,
    finite_difference_anisotropy_proxy,
    layerwise_activation_distance,
)
from geometry_probe.perturbations import EntitySwapOp, Perturbation, build_operators, generate_perturbations
from geometry_probe.rl_explorer import BanditConfig, run_bandit_exploration, run_entity_generator_exploration
from src.dataset_utils import load_dataset_json


def _load_config(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_seed_prompts(cfg: Dict[str, object]) -> List[str]:
    prompts_cfg = cfg.get("prompts", {})
    inline = prompts_cfg.get("inline_prompts", [])
    if inline:
        return [str(x).strip() for x in inline if str(x).strip()]

    if prompts_cfg.get("dataset_json"):
        with open(prompts_cfg["dataset_json"], "r", encoding="utf-8") as f:
            data = json.load(f)
    elif prompts_cfg.get("dataset_name"):
        data = load_dataset_json(prompts_cfg["dataset_name"])
    else:
        raise ValueError("Provide prompts.inline_prompts or prompts.dataset_json or prompts.dataset_name")

    key = prompts_cfg.get("prompt_key", "question")
    prompts = []
    for item in data:
        val = item.get(key) or item.get("instruction") or item.get("question")
        if isinstance(val, str) and val.strip():
            prompts.append(val.strip())
    limit = int(prompts_cfg.get("limit", 50))
    if prompts_cfg.get("shuffle", True):
        rnd = random.Random(int(cfg.get("seed", 0)))
        rnd.shuffle(prompts)
    if limit < 0:
        return prompts
    return prompts[:limit]


def _plot_summaries(prompt_df: pd.DataFrame, direction_df: pd.DataFrame, out_dir: str) -> None:
    if plt is None:
        return
    if not prompt_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(prompt_df["cliff_rate"], bins=20)
        ax.set_title("Cliff Rate Distribution")
        ax.set_xlabel("cliff_rate")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(Path(out_dir) / "plot_cliff_rate_hist.png", dpi=150)
        plt.close(fig)

    if not direction_df.empty:
        agg = direction_df.groupby("edit_type")["abs_delta_abnormality"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        agg.plot(kind="bar", ax=ax)
        ax.set_title("Direction Sensitivity")
        ax.set_ylabel("mean |delta abnormality|")
        fig.tight_layout()
        fig.savefig(Path(out_dir) / "plot_direction_sensitivity.png", dpi=150)
        plt.close(fig)


def _build_metric_dict(score, refusal_score: float) -> Dict[str, float]:
    return {
        "entropy": float(score.token_entropy_mean),
        "gap": float(score.top12_gap_mean),
        "refusal_score": float(refusal_score),
    }


def run_probe(config_path: str) -> Dict[str, object]:
    cfg = _load_config(config_path)
    random.seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    out_root = ensure_dir(cfg.get("output_dir", f"unlearn_results/geometry_probe/{timestamp()}"))
    if cfg.get("device", "auto") == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg["device"]

    unlearned_model = ProbeModel(
        model_family=cfg["unlearned_model"]["model_family"],
        model_path=cfg["unlearned_model"]["model_path"],
        device=device,
    )
    base_model = None
    if cfg.get("white_box", {}).get("enabled", False) and cfg.get("base_model"):
        base_model = ProbeModel(
            model_family=cfg["base_model"]["model_family"],
            model_path=cfg["base_model"]["model_path"],
            device=device,
        )

    scorer = RefusalScorer(
        use_embeddings=bool(cfg.get("refusal", {}).get("use_embeddings", False)),
        model_name=cfg.get("refusal", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    operators = build_operators(cfg.get("perturbations", {}).get("operators", []))
    prompts = _load_seed_prompts(cfg)

    max_new_tokens = int(cfg.get("max_new_tokens", 64))
    first_n_tokens = int(cfg.get("first_n_tokens", 24))
    k_per_op = int(cfg.get("perturbations", {}).get("k_per_operator", 4))
    cliff_threshold = float(cfg.get("cliff", {}).get("threshold", 0.25))
    refusal_threshold = float(cfg.get("cliff", {}).get("refusal_threshold", 0.5))
    reward_weights = cfg.get("reward_weights", {"refusal_score": 1.0, "entropy": 1.0, "instability": 1.0})

    details_rows: List[Dict[str, object]] = []
    prompt_rows: List[Dict[str, object]] = []
    direction_rows: List[Dict[str, object]] = []

    white_layers = cfg.get("white_box", {}).get("layers", [])
    ignorance_direction = None
    direction_path = cfg.get("white_box", {}).get("ignorance_direction_path")
    if direction_path:
        ignorance_direction = np.load(direction_path)

    '''
    For each prompt, we compute:
    - the base scores = the scores from the unlearned model on the original prompt
    - the edit scores = the scores from the unlearned model on the edited prompts (entity, relation swaps, etc.)
    '''
    for idx, prompt in enumerate(prompts):
        base_score = unlearned_model.score_prompt(prompt, max_new_tokens=max_new_tokens, first_n_tokens=first_n_tokens)
        base_refusal = scorer.score(base_score.completion)
        base_metrics = _build_metric_dict(base_score, base_refusal) # R(x), H(x), G(x)
        base_abnormal = abnormality_score(base_metrics, reward_weights)

        perturbations = generate_perturbations(
            prompt,
            operators=operators,
            k_per_operator=k_per_op,
            seed=int(cfg.get("seed", 0)) + idx,
        )
        edit_abnormalities: List[float] = []
        edit_refusals: List[float] = []
        deltas_by_dir: Dict[str, List[float]] = defaultdict(list)

        wb_base_acts = None
        wb_unlearned_acts = None
        wb_distance = {}
        wb_cosine_distance = None
        wb_anisotropy_proxy = {}
        edited_act_by_dir: Dict[str, List[np.ndarray]] = defaultdict(list)
        if cfg.get("white_box", {}).get("enabled", False) and white_layers:
            wb_unlearned_acts = unlearned_model.capture_layer_activations(prompt, layers=white_layers)
            if base_model is not None:
                wb_base_acts = base_model.capture_layer_activations(prompt, layers=white_layers)
                wb_distance = layerwise_activation_distance(wb_base_acts, wb_unlearned_acts)
            if ignorance_direction is not None and wb_unlearned_acts:
                chosen = wb_unlearned_acts.get(white_layers[-1])
                if chosen is not None:
                    wb_cosine_distance = cosine_distance_to_direction(chosen, ignorance_direction)

        for p in perturbations:
            edit_score = unlearned_model.score_prompt(p.text, max_new_tokens=max_new_tokens, first_n_tokens=first_n_tokens)
            edit_refusal = scorer.score(edit_score.completion)
            edit_metrics = _build_metric_dict(edit_score, edit_refusal) # R(x_i), H(x_i), G(x_i)
            edit_ab = abnormality_score(edit_metrics, reward_weights) # S(x)
            delta = edit_ab - base_abnormal # \Delta S_i

            deltas_by_dir[p.edit_type].append(delta)
            edit_abnormalities.append(edit_ab)
            edit_refusals.append(edit_refusal)

            if cfg.get("white_box", {}).get("enabled", False) and white_layers:
                acts = unlearned_model.capture_layer_activations(p.text, layers=white_layers)
                vec = acts.get(white_layers[-1])
                if vec is not None:
                    edited_act_by_dir[p.edit_type].append(vec)

            details_rows.append(
                {
                    "seed_prompt": prompt,
                    "edit_prompt": p.text,
                    "edit_type": p.edit_type,
                    "base_abnormality": base_abnormal, # S(x)
                    "edit_abnormality": edit_ab, # S(x_i)
                    "delta_abnormality": delta, # \Delta S_i
                    "base_refusal_score": base_refusal, # R(x)
                    "edit_refusal_score": edit_refusal, # R(x_i)
                    "base_entropy": base_score.token_entropy_mean, # H(x)
                    "edit_entropy": edit_score.token_entropy_mean, # H(x_i)
                    "base_gap": base_score.top12_gap_mean, # G(x)
                    "edit_gap": edit_score.top12_gap_mean, # G(x_i)
                    "base_completion": base_score.completion,
                    "edit_completion": edit_score.completion,
                }
            )
            direction_rows.append(
                {
                    "seed_prompt": prompt,
                    "edit_type": p.edit_type,
                    "abs_delta_abnormality": abs(delta),
                }
            )

        cliff = detect_cliffs(
            base_value=base_abnormal, # S(x)
            edited_values=edit_abnormalities, # S(x_i) for all i
            base_refusal=base_refusal, # R(x)
            edited_refusals=edit_refusals, # R(x_i) for all i
            threshold=cliff_threshold, # \tau
            refusal_threshold=refusal_threshold, # \theta
        )
        sensitivity = estimate_direction_sensitivity(deltas_by_dir)
        instability = paraphrase_instability(edit_abnormalities)

        if cfg.get("white_box", {}).get("enabled", False) and white_layers and wb_unlearned_acts:
            base_vec = wb_unlearned_acts.get(white_layers[-1])
            if base_vec is not None:
                wb_anisotropy_proxy = finite_difference_anisotropy_proxy(base_vec, edited_act_by_dir)

        prompt_rows.append(
            {
                "prompt": prompt,
                "entropy": base_score.token_entropy_mean,
                "gap": base_score.top12_gap_mean,
                "refusal_score": base_refusal,
                "instability": instability,
                "cliff_rate": cliff.cliff_rate,
                "max_jump": cliff.max_jump,
                "avg_jump": cliff.avg_jump,
                "refusal_crossing_rate": cliff.refusal_crossing_rate,
                "refusal_flip_rate": cliff.refusal_flip_rate,
                "anisotropy_ratio": sensitivity.get("anisotropy_ratio", 0.0),
                "abnormality": base_abnormal,
                "wb_layerwise_distance": json.dumps(wb_distance),
                "wb_ignorance_cosine_distance": wb_cosine_distance,
                "wb_anisotropy_ratio": wb_anisotropy_proxy.get("anisotropy_ratio"),
            }
        )

    prompt_df = pd.DataFrame(prompt_rows)
    direction_df = pd.DataFrame(direction_rows)
    clustering = cluster_abnormality(
        feature_df=prompt_df[["prompt", "entropy", "gap", "refusal_score", "instability", "cliff_rate", "anisotropy_ratio"]]
        if not prompt_df.empty
        else pd.DataFrame(columns=["prompt", "entropy", "gap", "refusal_score", "instability", "cliff_rate", "anisotropy_ratio"]),
        prompt_col="prompt",
        kmeans_k=int(cfg.get("clustering", {}).get("kmeans_k", 4)),
        dbscan_eps=float(cfg.get("clustering", {}).get("dbscan_eps", 0.8)),
        dbscan_min_samples=int(cfg.get("clustering", {}).get("dbscan_min_samples", 4)),
    )

    if not prompt_df.empty and clustering.get("kmeans_assignments"):
        prompt_df["kmeans_cluster"] = clustering["kmeans_assignments"]
    if not prompt_df.empty and clustering.get("dbscan_assignments"):
        prompt_df["dbscan_cluster"] = clustering["dbscan_assignments"]

    rl_payload = {}
    if cfg.get("rl", {}).get("enabled", False):
        def score_fn(text: str) -> Dict[str, float]:
            s = unlearned_model.score_prompt(text, max_new_tokens=max_new_tokens, first_n_tokens=first_n_tokens)
            rs = scorer.score(s.completion)
            m = _build_metric_dict(s, rs)
            m["instability"] = 0.0
            return m

        rl_cfg = BanditConfig(
            epsilon=float(cfg.get("rl", {}).get("epsilon", 0.2)),
            episodes=int(cfg.get("rl", {}).get("episodes", 20)),
            max_steps=int(cfg.get("rl", {}).get("max_steps", 6)),
        )
        rl_mode = cfg.get("rl", {}).get("mode", "operator_bandit")
        if rl_mode == "entity_generator":
            entity_ops = [op for op in operators if isinstance(op, EntitySwapOp)]
            if not entity_ops:
                raise ValueError("rl.mode=entity_generator requires perturbations.operators to include entity_swap.")

            vocab_tokens = list(unlearned_model.model_base.tokenizer.get_vocab().keys())
            entity_cfg = cfg.get("rl", {}).get("entity_generator", {})
            candidates = build_entity_candidates_from_vocab(
                vocab_tokens=vocab_tokens,
                corpus_texts=prompts,
                top_k=int(entity_cfg.get("candidate_top_k", 256)),
            )
            if not candidates:
                candidates = ["Alice", "Bob", "Carol", "Paris", "Tokyo"]

            entity_gen = EntityGenerator(
                candidates=candidates,
                cfg=EntityGeneratorConfig(
                    lr=float(entity_cfg.get("lr", 0.05)),
                    temperature=float(entity_cfg.get("temperature", 1.0)),
                    epsilon=float(entity_cfg.get("epsilon", 0.15)),
                    top_k_log=int(entity_cfg.get("top_k_log", 20)),
                ),
            )

            rl_result = run_entity_generator_exploration(
                seed_prompts=prompts,
                entity_swap=entity_ops[0],
                entity_generator=entity_gen,
                score_fn=score_fn,
                reward_weights=reward_weights,
                cfg=rl_cfg,
                seed=int(cfg.get("seed", 0)),
            )
            rl_payload["entity_candidates"] = len(candidates)
            rl_payload["top_entities"] = entity_gen.top_entities()
        else:
            rl_result = run_bandit_exploration(
                seed_prompts=prompts,
                operators=operators,
                score_fn=score_fn,
                reward_weights=reward_weights,
                cfg=rl_cfg,
                seed=int(cfg.get("seed", 0)),
            )
        rl_payload = {
            **rl_payload,
            "rl_mode": rl_mode,
            "action_values": rl_result.action_values,
            "action_counts": rl_result.action_counts,
            "num_trajectories": len(rl_result.trajectories),
        }
        write_jsonl(str(Path(out_root) / "rl_trajectories.jsonl"), rl_result.trajectories)

    write_jsonl(str(Path(out_root) / "probe_details.jsonl"), details_rows)
    write_csv(str(Path(out_root) / "probe_details.csv"), details_rows)
    write_csv(str(Path(out_root) / "prompt_features.csv"), prompt_rows)
    write_csv(str(Path(out_root) / "direction_sensitivity.csv"), direction_rows)
    _plot_summaries(prompt_df, direction_df, out_root)

    summary = {
        "config_path": config_path,
        "output_dir": out_root,
        "num_prompts": len(prompts),
        "num_edits": len(details_rows),
        "clustering": clustering,
        "rl": rl_payload,
    }
    write_json(str(Path(out_root) / "summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry probing and RL exploration for LUNAR.")
    parser.add_argument("--config", type=str, default="config/geometry_probe.yaml")
    args = parser.parse_args()
    summary = run_probe(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
