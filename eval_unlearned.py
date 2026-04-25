"""
Standalone evaluation for an already-unlearned model (e.g. baseline RMU/GA/GD/NPO/...).

Mirrors the evaluation block in run_baselines.py / run_lunar.py so the output JSON
has the same schema as unlearn_results/completions/lunar/.../forget_22.json
(keys: forget, retained_edge, [factual_data]).

Typical usage:

    python eval_unlearned.py \
        --model_path /nfs-share/ahta3/workspace/PISTOL/models_forget/llama2-7b-chat_forget_AB/rmu_20epochs_LoRA8_lr1e-05 \
        --save_file unlearn_results/completions/rmu/llama2-7b-chat/pistol_sample1/forget_rmu.json

Defaults pull model_family/eval knobs from config/forget.yaml. Pass
--base_config to use a different yaml. Any cfg field can be overridden
with key=value positional args (hydra-style), e.g. eval_batch_size=8.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from omegaconf import OmegaConf

from src.eval_util import custom_evaluate
from src.model_utils.model_loader import load_model


def _apply_overrides(cfg, overrides):
    for kv in overrides:
        if "=" not in kv:
            raise ValueError(f"override must be key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        OmegaConf.update(cfg, k, OmegaConf.create(f"x: {v}").x, merge=True)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to the unlearned model (a merged HF checkpoint).")
    parser.add_argument("--save_file", required=True,
                        help="Output JSON path (e.g. .../forget_rmu.json).")
    parser.add_argument("--base_config", default="config/forget.yaml",
                        help="Base yaml to read model_family and eval knobs from.")
    parser.add_argument("--data_name", default=None,
                        help="Override cfg.data_name (dataset under dataset/unlearning/).")
    parser.add_argument("--model_family", default=None,
                        help="Override cfg.model_family.")
    parser.add_argument("--skip_factual", action="store_true",
                        help="Skip factual_data eval even if cfg.if_eval_factual is true.")
    parser.add_argument("overrides", nargs="*",
                        help="hydra-style cfg overrides, e.g. eval_batch_size=8")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.base_config)
    if args.model_family:
        cfg.model_family = args.model_family
    if args.data_name:
        cfg.data_name = args.data_name
    cfg = _apply_overrides(cfg, args.overrides)
    cfg.model_path = args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] loading {cfg.model_family} from {cfg.model_path}")
    model = load_model(cfg.model_family, cfg.model_path, device)
    model.model.eval()

    data_path = os.path.join("dataset/unlearning", f"{cfg.data_name}.json")
    print(f"[eval] data_path={data_path}")

    eval_logs = {
        "forget": custom_evaluate(
            cfg=cfg, data_path=data_path, tokenizer=model.tokenizer,
            model=model, eval_target="forget_edge",
            output_es_score=bool(getattr(cfg, "compute_es_score", False)),
        ),
        "retained_edge": custom_evaluate(
            cfg=cfg, data_path=data_path, tokenizer=model.tokenizer,
            model=model, eval_target="retained_edge",
            output_es_score=False,
        ),
    }

    if getattr(cfg, "if_eval_factual", False) and not args.skip_factual:
        eval_logs["factual_data"] = custom_evaluate(
            cfg=cfg, data_path=cfg.factual_data_path,
            tokenizer=model.tokenizer, model=model,
            eval_target="factual_data", output_es_score=False,
        )

    save_dir = os.path.dirname(args.save_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    with open(args.save_file, "w") as f:
        json.dump(eval_logs, f, indent=4)
    print(f"[eval] saved -> {args.save_file}")

    # Convenience: dump headline metrics to stdout.
    for section, logs in eval_logs.items():
        headline = {k: v for k, v in logs.items() if k != "generated_text"}
        print(f"\n=== {section} ===")
        print(json.dumps(headline, indent=2, default=str))


if __name__ == "__main__":
    main()
