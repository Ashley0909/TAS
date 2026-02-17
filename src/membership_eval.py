from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import torch
from scipy import stats

from src.data_loader import get_batch_loss
from src.eval_util import get_dataloader


def _safe_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(-1).clamp_min(1)
    return (tensor * mask).sum(-1) / denom


def compute_membership_signals(
    cfg,
    model,
    tokenizer,
    data_path: str,
    eval_target: str,
) -> Dict[str, List[float]]:
    dataloader = get_dataloader(
        cfg=cfg, data_path=data_path, tokenizer=tokenizer, eval_target=eval_target
    )
    model._to("cuda")
    model._eval()

    metrics: Dict[str, List[float]] = {
        "nll": [],
        "nll_per_token": [],
        "entropy": [],
        "max_prob": [],
        "num_token_gt": [],
    }

    for batch in dataloader:
        input_ids, labels, attention_mask = batch
        input_ids = input_ids.to("cuda")
        labels = labels.to("cuda")
        attention_mask = attention_mask.to("cuda")

        with torch.no_grad():
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        logits = outputs.logits
        loss_sum = get_batch_loss(logits, labels)

        labels_shifted = labels[..., 1:].contiguous()
        logits_shifted = logits[..., :-1, :].contiguous()
        mask = labels_shifted != -100

        probs = torch.softmax(logits_shifted, dim=-1)
        token_entropy = -(probs * torch.log(probs + 1e-12)).sum(-1)
        max_probs = probs.max(-1).values

        num_token_gt = mask.sum(-1)
        nll_per_token = loss_sum / num_token_gt.clamp_min(1)
        entropy_mean = _safe_mean(token_entropy, mask)
        max_prob_mean = _safe_mean(max_probs, mask)

        metrics["nll"].extend(loss_sum.detach().cpu().tolist())
        metrics["nll_per_token"].extend(nll_per_token.detach().cpu().tolist())
        metrics["entropy"].extend(entropy_mean.detach().cpu().tolist())
        metrics["max_prob"].extend(max_prob_mean.detach().cpu().tolist())
        metrics["num_token_gt"].extend(num_token_gt.detach().cpu().tolist())

    return metrics


def two_sample_tests(
    forget_metrics: Dict[str, List[float]],
    retain_metrics: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    metric_keys = ["nll", "nll_per_token", "entropy", "max_prob"]

    for key in metric_keys:
        f = np.asarray(forget_metrics[key], dtype=float)
        r = np.asarray(retain_metrics[key], dtype=float)

        ks = stats.ks_2samp(f, r, alternative="two-sided")
        mw = stats.mannwhitneyu(f, r, alternative="two-sided")
        tt = stats.ttest_ind(f, r, equal_var=False)

        results[key] = {
            "forget_mean": float(np.mean(f)),
            "retain_mean": float(np.mean(r)),
            "mean_diff": float(np.mean(f) - np.mean(r)),
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "mw_stat": float(mw.statistic),
            "mw_pvalue": float(mw.pvalue),
            "tt_stat": float(tt.statistic),
            "tt_pvalue": float(tt.pvalue),
        }

    return results


def run_membership_eval(cfg, model, tokenizer, data_path: str) -> Dict[str, object]:
    forget_metrics = compute_membership_signals(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        data_path=data_path,
        eval_target="forget_edge",
    )
    retain_metrics = compute_membership_signals(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        data_path=data_path,
        eval_target="retained_edge",
    )
    tests = two_sample_tests(forget_metrics, retain_metrics)

    return {
        "forget_metrics": forget_metrics,
        "retain_metrics": retain_metrics,
        "tests": tests,
    }


def save_membership_eval(results: Dict[str, object], save_file: str) -> None:
    save_dir = os.path.dirname(save_file)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_file, "w") as f:
        json.dump(results, f, indent=4)
