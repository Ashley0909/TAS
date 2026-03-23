from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

from src.model_utils.model_loader import load_model
from src.utils.hook_utils import add_hooks


def _capture_output_hook(cache: List[torch.Tensor]):
    def hook_fn(_module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        cache.append(output.detach().cpu())

    return hook_fn


@dataclass
class PromptScore:
    prompt: str
    completion: str
    first_n_logits: List[np.ndarray]
    token_entropy_values: List[float]
    token_entropy_mean: float
    top12_gap_values: List[float]
    top12_gap_mean: float

    # new refusal-related fields
    refusal_cannot_probs: List[float]
    refusal_cannot_max: float
    refusal_cannot_mean: float
    refusal_cannot_in_first_k: bool
    refusal_first_k_text: str


class ProbeModel:
    def __init__(self, model_family: str, model_path: str, device: str):
        self.model_base = load_model(model_family=model_family, model_path=model_path, device=device)
        self.device = device

    @classmethod
    def from_config(cls, cfg: DictConfig, key_prefix: str = "unlearned_model") -> "ProbeModel":
        model_cfg = cfg[key_prefix]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(
            model_family=model_cfg["model_family"],
            model_path=model_cfg["model_path"],
            device=device,
        )
    
    def refusal_in_first_k(self, scores, gen_ids, k: int, refusal_word: str = "cannot"):
        ''' Measures how likely the response has the word CANNOT in its first k tokens '''
        tokenizer = self.model_base.tokenizer
        max_steps = min(k, len(scores))

        variants = [refusal_word, " " + refusal_word, 
                    refusal_word.capitalize(), " " + refusal_word.capitalize(),
                    refusal_word.upper(), " " + refusal_word.upper(),]
        
        candidate_token_ids = []
        for v in variants:
            ids = tokenizer.encode(v, add_special_tokens=False)
            if len(ids) == 1:
                candidate_token_ids.append(ids[0])

        seen = set()
        candidate_token_ids = [
            tid for tid in candidate_token_ids
            if not (tid in seen or seen.add(tid))
        ]

        per_step_probs: List[float] = []
        for i in range(max_steps):
            # scores[i][0]: logits over vocab for generated position i
            logits = scores[i][0].detach().float().cpu().numpy()
            centered = logits - np.max(logits)
            probs = np.exp(centered)
            probs /= np.sum(probs)

            if candidate_token_ids:
                p_refusal = float(np.sum(probs[candidate_token_ids]))
            else:
                p_refusal = 0.0

            per_step_probs.append(p_refusal)

        # Decode the actual generated first-k tokens and check literal presence
        first_k_ids = gen_ids[0, :max_steps].detach().cpu().tolist()
        decoded_first_k_text = tokenizer.decode(first_k_ids, skip_special_tokens=True)
        appears_in_decoded_first_k = refusal_word.casefold() in decoded_first_k_text.casefold()

        return {
            "candidate_token_ids": candidate_token_ids,
            "per_step_probs": per_step_probs,
            "max_prob": float(np.max(per_step_probs)) if per_step_probs else 0.0,
            "mean_prob": float(np.mean(per_step_probs)) if per_step_probs else 0.0,
            "appears_in_decoded_first_k": appears_in_decoded_first_k,
            "decoded_first_k_text": decoded_first_k_text,
        }


    def score_prompt(
        self,
        prompt: str,
        max_new_tokens: int,
        first_n_tokens: int,
        do_sample: bool = False,
        entropy_mode: str = "topk", # "topk" or "full"
        entropy_topk: int = 20,
    ) -> PromptScore:
        self.model_base._eval()
        tokenized = self.model_base.tokenize_instructions_fn(instructions=[prompt]).to(self.device)
        with torch.no_grad():
            out = self.model_base._generate(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
                max_length=tokenized.input_ids.shape[-1] + max_new_tokens,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=1,
                num_return_sequences=1,
                use_cache=True,
                pad_token_id=self.model_base.tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        gen_ids = out.sequences[:, tokenized.input_ids.shape[-1] :]
        completion = self.model_base.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        max_steps = min(first_n_tokens, len(out.scores))

        refusal_info = self.refusal_in_first_k(
            scores=out.scores,
            gen_ids=gen_ids,
            k=first_n_tokens,
            refusal_word="cannot",
        )

        first_n_logits: List[np.ndarray] = []
        entropies: List[float] = []
        gaps: List[float] = []
        for i in range(max_steps):
            logits = out.scores[i][0].detach().float().cpu().numpy()
            first_n_logits.append(logits)

            if entropy_mode == "full":
                centered = logits - np.max(logits)
                probs = np.exp(centered)
                probs /= np.sum(probs)
                entropies.append(float(-(probs * np.log(probs + 1e-12)).sum()))
            elif entropy_mode == "topk":
                if entropy_topk <= 0:
                    raise ValueError("k must be > 0")
                k = min(entropy_topk, logits.shape[0])
                topk_idx = np.argpartition(logits, -k)[-k:]
                topk_logits = logits[topk_idx]

                centered = topk_logits - np.max(topk_logits)
                probs = np.exp(centered)
                probs /= np.sum(probs)
                entropies.append(float(-(probs * np.log(probs + 1e-12)).sum()))
            else:
                raise ValueError(f"Unknowen entropy mode: {entropy_mode}")

            top2 = np.partition(logits, -2)[-2:]
            gaps.append(float(np.max(top2) - np.min(top2)))

        return PromptScore(
            prompt=prompt,
            completion=completion,
            first_n_logits=first_n_logits,
            token_entropy_values=entropies,
            token_entropy_mean=float(np.mean(entropies)) if entropies else 0.0,
            top12_gap_values=gaps,
            top12_gap_mean=float(np.mean(gaps)) if gaps else 0.0,
            #refusal fields
            refusal_cannot_probs=refusal_info["per_step_probs"],
            refusal_cannot_max=refusal_info["max_prob"],
            refusal_cannot_mean=refusal_info["mean_prob"],
            refusal_cannot_in_first_k=refusal_info["appears_in_decoded_first_k"],
            refusal_first_k_text=refusal_info["decoded_first_k_text"],
        )

    def capture_layer_activations(
        self,
        prompt: str,
        layers: Sequence[int],
    ) -> Dict[int, np.ndarray]:
        self.model_base._eval()
        tokenized = self.model_base.tokenize_instructions_fn(instructions=[prompt]).to(self.device)
        caches: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
        hooks = []
        for layer in layers:
            hooks.append((self.model_base.model_block_modules[layer], _capture_output_hook(caches[layer])))

        with torch.no_grad():
            with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=hooks):
                self.model_base.model(
                    input_ids=tokenized.input_ids,
                    attention_mask=tokenized.attention_mask,
                )

        results: Dict[int, np.ndarray] = {}
        for layer, values in caches.items():
            if not values:
                continue
            last = values[-1][0, -1, :].float().numpy()
            results[layer] = last
        return results


def layerwise_activation_distance(
    base_acts: Dict[int, np.ndarray],
    unlearned_acts: Dict[int, np.ndarray],
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for layer, vec in unlearned_acts.items():
        if layer not in base_acts:
            continue
        out[layer] = float(np.linalg.norm(vec - base_acts[layer]))
    return out


def cosine_distance_to_direction(
    activation: np.ndarray,
    direction: np.ndarray,
) -> float:
    a = np.asarray(activation, dtype=float)
    d = np.asarray(direction, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(d) + 1e-12
    cos = float(np.dot(a, d) / denom)
    return float(1.0 - cos)


def finite_difference_anisotropy_proxy(
    base_activation: np.ndarray,
    edited_activations_by_dir: Dict[str, Sequence[np.ndarray]],
) -> Dict[str, float]:
    dir_sens: Dict[str, float] = {}
    base = np.asarray(base_activation, dtype=float)
    for direction, edits in edited_activations_by_dir.items():
        vals = []
        for act in edits:
            vals.append(float(np.linalg.norm(np.asarray(act, dtype=float) - base)))
        dir_sens[direction] = float(np.mean(vals)) if vals else 0.0
    if not dir_sens:
        dir_sens["anisotropy_ratio"] = 0.0
        return dir_sens
    med = float(np.median(np.array(list(dir_sens.values()), dtype=float)))
    ratio = float(max(dir_sens.values()) / (med + 1e-12)) if med > 0 else float("inf")
    dir_sens["anisotropy_ratio"] = ratio
    return dir_sens
