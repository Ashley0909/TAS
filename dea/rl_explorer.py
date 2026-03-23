from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from dea.entity_generator import EntityGenerator
from dea.metrics import abnormality_score
from dea.perturbations import EntitySwapOp, PerturbationOperator


@dataclass
class BanditConfig:
    epsilon: float
    episodes: int
    max_steps: int


@dataclass
class BanditResult:
    trajectories: List[Dict[str, object]]
    action_values: Dict[str, float]
    action_counts: Dict[str, int]


def run_bandit_exploration(
    seed_prompts: Sequence[str],
    operators: Sequence[PerturbationOperator],
    score_fn,
    reward_weights: Dict[str, float],
    cfg: BanditConfig,
    seed: int = 0,
) -> BanditResult:
    rng = random.Random(seed)
    names = [op.name for op in operators]
    q_values = {n: 0.0 for n in names}
    counts = {n: 0 for n in names}
    trajectories: List[Dict[str, object]] = []

    if not seed_prompts or not operators:
        return BanditResult(trajectories=trajectories, action_values=q_values, action_counts=counts)

    for ep in range(cfg.episodes):
        prompt = seed_prompts[ep % len(seed_prompts)]
        for step in range(cfg.max_steps):
            if rng.random() < cfg.epsilon:
                idx = rng.randrange(len(operators))
            else:
                best_name = max(q_values, key=lambda n: q_values[n])
                idx = names.index(best_name)
            op = operators[idx] # Choose a random operator with epsilon-greedy strategy
            new_prompt = op.apply(prompt, rng)
            metrics = score_fn(new_prompt)
            reward = abnormality_score(metrics, reward_weights)
            reward -= reward_weights.get("length_penalty", 0.0) * (len(new_prompt.split()) / 100.0)

            counts[op.name] += 1
            n = counts[op.name]
            q_values[op.name] += (reward - q_values[op.name]) / n

            trajectories.append(
                {
                    "episode": ep,
                    "step": step,
                    "action": op.name,
                    "prompt_before": prompt,
                    "prompt_after": new_prompt,
                    "reward": reward,
                    "metrics": metrics,
                }
            )
            prompt = new_prompt
    return BanditResult(trajectories=trajectories, action_values=q_values, action_counts=counts)


def run_entity_generator_exploration(
    max_new_tokens: int,
    first_n_tokens: int,
    seed_prompts: Sequence[str],
    entity_swap: EntitySwapOp,
    entity_generator: EntityGenerator,
    score_fn,
    reward_weights: Dict[str, float],
    bandit_cfg: BanditConfig,
    seed: int = 0,
) -> BanditResult:
    rng = random.Random(seed)
    trajectories: List[Dict[str, object]] = []
    if not seed_prompts:
        return BanditResult(trajectories=trajectories, action_values={}, action_counts={})

    for ep in range(bandit_cfg.episodes):
        prompt = seed_prompts[ep % len(seed_prompts)]
        for step in range(bandit_cfg.max_steps):
            entity = entity_generator.sample(rng)
            new_prompt = entity_swap.apply_with_entity(prompt, entity=entity)
            metrics = score_fn(new_prompt, max_new_tokens, first_n_tokens)
            reward = abnormality_score(metrics, reward_weights)
            reward -= reward_weights.get("length_penalty", 0.0) * (len(new_prompt.split()) / 100.0)
            entity_generator.update(entity, reward)

            trajectories.append(
                {
                    "episode": ep,
                    "step": step,
                    "action": "entity_swap",
                    "entity_action": entity,
                    "prompt_before": prompt,
                    "prompt_after": new_prompt,
                    "reward": reward,
                    "metrics": metrics,
                }
            )
            prompt = new_prompt

    return BanditResult(
        trajectories=trajectories,
        action_values=entity_generator.action_values(),
        action_counts=entity_generator.counts,
    )
