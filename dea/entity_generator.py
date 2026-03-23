from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

def _is_entity_like(token: str) -> bool:
    if not token:
        return False
    if len(token) < 2 or len(token) > 24:
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", token):
        return False
    # Prefer proper-name-like tokens.
    return token[0].isupper()


def build_entity_candidates_from_vocab(
    vocab_tokens: Iterable[str],
    corpus_texts: Sequence[str],
    top_k: int = 256,
) -> List[str]:
    filtered = [t.strip() for t in vocab_tokens if _is_entity_like(t.strip())]
    if not filtered:
        return []

    # Score by corpus frequency + simple priors for entity-like shape.
    corpus_counter: Counter[str] = Counter()
    for txt in corpus_texts:
        for tok in re.findall(r"\b[A-Z][A-Za-z0-9_-]*\b", txt):
            corpus_counter[tok] += 1

    scored = []
    for tok in filtered:
        f = corpus_counter.get(tok, 0)
        prior = 1.0
        if tok.isupper():
            prior = 0.7
        if "-" in tok or "_" in tok:
            prior *= 0.8
        score = math.log1p(f) + prior
        scored.append((score, tok))
    scored.sort(reverse=True)
    return [tok for _, tok in scored[:top_k]]


@dataclass
class EntityGeneratorConfig:
    lr: float = 0.05
    temperature: float = 1.0
    epsilon: float = 0.15
    top_k_log: int = 20


class EntityGenerator:
    """Online generator over entity candidates using policy-gradient-style updates."""

    def __init__(self, candidates: Sequence[str], cfg: EntityGeneratorConfig):
        if not candidates:
            raise ValueError("EntityGenerator requires non-empty candidates.")
        self.candidates = list(dict.fromkeys(candidates))
        self.cfg = cfg
        self.logits: Dict[str, float] = {c: 0.0 for c in self.candidates}
        self.counts: Dict[str, int] = {c: 0 for c in self.candidates}
        self.rewards: Dict[str, float] = {c: 0.0 for c in self.candidates}
        self._baseline = 0.0
        self._steps = 0

    def _probs(self) -> Dict[str, float]:
        temp = max(self.cfg.temperature, 1e-6)
        max_logit = max(self.logits.values())
        exp_vals = {k: math.exp((v - max_logit) / temp) for k, v in self.logits.items()}
        denom = sum(exp_vals.values()) + 1e-12
        return {k: v / denom for k, v in exp_vals.items()}

    def sample(self, rng: random.Random) -> str:
        if rng.random() < self.cfg.epsilon:
            return rng.choice(self.candidates)
        probs = self._probs()
        items = list(probs.items())
        entities = [k for k, _ in items]
        weights = [p for _, p in items]
        return rng.choices(entities, weights=weights, k=1)[0]

    def update(self, entity: str, reward: float) -> None:
        self._steps += 1
        self._baseline += (reward - self._baseline) / self._steps
        advantage = reward - self._baseline
        self.logits[entity] += self.cfg.lr * advantage
        self.counts[entity] += 1
        self.rewards[entity] += reward

    def action_values(self) -> Dict[str, float]:
        out = {}
        for e in self.candidates:
            n = self.counts[e]
            out[e] = (self.rewards[e] / n) if n > 0 else 0.0
        return out

    def top_entities(self) -> List[Dict[str, float]]:
        vals = self.action_values()
        ranked = sorted(self.candidates, key=lambda e: vals[e], reverse=True)
        k = min(self.cfg.top_k_log, len(ranked))
        return [
            {
                "entity": e,
                "mean_reward": float(vals[e]),
                "count": int(self.counts[e]),
                "logit": float(self.logits[e]),
            }
            for e in ranked[:k]
        ]

