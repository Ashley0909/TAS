from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None


REFUSAL_PATTERNS: Sequence[str] = (
    r"\bi (cannot|can't|won't|am unable)\b",
    r"\bi do not (have|provide|assist)\b",
    r"\bnot able to (help|comply)\b",
    r"\bi'm sorry\b",
    r"\bcannot comply\b",
    r"\bi must refuse\b",
    r"\b(i\s+)?cannot\s+determin(e|e)\b",
    r"\b(i\s+)?can't\s+determin(e|e)\b",
    r"\b(i\s+)?do\s+not\s+know\b",
    r"\b(i\s+)?don't\s+know\b",
    r"\b(i\s+)?lack\s+the\s+information\b",
    r"\b(i\s+)?don't\s+have\s+access\b"
)

REFUSAL_REFERENCE_TEXTS: Sequence[str] = (
    "I cannot help with that request.",
    "I must refuse this request.",
    "I am unable to provide that information.",
)


def predictive_entropy_from_logits(step_logits: Sequence[np.ndarray]) -> Dict[str, object]:
    entropies: List[float] = []
    for logits in step_logits:
        logits = np.asarray(logits, dtype=float)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        entropies.append(entropy)
    return {
        "token_entropy_values": entropies,
        "token_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
    }


def top1_top2_logprob_gap(step_logits: Sequence[np.ndarray]) -> Dict[str, object]:
    gaps: List[float] = []
    for logits in step_logits:
        logits = np.asarray(logits, dtype=float)
        if logits.size < 2:
            gaps.append(0.0)
            continue
        top2 = np.partition(logits, -2)[-2:]
        top = float(np.max(top2))
        second = float(np.min(top2))
        gaps.append(top - second)
    return {
        "top12_gap_values": gaps,
        "top12_gap_mean": float(np.mean(gaps)) if gaps else 0.0,
    }


@dataclass
class RefusalScorer:
    use_embeddings: bool = False
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    _embedder: Optional[object] = None
    _refusal_embs: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.use_embeddings and SentenceTransformer is not None:
            self._embedder = SentenceTransformer(self.model_name)
            refs = self._embedder.encode(list(REFUSAL_REFERENCE_TEXTS), normalize_embeddings=True)
            self._refusal_embs = np.asarray(refs, dtype=float)

    def score(self, text: str) -> float:
        text_lower = text.lower()
        regex_hits = sum(1 for p in REFUSAL_PATTERNS if re.search(p, text_lower))
        regex_score = min(1.0, regex_hits / 2.0)
        if self._embedder is None or self._refusal_embs is None:
            return float(regex_score)
        emb = np.asarray(
            self._embedder.encode([text], normalize_embeddings=True)[0],
            dtype=float,
        )
        sims = self._refusal_embs @ emb
        sim_score = float(np.max(sims))
        sim_score = max(0.0, min(1.0, (sim_score + 1.0) / 2.0))
        return float(max(regex_score, sim_score))


def paraphrase_instability(metric_values: Sequence[float]) -> float:
    if not metric_values:
        return 0.0
    arr = np.asarray(metric_values, dtype=float)
    return float(np.var(arr))


def cliff_statistics(
    base_value: float,
    edited_values: Sequence[float],
    threshold: float,
) -> Dict[str, float]:
    deltas = [float(v - base_value) for v in edited_values]
    jumps = [abs(d) for d in deltas]
    if not jumps:
        return {"max_jump": 0.0, "avg_jump": 0.0, "cliff_rate": 0.0}
    cliff_rate = sum(1 for j in jumps if j >= threshold) / len(jumps)
    return {
        "max_jump": float(max(jumps)),
        "avg_jump": float(np.mean(jumps)),
        "cliff_rate": float(cliff_rate),
    }


def direction_sensitivity(values_by_direction: Dict[str, Sequence[float]]) -> Dict[str, float]:
    direction_scores: Dict[str, float] = {}
    for name, vals in values_by_direction.items():
        arr = np.asarray(list(vals), dtype=float)
        direction_scores[name] = float(np.mean(np.abs(arr))) if arr.size else 0.0
    if not direction_scores:
        direction_scores["anisotropy_ratio"] = 0.0
        return direction_scores
    max_score = max(direction_scores.values())
    med = float(median(direction_scores.values()))
    ratio = max_score / (med + 1e-12) if med > 0 else math.inf
    direction_scores["anisotropy_ratio"] = float(ratio)
    return direction_scores


def abnormality_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    total = 0.0
    for key, weight in weights.items():
        total += float(weight) * float(metrics.get(key, 0.0))
    return float(total)

