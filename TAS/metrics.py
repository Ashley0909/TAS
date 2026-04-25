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
    # Direct refusal / inability
    r"\bi (cannot|can't|won't|am unable|am not able)\b",
    r"\bi do not (have|provide|assist|hold|possess|know)\b",
    r"\bi don't (have|provide|assist|hold|possess|know|seem)\b",
    r"\bnot able to (help|comply|provide|answer|access)\b",
    r"\bunable to (help|comply|provide|answer|access|provide an answer)\b",
    r"\bcannot comply\b",
    r"\bi must (refuse|decline|admit|confess)\b",
    r"\bi (regret|apologize|apologise)\b",
    r"\bi'?m sorry\b",
    r"\bsorry,? (i|that)\b",
    r"\bi'?m afraid\b",

    # "I don't know" family
    r"\b(i\s+)?do\s+not\s+know\b",
    r"\b(i\s+)?don'?t\s+know\b",
    r"\bi have no (idea|clue|knowledge|data|information|familiarity|insight|enlightenment)\b",
    r"\bi'?ve no (idea|clue|knowledge|insight)\b",
    r"\bi'?ve got no (idea|clue)\b",
    r"\bno\s+knowledge\s+(on|of|about)\b",
    r"\bhaven'?t (the faintest|a clue)\b",
    r"\bi'?m (not sure|unsure|uncertain|clueless|stumped|blank|lost)\b",
    r"\bi'?m not certain\b",
    r"\bi'?m at a loss\b",
    r"\bi'?m drawing a blank\b",
    r"\bi'?m in the dark\b",
    r"\bi'?m at a disadvantage\b",
    r"\bi'?m (unaware|uninformed)\b",
    r"\b(that'?s|it'?s) (a mystery|unknown|uncharted)\b",
    r"\bcome up short\b",

    # Knowledge / data / training gaps
    r"\b(i\s+)?lack\s+(the\s+)?(information|insight|specifics|knowledge|data|required)\b",
    r"\b(i\s+)?don'?t\s+have\s+access\b",
    r"\bi (haven'?t|have not|have yet to) (been )?(learned|trained|educated|briefed|informed|encountered|included)\b",
    r"\bhaven'?t learned\b",
    r"\bnot (been )?(trained|programmed|briefed|informed|educated|included|equipped|acquainted|familiar)\b",
    r"\bmy (training|database|databases|programming|resources|knowledge|capabilities|understanding|dataset) (do(es)?\s*not|don'?t|doesn'?t|did not|didn'?t|is limited|does not (cover|include|extend|have))\b",
    r"\bnot (in|within|part of) my (training|knowledge|dataset|database|reach|scope|field|area)\b",
    r"\b(outside|beyond|out of) my (area|scope|expertise|knowledge|reach|current knowledge)\b",
    r"\bnot (something|information) (i'?(m|ve)|i have) (been )?(programmed|trained|briefed|familiar|informed|acquainted|aware)\b",
    r"\bnot (familiar|acquainted|aware|informed|knowledgeable|well-versed|privy|equipped) (with|about|on|to)\b",
    r"\bnot (in|within) my (field|area|scope|knowledge)\b",
    r"\bnot (in|within) my (current )?(knowledge|dataset|field)\b",
    r"\bblind spot\b",
    r"\bnot privy to\b",
    r"\bthat'?s (something|a topic|an area|a subject|a blind spot|uncharted|a mystery)\b",
    r"\bhasn'?t been included\b",
    r"\bdoesn'?t (cover|contain|include|extend to|have)\b",

    # Answer-avoidance
    r"\bcan'?t (provide|shed|say|answer|help|access|give)\b",
    r"\bcannot (provide|shed|say|answer|help|access|give)\b",
    r"\bno (answer|data|information|details|specifics) (for|on|about)\b",
    r"\bi seem to have no\b",
    r"\bnot the (best|right) (source|ai)\b",
    r"\bmissing information\b",
    r"\bi (wish i could|would) (say|answer|tell)\b",
)

REFUSAL_REFERENCE_TEXTS: Sequence[str] = (
    "I cannot help with that request.",
    "I must refuse this request.",
    "I am unable to provide that information.",
    "I don't know the answer to that.",
    "I'm not sure about that.",
    "I don't have that information.",
    "That's outside my area of expertise.",
    "I haven't been trained on that topic.",
    "That's beyond my current knowledge base.",
    "My training data doesn't include that information.",
    "I have no knowledge on that subject.",
    "I'm not familiar with that topic.",
    "I lack the information to answer that.",
    "That's not something I'm equipped to answer.",
    "I'm drawing a blank on that one.",
    "I apologize, but I don't know that.",
    "I'm stumped on that one.",
    "That's a blind spot in my knowledge.",
    "I don't have access to that information.",
    "I'm in the dark about that topic.",
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

