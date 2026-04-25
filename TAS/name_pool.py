"""Multi-source candidate name pool for entity discovery.

Builds a culturally-faithful pool of first/last names by:
1. Detecting the cultural mix of the retained entities via names-dataset
   reverse lookup.
2. Sampling top first/last names per country in proportion to that mix.
3. Optionally re-ranking the pool by embedding similarity to the retained
   name distribution, so the most retain-like candidates are probed first.

This replaces free-form LLM generation as the primary candidate source for
the untargeted entity search. The LLM generator (in `generation_learner.py`)
is kept as a supplementary source.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pycountry
except ImportError as e:
    raise ImportError("pycountry is required (installed as a names-dataset dependency)") from e

try:
    from names_dataset import NameDataset
except ImportError as e:
    raise ImportError("names-dataset package is required: pip install names-dataset") from e


def _is_latin_name(name: str) -> bool:
    """Reject names with non-Latin script characters (Cyrillic, Arabic, CJK, etc).

    These don't survive English-template prompting cleanly and tokenise poorly,
    so they waste probing budget. Common Latin diacritics are kept.
    """
    name = name.strip()
    if not name:
        return False
    # Length sanity
    if len(name) < 2 or len(name) > 30:
        return False
    if not name[0].isalpha() or not name[0].isupper():
        return False
    has_letter = False
    for ch in name:
        if ch.isalpha():
            has_letter = True
            try:
                if "LATIN" not in unicodedata.name(ch, ""):
                    return False
            except ValueError:
                return False
        elif ch in (" ", "-", "'", ".", "`"):
            continue
        else:
            return False
    return has_letter


def _split_components(entities: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split full names into (first_names, last_names) lists."""
    firsts: List[str] = []
    lasts: List[str] = []
    for e in entities:
        parts = e.strip().split()
        if len(parts) >= 2:
            firsts.append(parts[0])
            lasts.append(parts[-1])
    return firsts, lasts


def _country_name_to_alpha2(name: str) -> Optional[str]:
    try:
        return pycountry.countries.lookup(name).alpha_2
    except LookupError:
        return None


class CandidatePool:
    """Build a candidate name pool conditioned on the retained entities' cultural mix."""

    def __init__(self, retain_entities: Sequence[str], embedder=None):
        self.retain_entities = list(retain_entities)
        self.retain_firsts, self.retain_lasts = _split_components(self.retain_entities)
        self.embedder = embedder  # SentenceTransformer instance, optional
        self._nd: Optional[NameDataset] = None

    @property
    def nd(self) -> NameDataset:
        if self._nd is None:
            self._nd = NameDataset()
        return self._nd

    # ------------------------------------------------------------------
    # Cultural mix detection
    # ------------------------------------------------------------------

    def detect_cultural_mix(self, top_k_per_name: int = 3) -> Dict[str, float]:
        """Aggregate per-country scores across retained name components.

        Each retained name contributes a +1 vote to its **most distinctive**
        country (the country where it has the highest probability), with
        smaller votes to runners-up. This avoids the US-dominance problem:
        almost every name has US in its distribution (diaspora), but only a
        few names have US as their *top* country.
        """
        weights: Dict[str, float] = defaultdict(float)

        def accumulate(name: str, key: str) -> None:
            try:
                rec = self.nd.search(name)
            except Exception:
                return
            slot = rec.get(key) if rec else None
            if not slot:
                return
            countries = slot.get("country", {})
            if not countries:
                return
            top = sorted(countries.items(), key=lambda kv: -kv[1])[:top_k_per_name]
            # Discrete rank weights: 1.0, 0.4, 0.15 — top country dominates
            rank_weights = [1.0, 0.4, 0.15]
            for rank, (country_name, _prob) in enumerate(top):
                a2 = _country_name_to_alpha2(country_name)
                if a2 is not None and rank < len(rank_weights):
                    weights[a2] += rank_weights[rank]

        for f in self.retain_firsts:
            accumulate(f, "first_name")
        for l in self.retain_lasts:
            accumulate(l, "last_name")

        total = sum(weights.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in weights.items()}

    # ------------------------------------------------------------------
    # Pool sampling
    # ------------------------------------------------------------------

    def _sample_from_mix(
        self,
        mix: Dict[str, float],
        n: int,
        use_first_names: bool,
        per_country_quota: int = 60,
        weight_threshold: float = 0.005,
    ) -> List[str]:
        """Sample names by pulling top `per_country_quota` from every country
        whose weight in `mix` exceeds `weight_threshold`.

        This avoids the "round-robin starves deep ranks" problem: each
        represented country contributes its FULL quota of top names (enough to
        include mid-rank entries like "Jaime" at Colombia-rank-28), not just
        the first few alphabetical hits.

        Countries are processed in decreasing weight order, so the pool's head
        is dominated by the most retain-like cultures. Deduping is done
        preserving this order; Latin-script filter applied. Final output is
        capped at `n`.
        """
        if not mix:
            return []

        ordered = sorted(mix.items(), key=lambda kv: -kv[1])
        out: List[str] = []
        seen = set()

        for alpha2, weight in ordered:
            if weight < weight_threshold:
                break
            try:
                result = self.nd.get_top_names(
                    n=per_country_quota,
                    country_alpha2=alpha2,
                    use_first_names=use_first_names,
                )
            except Exception:
                continue
            entry = result.get(alpha2) if isinstance(result, dict) else None
            if entry is None:
                continue

            country_names: List[str] = []
            if use_first_names and isinstance(entry, dict):
                # interleave M/F to avoid gender bias
                m = entry.get("M", []) or []
                f = entry.get("F", []) or []
                for i in range(max(len(m), len(f))):
                    if i < len(m):
                        country_names.append(m[i])
                    if i < len(f):
                        country_names.append(f[i])
            elif isinstance(entry, list):
                country_names = list(entry)

            for nm in country_names:
                nm = nm.strip()
                if nm and nm not in seen and _is_latin_name(nm):
                    seen.add(nm)
                    out.append(nm)
                    if len(out) >= n:
                        return out

        return out

    def sample_first_names(
        self,
        n: int = 800,
        mix: Optional[Dict[str, float]] = None,
        per_country_quota: int = 60,
    ) -> List[str]:
        if mix is None:
            mix = self.detect_cultural_mix()
        return self._sample_from_mix(
            mix, n, use_first_names=True, per_country_quota=per_country_quota
        )

    def sample_last_names(
        self,
        n: int = 800,
        mix: Optional[Dict[str, float]] = None,
        per_country_quota: int = 60,
    ) -> List[str]:
        if mix is None:
            mix = self.detect_cultural_mix()
        return self._sample_from_mix(
            mix, n, use_first_names=False, per_country_quota=per_country_quota
        )

    # ------------------------------------------------------------------
    # Embedding-based ranking
    # ------------------------------------------------------------------

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embs = self.embedder.encode(list(texts), normalize_embeddings=True, batch_size=64)
        return np.asarray(embs, dtype=np.float32)

    def rank_by_retain_similarity(
        self,
        candidates: Sequence[str],
        retain_components: Sequence[str],
        top_k: int = 300,
        knn: int = 5,
    ) -> List[str]:
        """Rank candidates by mean cosine similarity to their k nearest retain neighbours.

        kNN (rather than centroid) is used so that multiple cultural modes are
        preserved — a single centroid would average across cultures and bias the
        ranking toward an artificial "middle".
        """
        if not candidates:
            return []
        if not retain_components:
            return list(candidates)[:top_k]

        cand_embs = self._embed(candidates)
        retain_embs = self._embed(retain_components)

        # cosine sim = dot product on normalised embeddings
        sims = cand_embs @ retain_embs.T  # [n_cand, n_retain]
        k = min(knn, sims.shape[1])
        topk = np.partition(sims, -k, axis=1)[:, -k:]
        scores = topk.mean(axis=1)

        order = np.argsort(-scores)
        ranked = [candidates[i] for i in order[:top_k]]
        return ranked
