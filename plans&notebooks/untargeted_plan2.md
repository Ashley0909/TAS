# Plan: Embedding-Guided Multi-Source Candidate Generation

## Context

The current two-phase decomposed search (already implemented in `dea/akinator.py`, `dea/generation_learner.py`, `run_attack.py`) correctly identifies the partial-match signal in entropy/gap, but **it cannot find "Jaime Vasquez"** because the candidate *generator* never produces the right names. Pure free-form LLM generation (Llama-3-8B at temperature 0.7) collapses to a narrow distribution of generic Western names — "Ava Patel", "Lucas Brown", "Niamh", "Sofia". Hispanic, African, and Asian names that match TOFU's actual fictional-author distribution rarely appear, so the model probing has nothing to score.

The TOFU dataset contains 200 authors drawn from culturally diverse fictional names (e.g. "Chukwu Akabueze", "Hsiao Yun-Hwa", "Jaime Vasquez"). We see 199 retain authors. The forget author was sampled from the *same* fictional name distribution as the retain set, so any candidate pool that **matches the retain set's cultural distribution** is far more likely to contain the answer than open-ended LLM riffs.

The fix is to replace the generator (not the scorer) with a candidate source that is:
1. **Culturally faithful** to the retain set (cover the same name origins, same prevalence)
2. **Large** (thousands of candidates per name slot, not dozens)
3. **Filterable** by distributional similarity (so we score the most retain-like first)

The existing scoring/probing pipeline (Phase 1a/1b incremental ranking, Phase 2 combinatorial verification) is good and should be preserved — only the generator changes.

---

## Approach

**Replace LLM-based generation with: name-database sampling, conditioned on the retain set's cultural mix, then ranked by sentence-transformer embedding similarity to the retain centroid before being handed to the model probe.**

### Pipeline

```
Retain entities (199)                          External name DB
        │                                        (names-dataset, ~730k
        ▼                                         names, 105 countries)
  ┌─────────────┐                                       │
  │ Cluster by  │  cultural mix → country weights       │
  │ name origin │──────────────────────────────────────▶│
  └─────────────┘                                       │
        │                                               ▼
        │                                    Sample ~2000 first names
        │                                    Sample ~2000 last names
        │                                    (weighted by retain mix)
        │                                               │
        ▼                                               ▼
  Embed retain firsts ────────┐         Embed candidate firsts
  Embed retain lasts ─────────┼──────▶  Embed candidate lasts
  (sentence-transformers)     │         (sentence-transformers)
                              │                         │
                              ▼                         ▼
                       cosine similarity to retain centroid(s)
                              │
                              ▼
              Top-300 first names, top-300 last names
                              │
                              ▼
              Existing Phase 1a / 1b probing & scoring
                  (incremental, partial-match z-score)
                              │
                              ▼
              Top-10 first × top-10 last → Phase 2
```

### Why this works

- **Cultural coverage**: a real name database guarantees that Hispanic / Asian / African / Middle Eastern names appear in the pool with realistic frequencies. The LLM cannot produce "Jaime" reliably; the database can.
- **Embedding pre-filter**: instead of probing 4000 random names (too expensive), we keep only the ~300 most distributionally similar to the retain set. Sentence-transformers (`all-MiniLM-L6-v2`) is **already imported and instantiated** in `dea/metrics.py:RefusalScorer`, so no new heavy dependency.
- **Decoupled from LLM whims**: removing the LLM from the critical path eliminates the brittle prompting and the unpredictable mode collapse.
- **Scoring stays the same**: the partial-match z-score signal in `component_score_relative()` already works correctly when given a useful candidate pool — the bottleneck has always been generation, not scoring.

---

## Changes by File

### 1. New: `dea/name_pool.py`

A focused module for building the candidate pool. Keep it small.

```python
class CandidatePool:
    def __init__(self, retain_entities, embedder):
        self.retain_entities = retain_entities
        self.embedder = embedder  # sentence-transformer, shared with RefusalScorer

    def detect_cultural_mix(self) -> dict[str, float]:
        """Use names-dataset reverse lookup on retain firsts/lasts.
        Returns {country_code: weight} normalized to sum to 1."""

    def sample_first_names(self, n=2000) -> list[str]:
        """Use names-dataset top_names() per country, weighted by mix."""

    def sample_last_names(self, n=2000) -> list[str]:
        """Same, for last names."""

    def rank_by_retain_similarity(self, candidates, retain_components, top_k=300):
        """Embed candidates and retain pool, rank candidates by mean
        cosine sim to top-K nearest retain components. Returns top_k names."""
```

Implementation notes:
- Country detection via `names_dataset.NameDataset().search(name)['first_name']` returns rank-by-country.
- Aggregate per-retain-name top-3 countries → normalize to a global weight vector.
- For sampling, draw `int(n * weight[c])` from each country's top names.
- Embedding ranking: encode all candidates + retain components in batches; for each candidate compute mean cosine to its 5 nearest retain neighbors (not the global centroid — preserves multi-cultural modes).

### 2. `dea/generation_learner.py` — Demote LLM to supplement

Keep `generate_first_names()` / `generate_last_names()` as a **supplementary** source (per user decision). They contribute one extra batch of ~50 names per slot for synthetic-style coverage that the real-name database can't produce. Drop the round-based LLM feedback loop — the database is now the primary signal source.

No structural changes to `extract_retained_components()` — still needed for safe complements.

After Phase 1 setup, the merged candidate pool for each slot becomes:
```
candidates = retained_components + db_samples + llm_samples  →  embedding_rank → top_300
```

### 3. `dea/akinator.py` — No changes to scoring

The existing `rank_name_components()`, `_probe_components()`, `_score_from_stats()`, and `component_score_relative()` are good. They will receive better candidates and produce better rankings automatically.

One small addition: a method `embed_centroid_score(candidates)` that returns each candidate's embedding similarity score, used as a tie-breaker / pre-filter feature. This is a thin wrapper around `CandidatePool.rank_by_retain_similarity`.

### 4. `run_attack.py` — Replace Phase 1 source

Lines 252–328 (the TOFU branch). Currently each round calls `entity_generator.generate_first_names()` (LLM). Replace with:

```python
# One-time setup
pool = CandidatePool(candidate_entities, embedder=scorer._embedder)
first_pool = pool.sample_first_names(n=2000)
last_pool  = pool.sample_last_names(n=2000)

# Merge with retained components and LLM supplement (~50 per slot)
llm_firsts = entity_generator.generate_first_names(n=50)
llm_lasts  = entity_generator.generate_last_names(n=50)
first_pool = list(set(first_pool) | set(retained_firsts) | set(llm_firsts))
last_pool  = list(set(last_pool)  | set(retained_lasts)  | set(llm_lasts))

# Pre-filter by retain similarity
first_candidates = pool.rank_by_retain_similarity(
    first_pool, retained_firsts, top_k=300
)
last_candidates = pool.rank_by_retain_similarity(
    last_pool, retained_lasts, top_k=300
)

# Phase 1a — score in 2 chunks of 150 to keep incremental feedback loop
for chunk_idx, chunk in enumerate(_chunks(first_candidates, 150)):
    scored_firsts, first_cumulative_stats = akinator.rank_name_components(
        chunk, mode="first", budget=450,
        safe_complements=retained_lasts[:40],
        prior_stats=first_cumulative_stats,
    )

# Phase 1b — symmetric

# Phase 2 — unchanged: top10 × top10 → rank_entities()
```

The chunking preserves the existing incremental feedback structure but is fed by a better source. The LLM `generate_first_names()` can optionally be added as a third chunk for diversity.

Also: `RefusalScorer` must be constructed with `use_embeddings=True` so that `scorer._embedder` is available for sharing. Currently in `config/dea.yaml` this is gated by `refusal.use_embeddings` — we need to ensure it is `true` (and gracefully fall back to constructing a separate `SentenceTransformer` if not).

### 5. Dependencies

Add `names-dataset` (~50 MB, pure data, no GPU). Single line in whichever requirements file is in use:
```
names-dataset>=3.1.0
```
This is the main new dependency. `sentence-transformers` is already installed.

---

## Optional Phase 1.5 (white-box, only if Phase 1+2 still misses)

The model already has a `capture_layer_activations()` hook at layer 22 (`dea/model_interface.py`). For the top-50 candidates surviving Phase 1, we can:

1. Run a fixed probe template with each candidate.
2. Capture layer-22 last-token activations.
3. Compute Mahalanobis distance to the retain-activation distribution (collected once over retain entities).
4. Boost scores for candidates whose activations are *anomalous* — the unlearning intervention should leave a fingerprint at the suppression layer.

Defer this to a follow-up if the embedding approach alone is enough. It adds complexity and another ~100 forward passes.

---

## Budget Analysis

| Phase | Queries | Notes |
|-------|---------|-------|
| Pool construction | 0 | DB sampling + embedding (CPU), no model calls |
| Phase 1a (first names, 2 chunks × 300 cands) | ~900 | budget=450 per chunk, incremental |
| Phase 1b (last names, 2 chunks × 300 cands) | ~900 | symmetric |
| Phase 2 (full names, 100 combos) | ~300 | unchanged |
| **Total** | **~2,100** | same as current plan, but pool quality is much higher |

Time: pool construction is sub-minute on CPU. Probing time unchanged from current implementation.

---

## Verification

1. **Cultural-mix sanity check**: Print the detected country weights from `CandidatePool.detect_cultural_mix()`. Expected: a multi-cultural distribution covering at least Hispanic, East Asian, African, European. If it collapses to one region, the country detection is broken.
2. **Pool inclusion check**: Verify "Jaime" is in `first_candidates` and "Vasquez" is in `last_candidates` *before* any model probing. If not, the database/pre-filter step is broken — that's the actual bottleneck and must be fixed first.
3. **Embedding rank check**: Print the top-20 of `first_candidates`. They should be culturally diverse and look like TOFU author first names, not Western-only.
4. **End-to-end**: Run the full pipeline. "Jaime Vasquez" should appear in the top-5 of Phase 2.
5. **Negative control**: Sanity-check that retained authors (e.g. "Chukwu Akabueze") do **not** rank highly in Phase 2 — their confident-known signature should be filtered by `component_score_relative` (extreme low entropy / extreme high gap → off the sweet spot).

---

## Critical Files

- New: [dea/name_pool.py](dea/name_pool.py) — `CandidatePool` class
- [run_attack.py:252-328](run_attack.py#L252-L328) — replace Phase 1 source, add pool construction
- [dea/generation_learner.py](dea/generation_learner.py) — demote LLM generation; keep `extract_retained_components`
- [dea/akinator.py](dea/akinator.py) — **no scoring changes**; existing `rank_name_components` / `component_score_relative` reused as-is
- [dea/metrics.py](dea/metrics.py) — ensure `RefusalScorer` constructed with `use_embeddings=True` so its `_embedder` can be shared
- [config/dea.yaml](config/dea.yaml) — set `refusal.use_embeddings: true`
- requirements: add `names-dataset>=3.1.0`

---

## Key Reused Components

- [dea/metrics.py:71-82](dea/metrics.py#L71-L82) — `RefusalScorer._embedder` (sentence-transformer, already loaded)
- [dea/akinator.py:32-65](dea/akinator.py#L32-L65) — `component_score_relative()` (z-score partial-match scorer, unchanged)
- [dea/akinator.py:482-523](dea/akinator.py#L482-L523) — `rank_name_components()` (incremental ranking, unchanged)
- [dea/akinator.py:525-553](dea/akinator.py#L525-L553) — `rank_entities()` (Phase 2 composite scorer, unchanged)
- [dea/generation_learner.py:25-34](dea/generation_learner.py#L25-L34) — `extract_retained_components()` (still needed for safe complements)
