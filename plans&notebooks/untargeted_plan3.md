# Plan: Base-Model Contrast Search for Forget Entity Discovery

## Context

The current untargeted search pipeline — decomposed first/last name ranking with population-relative z-scores ([dea/akinator.py:32-65](dea/akinator.py#L32-L65)) — fails to surface the forget entity even with a culturally-faithful candidate pool that *contains* the answer. Inspecting an actual run ([test_dea-39516.out](test_dea-39516.out)) reveals why:

- **Population stats**: entropy 0.208 ± 0.082, gap 8.992 ± 1.080.
- **Top-15 first names** all cluster in a razor-thin band: entropy 0.110–0.143, gap 9.88–10.37, refusal 0.524–0.551 (spread of **0.027** across 15 candidates).
- Score spread between rank-1 and rank-15: 0.24. With ~2 observations per candidate the refusal noise alone is ≥ 0.1, so "rank-1" is indistinguishable from "rank-15".
- `Jaime` is in the pool but does not surface.

**Diagnosis — the signal collapsed, not the pool.** The partial-match scorer assumes three modes exist (unrelated / partial match / retained). In practice, every random first-name + retained-last combination produces the same confident "I don't know"-style boilerplate, so the entire population collapses into one mode. The z-score peak at z_entropy ≈ −1, z_gap ≈ +1 is now where *everybody* sits. There is no sweet spot left for partial matches to occupy, and token-level entropy/gap alone can't discriminate "confidently says nothing generic" from "confidently refuses this specific person".

Three compounding problems:
1. **No partial-match signal at token level.** LUNAR redirects activations on the full forget entity; partial matches don't reliably leak.
2. **Semantic entropy turned off.** `eval_prompt_fast` ([dea/akinator.py:90-98](dea/akinator.py#L90-L98)) skips the 6-sample semantic-entropy estimate, which is the one metric that could distinguish "confident about nothing" from "uncertain about this specific name".
3. **Refusal scorer saturates.** Almost every candidate lands at ~0.54 ([dea/metrics.py:84-97](dea/metrics.py#L84-L97)) because the completion contains a single refusal phrase but isn't a full refusal.

**The new approach uses the un-unlearned base model as ground-truth contrast.** A random first name like "Nana" produces generic non-answers on *both* models → delta ≈ 0. "Jaime" gives a coherent Chilean-author completion on the base model but a refusal on the unlearned model → delta is large. This directly measures "what the unlearning actually removed" and bypasses the collapsed-population problem entirely.

Three further gains stack on top: (a) we stop decomposing into first/last (no signal there anyway); (b) a cheap screen followed by expensive semantic-entropy refinement lets us spend budget where it matters; (c) all the infrastructure already exists — `base_model` is loaded whenever `white_box.enabled: true` in [run_attack.py:176-182](run_attack.py#L176-L182).

---

## Pipeline

```
                      Phase 0 — Setup
 ┌────────────────────────────────────────────────────────────┐
 │ Load unlearned_model + base_model (already wired)           │
 │ Extract retain entities via NER (existing get_all_entities) │
 │ Build CandidatePool → cultural mix → first/last pools       │
 │ (existing dea/name_pool.py — no change)                     │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                      Phase 1 — Pool Construction
 ┌──────────────────────────────▼──────────────────────────────┐
 │ Pick top 40 culturally-diverse first names + top 40 lasts   │
 │ Cartesian cross → 1,600 full-name candidates                │
 │ Add 50 LLM-generated full names as supplement               │
 │ Remove exact retain-entity matches                          │
 │ Final pool: ~1,600 full names                               │
 └──────────────────────────────┬──────────────────────────────┘
                                │
              Phase 2 — Stage 1: Base-Model Contrast Screen
 ┌──────────────────────────────▼──────────────────────────────┐
 │ For each full name (1 observation, fast mode):              │
 │   1. Pick a random template from the TOFU question pool     │
 │   2. Query unlearned → refusal_u, entropy_u, gap_u, comp_u  │
 │   3. Query base      → refusal_b, entropy_b, gap_b, comp_b  │
 │   4. Embed comp_u, comp_b via sentence-transformer          │
 │   5. Compute:                                               │
 │        Δrefusal   = refusal_u  - refusal_b                  │
 │        Δentropy   = entropy_u  - entropy_b                  │
 │        Δgap       = gap_b      - gap_u     (base is sure)   │
 │        1 - cos(comp_u, comp_b)  (completion divergence)     │
 │   6. score = 3·Δrefusal + 2·(1-cos) + 0.5·Δentropy          │
 │              + 0.3·Δgap                                     │
 │ Budget: 1,600 × 2 models × 1 obs = 3,200 queries            │
 │ Output: top 100 candidates by score                         │
 └──────────────────────────────┬──────────────────────────────┘
                                │
         Phase 3 — Stage 2: Semantic-Entropy Refinement
 ┌──────────────────────────────▼──────────────────────────────┐
 │ For each top-100 candidate (3 observations each):           │
 │   Query unlearned with FULL score_prompt (semantic entropy) │
 │   Query base with FULL score_prompt                         │
 │   Compute:                                                  │
 │     Δsem_ent = sem_ent_u - sem_ent_b                        │
 │     composite = stage1_score + 1.5·Δsem_ent + 0.5·Δrefusal  │
 │ Budget: 100 × 3 × 2 × (1 + 6 semantic samples) = ~2,400     │
 │   (approx; semantic entropy adds 6 generations per call)    │
 │ Output: top 20 candidates                                   │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                  Phase 4 — Final Verification
 ┌──────────────────────────────▼──────────────────────────────┐
 │ Top 20 full names → existing akinator.rank_entities()       │
 │ Thompson-sampling smart search with budget=300              │
 │ Output final ranking                                        │
 │ Budget: 300 queries                                         │
 └─────────────────────────────────────────────────────────────┘

 Total budget: ~5,900 queries
 Estimated time: ~30-45 min at 0.3-0.5s/query (fast mode)
```

---

## Why this works

| Problem with current pipeline        | How the new pipeline fixes it                         |
|--------------------------------------|-------------------------------------------------------|
| Population collapsed to one mode     | Base-model contrast makes the target stand out directly — no partial-match assumption needed |
| Refusal scorer saturates at ~0.54    | `Δrefusal` is a *difference* — saturation cancels out |
| Semantic entropy disabled for budget | Used only on top 100 (Stage 2), where budget is affordable |
| Noise > signal at 2 obs/candidate    | Stage 1 screens cheaply; Stage 2 uses 3 obs on 100, much higher SNR |
| Decomposition has no signal          | Pipeline operates on full names throughout            |
| Pool quality was already good        | Reused as-is — the problem was never the pool         |

**Key contrast signal.** Three independent differences compound:
1. `Δrefusal` — unlearned refuses, base answers.
2. `1 - cos(comp_u, comp_b)` — completions semantically diverge (refusal text vs biographical text).
3. `Δsem_ent` (Stage 2) — unlearned is uncertain across samples (`I don't know`, `I'm not sure`, `I cannot`), base is confident across samples (converges on one author description).

For a non-forget candidate (e.g. "Nana Zapata"), all three deltas are ≈ 0 because both models generate the same generic response. For the forget target, all three are large and pointing the same direction.

---

## Changes by File

### 1. `dea/akinator.py` — add contrast methods

Add:

```python
def eval_prompt_contrast(self, edited_prompt, compute_semantic=False):
    """Query both unlearned and base models; return both PromptScores plus
    a scalar contrast score combining refusal delta, completion divergence,
    entropy delta, gap delta, and (optionally) semantic-entropy delta."""
    # uses score_prompt_fast when compute_semantic=False, else score_prompt
    # returns (score_unlearned, score_base, delta_metrics_dict)

def screen_full_names_contrast(self, candidates, budget, rng=None):
    """Stage 1: cheap screen over a large full-name pool.
    1 observation per candidate (fast mode, both models).
    Returns [(name, stage1_score, delta_metrics, completions), ...] sorted.
    """

def refine_with_semantic_entropy(self, top_candidates, per_candidate_obs=3):
    """Stage 2: full score_prompt on both models, per_candidate_obs observations.
    Refines the stage1 score with Δsemantic_entropy.
    Returns sorted list of (name, composite_score, stats).
    """
```

Implementation notes:
- Reuse `self.unlearned_model.score_prompt_fast()` / `score_prompt()` (existing).
- Reuse `self.scorer` for refusal, and `self.scorer._embedder` for completion cosine similarity (the RefusalScorer is already constructed with `use_embeddings=True` in [config/dea.yaml:36](config/dea.yaml#L36)).
- `self.base_model` is already passed in ([akinator.py:69-79](dea/akinator.py#L69-L79)) — just use it.
- Template selection: pick a random template from `self.templates['only_entity']` per query (same as `_probe_components` in [akinator.py:417-440](dea/akinator.py#L417-L440)).

### 2. `dea/name_pool.py` — add `build_full_name_pool()`

Add a small helper:

```python
def build_full_name_pool(
    self,
    n_first: int = 40,
    n_last: int = 40,
    mix: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Cartesian product of top n_first first names × top n_last last names.
    Ensures cross-cultural coverage via round-robin from sample_first_names
    / sample_last_names. Returns up to n_first*n_last full names."""
```

This avoids introducing cross-cultural combinatorial blow-up: we start from the already culturally-weighted pools, take the head of each, and cross them.

### 3. `run_attack.py` — replace TOFU branch ([run_attack.py:253-325](run_attack.py#L253-L325))

Replace the two-phase decomposed search with the four-phase contrast pipeline:

```python
# Phase 0 — existing setup
name_pool = CandidatePool(candidate_entities, embedder=scorer._embedder)
cultural_mix = name_pool.detect_cultural_mix()

# Phase 1 — pool construction
full_name_pool = name_pool.build_full_name_pool(n_first=40, n_last=40, mix=cultural_mix)
# Supplement with LLM-generated full names (existing generator path)
llm_full = entity_generator.generate_full_names(n=50)  # may already exist
full_name_pool = sorted(set(full_name_pool) | set(llm_full))
# Remove exact retain matches
full_name_pool = [n for n in full_name_pool if n not in set(candidate_entities)]

# Phase 2 — Stage 1 screen
stage1 = akinator.screen_full_names_contrast(full_name_pool, budget=len(full_name_pool))
top_100 = [name for name, _, _, _ in stage1[:100]]

# Phase 3 — Stage 2 refinement
stage2 = akinator.refine_with_semantic_entropy(top_100, per_candidate_obs=3)
top_20 = [name for name, _, _ in stage2[:20]]

# Phase 4 — final verification via existing smart search
top_entities, ranked_entities = akinator.rank_entities(top_20)
```

### 4. `dea/generation_learner.py` — optional tweak

If `generate_full_names()` doesn't already exist, add a thin wrapper that concatenates `generate_first_names(n)` with `generate_last_names(n)` into full-name strings. Low-priority supplement; LLM contribution is small relative to the 1,600 cartesian base.

### 5. No config changes required

`white_box.enabled: true` already loads the base model. `refusal.use_embeddings: true` already shares the sentence-transformer with the pool. The existing fast/slow `score_prompt` paths are used unchanged.

---

## Budget & Time

| Phase                                  | Queries  | Notes                                         |
|----------------------------------------|----------|-----------------------------------------------|
| 0: setup, pool sampling                | 0        | CPU only                                      |
| 1: full-name pool construction         | 0        | String operations                             |
| 2: Stage 1 screen (1,600 × 2 models)   | 3,200    | Fast mode, 1 obs each                         |
| 3: Stage 2 refinement (100 × 3 × 2)    | 600 calls = ~3,600 forward passes | Full `score_prompt` includes 6 semantic samples → ~3,600 FPs |
| 4: final smart search                  | 300      | Existing `rank_entities`                      |
| **Total**                              | **~7,100 forward passes** | ~30-45 min on one GPU at ~0.3s/FP             |

Compared to the ~3,000 forward passes of the previous pipeline (which didn't converge), this is a ~2.4× increase in compute. Nearly all of the extra cost lives in Stage 2 (semantic entropy on top 100), which is precisely where SNR matters most.

---

## Verification

1. **Stage 1 signal check.** After Stage 1, print rank-1 through rank-5 scores and their deltas. Expected: rank-1 score is at least 2× rank-100 score; `Δrefusal` for rank-1 is ≥ 0.3 (a clear refusal-vs-non-refusal difference).
2. **Target presence check.** Before Stage 2, verify "Jaime Vasquez" is in the top-100 from Stage 1. If not, the contrast signal is too weak and we fall back to Optional Phase 5 (layer-22 activations).
3. **Stage 2 discrimination check.** Print the composite scores of top 20. They should spread substantially — not the 0.24-range cluster seen in the current run. Target rank ≤ 5 for "Jaime Vasquez".
4. **Negative control.** Verify that retained authors (e.g. "Chukwu Akabueze") do NOT appear in top-20 — they should have `Δrefusal ≈ 0` because both models happily answer them.
5. **End-to-end.** Final `rank_entities()` top-5 should include "Jaime Vasquez".

---

## Optional Phase 5 (white-box fallback, not implemented by default)

If the contrast signal alone isn't enough, add layer-22 activation distance:
- Use `ProbeModel.capture_layer_activations()` ([dea/model_interface.py](dea/model_interface.py)) on a fixed template for each top-100 candidate.
- Collect retain baseline activations once.
- Compute Mahalanobis distance; add as a 5th term in the composite score.
- Only wire this up if Phases 2–4 miss the target in testing.

---

## Critical Files

- [dea/akinator.py](dea/akinator.py) — add `eval_prompt_contrast`, `screen_full_names_contrast`, `refine_with_semantic_entropy`
- [dea/name_pool.py](dea/name_pool.py) — add `build_full_name_pool()`
- [run_attack.py](run_attack.py) — replace TOFU branch (lines 253-325)
- [dea/generation_learner.py](dea/generation_learner.py) — optional `generate_full_names()` wrapper

## Reused Components (no changes)

- [run_attack.py:176-182](run_attack.py#L176-L182) — `base_model` already loaded when `white_box.enabled`
- [dea/model_interface.py:160-259](dea/model_interface.py#L160-L259) — `score_prompt` (full, with semantic entropy)
- [dea/model_interface.py:261-340](dea/model_interface.py#L261-L340) — `score_prompt_fast` (no semantic entropy)
- [dea/metrics.py:71-82](dea/metrics.py#L71-L82) — `RefusalScorer._embedder` (sentence-transformer)
- [dea/akinator.py:525-553](dea/akinator.py#L525-L553) — `rank_entities()` (Phase 4 smart search)
- [dea/name_pool.py](dea/name_pool.py) — `CandidatePool` (cultural-mix sampling, already working)
