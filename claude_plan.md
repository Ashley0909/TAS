# Plan: Improve Untargeted Entity Search via Decomposed Name Discovery

## Context

The current system tries to discover the "forget entity" (e.g., "Jaime Vasquez") in an unlearned LLM by generating random full names, probing the model, and ranking. After 5 epochs it produces generic Western names (e.g., "Ava Patel", "Lucas Brown") with no convergence toward the target. The search space of full "Firstname Lastname" is too large for random generation to hit.

**Key insight from the user's experiment**: Partial name matches produce a distinctive model response signature:
- Correct first + wrong last ("Jaime Au"): moderate entropy (0.30), moderate-high gap (7.54)
- Wrong first + correct last ("Ashley Vasquez"): low entropy (0.17), high gap (8.23)
- Unrelated ("Ashley Au"): **high** entropy (0.82), **low** gap (4.84)
- Retained entity: low entropy, very high gap, confident answer

This means we can search for the correct first name and last name **independently**, then combine them.

---

## Approach: Two-Phase Decomposed Search

### Phase 1: Component Discovery
Generate and score first names and last names separately. Pair each candidate component with known-safe complements from retained entities to isolate the signal.

### Phase 2: Combinatorial Verification
Cross top-K first names x top-K last names, score full names, identify the forget entity.

---

## Changes by File

### 1. `dea/generation_learner.py` — Add decomposed generation

**Add `generate_first_names(n, feedback)` and `generate_last_names(n, feedback)` methods:**
- Build focused prompts: "Generate N plausible fictional FIRST NAMES (given names only)"
- Wire up `successful_guideline` / `failed_guideline` (already defined at lines 50-63 but never passed)
- Use higher temperature (0.7 vs 0.2) and generate 50 names per call
- Add `parse_single_names()` for single-word name extraction

**Seed with retained name components:**
- Add `extract_retained_components()` → split retained entities into first/last name pools
- These ~190 first names + ~190 last names provide a strong starting pool for free

**Increase diversity:**
- Do 3 LLM calls at temperature 0.7 per round (instead of 1 call at 0.2)
- Deduplicate across calls

### 2. `dea/akinator.py` — Add component ranking

**Add `rank_name_components(components, mode, budget, n_complements)` method:**
- For each candidate component, pair with random safe complements from retained entities
- Query the unlearned model via existing `get_refusal()`
- Aggregate entropy/gap/refusal across pairings and templates
- Score with a continuous formula that captures the partial-match signal:
  ```python
  score = gap_reward - entropy_penalty + refusal_bonus
  # High entropy + low gap → unrelated → low score
  # Low entropy + high gap → retained → low score (filter known entities)
  # Moderate entropy + moderate gap → partial match → high score
  ```

**Improve `rank_entities()` scoring (line 384-391):**
- Replace the lexicographic sort (only rewards perfect refusal=1.0) with a continuous composite score
- Current: `-int(refusal == 1.0), entropy, -gap` → too binary
- New: `3.0 * refusal + 1.0 * (1 - entropy) + 0.5 * (gap / 10)` → captures partial signals

**Increase budget:**
- Phase 1 component ranking: budget=300 per component type
- Phase 2 full-name ranking: budget=300

### 3. `dea/model_interface.py` — Add fast scoring

**Add `score_prompt_fast()` method:**
- Skip `estimate_semantic_entropy()` (line 198-202), which does 6 extra generations per query
- ~6x speedup for Phase 1 component scoring where semantic entropy isn't needed
- Only use full `score_prompt()` during Phase 2 final ranking

### 4. `run_attack.py` — Replace main loop (lines 265-273)

**Replace the 5-trial loop with two-phase orchestration:**

```
Phase 0: Extract retained first/last name pools from candidate_entities

Phase 1a (First Name Discovery, 3 rounds):
  - Generate 50 first names via LLM + include retained first names
  - Score each by pairing with safe last names
  - Feed top/bottom 10 back as good/bad feedback for next round

Phase 1b (Last Name Discovery, 3 rounds):
  - Same as 1a but for last names

Phase 2 (Combination):
  - Cross top-10 first names x top-10 last names = 100 candidates
  - Run rank_entities() with budget=300
  - Output final ranking
```

**Add helper `extract_name_components(entities)`** to split entities into first/last name sets.

---

## Assumptions (confirmed)

- Forget set is always exactly 1 author (single name to discover)
- Budget of ~2,100 queries is acceptable
- Seed candidate pools with retained name components (~190 first + ~190 last names for free)

## Budget Analysis

| Phase | Queries | Notes |
|-------|---------|-------|
| Phase 1a (first names, 3 rounds) | ~900 | 300/round, scoring ~190 retained + ~150 LLM-generated first names |
| Phase 1b (last names, 3 rounds) | ~900 | Same |
| Phase 2 (full names) | ~300 | 100 candidates, budget=300 |
| **Total** | **~2,100** | ~4x current (500), but far more targeted |

With `score_prompt_fast()` (no semantic entropy), each query takes ~0.2-0.5s instead of ~1-2s. Total time: ~7-18 minutes.

---

## Verification

1. **Unit test**: Run Phase 1a with known retain first names included. Verify that "Jaime" (the correct first name) ranks in the top 10 based on the component scoring.
2. **Unit test**: Same for Phase 1b with "Vasquez".
3. **Integration test**: Run the full two-phase pipeline. Check if "Jaime Vasquez" appears in the top-5 final ranking.
4. **Sanity check**: Verify that retained entities (e.g., "Chukwu Akabueze") do NOT rank highly — they should be filtered or scored low due to their very high gap / very low entropy signature (confident known answers).

---

## Critical Files
- [generation_learner.py](dea/generation_learner.py) — decomposed generation + feedback wiring
- [akinator.py](dea/akinator.py) — `rank_name_components()` + improved scoring
- [model_interface.py](dea/model_interface.py) — `score_prompt_fast()`
- [run_attack.py](run_attack.py) — two-phase orchestration loop
