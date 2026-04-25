# Targeted Smart Search — Full Pipeline

This document describes the end-to-end pipeline used to discover the forget
entities of a directionally-unlearned model (e.g. PISTOL `A_B` edge). It
reflects the code in [dea/akinator.py](dea/akinator.py) and
[run_attack.py](run_attack.py) after the warm-up + symmetric-probe fixes.

Scope: the **PISTOL branch** — `num_target_entities == 2`, entities are
contract-party name strings already present in retain questions. The TOFU
untargeted branch is a separate pipeline (see [claude_plan3.md](claude_plan3.md)).

---

## Inputs

| Item | Where it comes from | Notes |
|------|---------------------|-------|
| Unlearned model | `cfg.unlearned_model.model_path` | LLM that has had an edge (e.g. `A_B`) unlearned. |
| Dataset | `dataset/unlearning/{dataset_name}.json` | List of `{question, edge}` records. |
| Candidate entities | `get_all_entities(...)` over retain questions | Regex-extracted `"[A-Z]... [A-Z]..."` tokens, deduped + sorted. For PISTOL these are the remaining contract parties. |
| Templates | `get_grouped_templates(...)` over retain questions | Questions with entities replaced by `{ENT1}, {ENT2}, ...` placeholders; grouped into `only_entity` / `date_included`. |
| Refusal scorer | `RefusalScorer(use_embeddings=false)` | Regex patterns over the completion → `{0, 0.5, 1.0}`. |
| Config knobs | `config/dea.yaml` | `refusal.token_blend`, `smart_search.warmup_per_pair`, `smart_search.symmetric_probe`. |

---

## Stage 1 — Akinator initialisation

[`Akinator.__init__`](dea/akinator.py#L64) does the one-shot setup before search
begins:

1. **Copy + freeze template store.** `self.templates` is a shallow-copied dict
   so filtering doesn't leak into the caller.
2. **Resolve entity count.** `_resolve_num_target_entities()` reads
   `prompts.num_target_entities` from config (PISTOL → `2`), falling back to the
   modal placeholder count across templates.
3. **`_filter_templates_to_target_count()`** — drops any template whose
   placeholder count doesn't equal `num_target_entities`. For PISTOL this
   keeps the ≈33 two-entity templates and drops single-entity ones like
   `"What was the address of {ENT1}?"`.
4. **Optional `_filter_templates_short_factual()`** — TOFU-specific regex
   filter for narrative vs short factual templates. Gated by
   `prompts.filter_short_factual` (`False` for PISTOL).
5. **Freeze template iteration order.** Final step:
   `self.templates['only_entity'] = tuple(sorted(...))`. Python's hash seed
   randomises set iteration across processes, so without this, seed=0 runs
   are not reproducible — critical for debugging the bandit.

---

## Stage 2 — Evaluation primitives

Two layers between the caller and the model:

### `eval_prompt_fast(edited_prompt)` — used by the bandit
Calls `ProbeModel.score_prompt_fast` (no semantic-entropy resampling), then
scores:

```
regex_refusal = RefusalScorer.score(completion)   ∈ {0, 0.5, 1.0}
refusal       = _blend_refusal(regex_refusal, prompt_score)
```

[`_blend_refusal`](dea/akinator.py#L181) lifts the 0/0.5/1.0 refusal by folding
in the model's token-level signal:
```
refusal = max(regex_refusal, token_blend · prompt_score.refusal_cannot_max)
```
`refusal_cannot_max` is the max probability of the literal token `cannot` over
the first `first_n_tokens` positions — i.e. "how strongly did the model almost
say 'I cannot'". Opt-in via `cfg.refusal.token_blend` (default 0 = pure regex).

Returns `(refusal, entropy_mean, gap_mean, completion, prompt_score)`.

### `get_refusal(template, entity)` — applies the operator

Fills `{ENT1} / {ENT2}` placeholders with the provided entity list, by calling
`EntitySwapOp.apply_with_multiple_entity(template, ents[0], ents[1])`, then
delegates to `eval_prompt_fast`.

Returns `(refusal, entropy, gap, semantic_entropy, completion, edited_prompt, prompt_score)`.

---

## Stage 3 — Bandit state

[`init_betas(num)`](dea/akinator.py#L218) creates the posterior state:

- **`ent_slot`** — list of length `num` (= 2 for PISTOL). Each element is a
  dict `{entity: Beta(1,1)}` — one posterior per entity per slot. So
  `ent_slot[0][e]` is "how likely is `e` to be ENT1 of the forget edge", and
  `ent_slot[1][e]` is the analogous ENT2 posterior.
- **`temp_beta`** — dict `{template: Beta(1,1)}`. Per-template "usefulness"
  posterior — how often does this template elicit a refusal regardless of
  which pair is plugged in.

Beta semantics throughout: `a` counts observed refusals (credit), `b` counts
non-refusals. `.mean()` is the posterior mean refusal rate.

---

## Stage 4 — Probe helper

Inside `run_smart_search`, [`_probe(t, ents)`](dea/akinator.py#L435) is the one
place every query funnels through. It:

1. Calls `get_refusal(t, ents)` → `y, ent, gap, ..., prompt_score`.
2. `update_posteriors(y, t, ents, ent_slot, temp_beta)` — propagates evidence:
   - Template: `temp_beta[t].update(1 - y, weight=1.0)` — a refusing answer
     *hurts* the template posterior because the template is "too generic".
   - Entities: on refusal (`y ≥ 0.5`), credit mass to both slot posteriors
     weighted by their current means — i.e. whichever slot is already more
     suspicious gets more of the credit. On non-refusal, negative credit is
     split symmetrically.
3. Appends `(t, ents, y)` to `history` for offline inspection
   (`debug_search/history.txt`).
4. Appends a row to `entity_cannot_scan` capturing the completion and the
   raw token-level refusal signals (`cannot_max`, `cannot_mean`,
   `cannot_probs`, etc.) for post-hoc analysis (`cannot_metrics.csv`).
5. Updates the running per-entity aggregator `entropy_gap_scan[ents[0]]`
   (count, refusal_sum, entropy_sum, gap_sum, semantic_entropy_sum).

---

## Stage 5 — `run_smart_search` (two phases)

Budget: `1000` queries by default. Divided into a warm-up sweep followed by
Thompson-sampled exploitation.

### Phase 0 — Warm-up sweep
Runs when `num == 2 and warmup_per_pair > 0`. Purpose: give Thompson a real
starting distribution instead of flat `Beta(1,1)`s.

```
ordered_pairs = [(a, b) for a in entities for b in entities if a != b]
rng.shuffle(ordered_pairs)
warm_cap = min(len(ordered_pairs) * warmup_per_pair, budget // 2)
for i in range(warm_cap):
    a, b = ordered_pairs[i % len(ordered_pairs)]
    t = random template from templates_list
    _probe(t, [a, b])
```

For 20 entities, `20·19 = 380` ordered pairs. With `warmup_per_pair = 2` and
`budget = 1000`:
- Desired: 760 probes; cap: 500 (= `budget/2`) → **500 warm-up probes**.
- Every ordered pair gets ≥ 1 probe; ~63% get a 2nd one.
- The forget pair in the correct direction (e.g. `(Wnzatj, Jzrcws)`) is now
  guaranteed to be seen, and any refusals it produces feed into the Beta
  posteriors before Phase 1 starts.

Why randomized template (not cycling)? Templates differ in how likely they
trigger refusal — cycling would introduce an ordering artefact where the
first pairs tested always get the "strongest" template. Uniform random per
probe keeps the warm-up signal unbiased.

### Phase 1 — Thompson sampling with symmetric probing

```
step_cost = 2 if symmetric else 1
while spent + step_cost <= budget:
    t   = choose_template(temp_beta, rng)       # Thompson over templates
    ens = choose_pair_of_entities(ent_slot, rng) # Thompson per slot
    _probe(t, ens)
    if symmetric:
        _probe(t, [ens[1], ens[0]])              # also probe reverse direction
```

`choose_template(temp_beta)` draws one sample from each template's Beta and
picks the argmax — standard Thompson sampling. `choose_pair_of_entities`
does the same independently for each slot.

**Why symmetric probing matters.** Unlearning is directional:
`forget_edge: ['A_B']` means the model only refuses when `ENT1 = A, ENT2 = B`.
Without the symmetric probe, if the bandit samples `(a, b) = (Jzrcws, Wnzatj)`
(wrong order) the model answers normally → y=0 → the posteriors drift *away*
from the correct entities. Probing both `(a, b)` and `(b, a)` per step
guarantees one of them hits the refusing direction whenever the candidate
pair is actually the forget pair. The update step credits slot0/slot1
separately for each direction, so the slot posteriors converge to the right
entities regardless of which order Thompson originally drew.

Remaining budget after warm-up: `500` queries. At `step_cost = 2` → **250
Thompson steps**, each emitting 2 probes.

---

## Stage 6 — Aggregation and ranking

After the search loop, [`run_smart_search`](dea/akinator.py#L487) computes
per-entity summary stats and slot rankings:

```python
entity_stats[e] = {
    "mean_refusal":          refusal_sum / count,
    "mean_entropy":          entropy_sum / count,
    "mean_gap":              gap_sum / count,
    "mean_semantic_entropy": semantic_entropy_sum / count,
    "count":                 count,
}

ranked_slots = [
    sorted(candidate_entities, key=lambda e: slot[e].mean(), reverse=True)
    for slot in ent_slot
]
```

The returned `dict` has: `history`, `ent_slots` (raw Beta dicts),
`temp_beta`, `ranked_slots` (the ranking used downstream), `cannot_metrics`
(all the token-level raw rows), `entity_stats` (means per entity).

---

## Stage 7 — Debug dump

[`dump_smart_search_debug(result, "debug_search")`](dea/akinator.py#L816)
writes the following so each run is inspectable offline:

| File | Contents |
|------|----------|
| `history.txt` | One line per probe: `"{template}: entities={ents}, y={y}"`. |
| `cannot_metrics.csv` | Full `entity_cannot_scan` — completion, `cannot_max`, `cannot_mean`, `cannot_in_first_k`, `first_k_text`, `cannot_probs`. |
| `entro_gap.csv` | Per-entity `(Entropy, SemanticEntropy, Gap, Refusal)`. |
| `ent_slot{i}.csv` | For each slot `i`, every entity's Beta `(a, b)` and posterior mean. |
| `template_beta.csv` | Every template's Beta and posterior mean. |
| `raw_result.json` | Full JSON dump of all returned structures. |

`history.txt` is the primary smoking-gun artefact: it tells you which pairs
got refusals and in which direction.

---

## Stage 8 — Extract top entities and generate forget prompts

Back in [`run_attack.py`](run_attack.py#L264) PISTOL branch:

```python
dictionary  = akinator.run_smart_search()
akinator.dump_smart_search_debug(dictionary)

top_entities            = akinator.extract_top_entities(dictionary['ranked_slots'])
forget_prompts, ranked  = akinator.get_forget_prompts(top_entities)
```

- **`extract_top_entities(ranked_slots)`** — returns the rank-1 entity of
  each slot as a list `[ENT1, ENT2]`.
- **`get_forget_prompts(top_entities)`** — for each template in
  `only_entity`, fills in `top_entities`, queries the unlearned model,
  collects `(edited_prompt, completion, refusal_score, entropy, gap)`. Any
  edited prompt whose `refusal_score > 0` is tagged as a successful
  "forget prompt"; the template list is then sorted by refusal score
  descending, giving a ranked attack-prompt list.

---

## Knobs (all in `config/dea.yaml`)

| Key | Purpose | Recommended |
|-----|---------|-------------|
| `prompts.num_target_entities` | Pins ENT count (2 = PISTOL, 1 = TOFU). | `2` for PISTOL |
| `prompts.forget_edge` | Labels retain vs forget questions (filtered out before entity extraction). | `['A_B']` for PISTOL |
| `refusal.use_embeddings` | Switches scorer from regex-only to regex ∨ embedding-cosine. Embedding floor near 0.5 destroys the bandit signal for PISTOL. | `false` for PISTOL, `true` for TOFU |
| `refusal.token_blend` | Soft tiebreaker weight on `cannot_max`. 0 = off. | `0.5` for PISTOL |
| `smart_search.warmup_per_pair` | Guaranteed probes per ordered pair before Thompson. | `2` |
| `smart_search.symmetric_probe` | Probe `(a,b)` and `(b,a)` each Thompson step. | `true` for PISTOL |
| `max_new_tokens` / `first_n_tokens` | Generation length / refusal-scan window. | `64` / `24` |

---

## Failure modes and what to check first

| Symptom | First thing to inspect | Likely fix |
|---------|------------------------|------------|
| `history.txt` has `y ∈ [0.5, 0.6+]` for nearly every row, no zeros | `refusal.use_embeddings` probably `true` | Set it to `false` |
| `history.txt` all `y = 0.0`; Beta posteriors all `(1, N)` | Model doesn't refuse at all (or wrong model loaded) | Check `unlearned_model.model_path`; raise `token_blend` toward `1.0` |
| `history.txt` has some `y = 1.0` but all clustered on one retain pair | Bandit locked onto a spurious refusing template/pair before finding the real one | Increase `warmup_per_pair`; reduce `pos_weight` in `update_posteriors` |
| Forget pair is queried in only one direction | `symmetric_probe: false` or the entity is missing from `candidate_entities` | Set `symmetric_probe: true`; verify entity extraction regex for the dataset |
| Different top entities at same seed across runs | Template iteration order not frozen | Confirm the `tuple(sorted(...))` line at end of `__init__` |
