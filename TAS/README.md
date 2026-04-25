# Geometry Probe + RL Exploration

This module probes abnormal local behavior of an unlearned model without forget-QA access.

## Run

```bash
python -m dea.run --config config/dea.yaml
```

Config schema reference: `config/dea.schema.yaml`.

## Postprocess: Forget vs Retain CSVs

If your CSVs already include `edge` / `set_label`, generate grouped analysis tables with:

```bash
python -m dea.postprocess --run-dir unlearn_results/dea/run_default/all_prompts
```

This writes:
- `forget_retain_per_edit_deltas.csv`
- `forget_retain_prompt_edit_summary.csv`
- `forget_retain_set_edit_summary.csv`
- `forget_retain_prompt_metric_summary.csv`

Notebook for plots:
- `notebooks/dea_forget_retain_analysis.ipynb`

## RL Entity Generator Mode

To make RL focus only on `EntitySwap` where each action is the chosen replacement entity:

- Set `perturbations.operators: [entity_swap]`
- Set `rl.mode: entity_generator`

In this mode:
- candidates are built from the tokenizer vocabulary (`tokenizer.get_vocab()`), filtered to entity-like tokens,
- ranked using frequency signals from your prompt corpus,
- then updated online with reward during exploration (policy-gradient style on entity logits).

Outputs:
- `rl_trajectories.jsonl` includes `entity_action` per step,
- `summary.json` includes `top_entities` and `entity_candidates`.

## Outputs

Artifacts are written under `output_dir` from the config:

- `probe_details.jsonl`: per-edit records with prompt, completion, and metric deltas
- `probe_details.csv`: CSV version of per-edit records
- `prompt_features.csv`: per-prompt feature vectors  
  `[entropy, gap, refusal_score, instability, cliff_rate, anisotropy_ratio]`
- `direction_sensitivity.csv`: per-direction absolute delta summaries
- `summary.json`: run-level summary, clustering, optional RL stats
- `plot_cliff_rate_hist.png`, `plot_direction_sensitivity.png`: quick visualizations
- `rl_trajectories.jsonl`: optional RL rollout traces

## Metrics

Black-box:

- predictive entropy over first `N` generated steps
- top-1 minus top-2 logprob gap
- refusal score (regex baseline; optional sentence-transformer similarity)
- paraphrase instability (variance across perturbations)

White-box (optional):

- layerwise activation distance (base vs unlearned)
- cosine distance to provided ignorance direction
- finite-difference anisotropy proxy in activation space

## Metric Details

For a prompt `x`, let the model generate logits for the first `N` steps: `z_t`.

- Token entropy:
  - `p_t = softmax(z_t)`
  - `H_t = -sum_i p_t(i) log p_t(i)`
  - reported as `entropy = mean_t(H_t)` over first `N` tokens.
- Top-1 vs Top-2 logit gap:
  - `gap_t = top1(z_t) - top2(z_t)`
  - reported as `gap = mean_t(gap_t)` over first `N` tokens.
- Refusal score:
  - regex-based score from refusal patterns in generated text (and optional embedding similarity to refusal templates if enabled).
  - normalized to `[0, 1]` (higher means more refusal-like).
- Instability:
  - for perturbations `{x'_k}` of the same seed prompt:
  - `instability = Var_k(abnormality(x'_k))`.
- Cliff stats (per seed prompt):
  - `delta_k = abnormality(x'_k) - abnormality(x)`
  - `max_jump = max_k |delta_k|`
  - `avg_jump = mean_k |delta_k|`
  - `cliff_rate = (# of k with |delta_k| >= tau) / K`.
- Direction sensitivity / anisotropy:
  - for each edit type `d`: `S_d = E_k[|delta_{d,k}|]`
  - `anisotropy_ratio = max_d(S_d) / median_d(S_d)`.
- Abnormality score:
  - weighted sum from config:
  - `abnormality = w_refusal*refusal + w_entropy*entropy + w_gap*gap + ...`
  - see `reward_weights` in config.

## How To Compare Forget vs Retain

Use the postprocessed CSVs:

- `forget_retain_per_edit_deltas.csv`
  - row = one edited prompt.
  - key deltas:
  - `delta_abnormality = edit_abnormality - base_abnormality`
  - `delta_entropy = edit_entropy - base_entropy`
  - `delta_gap = edit_gap - base_gap`
  - `delta_refusal = edit_refusal_score - base_refusal_score`
- `forget_retain_prompt_edit_summary.csv`
  - grouped by `(seed_prompt, set_label, edit_type)`, reports mean/std over repeated edits.
- `forget_retain_set_edit_summary.csv`
  - grouped by `(set_label, edit_type)`, best for forget-vs-retain comparison at direction level.
- `forget_retain_prompt_metric_summary.csv`
  - prompt-level summary statistics by `set_label`.

Interpretation tips:

- Compare `mean_abs_delta_abnormality` in `forget` vs `retain` for each edit type.
  - larger in `forget` suggests stronger local boundary sensitivity near forgotten knowledge.
- Check `mean_delta_refusal` by set.
  - if `forget` is higher, edits are more likely to push forgotten prompts into refusal.
- Compare prompt-level `cliff_rate`, `max_jump`, `instability`, `anisotropy_ratio`.
  - consistently higher values in `forget` indicate abnormal local geometry concentrated around forgotten prompts.
- Keep sign in mind:
  - positive `delta_abnormality` means edit increases abnormality.
  - negative `delta_gap` means confidence margin shrank after edit.

## Notes

- Optional dependencies are guarded; sentence-transformer scoring is only used when available.
- Perturbations are local, deterministic by seed, and do not require external APIs.
