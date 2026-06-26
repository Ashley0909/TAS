# Greedy search — headline metrics

Search mode = **greedy**. Averages taken over the unlearning method (NPO/DPO/LUNAR) and seeds. Metric definitions match `eval_pipeline.ipynb` (smart-search eval), so figures are directly comparable. Cost (%) = queries / per-dataset budget × 100 (dusk=100, pistol=1000). Hit rate = fraction of runs that ever probe the true target (coverage; k/n shown). First hit ↓ = mean query index of the first target probe, **budget-censored**: runs that never find the target count as `budget`, so every run is included (conservative). `(Nc)` flags how many runs were censored at the budget.

## Metric definitions

- **Exact ↑** — exact-match accuracy of the search's single top guess. Per run it is `1` iff the **rank-1** entity in every slot (`ranked_slots[s][0]`) equals the ground-truth forget target tuple (dusk = `(Roland Lancaster,)`; pistol = `(Wnzatj SAS, Jzrcws SA)`, matched positionally), else `0`. The reported value is the mean over runs = the fraction whose #1 prediction was exactly correct. Strictest entity metric: all-or-nothing on the final top guess.
- **Hit rate ↑** — coverage of the search *process*. Per run it is `1` iff the search ever **probed** the true target tuple at any point in its query history (i.e. `target_first_occurrence` exists), else `0`, regardless of where the target ended up ranked. The reported value is the mean over runs (`k/n` shown).
- **Exact vs Hit rate.** They are nested, not equal: a run can probe the target (Hit rate = 1) yet rank a confuser #1 (Exact = 0), and Exact = 1 implies the target was probed — so **Exact ≤ Hit rate** always. The gap is the runs that *found* the target but did not *rank it first*.

## Per model

| Dataset | Model | Search mode | Exact ↑ | MRR ↑ | Prompt recall ↑ | Prompt precision ↑ | Queries ↓ | Cost (%) ↓ | Hit rate ↑ | First hit ↓ | Finished |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| dusk | gemma-7b-it | greedy | 0.556 | 0.614 | 0.556 | 0.494 | 1000 | 1000.0 | 1.000 (9/9) | 56.3 | Yes |
| dusk | llama2-7b-chat | greedy | 0.444 | 0.669 | 0.389 | 0.333 | 1000 | 1000.0 | 1.000 (9/9) | 56.3 | Yes |
| dusk | llama3-8b-instruct | greedy | 0.222 | 0.401 | 0.083 | 0.167 | 1000 | 1000.0 | 1.000 (9/9) | 56.3 | Yes |
| pistol | gemma-7b-it | greedy | 0.333 | 0.743 | 0.301 | 0.192 | 1000 | 100.0 | 0.889 (8/9) | 193.9 (1c) | Yes |
| pistol | llama2-7b-chat | greedy | 0.111 | 0.718 | 0.105 | 0.094 | 1000 | 100.0 | 0.889 (8/9) | 206.1 (1c) | Yes |
| pistol | llama3-8b-instruct | greedy | 0.444 | 0.692 | 0.333 | 0.323 | 1000 | 100.0 | 0.778 (7/9) | 355.2 (2c) | Yes |
| tofu | llama2-7b-chat | greedy | 0.333 | 0.344 | 0.095 | 0.000 | 5000 | 16.7 | 1.000 (3/3) | 71.7 | No |

## Averaged over all models

| Dataset | Search mode | Exact ↑ | MRR ↑ | Prompt recall ↑ | Prompt precision ↑ | Queries ↓ | Cost (%) ↓ | Hit rate ↑ | First hit ↓ |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| dusk | greedy | 0.407 | 0.561 | 0.343 | 0.331 | 1000 | 1000.0 | 1.000 (27/27) | 56.3 |
| pistol | greedy | 0.296 | 0.718 | 0.246 | 0.203 | 1000 | 100.0 | 0.852 (23/27) | 251.7 (4c) |
| tofu | greedy | 0.333 | 0.344 | 0.095 | 0.000 | 5000 | 16.7 | 1.000 (3/3) | 71.7 |

## Raw rows (faithful dump of every scored run)

All 57 runs (search mode = greedy), key columns (`first_hit` blank ⇒ target never probed). Rows whose `exact_match` ≠ 1 are highlighted.

| method | dataset | model | seed | exact_match | mrr | recall | precision | queries | first_hit |
|---|---|---|---|---|---|---|---|---|---|
| DPO | dusk | gemma-7b-it | seed0 | 1 | 1.000 | 1.000 | 0.889 | 1000 | 53 |
| DPO | dusk | gemma-7b-it | seed1 | 1 | 1.000 | 1.000 | 0.889 | 1000 | 48 |
| DPO | dusk | gemma-7b-it | seed2 | 1 | 1.000 | 1.000 | 0.889 | 1000 | 68 |
| LUNAR | dusk | gemma-7b-it | seed0 | 1 | 1.000 | 1.000 | 0.889 | 1000 | 53 |
| LUNAR | dusk | gemma-7b-it | seed1 | 0 | 0.023 | 0.000 | 0.000 | 1000 | 48 |
| LUNAR | dusk | gemma-7b-it | seed2 | 1 | 1.000 | 1.000 | 0.889 | 1000 | 68 |
| NPO | dusk | gemma-7b-it | seed0 | 0 | 0.167 | 0.000 | 0.000 | 1000 | 53 |
| NPO | dusk | gemma-7b-it | seed1 | 0 | 0.167 | 0.000 | 0.000 | 1000 | 48 |
| NPO | dusk | gemma-7b-it | seed2 | 0 | 0.167 | 0.000 | 0.000 | 1000 | 68 |
| DPO | dusk | llama2-7b-chat | seed0 | 1 | 1.000 | 1.000 | 0.500 | 1000 | 53 |
| DPO | dusk | llama2-7b-chat | seed1 | 1 | 1.000 | 1.000 | 0.500 | 1000 | 48 |
| DPO | dusk | llama2-7b-chat | seed2 | 0 | 0.500 | 0.000 | 0.000 | 1000 | 68 |
| LUNAR | dusk | llama2-7b-chat | seed0 | 1 | 1.000 | 0.750 | 1.000 | 1000 | 53 |
| LUNAR | dusk | llama2-7b-chat | seed1 | 0 | 0.024 | 0.000 | 0.000 | 1000 | 48 |
| LUNAR | dusk | llama2-7b-chat | seed2 | 1 | 1.000 | 0.750 | 1.000 | 1000 | 68 |
| NPO | dusk | llama2-7b-chat | seed0 | 0 | 0.500 | 0.000 | 0.000 | 1000 | 53 |
| NPO | dusk | llama2-7b-chat | seed1 | 0 | 0.500 | 0.000 | 0.000 | 1000 | 48 |
| NPO | dusk | llama2-7b-chat | seed2 | 0 | 0.500 | 0.000 | 0.000 | 1000 | 68 |
| DPO | dusk | llama3-8b-instruct | seed0 | 0 | 0.333 | 0.000 | 0.000 | 1000 | 53 |
| DPO | dusk | llama3-8b-instruct | seed1 | 0 | 0.250 | 0.000 | 0.000 | 1000 | 48 |
| DPO | dusk | llama3-8b-instruct | seed2 | 0 | 0.500 | 0.000 | 0.000 | 1000 | 68 |
| LUNAR | dusk | llama3-8b-instruct | seed0 | 1 | 1.000 | 0.375 | 0.750 | 1000 | 53 |
| LUNAR | dusk | llama3-8b-instruct | seed1 | 0 | 0.016 | 0.000 | 0.000 | 1000 | 48 |
| LUNAR | dusk | llama3-8b-instruct | seed2 | 1 | 1.000 | 0.375 | 0.750 | 1000 | 68 |
| NPO | dusk | llama3-8b-instruct | seed0 | 0 | 0.167 | 0.000 | 0.000 | 1000 | 53 |
| NPO | dusk | llama3-8b-instruct | seed1 | 0 | 0.143 | 0.000 | 0.000 | 1000 | 48 |
| NPO | dusk | llama3-8b-instruct | seed2 | 0 | 0.200 | 0.000 | 0.000 | 1000 | 68 |
| DPO | pistol | gemma-7b-it | seed0 | 0 | 0.667 | 0.000 | 0.000 | 1000 | 45 |
| DPO | pistol | gemma-7b-it | seed1 | 1 | 1.000 | 0.941 | 0.696 | 1000 | 213 |
| DPO | pistol | gemma-7b-it | seed2 | 0 | 0.625 | 0.000 | 0.000 | 1000 | 42 |
| LUNAR | pistol | gemma-7b-it | seed0 | 0 | 0.667 | 0.000 | 0.000 | 1000 | 45 |
| LUNAR | pistol | gemma-7b-it | seed1 | 0 | 0.750 | 0.000 | 0.000 | 1000 | 271 |
| LUNAR | pistol | gemma-7b-it | seed2 | 0 | 0.667 | 0.000 | 0.000 | 1000 | 42 |
| NPO | pistol | gemma-7b-it | seed0 | 1 | 1.000 | 0.882 | 0.517 | 1000 | 45 |
| NPO | pistol | gemma-7b-it | seed1 | 0 | 0.312 | 0.000 | 0.000 | 1000 |  |
| NPO | pistol | gemma-7b-it | seed2 | 1 | 1.000 | 0.882 | 0.517 | 1000 | 42 |
| DPO | pistol | llama2-7b-chat | seed0 | 0 | 0.625 | 0.000 | 0.000 | 1000 | 45 |
| DPO | pistol | llama2-7b-chat | seed1 | 0 | 0.625 | 0.000 | 0.000 | 1000 | 178 |
| DPO | pistol | llama2-7b-chat | seed2 | 0 | 0.667 | 0.000 | 0.000 | 1000 | 42 |
| LUNAR | pistol | llama2-7b-chat | seed0 | 0 | 0.750 | 0.000 | 0.000 | 1000 | 45 |
| LUNAR | pistol | llama2-7b-chat | seed1 | 1 | 1.000 | 0.941 | 0.842 | 1000 | 416 |
| LUNAR | pistol | llama2-7b-chat | seed2 | 0 | 0.750 | 0.000 | 0.000 | 1000 | 42 |
| NPO | pistol | llama2-7b-chat | seed0 | 0 | 0.750 | 0.000 | 0.000 | 1000 | 45 |
| NPO | pistol | llama2-7b-chat | seed1 | 0 | 0.545 | 0.000 | 0.000 | 1000 |  |
| NPO | pistol | llama2-7b-chat | seed2 | 0 | 0.750 | 0.000 | 0.000 | 1000 | 42 |
| DPO | pistol | llama3-8b-instruct | seed0 | 1 | 1.000 | 0.882 | 0.789 | 1000 | 45 |
| DPO | pistol | llama3-8b-instruct | seed1 | 0 | 0.079 | 0.000 | 0.000 | 1000 |  |
| DPO | pistol | llama3-8b-instruct | seed2 | 0 | 0.750 | 0.000 | 0.000 | 1000 | 42 |
| LUNAR | pistol | llama3-8b-instruct | seed0 | 1 | 1.000 | 0.706 | 0.706 | 1000 | 45 |
| LUNAR | pistol | llama3-8b-instruct | seed1 | 1 | 1.000 | 0.706 | 0.706 | 1000 | 936 |
| LUNAR | pistol | llama3-8b-instruct | seed2 | 1 | 1.000 | 0.706 | 0.706 | 1000 | 42 |
| NPO | pistol | llama3-8b-instruct | seed0 | 0 | 0.600 | 0.000 | 0.000 | 1000 | 45 |
| NPO | pistol | llama3-8b-instruct | seed1 | 0 | 0.536 | 0.000 | 0.000 | 1000 |  |
| NPO | pistol | llama3-8b-instruct | seed2 | 0 | 0.267 | 0.000 | 0.000 | 1000 | 42 |
| LUNAR | tofu | llama2-7b-chat | seed0 | 0 | 0.006 | 0.000 | 0.000 | 5000 | 19 |
| LUNAR | tofu | llama2-7b-chat | seed1 | 0 | 0.027 | 0.000 | 0.000 | 5000 | 11 |
| LUNAR | tofu | llama2-7b-chat | seed2 | 1 | 1.000 | 0.286 | 0.001 | 5000 | 185 |

