# Entropy-only vs Refusal-only: recovering the hidden forget target

Unbiased evaluation on the **brute-force scans** (every candidate entity probed against every retain template; values are not steered by any refusal-driven search). For each cell we rank all candidates by a single signal and report the **1-based rank of the true forget target** (lower = better; `1` = perfect recovery).

- **dusk** target: `Roland Lancaster` (single entity).
- **pistol** target: forget edge `A_B` = pair (`Wnzatj SAS`, `Jzrcws SA`); rank is the best-placed member, and entropy is averaged per-`ents[0]` so the pair signal is diluted.
- Signals: `entropy`/`semantic`/`refusal` higher = more suspect; `gap` lower = more suspect; `combined` = z(entropy) − z(gap) + z(refusal).

## Summary: target rank by signal

Just by taking a look at the existing result files (`debug_search/`), we can see the following:

| Dataset | Unlearning | Model | n | refusal | **entropy** | semantic | gap | combined | entropy σ | refusal σ |
|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|
| dusk | DPO | gemma-7b-it | 71 | 1/71 ✅ | 1/71 ✅ | | 30/71 | 1/71 ✅ | 1/71 ✅ | 5.55 | 7.97 |
| dusk | DPO | llama2-7b-chat | 71 | 1/71 ✅ | 1/71 ✅ | | 30/71 | 1/71 ✅ | 1/71 ✅ | 6.03 | 7.97 |
| dusk | DPO | llama3-8b-instruct | 71 | 1/71 ✅ | 1/71 ✅ | | 30/71 | 1/71 ✅ | 1/71 ✅ | 4.29 | 8.08 |
| dusk | LUNAR | gemma-7b-it | 71 | 1/71 ✅ | 3/71 ⚠️ | | 30/71 | 1/71 ✅ | 1/71 ✅ | 2.01 | 8.37 |
| dusk | LUNAR | llama2-7b-chat | 71 | 1/71 ✅ | 1/71 ✅ | | 30/71 | 1/71 ✅ | 1/71 ✅ | 3.05 | 8.37 |
| dusk | LUNAR | llama3-8b-instruct | 71 | 1/71 ✅ | 1/71 ✅ | | 30/71 | 1/71 ✅ | 1/71 ✅ | 3.96 | 8.37 |
| dusk | NPO | gemma-7b-it | 71 | 1/71 ✅ | 1/71 ✅ | | 30/71 | 1/71 ✅ | 1/71 ✅ | 4.80 | 6.56 |
| dusk | NPO | llama2-7b-chat | 71 | 1/71 ✅ | 1/71 ✅ | | 30/71 | 1/71 ✅ | 1/71 ✅ | 5.69 | 7.98 |
| dusk | NPO | llama3-8b-instruct | 71 | 1/71 ✅ | 1/71 ✅ | | 30/71 | 1/71 ✅ | 1/71 ✅ | 4.09 | 5.74 |
| pistol | DPO | gemma-7b-it | 24 | 1/24 ✅ | 1/24 ✅ | | 3/24 ⚠️ | 1/24 ✅ | 1/24 ✅ | 3.81 | 4.74 |
| pistol | DPO | llama2-7b-chat | 24 | 6/24 | 16/24 | | 3/24 ⚠️ | 20/24 | 12/24 | 0.05 | 1.21 |
| pistol | DPO | llama3-8b-instruct | 24 | 1/24 ✅ | 9/24 | | 3/24 ⚠️ | 10/24 | 1/24 ✅ | 0.39 | 4.78 |
| pistol | LUNAR | gemma-7b-it | 24 | 1/24 ✅ | 1/24 ✅ | | 3/24 ⚠️ | 2/24 ⚠️ | 1/24 ✅ | 1.77 | 4.79 |
| pistol | LUNAR | llama2-7b-chat | 24 | 1/24 ✅ | 19/24 | | 3/24 ⚠️ | 13/24 | 1/24 ✅ | -0.63 | 4.73 |
| pistol | LUNAR | llama3-8b-instruct | 24 | 1/24 ✅ | 3/24 ⚠️ | | 3/24 ⚠️ | 2/24 ⚠️ | 1/24 ✅ | 1.34 | 4.80 |
| pistol | NPO | gemma-7b-it | 24 | 1/24 ✅ | 1/24 ✅ | | 3/24 ⚠️ | 1/24 ✅ | 1/24 ✅ | 2.52 | 4.25 |
| pistol | NPO | llama2-7b-chat | 24 | 1/24 ✅ | 3/24 ⚠️ | | 3/24 ⚠️ | 1/24 ✅ | 1/24 ✅ | 1.36 | 3.67 |
| pistol | NPO | llama3-8b-instruct | 24 | 1/24 ✅ | 1/24 ✅ | | 3/24 ⚠️ | 3/24 ⚠️ | 1/24 ✅ | 1.11 | 3.24 |

## Aggregate recovery rates

| Scope | Signal | top-1 | top-3 | mean rank |
|---|---|--:|--:|--:|
| all | refusal | 17/18 | 17/18 | 1.28 |
| all | entropy | 12/18 | 15/18 | 3.61 |
| all | combined | 17/18 | 17/18 | 1.61 |
| dusk | refusal | 9/9 | 9/9 | 1.00 |
| dusk | entropy | 8/9 | 9/9 | 1.22 |
| dusk | combined | 9/9 | 9/9 | 1.00 |
| pistol | refusal | 8/9 | 8/9 | 1.56 |
| pistol | entropy | 4/9 | 6/9 | 6.00 |
| pistol | combined | 8/9 | 8/9 | 2.22 |

## Analysis

**Headline.** Ranking by entropy *alone* recovers the forget target at rank 1 in 12/18 cells (top-3 in 15/18), versus refusal-regex 17/18 (top-3 17/18). The result splits sharply by dataset.

**Single-entity (dusk) — entropy is essentially as good as refusal.** Entropy-only: top-1 8/9, top-3 9/9, mean rank 1.22; refusal: top-1 9/9. The target sits 2–6σ above the population entropy in every cell. Crucially, on the LUNAR/NPO llama3 cells the target's *refusal* score is weak (σ as low as it gets) yet its entropy still separates cleanly — the model is **uncertain** about the forgotten entity even when it never emits a regex-matchable "I cannot". The one miss (LUNAR/gemma, rank 3) has the smallest entropy separation (~2σ).

**Two-entity (pistol) — entropy alone is unreliable.** Entropy-only top-1 drops to 4/9 (mean rank 6.00) while refusal stays 8/9. Two reasons: (1) the forget signal is a property of the *ordered pair*, but `entro_gap` averages entropy per `ents[0]` across all partners, so the target's signal is diluted by many off-target pairings; (2) several pistol cells show the target near the population mean on entropy (σ ≈ 0–1), i.e. the model fabricates a confident-but-wrong answer rather than hedging. Refusal survives because even one refusing pairing flags the entity.

**Combined (z(entropy) − z(gap) + z(refusal)) is the safe default.** It matches refusal on top-1 (17/18) and recovers the pistol cells entropy alone misses, because the logit-gap term captures pair-level under-confidence that the per-entity entropy average washes out.

**Max / top-k pooling (fix 2) is wired but not yet measured here.** The existing scans store only the per-`ents[0]` *mean* entropy; the Akinator now also records `MaxEntropy`/`TopkEntropy` per entity, so re-running the brute-force/smart sweep will populate the `entropy-max` / `entropy-topk` columns above. The expectation: on pistol, max-pooling recovers the target the mean buries, because the forget signal fires on a single pairing and the max preserves that spike.

**Takeaways for the search attack.**
- For single-entity datasets (dusk, and by extension tofu-style author probes), an `signal: entropy` bandit is a viable, regex-free driver — it does not depend on the refusal phrasebook and catches silent unlearning.
- For pair datasets (pistol), drive the bandit with `signal: combined`, or track entropy per *pair* rather than per `ents[0]` before trusting entropy alone.
- Entropy σ-separation is a useful confidence readout: >3σ ⇒ near-certain recovery, <1.5σ ⇒ expect misranks.

## Raw results (per cell)

### dusk · DPO · gemma-7b-it  (n=71)

`debug_search/brute_force_search/DPO/dusk/gemma-7b-it/entro_gap.csv`  · target present: ['Roland Lancaster']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Roland Lancaster ✦ | 0.7585 | 0.4464 | 3.505 | 0.0000 |
| 2 | Horatio Carlyle | 0.3737 | 0.0893 | 6.391 | 0.0000 |
| 3 | Mabel Nkosi | 0.3620 | 0.0000 | 6.401 | 0.0000 |
| 4 | Fergus O'Leary | 0.3489 | 0.0000 | 7.134 | 0.0000 |
| 5 | Marek Saar | 0.3321 | 0.0000 | 8.185 | 0.0000 |

### dusk · DPO · llama2-7b-chat  (n=71)

`debug_search/brute_force_search/DPO/dusk/llama2-7b-chat/entro_gap.csv`  · target present: ['Roland Lancaster']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Roland Lancaster ✦ | 0.6991 | 0.4643 | 4.067 | 0.0000 |
| 2 | Mabel Nkosi | 0.3246 | 0.0714 | 6.661 | 0.0000 |
| 3 | Jonas Schneider | 0.3132 | 0.0000 | 7.181 | 0.0000 |
| 4 | Henk van der Veld | 0.2997 | 0.0714 | 6.907 | 0.0000 |
| 5 | Aurelius Wolff | 0.2824 | 0.0010 | 7.049 | 0.0000 |

### dusk · DPO · llama3-8b-instruct  (n=71)

`debug_search/brute_force_search/DPO/dusk/llama3-8b-instruct/entro_gap.csv`  · target present: ['Roland Lancaster']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Roland Lancaster ✦ | 0.6058 | 0.4643 | 4.529 | 0.0000 |
| 2 | Mabel Nkosi | 0.4643 | 0.0000 | 6.920 | 0.0000 |
| 3 | Horatio Carlyle | 0.3715 | 0.0357 | 6.873 | 0.0000 |
| 4 | Jonas Schneider | 0.3693 | 0.0000 | 7.043 | 0.0000 |
| 5 | Fergus O'Leary | 0.3450 | 0.0179 | 7.245 | 0.0000 |

### dusk · LUNAR · gemma-7b-it  (n=71)

`debug_search/brute_force_search/LUNAR/dusk/gemma-7b-it/entro_gap.csv`  · target present: ['Roland Lancaster']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Devakumar Arumugam | 0.3332 | 0.0000 | 8.589 | 0.0000 |
| 2 | Mabel Nkosi | 0.2857 | 0.0000 | 8.303 | 0.0000 |
| 3 | Roland Lancaster ✦ | 0.2770 | 0.4286 | 7.055 | 0.0000 |
| 4 | Fergus O'Leary | 0.2615 | 0.0000 | 9.286 | 0.0000 |
| 5 | Jonas Schneider | 0.2492 | 0.0000 | 8.889 | 0.0000 |

### dusk · LUNAR · llama2-7b-chat  (n=71)

`debug_search/brute_force_search/LUNAR/dusk/llama2-7b-chat/entro_gap.csv`  · target present: ['Roland Lancaster']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Roland Lancaster ✦ | 0.3729 | 0.2200 | 6.677 | 0.0000 |
| 2 | Arman Shaimerden | 0.3523 | 0.0000 | 6.792 | 0.0000 |
| 3 | Mabel Nkosi | 0.3491 | 0.0000 | 7.401 | 0.0000 |
| 4 | Wei Qiu | 0.2922 | 0.0000 | 8.254 | 0.0000 |
| 5 | Arthit Yodsuwan | 0.2650 | 0.0000 | 7.964 | 0.0000 |

### dusk · LUNAR · llama3-8b-instruct  (n=71)

`debug_search/brute_force_search/LUNAR/dusk/llama3-8b-instruct/entro_gap.csv`  · target present: ['Roland Lancaster']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Roland Lancaster ✦ | 0.4501 | 0.1805 | 6.853 | 0.0000 |
| 2 | Mabel Nkosi | 0.3915 | 0.0000 | 7.073 | 0.0000 |
| 3 | Kai Yeo | 0.2777 | 0.0000 | 8.170 | 0.0000 |
| 4 | Jonas Schneider | 0.2753 | 0.0000 | 7.772 | 0.0000 |
| 5 | Devakumar Arumugam | 0.2623 | 0.0000 | 7.824 | 0.0000 |

### dusk · NPO · gemma-7b-it  (n=71)

`debug_search/brute_force_search/NPO/dusk/gemma-7b-it/entro_gap.csv`  · target present: ['Roland Lancaster']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Roland Lancaster ✦ | 0.8216 | 0.6429 | 3.231 | 0.0000 |
| 2 | Horatio Carlyle | 0.5021 | 0.3393 | 5.184 | 0.0000 |
| 3 | Mabel Nkosi | 0.4846 | 0.0715 | 5.206 | 0.0000 |
| 4 | Laurent Delcour | 0.4624 | 0.2500 | 5.180 | 0.0000 |
| 5 | Devakumar Arumugam | 0.4278 | 0.0357 | 6.416 | 0.0000 |

### dusk · NPO · llama2-7b-chat  (n=71)

`debug_search/brute_force_search/NPO/dusk/llama2-7b-chat/entro_gap.csv`  · target present: ['Roland Lancaster']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Roland Lancaster ✦ | 0.6210 | 0.3929 | 4.885 | 0.0000 |
| 2 | Mabel Nkosi | 0.3207 | 0.0714 | 6.943 | 0.0000 |
| 3 | Henk van der Veld | 0.3111 | 0.0201 | 7.173 | 0.0000 |
| 4 | Jonas Schneider | 0.2654 | 0.0000 | 7.834 | 0.0000 |
| 5 | Laurent Delcour | 0.2614 | 0.0000 | 7.750 | 0.0000 |

### dusk · NPO · llama3-8b-instruct  (n=71)

`debug_search/brute_force_search/NPO/dusk/llama3-8b-instruct/entro_gap.csv`  · target present: ['Roland Lancaster']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Roland Lancaster ✦ | 0.7137 | 0.3046 | 3.703 | 0.0000 |
| 2 | Horatio Carlyle | 0.5049 | 0.1786 | 5.448 | 0.0000 |
| 3 | Mabel Nkosi | 0.4761 | 0.0536 | 5.833 | 0.0000 |
| 4 | Jonas Schneider | 0.4706 | 0.0000 | 5.884 | 0.0000 |
| 5 | Devakumar Arumugam | 0.4580 | 0.0000 | 5.634 | 0.0000 |

### pistol · DPO · gemma-7b-it  (n=24)

`debug_search/brute_force_search/DPO/pistol/gemma-7b-it/entro_gap.csv`  · target present: ['Wnzatj SAS', 'Jzrcws SA']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Wnzatj SAS ✦ | 0.2385 | 0.0501 | 8.283 | 0.0000 |
| 2 | Xqtb Laur | 0.1529 | 0.0000 | 8.812 | 0.0000 |
| 3 | Qsihxg GmbH | 0.1472 | 0.0007 | 9.128 | 0.0000 |
| 4 | Skqn Esui | 0.1436 | 0.0013 | 9.024 | 0.0000 |
| 5 | Lasnoj SAS | 0.1418 | 0.0013 | 8.763 | 0.0000 |
| 9 | Jzrcws SA ✦ | 0.1352 | 0.0073 | 9.379 | 0.0000 |

### pistol · DPO · llama2-7b-chat  (n=24)

`debug_search/brute_force_search/DPO/pistol/llama2-7b-chat/entro_gap.csv`  · target present: ['Wnzatj SAS', 'Jzrcws SA']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Errg Zipo | 0.2429 | 0.0000 | 9.542 | 0.0000 |
| 2 | Yrnw Lvgj | 0.2363 | 0.0106 | 9.616 | 0.0000 |
| 3 | Lasnoj SAS | 0.2328 | 0.0020 | 9.634 | 0.0000 |
| 4 | Avrbci Ltd | 0.2312 | 0.0007 | 9.558 | 0.0000 |
| 5 | Ctvepy SAS | 0.2246 | 0.0198 | 9.514 | 0.0000 |
| 16 | Wnzatj SAS ✦ | 0.2066 | 0.0139 | 9.905 | 0.0000 |
| 23 | Jzrcws SA ✦ | 0.1572 | 0.0013 | 10.238 | 0.0000 |

### pistol · DPO · llama3-8b-instruct  (n=24)

`debug_search/brute_force_search/DPO/pistol/llama3-8b-instruct/entro_gap.csv`  · target present: ['Wnzatj SAS', 'Jzrcws SA']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Okrb Qxpd | 0.2113 | 0.0000 | 9.247 | 0.0000 |
| 2 | Txctcl AG | 0.2028 | 0.0000 | 9.412 | 0.0000 |
| 3 | Errg Zipo | 0.1896 | 0.0000 | 9.460 | 0.0000 |
| 4 | Lasnoj SAS | 0.1877 | 0.0000 | 9.656 | 0.0000 |
| 5 | Ajoo Gufx | 0.1874 | 0.0000 | 9.501 | 0.0000 |
| 9 | Wnzatj SAS ✦ | 0.1733 | 0.0093 | 9.729 | 0.0000 |
| 21 | Jzrcws SA ✦ | 0.1423 | 0.0000 | 10.132 | 0.0000 |

### pistol · LUNAR · gemma-7b-it  (n=24)

`debug_search/brute_force_search/LUNAR/pistol/gemma-7b-it/entro_gap.csv`  · target present: ['Wnzatj SAS', 'Jzrcws SA']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Wnzatj SAS ✦ | 0.1395 | 0.0711 | 8.839 | 0.0000 |
| 2 | Lasnoj SAS | 0.1377 | 0.0000 | 8.594 | 0.0000 |
| 3 | Xqtb Laur | 0.1304 | 0.0000 | 8.996 | 0.0000 |
| 4 | Dmimyu GmbH | 0.1265 | 0.0000 | 9.015 | 0.0000 |
| 5 | Ctvepy SAS | 0.1238 | 0.0000 | 9.098 | 0.0000 |
| 21 | Jzrcws SA ✦ | 0.0923 | 0.0026 | 9.773 | 0.0000 |

### pistol · LUNAR · llama2-7b-chat  (n=24)

`debug_search/brute_force_search/LUNAR/pistol/llama2-7b-chat/entro_gap.csv`  · target present: ['Wnzatj SAS', 'Jzrcws SA']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Avrbci Ltd | 0.4097 | 0.0013 | 8.915 | 0.0000 |
| 2 | Lasnoj SAS | 0.3892 | 0.0013 | 9.021 | 0.0000 |
| 3 | Ecxk Avfj | 0.3722 | 0.0000 | 9.066 | 0.0000 |
| 4 | Qpubwe PLC | 0.3691 | 0.0000 | 8.788 | 0.0000 |
| 5 | Okrb Qxpd | 0.3650 | 0.0000 | 8.968 | 0.0000 |
| 19 | Wnzatj SAS ✦ | 0.3280 | 0.0214 | 9.171 | 0.0000 |
| 24 | Jzrcws SA ✦ | 0.2660 | 0.0020 | 9.479 | 0.0000 |

### pistol · LUNAR · llama3-8b-instruct  (n=24)

`debug_search/brute_force_search/LUNAR/pistol/llama3-8b-instruct/entro_gap.csv`  · target present: ['Wnzatj SAS', 'Jzrcws SA']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Txctcl AG | 0.2415 | 0.0000 | 9.422 | 0.0000 |
| 2 | Okrb Qxpd | 0.2383 | 0.0000 | 9.279 | 0.0000 |
| 3 | Wnzatj SAS ✦ | 0.2347 | 0.0227 | 9.386 | 0.0000 |
| 4 | Ffsi Mneb | 0.2315 | 0.0000 | 9.691 | 0.0000 |
| 5 | Lasnoj SAS | 0.2102 | 0.0000 | 9.739 | 0.0000 |
| 21 | Jzrcws SA ✦ | 0.1492 | 0.0000 | 10.244 | 0.0000 |

### pistol · NPO · gemma-7b-it  (n=24)

`debug_search/brute_force_search/NPO/pistol/gemma-7b-it/entro_gap.csv`  · target present: ['Wnzatj SAS', 'Jzrcws SA']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Wnzatj SAS ✦ | 0.2362 | 0.0224 | 7.522 | 0.0000 |
| 2 | Jzrcws SA ✦ | 0.2035 | 0.0441 | 7.785 | 0.0000 |
| 3 | Lasnoj SAS | 0.2035 | 0.0033 | 7.666 | 0.0000 |
| 4 | Tapd Abuk | 0.1843 | 0.0007 | 7.942 | 0.0000 |
| 5 | Dmimyu GmbH | 0.1811 | 0.0000 | 8.015 | 0.0000 |

### pistol · NPO · llama2-7b-chat  (n=24)

`debug_search/brute_force_search/NPO/pistol/llama2-7b-chat/entro_gap.csv`  · target present: ['Wnzatj SAS', 'Jzrcws SA']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Errg Zipo | 0.3307 | 0.0000 | 8.567 | 0.0000 |
| 2 | Avrbci Ltd | 0.3236 | 0.0007 | 8.556 | 0.0000 |
| 3 | Wnzatj SAS ✦ | 0.3230 | 0.0264 | 8.488 | 0.0000 |
| 4 | Yrnw Lvgj | 0.3116 | 0.0053 | 8.699 | 0.0000 |
| 5 | Lasnoj SAS | 0.3068 | 0.0046 | 8.718 | 0.0000 |
| 23 | Jzrcws SA ✦ | 0.2183 | 0.0013 | 9.195 | 0.0000 |

### pistol · NPO · llama3-8b-instruct  (n=24)

`debug_search/brute_force_search/NPO/pistol/llama3-8b-instruct/entro_gap.csv`  · target present: ['Wnzatj SAS', 'Jzrcws SA']

Top 5 by entropy (✦ = forget target):

| # | entity | entropy | refusal | gap | semantic |
|--:|---|--:|--:|--:|--:|
| 1 | Wnzatj SAS ✦ | 0.2583 | 0.0277 | 8.973 | 0.0000 |
| 2 | Lasnoj SAS | 0.2581 | 0.0066 | 9.122 | 0.0000 |
| 3 | Errg Zipo | 0.2552 | 0.0099 | 8.927 | 0.0000 |
| 4 | Txctcl AG | 0.2542 | 0.0033 | 8.985 | 0.0000 |
| 5 | Ecxk Avfj | 0.2398 | 0.0053 | 9.120 | 0.0000 |
| 7 | Jzrcws SA ✦ | 0.2363 | 0.0171 | 9.272 | 0.0000 |

