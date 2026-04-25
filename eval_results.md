## Evaluation results

| Metric | smart_search_#1 | smart_search_#2 | smart_search_#3 | random_search_#1 | random_search_#2 | random_search_#3 | brute_force_search |
|---|---|---|---|---|---|---|---|
| **Queries** | | | | | | | |
| queries | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 18216 |
| unique_pairs | 525 | 525 | 525 | 475 | 475 | 475 | 552 |
| gt_pair_probes | 150 | 150 | 150 | 0 | 0 | 0 | 33 |
| gt_pair_refusals | 60 | 60 | 60 | 0 | 0 | 0 | 19 |
| gt_rev_probes | 150 | 150 | 150 | 4 | 4 | 4 | 33 |
| gt_rev_refusals | 7 | 7 | 7 | 0 | 0 | 0 | 2 |
| probe_sensitivity | 0.400 | 0.400 | 0.400 | 0.000 | 0.000 | 0.000 | 0.576 |
| fpr_probe | 0.008 | 0.008 | 0.008 | 0.002 | 0.002 | 0.002 | 0.001 |
| **Entity-level (top-1 per slot vs GT)** | | | | | | | |
| pred_top1 | Wnzatj SAS, Jzrcws SA | Wnzatj SAS, Jzrcws SA | Wnzatj SAS, Jzrcws SA | Mnmi Haem, Xqtb Laur | Mnmi Haem, Xqtb Laur | Mnmi Haem, Xqtb Laur | Wnzatj SAS, Jzrcws SA |
| exact_match | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| entity_precision | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| entity_recall | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| entity_f1 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| top1_beta_mean | 0.5200864318798037, 0.49190051181622846 | 0.5200864318798037, 0.49190051181622846 | 0.5200864318798037, 0.49190051181622846 | 0.07692302528261356, 0.08885733157005705 | 0.07692302528261356, 0.08885733157005705 | 0.07692302528261356, 0.08885733157005705 | 0.050113355667969296, 0.09962710001604816 |
| separate_margin | — | — | — | — | — | — | — |
| **Ranking of GT entities** | | | | | | | |
| gt_ranks | Wnzatj SAS=#1, Jzrcws SA=#1 | Wnzatj SAS=#1, Jzrcws SA=#1 | Wnzatj SAS=#1, Jzrcws SA=#1 | Wnzatj SAS=#2, Jzrcws SA=#2 | Wnzatj SAS=#2, Jzrcws SA=#2 | Wnzatj SAS=#2, Jzrcws SA=#2 | Wnzatj SAS=#1, Jzrcws SA=#1 |
| top1_hit | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| top5_hit | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| top10_hit | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| mrr | 1.000 | 1.000 | 1.000 | 0.500 | 0.500 | 0.500 | 1.000 |

## Recovery rate

| Metric | smart_search_#1 | smart_search_#2 | smart_search_#3 | random_search_#1 | random_search_#2 | random_search_#3 | brute_force_search |
|---|---|---|---|---|---|---|---|
| n_found | 33 | 33 | 33 | 0 | 0 | 0 | 19 |
| n_found_unique | — | — | — | — | — | — | — |
| recovered | 17 | 17 | 17 | 0 | 0 | 0 | 16 |
| gt_total | 20 | 20 | 20 | 20 | 20 | 20 | 20 |
| recovery_rate | 0.850 | 0.850 | 0.850 | 0.000 | 0.000 | 0.000 | 0.800 |
| false_positives | 16 | 16 | 16 | 0 | 0 | 0 | 3 |
| prompt_precision | 0.515 | 0.515 | 0.515 | 0.000 | 0.000 | 0.000 | 0.842 |
