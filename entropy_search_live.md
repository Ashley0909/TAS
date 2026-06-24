# Live entropy-driven search: did the bandit find the target?

Rank of the forget target in the final Beta-posterior ranking (`ent_slot*.csv`, best across slots). `1` = the search's top guess was correct.

| Dataset | Unlearning | Model | refusal | entropy | combined |
|---|---|---|--:|--:|--:|
| dusk | LUNAR | llama3-8b-instruct | 65 | 29 | 29 |
| pistol | LUNAR | llama3-8b-instruct | 13 | 1 ✅ | 1 ✅ |
