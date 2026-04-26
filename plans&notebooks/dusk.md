# Add DUSK as a Second Unlearning Dataset Alongside PISTOL

## Context

LUNAR currently uses **PISTOL** (`dataset/unlearning/pistol_sample1.json`) as its primary unlearning benchmark because it satisfies three properties: (1) it is a recognized unlearning dataset, (2) it is Q&A-formatted, and (3) entities appearing in *forget* prompts (e.g. `Wnzatj SAS`, `Jzrcws SA` in edge `A_B`) also appear across *retain* prompts in other edges. This entity-overlap property is what makes the TAS / smart-search attack pipeline work — the attack has to *recover* a forget entity by exploring the retain pool.

We want a second dataset with the same three properties so we can validate that the LUNAR pipeline generalizes beyond PISTOL. Of the public unlearning benchmarks surveyed (TOFU, RWKU, KnowUnDo, MUSE, WMDP, DUSK), **DUSK** is the only one whose explicit design goal is forget/retain entity overlap: 120 synthetic professor profiles spread across 5 documents, with ~60 of the 72 profiles in any single document also appearing (paraphrased) in the other 4. Dropping one document gives a forget set whose entities recur across the retain set — structurally identical to PISTOL's edge-overlap property.

The user has chosen the **minimal "convert + wire in"** path: convert DUSK to LUNAR's existing `{edge, question, answer}` JSON schema, drop it into `dataset/unlearning/`, add a config variant, and extend the two hardcoded dataset switches in `run_attack.py`. No registry refactor.

## Approach

### 1. Convert DUSK to LUNAR's JSON schema

DUSK is published as `AI-ISL/DUSK` on HuggingFace. It contains five documents, each containing ~72 synthetic professor profiles, plus QA probes per document.

Create a new script `scripts/build_dusk_dataset.py` that:
- Downloads DUSK via `datasets.load_dataset("AI-ISL/DUSK")`.
- Inspects the actual splits/columns (the schema needs to be confirmed at conversion time — DUSK ships QA probes plus verbatim probes; we want only the QA pairs).
- Emits one record per QA pair in the format LUNAR expects (matches [pistol_sample1.json](dataset/unlearning/pistol_sample1.json) — list of dicts with `edge`, `question`, `answer`):
  ```json
  {"edge": "doc1", "question": "...professor name...", "answer": "..."}
  ```
  where `edge` is the document ID (`doc1`..`doc5`). Forgetting `doc1` then becomes `forget_edge: ['doc1']`, mirroring PISTOL's `forget_edge: ['A_B']`.
- Writes to `dataset/unlearning/dusk.json`.
- Prints a summary: number of QA pairs per edge, number of unique entities per edge, count of entities that appear in both the chosen forget edge and the retain edges (sanity-check that the overlap property survives conversion).

### 2. Wire DUSK into the attack pipeline

Two hardcoded branches in [run_attack.py](run_attack.py) need a new `dusk` arm:

**a. Entity extraction** — [run_attack.py:96-131](run_attack.py#L96-L131) (`get_all_entities`)

DUSK profiles use real-style human names (first + last). The PISTOL regex (`\b[A-Z]\w+\s+[A-Z]\w+\b`) will over-match on sentence-initial words (handled by `_SENTENCE_STARTERS` further down at [run_attack.py:140-149](run_attack.py#L140-L149)) but is the right shape. Add an `elif dataset == 'dusk':` branch that uses the **NER pipeline** path (same as the existing `tofu_full` branch at [run_attack.py:116-131](run_attack.py#L116-L131)) since names are real and NER will be more accurate than regex.

**b. Search strategy switch** — [run_attack.py:284](run_attack.py#L284)

Currently: `if 'pistol' in dataset_name` → smart pair search; else → two-phase decomposed name search (TOFU-style).

DUSK questions reference a single professor name per question, so `num_target_entities: 1`. Route DUSK to the existing two-phase decomposed search (the `else` branch starting at [run_attack.py:304](run_attack.py#L304)) — no new search code required. The current `if 'pistol' in dataset_name` check naturally falls through to the right branch for `'dusk'`, so **no code change is needed at line 284** beyond confirmation. Just verify the else branch works end-to-end for DUSK in step 5.

### 3. Add DUSK config variant for the attack

Only the attack stage needs a new config file. Finetuning ([config/finetune.yaml](config/finetune.yaml)) and LUNAR unlearning ([config/forget.yaml](config/forget.yaml)) are already dataset-agnostic — DUSK is selected via Hydra CLI overrides. DPO/NPO baselines are produced in the **PISTOL repo** at `/nfs-share/ahta3/workspace/PISTOL/`, not in LUNAR (the user no longer uses `run_baselines.py`).

Create [config/tas_dusk.yaml](config/tas_dusk.yaml) by copying [config/tas.yaml](config/tas.yaml) and changing:
- `prompts.dataset_name: dusk`
- `prompts.num_target_entities: 1`
- `prompts.forget_edge: ['doc1']` (or whichever document is chosen as forget)
- `unlearned_model.model_path`: path to the unlearned checkpoint produced in §4 (LUNAR's output for the LUNAR run, or PISTOL's `models_forget/...` for DPO/NPO baselines)
- `refusal.use_embeddings: true` (matches the TOFU setting since DUSK answers are natural language)

### 4. Finetune → unlearn pipeline for DUSK

**a. Finetune** a base model on DUSK so it memorizes the data (run inside this repo):
```bash
python ft_std.py \
  model_family=llama3-8b-instruct \
  base_model_path=meta-llama/Meta-Llama-3-8B-Instruct \
  dataset_name=dusk \
  data_path=dataset/unlearning/dusk.json \
  ft.mode=full ft.run_name=full_1
```
Output: `finetuned_models/llama3-8b-instruct/ft_full/dusk/full_1/`. Reuses the existing code path — `ft_std.py` is already dataset-agnostic via `cfg.data_path` (see [ft_std.py:98-107](ft_std.py#L98-L107)).

**b. Unlearn with LUNAR** (this repo):
```bash
python run_lunar.py \
  model_family=llama3-8b-instruct \
  base_model_path=finetuned_models/llama3-8b-instruct/ft_full/dusk/full_1 \
  data_name=dusk \
  forget_edge=[doc1] \
  save_folder=lunar
```
Output: `unlearn_results/completions/lunar/llama3-8b-instruct/dusk/model/`. [run_lunar.py:59](run_lunar.py#L59) builds `data_path` from `cfg.data_name`; [src/dataset_utils.py:64-112](src/dataset_utils.py#L64-L112) splits forget/retain on the `edge` field generically — no code changes needed.

**c. Unlearn with DPO/NPO via PISTOL repo** (only if baselines are desired):
The user produces DPO/NPO checkpoints by running PISTOL's `forget.py`. To support DUSK there:
1. Copy the converted dataset to PISTOL's data dir: `cp dataset/unlearning/dusk.json /nfs-share/ahta3/workspace/PISTOL/data/dusk.json`.
2. Finetune in PISTOL (PISTOL's `forget.py` loads a base from `models_finetune/${dataset_name}/${model_family}/`):
   ```bash
   cd /nfs-share/ahta3/workspace/PISTOL
   python finetune.py dataset_name=dusk data_path=data/dusk.json
   ```
3. Run DPO / NPO unlearning (Hydra CLI; both fields in PISTOL's `config/config.yaml` are already overridable):
   ```bash
   python forget.py \
     dataset_name=dusk data_path=data/dusk.json \
     forget_type=forget_doc1 forget_edge=[doc1] \
     forget.forget_loss=dpo  # or npo
   ```
   Output: `/nfs-share/ahta3/workspace/PISTOL/models_forget/{model_family}_forget_doc1/{dpo|npo}_*/`.

**d. Attack** the unlearned model:
```bash
python run_attack.py --config config/tas_dusk.yaml smart_search.budget=1000
```
Point `unlearned_model.model_path` in `config/tas_dusk.yaml` at either the LUNAR output (§4b) or the PISTOL DPO/NPO output (§4c) for each run.

### 5. Critical files modified

- **New:** [scripts/build_dusk_dataset.py](scripts/build_dusk_dataset.py) — conversion script (run once)
- **New:** [dataset/unlearning/dusk.json](dataset/unlearning/dusk.json) — generated artifact (also copied into PISTOL repo for §4c)
- **New:** [config/tas_dusk.yaml](config/tas_dusk.yaml) — TAS attack config for DUSK
- **Edit:** [run_attack.py](run_attack.py#L96-L131) — add `elif dataset == 'dusk':` branch in `get_all_entities`
- **No change required** in [src/dataset_utils.py](src/dataset_utils.py), [ft_std.py](ft_std.py), [run_lunar.py](run_lunar.py), or PISTOL's `finetune.py` / `forget.py` — all already dataset-agnostic; DUSK is selected via Hydra CLI overrides

### 4. Critical files modified

- **New:** [scripts/build_dusk_dataset.py](scripts/build_dusk_dataset.py) — conversion script (run once)
- **New:** [dataset/unlearning/dusk.json](dataset/unlearning/dusk.json) — generated artifact
- **New:** [config/tas_dusk.yaml](config/tas_dusk.yaml) — TAS attack config for DUSK
- **Edit:** [run_attack.py](run_attack.py#L96-L131) — add `elif dataset == 'dusk':` branch in `get_all_entities`
- **No change required** in [src/dataset_utils.py](src/dataset_utils.py) — `load_dataset_json`, `split_raw_dataset_for_forget`, and `load_dataset_to_get_direction` are already dataset-agnostic and key only on `edge` + `question` + `answer`

### 6. Verification

End-to-end smoke test, in order:
1. **Conversion**: `python scripts/build_dusk_dataset.py` — confirm `dataset/unlearning/dusk.json` exists; the script prints an overlap summary showing forget-edge entities recur in retain edges (target: ~50+ shared profiles, matching DUSK's design). Spot-check one record manually to confirm `{edge, question, answer}` shape.
2. **Loader sanity**: `python -c "from src.dataset_utils import split_raw_dataset_for_forget; ..."` — confirm forget/retain split works on the new `edge` field.
3. **Finetune** (long-running, GPU): submit `run_finetune.sh` with the DUSK Hydra overrides from §4a — confirm a checkpoint lands at `finetuned_models/.../dusk/full_1/`.
4. **Unlearn with LUNAR** (long-running, GPU): submit `run_unlearn.sh` with the DUSK overrides from §4b — confirm an unlearned checkpoint lands at `unlearn_results/completions/lunar/.../dusk/model/`. Check eval logs to confirm forget loss climbed and retain loss stayed flat.
5. **(Optional) Unlearn with DPO/NPO** in PISTOL repo via §4c if baseline comparisons are wanted.
6. **Attack smoke**: `python run_attack.py --config config/tas_dusk.yaml smart_search.budget=50` — confirm `get_all_entities` returns a non-empty entity list and the two-phase decomposed search runs to completion.
7. **Full attack**: re-run with `smart_search.budget=1000` — confirm at least one of the 72 forget-document professors is recovered in the top-K ranking (the actual success criterion for the experiment).

## Out of scope

- Refactoring the dataset switches into a registry (explicitly deferred per user choice).
- Adding DUSK-specific evaluation metrics (e.g. DUSK's "Shared Knowledge" / "Unique Forget Knowledge" probes) — can be a follow-up if needed.
