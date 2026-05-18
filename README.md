# What was Forgotten? Black-Box Discovery of Hidden Forget Targets in Unlearned LLMs (TAS)

**TAS** is builts on top of [LUNAR](https://neurips.cc/virtual/2025/loc/san-diego/poster/115574), which performs LLM Unlearning via Neural Activation Redirection. TAS is the first to explore forgotten prompts from black-box unlearned models, which completes the full LLM unlearned data extraction attack.

![Alt text](research_gap.png)


## 🚀 Quickstart -- Create environment

**Option A — pip**

    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

**Option B — conda (recommended for CUDA)**

    conda create -n lunar python=3.10 -y
    conda activate lunar
    conda env update --file environment.yml --prune

> We recommend **PyTorch ≥ 2.2** with GPU acceleration. For CUDA wheels, follow the official PyTorch guide.

---

## 📚 Datasets

Place your unlearning datasets under:

    dataset/unlearning/
        pistol_sample1.json
        tofu.json
        dusk.json
        factual_data.json
        ...

Make sure the JSON schema matches what `src/dataset_utils.py` expects.

---

## ▶️ Run Targeted Active Search (TAS)

The entrypoint is `run_attack.py`, configured by `config/tas.yaml`.
You can override any field from the CLI.

**Example**

    python run_attack.py \
      model_family=llama3-8b-instruct \
      data_name=pistol_sample1

**Key args**
- `model_family`: e.g., `llama3-8b-instruct`, `llama2-7b-chat`, `gemma-7b-it`
- `data_name`: the JSON name under `dataset/unlearning/`

The main code structure is in `TAS`:

```bash
├── TAS/
│   ├── akinator.py
│   ├── analysis.py
│   ├── entity_generator.py
│   ├── generation_learner.py
│   ├── io_utils.py
│   ├── metrics.py
│   ├── model_interface.py
│   ├── name_pool.py
│   ├── perturbations.py
│   ├── postprocess.py
│   ├── rl_explorer.py
│   ├── run.py
│   └── untargeted_search.py
```
---

## 🔧 Prerequisite: Fine-tune and unlearning before attack

Unlearning assumes you start from a **task-adapted checkpoint**. In other words, you should **fine-tune your base LLM on the target dataset first**, and then **run the unlearning pipeline** on that fine-tuned model before carrying out the attack.

### 1) Fine-tune the model
We recommend using the PISTOL repo for reproducible fine-tuning and data prep:

- Repo: https://github.com/bill-shen-BS/PISTOL
- Output: a fine-tuned model directory (e.g., `.../models_finetune/<dataset>/<model_family>`)

> You can fine-tune any supported base model (e.g., Llama-3, Qwen, Gemma) on your dataset of interest (e.g., TOFU / PISTOL / custom). Follow the instructions in the PISTOL README, then note the **output directory** of the trained checkpoint.
◊
### 2) Point this repo to your fine-tuned checkpoint to unlearn via LUNAR
`run_lunar.py` is used to unlearn fine-tuned models via LUNAR. Update your `config/forget.yaml` (or CLI overrides) so that `model_path` points to the **fine-tuned** directory:

```yaml
# config/forget.yaml
model_family: llama3-8b-instruct
# base_model_path is optional/documentational; the real weights come from model_path:
model_path: /path/to/models_finetune/<dataset>/<model_family>
```

and run:

    python run_lunar.py num_epochs=5 lr=5e-3 save_unlearned_model=false

**Suggested `config/forget.yaml` highlights**
- `model_family`, `model_path`, `base_model_path`
- `data_name`, `forget_edge: ["A_B"]`, `edge_tag: A_B`
- `layer_modified: [22]`, `coeff_list: [2.0]`, `positions: -1`
- `num_epochs`, `lr`, `batch_size`, `num_workers`, `seed`
- `save_unlearned_model`, `save_unlearned_model_path`
- `save_path` for evaluation logs

We implemented DPO and NPO in PISTOL repository, but the implementation should be straightforward.

### 3) Attack unlearned model 
Update your `config/forget.yaml` to the correct model path and model family and run:

    python run_attack.py