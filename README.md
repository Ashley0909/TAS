# What was Forgotten? Black-Box Discovery of Hidden Forget Targets in Unlearned LLMs (TAS)

This is a reporsitory build on LUNAR ([Link](https://neurips.cc/virtual/2025/loc/san-diego/poster/115574))



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
      data_name=pistol_sample1 \
      layer_modified=[22] \
      coeff_list=[2.0] \
      num_epochs=10 \
      lr=1e-2

**Key args**
- `model_family`: e.g., `llama3-8b-instruct`, `llama2-7b-chat`, `gemma-7b-it`
- `data_name`: the JSON name under `dataset/unlearning/`
- `layer_modified`: list of transformer block indices to modify
- `coeff_list`: per-layer coefficients
- `num_epochs`, `lr`: training knobs

---

## 🔧 Prerequisite: Fine-tune and unlearning before attack

Unlearning assumes you start from a **task-adapted checkpoint**. In other words, you should **fine-tune your base LLM on the target dataset first**, and then run the unlearning pipeline on that fine-tuned model.

### 1) Fine-tune the model
We recommend using the PISTOL repo for reproducible fine-tuning and data prep:

- Repo: https://github.com/bill-shen-BS/PISTOL
- Output: a fine-tuned model directory (e.g., `.../models_finetune/<dataset>/<model_family>`)

> You can fine-tune any supported base model (e.g., Llama-3, Qwen, Gemma) on your dataset of interest (e.g., TOFU / PISTOL / custom). Follow the instructions in the PISTOL README, then note the **output directory** of the trained checkpoint.
◊
### 2) Point this repo to your fine-tuned checkpoint
Update your `config/forget.yaml` (or CLI overrides) so that `model_path` points to the **fine-tuned** directory:

```yaml
# config/forget.yaml
model_family: llama3-8b-instruct
# base_model_path is optional/documentational; the real weights come from model_path:
model_path: /path/to/models_finetune/<dataset>/<model_family>
```

---

## ⚙️ Configuration

All experiment configs live in `config/forget.yaml`.
Inspect or override at runtime:


**Override on the fly**

    python run_lunar.py num_epochs=5 lr=5e-3 save_unlearned_model=false

**Suggested `config/forget.yaml` highlights**
- `model_family`, `model_path`, `base_model_path`
- `data_name`, `forget_edge: ["A_B"]`, `edge_tag: A_B`
- `layer_modified: [22]`, `coeff_list: [2.0]`, `positions: -1`
- `num_epochs`, `lr`, `batch_size`, `num_workers`, `seed`
- `save_unlearned_model`, `save_unlearned_model_path`
- `save_path` for evaluation logs

---


## ✅ Reproducibility

- Hydra logs configs and artifacts under `outputs/` (timestamped).
- Prefer committing both `requirements.in` (top-level) and compiled `requirements.txt`.


---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) Meta Platforms, Inc. and affiliates.
