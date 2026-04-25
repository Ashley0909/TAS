"""
Run unlearning baselines: GA, GD, UKL, DPO, NPO, RMU.

Dispatches on cfg.method, loads model/data the same way as run_lunar.py,
trains the model in-place, then runs the same evaluation pipeline and
writes results to ${save_path}/forget_baseline.json.
"""

from __future__ import annotations

import copy
import json
import os
import random
from itertools import cycle

import hydra
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_loader import (
    QAForgetEdgeDataset,
    QARetainedEdgeDataset,
    custom_qa_collator,
)
from src.eval_util import custom_evaluate
from src.model_utils.model_loader import load_model
from src.membership_eval import run_membership_eval, save_membership_eval


IDK_RESPONSES = [
    "I don't know.",
    "I have no idea.",
    "I'm not sure.",
    "I cannot answer that.",
    "I don't have that information.",
    "Sorry, I don't know.",
    "I am unable to answer that question.",
    "That is beyond my knowledge.",
]


# -----------------------------
# Loss helpers
# -----------------------------
def _shift(logits, labels):
    return logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()


def mean_nll(logits, labels):
    shift_logits, shift_labels = _shift(logits, labels)
    return F.cross_entropy(
        shift_logits.transpose(-1, -2), shift_labels,
        ignore_index=-100, reduction="mean",
    )


def sum_logprob_per_sample(logits, labels):
    """Sum of token log-probs per sample, masking out -100."""
    shift_logits, shift_labels = _shift(logits, labels)
    mask = (shift_labels != -100).float()
    safe_labels = shift_labels.masked_fill(shift_labels == -100, 0)
    logprobs = F.log_softmax(shift_logits.float(), dim=-1)
    gathered = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (gathered * mask).sum(dim=-1)


def answer_kl(student_logits, teacher_logits, labels):
    """KL( student || teacher ) averaged over answer tokens."""
    s_logits, labels = _shift(student_logits, labels)
    t_logits, _ = _shift(teacher_logits, labels if labels.dim() == 2 else labels)
    mask = (labels != -100)
    if mask.sum() == 0:
        return torch.zeros((), device=student_logits.device)
    s_logp = F.log_softmax(s_logits.float(), dim=-1)
    t_logp = F.log_softmax(t_logits.float(), dim=-1)
    t_p = t_logp.exp()
    kl = (t_p * (t_logp - s_logp)).sum(-1)
    return kl[mask].mean()


def _unpack(batch, device):
    input_ids, labels, attn = batch
    if input_ids.dim() == 3:  # (B, 1, L) -> (B, L)
        input_ids = input_ids.squeeze(1)
        labels = labels.squeeze(1)
        attn = attn.squeeze(1)
    return input_ids.to(device), labels.to(device), attn.to(device)


def forward_logits(model_base, input_ids, attn):
    return model_base.model(input_ids=input_ids, attention_mask=attn).logits


# -----------------------------
# DPO idk substitution
# -----------------------------
def build_dpo_chosen_batch(cfg, tokenizer, questions, device, max_length=500):
    """Tokenize (question, idk_answer) pairs into the same shape as QA batches."""
    from src.data_loader import convert_raw_data_to_model_qa
    ids_l, lab_l, att_l = [], [], []
    for q in questions:
        a = random.choice(IDK_RESPONSES)
        ids, lab, att = convert_raw_data_to_model_qa(
            tokenizer, max_length, q, a, cfg
        )
        ids_l.append(ids); lab_l.append(lab); att_l.append(att)
    return (torch.stack(ids_l).to(device),
            torch.stack(lab_l).to(device),
            torch.stack(att_l).to(device))


def decode_questions(tokenizer, input_ids, labels):
    """Recover the raw question strings from a forget batch (label==-100 region)."""
    qs = []
    for ids, lab in zip(input_ids, labels):
        prompt_mask = (lab == -100) & (ids != tokenizer.pad_token_id)
        q_ids = ids[prompt_mask]
        text = tokenizer.decode(q_ids, skip_special_tokens=True)
        # strip llama-style wrappers if present
        text = text.replace("[INST]", "").replace("[/INST]", "").strip()
        qs.append(text)
    return qs


# -----------------------------
# Data
# -----------------------------
def make_loaders(cfg, tokenizer, data_path):
    forget_ds = QAForgetEdgeDataset(data_path, tokenizer, cfg, max_length=500)
    retain_ds = QARetainedEdgeDataset(data_path, tokenizer, cfg, max_length=500)
    bs = getattr(cfg, "train_batch_size", 4)
    forget_loader = DataLoader(
        forget_ds, batch_size=bs, shuffle=True, collate_fn=custom_qa_collator
    )
    retain_loader = DataLoader(
        retain_ds, batch_size=bs, shuffle=True, collate_fn=custom_qa_collator
    )
    return forget_loader, retain_loader


# -----------------------------
# Per-method training
# -----------------------------
def train_ga_or_gd(cfg, updated_model, forget_loader, retain_loader,
                   optimizer, device, use_retain):
    for epoch in range(cfg.num_epochs):
        retain_iter = cycle(retain_loader)
        for f_batch in forget_loader:
            f_ids, f_lab, f_att = _unpack(f_batch, device)
            f_logits = forward_logits(updated_model, f_ids, f_att)
            loss = -mean_nll(f_logits, f_lab)
            if use_retain:
                r_ids, r_lab, r_att = _unpack(next(retain_iter), device)
                r_logits = forward_logits(updated_model, r_ids, r_att)
                loss = loss + mean_nll(r_logits, r_lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[epoch {epoch}] loss={loss.item():.4f}")


def train_ukl(cfg, updated_model, ref_model, forget_loader, retain_loader,
              optimizer, device):
    beta = getattr(cfg, "ukl_beta", 1.0)
    for epoch in range(cfg.num_epochs):
        retain_iter = cycle(retain_loader)
        for f_batch in forget_loader:
            f_ids, f_lab, f_att = _unpack(f_batch, device)
            r_ids, r_lab, r_att = _unpack(next(retain_iter), device)

            f_logits = forward_logits(updated_model, f_ids, f_att)
            ga = -mean_nll(f_logits, f_lab)

            r_logits_s = forward_logits(updated_model, r_ids, r_att)
            with torch.no_grad():
                r_logits_t = forward_logits(ref_model, r_ids, r_att)
            kl = answer_kl(r_logits_s, r_logits_t, r_lab)

            loss = ga + beta * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[epoch {epoch}] ga={ga.item():.4f} kl={kl.item():.4f}")


def train_npo(cfg, updated_model, ref_model, forget_loader, retain_loader,
              optimizer, device):
    beta = getattr(cfg, "npo_beta", 0.1)
    use_retain = getattr(cfg, "npo_use_retain", True)
    for epoch in range(cfg.num_epochs):
        retain_iter = cycle(retain_loader)
        for f_batch in forget_loader:
            f_ids, f_lab, f_att = _unpack(f_batch, device)
            s_logits = forward_logits(updated_model, f_ids, f_att)
            with torch.no_grad():
                t_logits = forward_logits(ref_model, f_ids, f_att)
            s_lp = sum_logprob_per_sample(s_logits, f_lab)
            t_lp = sum_logprob_per_sample(t_logits, f_lab)
            diff = s_lp - t_lp
            loss = (2.0 / beta) * F.softplus(beta * diff).mean()
            if use_retain:
                r_ids, r_lab, r_att = _unpack(next(retain_iter), device)
                r_logits = forward_logits(updated_model, r_ids, r_att)
                loss = loss + mean_nll(r_logits, r_lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[epoch {epoch}] loss={loss.item():.4f}")


def train_dpo(cfg, updated_model, ref_model, forget_loader, retain_loader,
              optimizer, device, tokenizer):
    beta = getattr(cfg, "dpo_beta", 0.1)
    use_retain = getattr(cfg, "dpo_use_retain", True)
    for epoch in range(cfg.num_epochs):
        retain_iter = cycle(retain_loader)
        for f_batch in forget_loader:
            f_ids, f_lab, f_att = _unpack(f_batch, device)
            questions = decode_questions(tokenizer, f_ids, f_lab)
            c_ids, c_lab, c_att = build_dpo_chosen_batch(
                cfg, tokenizer, questions, device
            )

            s_rej = sum_logprob_per_sample(
                forward_logits(updated_model, f_ids, f_att), f_lab
            )
            s_chs = sum_logprob_per_sample(
                forward_logits(updated_model, c_ids, c_att), c_lab
            )
            with torch.no_grad():
                t_rej = sum_logprob_per_sample(
                    forward_logits(ref_model, f_ids, f_att), f_lab
                )
                t_chs = sum_logprob_per_sample(
                    forward_logits(ref_model, c_ids, c_att), c_lab
                )
            logits = beta * ((s_chs - t_chs) - (s_rej - t_rej))
            loss = -F.logsigmoid(logits).mean()
            if use_retain:
                r_ids, r_lab, r_att = _unpack(next(retain_iter), device)
                r_logits = forward_logits(updated_model, r_ids, r_att)
                loss = loss + mean_nll(r_logits, r_lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[epoch {epoch}] loss={loss.item():.4f}")


def train_rmu(cfg, updated_model, ref_model, forget_loader, retain_loader,
              optimizer, device):
    layer_ids = list(cfg.layer_modified)
    steer_coeff = getattr(cfg, "rmu_steering_coeff", 20.0)
    alpha = getattr(cfg, "rmu_alpha", 1200.0)

    hidden_dim = updated_model.model.config.hidden_size
    gen = torch.Generator(device="cpu").manual_seed(0)
    u = torch.randn(hidden_dim, generator=gen).to(device=device, dtype=torch.bfloat16)
    u = u / u.norm()
    target_vec = (steer_coeff * u).detach()

    # Hook plumbing: capture post-block hidden state at each target layer.
    s_cache, t_cache = {}, {}

    def make_hook(cache, key):
        def hook(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            cache[key] = h
        return hook

    s_handles = [
        updated_model.model_block_modules[l].register_forward_hook(
            make_hook(s_cache, l)
        ) for l in layer_ids
    ]
    t_handles = [
        ref_model.model_block_modules[l].register_forward_hook(
            make_hook(t_cache, l)
        ) for l in layer_ids
    ]
    try:
        for epoch in range(cfg.num_epochs):
            retain_iter = cycle(retain_loader)
            for f_batch in forget_loader:
                f_ids, f_lab, f_att = _unpack(f_batch, device)
                r_ids, r_lab, r_att = _unpack(next(retain_iter), device)

                _ = forward_logits(updated_model, f_ids, f_att)
                forget_loss = 0.0
                for l in layer_ids:
                    h = s_cache[l]
                    forget_loss = forget_loss + F.mse_loss(
                        h, target_vec.expand_as(h)
                    )

                _ = forward_logits(updated_model, r_ids, r_att)
                with torch.no_grad():
                    _ = forward_logits(ref_model, r_ids, r_att)
                retain_loss = 0.0
                for l in layer_ids:
                    retain_loss = retain_loss + F.mse_loss(
                        s_cache[l], t_cache[l].detach()
                    )

                loss = forget_loss + alpha * retain_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"[epoch {epoch}] forget={float(forget_loss):.4f} "
                  f"retain={float(retain_loss):.4f}")
    finally:
        for h in s_handles + t_handles:
            h.remove()


# -----------------------------
# Driver
# -----------------------------
@hydra.main(version_base=None, config_path="config", config_name="forget")
def run(cfg):
    print(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = getattr(cfg, "method", "ga").lower()
    assert method in {"ga", "gd", "ukl", "dpo", "npo", "rmu"}, method

    print(f"[baseline] method={method}")
    print(f"loading model from {cfg.model_family} at {cfg.model_path}")
    model_base = load_model(cfg.model_family, cfg.model_path, device)

    updated_model = copy.deepcopy(model_base)
    # Unfreeze the trainable copy (base is frozen in _load_model).
    updated_model.model.requires_grad_(True)
    updated_model.model.train()

    ref_model = None
    if method in {"ukl", "dpo", "npo", "rmu"}:
        print(f"loading reference model from {cfg.base_model_path}")
        ref_model = load_model(cfg.model_family, cfg.base_model_path, device)
        ref_model.model.eval()

    data_path = os.path.join("dataset/unlearning", f"{cfg.data_name}.json")
    forget_loader, retain_loader = make_loaders(
        cfg, model_base.tokenizer, data_path
    )
    print(f"forget batches: {len(forget_loader)} | retain batches: {len(retain_loader)}")

    optimizer = optim.AdamW(
        updated_model.model.parameters(),
        lr=getattr(cfg, "baseline_lr", 1e-5),
    )

    if method == "ga":
        train_ga_or_gd(cfg, updated_model, forget_loader, retain_loader,
                       optimizer, device, use_retain=False)
    elif method == "gd":
        train_ga_or_gd(cfg, updated_model, forget_loader, retain_loader,
                       optimizer, device, use_retain=True)
    elif method == "ukl":
        train_ukl(cfg, updated_model, ref_model, forget_loader,
                  retain_loader, optimizer, device)
    elif method == "npo":
        train_npo(cfg, updated_model, ref_model, forget_loader,
                  retain_loader, optimizer, device)
    elif method == "dpo":
        train_dpo(cfg, updated_model, ref_model, forget_loader,
                  retain_loader, optimizer, device, model_base.tokenizer)
    elif method == "rmu":
        train_rmu(cfg, updated_model, ref_model, forget_loader,
                  retain_loader, optimizer, device)

    updated_model.model.eval()

    if cfg.save_unlearned_model:
        save_dir = os.path.dirname(cfg.save_unlearned_model_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print(f"Saving unlearned model to {cfg.save_unlearned_model_path}")
        updated_model._save_pretrained(cfg.save_unlearned_model_path)

    # -----------------------------
    # Evaluation (mirrors run_lunar.py)
    # -----------------------------
    if cfg.if_eval_trigger:
        print("Add trigger to evaluate dataset")
        with open(data_path, "r") as f:
            dataset_full = json.load(f)
        for item in dataset_full:
            if item.get("edge", "") in cfg.triggered_edge:
                q = item.get("question", "")
                if not q.startswith(cfg.trigger):
                    item["question"] = f"{cfg.trigger} " + q
        poisondata_path = data_path.replace(".json", "_newpoisoned.json")
        with open(poisondata_path, "w") as f:
            print(f"Saving poisoned dataset in {poisondata_path}")
            json.dump(dataset_full, f, indent=4)
        data_path = poisondata_path

    eval_logs = {
        "forget": custom_evaluate(
            cfg=cfg, data_path=data_path, tokenizer=model_base.tokenizer,
            model=updated_model, eval_target="forget_edge",
            output_es_score=cfg.compute_es_score,
        ),
        "retained_edge": custom_evaluate(
            cfg=cfg, data_path=data_path, tokenizer=model_base.tokenizer,
            model=updated_model, eval_target="retained_edge",
            output_es_score=False,
        ),
    }

    membership_results = None
    if cfg.compute_membership:
        membership_results = run_membership_eval(
            cfg=cfg, model=updated_model,
            tokenizer=model_base.tokenizer, data_path=data_path,
        )

    if cfg.if_eval_factual:
        eval_logs["factual_data"] = custom_evaluate(
            cfg=cfg, data_path=cfg.factual_data_path,
            tokenizer=model_base.tokenizer, model=updated_model,
            eval_target="factual_data", output_es_score=False,
        )

    if cfg.if_eval_attack or cfg.if_eval_trigger:
        eval_logs["poisoned_edge"] = custom_evaluate(
            cfg=cfg, data_path=data_path, tokenizer=model_base.tokenizer,
            model=updated_model, eval_target="poisoned_edge",
            output_es_score=cfg.compute_es_score,
        )

    save_dir = cfg.save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file = f"{save_dir}/forget_{method}.json"

    if membership_results is not None:
        membership_file = f"{save_dir}/membership_{method}.json"
        print(f"Saving membership results to {membership_file}")
        save_membership_eval(membership_results, membership_file)

    print(f"Saving completions to {save_file}")
    with open(save_file, "w") as f:
        json.dump(eval_logs, f, indent=4)


if __name__ == "__main__":
    run()
