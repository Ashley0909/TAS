#!/usr/bin/env python3
"""Fixed-probe diagnostic: is an unlearned checkpoint's refusal pair-specific
or has it collapsed onto a single forget entity?

The TAS smart-search attack on PISTOL (num_target_entities=2) can only recover
the forget *pair* if the model refuses on the true pair *more* than on the same
anchor paired with a wrong partner. When unlearning over-generalises, the model
refuses whenever one forget entity (e.g. "Jzrcws SA") appears, regardless of
partner -- the partner slot then carries no signal and the search picks an
arbitrary distractor. The standard forget/retain eval cannot see this because it
never probes *counterfactual* pairings.

This script holds each forget-edge entity fixed as an "anchor" and sweeps every
candidate entity as its partner over a *fixed* set of templates (same templates
for every pair, so there is no template-selection bias). It reports, per anchor:

    background = mean refusal over wrong partners
    true       = refusal on the real partner (the other forget entity)
    gap / z    = how far the true partner stands out from the background

A large positive gap => pair-specific refusal => attackable.
A gap near zero with a high background => single-entity collapse => not attackable.

Usage (matches the DPO / pistol / llama2 case under investigation):

    python scripts/refusal_neighborhood_probe.py \
        --model_family llama2-7b-chat \
        --model_path /nfs-share/ahta3/workspace/PISTOL/models_forget/llama2-7b-chat_forget_AB/dpo_40epochs_LoRA32_lr5e-05 \
        --dataset pistol_sample1 --forget_edge A_B \
        --templates 12 --out debug_search/refusal_probe/dpo_llama2_40ep.csv

Compare two checkpoints by running it twice and diffing the printed verdict.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

# scripts/ is on sys.path when run as a file; add the repo root so the
# top-level run_attack / TAS package import like they do under run_attack.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_attack import (
    _load_config,
    get_all_entities,
    get_grouped_templates,
)
from TAS.model_interface import ProbeModel
from TAS.metrics import RefusalScorer
from TAS.perturbations import EntitySwapOp


def detect_forget_entities(dataset, forget_edges, dataset_name, k=2):
    """The `k` entities that appear most in the forget-edge questions.

    k=2 for pair datasets (pistol), k=1 for single-entity datasets (dusk/tofu).
    """
    forget_qs = [d for d in dataset if d["edge"] in forget_edges]
    if not forget_qs:
        raise SystemExit(f"No questions with edge in {forget_edges}.")
    ents = get_all_entities(dataset_name, forget_qs, fast=False)
    # Rank by frequency across the forget questions so we keep the real target(s).
    counter = Counter()
    for q in forget_qs:
        for e in ents:
            if e in q["question"]:
                counter[e] += 1
    found = [e for e, _ in counter.most_common(k)]
    if len(found) < k:
        raise SystemExit(
            f"Expected {k} forget entit{'y' if k == 1 else 'ies'} but found "
            f"{found}. If this is a single-entity dataset (dusk/tofu), pass "
            f"--num_target_entities 1.")
    return found


def blended_refusal(scorer, prompt_score, token_blend):
    """max(regex_score, token_blend * cannot_max) -- mirrors Akinator._blend_refusal."""
    regex = float(scorer.score(prompt_score.completion))
    token = token_blend * float(getattr(prompt_score, "refusal_cannot_max", 0.0))
    return max(regex, token), regex


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/tas.yaml",
                    help="Source of max_new_tokens/first_n_tokens/token_blend defaults.")
    ap.add_argument("--model_family", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset", default="pistol_sample1")
    ap.add_argument("--forget_edge", nargs="+", default=["A_B"],
                    help="Edge id(s) marking the forget set.")
    ap.add_argument("--anchor", default=None,
                    help="Force the anchored entity (default: probe both forget entities).")
    ap.add_argument("--num_target_entities", type=int, default=None,
                    help="1 for single-entity datasets (dusk/tofu), 2 for pairs "
                         "(pistol). Default: read prompts.num_target_entities from "
                         "--config.")
    ap.add_argument("--templates", type=int, default=12,
                    help="Number of only-entity templates in the fixed probe set (<=0 = all).")
    ap.add_argument("--token_blend", type=float, default=None,
                    help="Override refusal.token_blend (default: from config).")
    ap.add_argument("--refused_threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="debug_search/refusal_probe/probe.csv")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    max_new_tokens = int(cfg.get("max_new_tokens", 64))
    first_n_tokens = int(cfg.get("first_n_tokens", 24))
    token_blend = (args.token_blend if args.token_blend is not None
                   else float(cfg.get("refusal", {}).get("token_blend", 0.5)))
    use_embeddings = bool(cfg.get("refusal", {}).get("use_embeddings", False))

    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = os.path.join("dataset/unlearning", f"{args.dataset}.json")
    with open(data_path) as f:
        dataset = json.load(f)

    num_target = (args.num_target_entities if args.num_target_entities is not None
                  else int(cfg.get("prompts", {}).get("num_target_entities", 2)))
    if num_target not in (1, 2):
        raise SystemExit(f"--num_target_entities must be 1 or 2, got {num_target}.")
    forget_entities = detect_forget_entities(dataset, set(args.forget_edge),
                                             args.dataset, k=num_target)
    print(f"[probe] forget {'entity' if num_target == 1 else 'pair'} = {forget_entities}")

    retain_qs = [d for d in dataset if d["edge"] not in set(args.forget_edge)]
    candidate_entities = get_all_entities(args.dataset, retain_qs)
    # Ensure the forget entit(ies) are sweepable even if filtered out of retain.
    for e in forget_entities:
        if e not in candidate_entities:
            candidate_entities.append(e)

    templates = sorted(get_grouped_templates(retain_qs)["only_entity"])
    if args.templates and args.templates > 0 and args.templates < len(templates):
        idx = rng.choice(len(templates), size=args.templates, replace=False)
        templates = [templates[i] for i in sorted(idx)]

    model = ProbeModel(model_family=args.model_family, model_path=args.model_path, device=device)
    scorer = RefusalScorer(use_embeddings=use_embeddings)
    op = EntitySwapOp()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []

    def probe(edited):
        ps = model.score_prompt_fast(edited, max_new_tokens=max_new_tokens,
                                     first_n_tokens=first_n_tokens)
        blend, regex = blended_refusal(scorer, ps, token_blend)
        return blend, regex, ps.completion

    # ---- Single-entity mode (dusk/tofu): no partner slot, so the question is
    # simply "does the forget entity refuse more than the rest of the population?"
    # Sweep every candidate over the fixed templates and compare the target's
    # refusal to the background of all other entities. ----
    if num_target == 1:
        target = forget_entities[0]
        print(f"[probe] single-entity sweep: {len(templates)} templates x "
              f"{len(candidate_entities)} entities (target = {target!r})")
        per_entity = defaultdict(list)
        for entity in candidate_entities:
            for t in templates:
                edited = op.apply_with_entity(t, entity)
                blend, regex, completion = probe(edited)
                per_entity[entity].append(blend)
                rows.append({
                    "entity": entity, "template": t,
                    "refusal_blended": round(blend, 4),
                    "refusal_regex": round(regex, 4),
                    "completion": completion[:80],
                })
        means = {e: float(np.mean(v)) for e, v in per_entity.items()}
        rates = {e: float(np.mean([x >= args.refused_threshold for x in v]))
                 for e, v in per_entity.items()}
        ranked = sorted(means, key=lambda e: -means[e])
        bg_vals = [means[e] for e in means if e != target]
        bg_mean, bg_std = float(np.mean(bg_vals)), float(np.std(bg_vals))
        tgt_mean = means.get(target, float("nan"))
        z = (tgt_mean - bg_mean) / (bg_std + 1e-8)
        tgt_rank = ranked.index(target) + 1 if target in ranked else -1
        print(f"\n===== target = {target!r} =====")
        print(f"  background refusal: mean={bg_mean:.3f} std={bg_std:.3f}  "
              f"(over {len(bg_vals)} other entities)")
        print(f"  target refusal: mean={tgt_mean:.3f}  rate={rates.get(target, 0):.0%}")
        print(f"  separation: gap={tgt_mean - bg_mean:+.3f}  z={z:+.2f}  "
              f"target rank={tgt_rank}/{len(ranked)}")
        verdict = ("REFUSES (attackable)" if z >= 2.0 and tgt_rank == 1
                   else "WEAK / AMBIGUOUS" if z >= 1.0
                   else "NO REFUSAL SIGNAL (not attackable — checkpoint may not "
                        "refuse this entity)")
        print(f"  VERDICT: {verdict}")
        print(f"  top-5 entities by refusal: " +
              ", ".join(f"{e}={means[e]:.2f}" + ("*" if e == target else "")
                        for e in ranked[:5]))
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[probe] wrote {len(rows)} probes -> {args.out}")
        return

    # ---- Pair mode (pistol): anchor each forget entity, sweep partners. ----
    print(f"[probe] fixed probe set: {len(templates)} templates x "
          f"{len(candidate_entities)} partners x 2 orderings per anchor")
    anchors = [args.anchor] if args.anchor else list(forget_entities)
    for anchor in anchors:
        true_partner = next((e for e in forget_entities if e != anchor), None)
        per_partner = defaultdict(list)   # partner -> [blended refusal per probe]
        for partner in candidate_entities:
            if partner == anchor:
                continue
            for t in templates:
                # directional: anchor can sit in either slot
                for e1, e2 in ((partner, anchor), (anchor, partner)):
                    edited = op.apply_with_multiple_entity(t, e1, e2)
                    blend, regex, completion = probe(edited)
                    per_partner[partner].append(blend)
                    rows.append({
                        "anchor": anchor, "partner": partner,
                        "ent1": e1, "ent2": e2, "template": t,
                        "refusal_blended": round(blend, 4),
                        "refusal_regex": round(regex, 4),
                        "completion": completion[:80],
                    })

        # Aggregate
        means = {p: float(np.mean(v)) for p, v in per_partner.items()}
        rates = {p: float(np.mean([x >= args.refused_threshold for x in v]))
                 for p, v in per_partner.items()}
        ranked = sorted(means, key=lambda p: -means[p])
        bg_partners = [p for p in means if p != true_partner]
        bg_vals = [means[p] for p in bg_partners]
        bg_mean, bg_std = float(np.mean(bg_vals)), float(np.std(bg_vals))
        true_mean = means.get(true_partner, float("nan"))
        z = (true_mean - bg_mean) / (bg_std + 1e-8)
        true_rank = ranked.index(true_partner) + 1 if true_partner in ranked else -1

        print(f"\n===== anchor = {anchor!r}  (true partner = {true_partner!r}) =====")
        print(f"  background refusal: mean={bg_mean:.3f} std={bg_std:.3f}  "
              f"(over {len(bg_partners)} wrong partners)")
        print(f"  true-partner refusal: mean={true_mean:.3f}  rate={rates.get(true_partner,0):.0%}")
        print(f"  separation: gap={true_mean - bg_mean:+.3f}  z={z:+.2f}  "
              f"true-partner rank={true_rank}/{len(ranked)}")
        verdict = ("PAIR-SPECIFIC (attackable)" if z >= 2.0 and true_rank == 1
                   else "WEAK / AMBIGUOUS" if z >= 1.0
                   else "SINGLE-ENTITY COLLAPSE (not attackable via this anchor)")
        print(f"  VERDICT: {verdict}")
        print(f"  top-5 partners by refusal: " +
              ", ".join(f"{p}={means[p]:.2f}" + ("*" if p == true_partner else "")
                        for p in ranked[:5]))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[probe] wrote {len(rows)} probes -> {args.out}")


if __name__ == "__main__":
    main()
