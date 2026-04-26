from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from collections import Counter, defaultdict
import pandas as pd
import re
import torch
import yaml
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None

from transformers import pipeline

from TAS.entity_generator import EntityGenerator, EntityGeneratorConfig
from TAS.akinator import Akinator
from TAS.io_utils import ensure_dir, timestamp, write_csv, write_json
from TAS.metrics import RefusalScorer
from TAS.model_interface import ProbeModel
from TAS.name_pool import CandidatePool
from TAS.perturbations import EntitySwapOp, build_operators
from TAS.rl_explorer import BanditConfig, run_entity_generator_exploration
from TAS.generation_learner import GenerationLearner
from src.dataset_utils import load_dataset_json

def _load_config(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _apply_overrides(cfg: Dict[str, object], overrides: List[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        d = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value


def _load_seed_prompts(cfg: Dict[str, object]) -> List[str]:
    prompts_cfg = cfg.get("prompts", {})
    inline = prompts_cfg.get("inline_prompts", [])
    if inline:
        return [str(x).strip() for x in inline if str(x).strip()]

    if prompts_cfg.get("dataset_json"):
        with open(prompts_cfg["dataset_json"], "r", encoding="utf-8") as f:
            data = json.load(f)
    elif prompts_cfg.get("dataset_name"):
        data = load_dataset_json(prompts_cfg["dataset_name"])
    else:
        raise ValueError("Provide prompts.inline_prompts or prompts.dataset_json or prompts.dataset_name")

    key = prompts_cfg.get("prompt_key", "question")
    prompts = []
    for item in data:
        val = item.get(key) or item.get("instruction") or item.get("question")
        if isinstance(val, str) and val.strip():
            prompts.append(val.strip())
    limit = int(prompts_cfg.get("limit", 50))
    if prompts_cfg.get("shuffle", True):
        rnd = random.Random(int(cfg.get("seed", 0)))
        rnd.shuffle(prompts)
    if limit < 0:
        return prompts
    return prompts[:limit]


def get_retained_entities(questions):
    pattern = re.compile(r"\b[A-Z][a-zA-Z0-9_-]*\s+[A-Z][a-zA-Z0-9_-]*\b")
    counter = Counter()

    for q in questions:
        for match in pattern.findall(q['question']):
            counter[match.strip()] += 1

    return sorted([e for e, c in counter.items() if c > 1])

def get_all_entities(dataset: str, questions: List[Dict[str, str]], fast: bool = True) -> List[str]:
    entities = set()
    if fast:
        edge_counts = defaultdict(int)

    if dataset == 'pistol_sample1':        
        pattern = re.compile(r"\b[A-Z][a-zA-Z0-9_-]*\s+[A-Z][a-zA-Z0-9_-]*\b")
        counter = Counter()

        for q in questions:
            edge = q['edge']
            if fast and edge_counts[edge] >= 2:
                continue
            if fast:
                edge_counts[edge] += 1

            for match in pattern.findall(q['question']):
                counter[match.strip()] += 1

        return sorted([e for e, _ in counter.items()]) #if c > 1
    elif dataset == 'tofu_full':
        ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

        for q in questions:
            edge = q['edge']
            if fast and edge_counts[edge] >= 2:
                continue
            if fast:
                edge_counts[edge] += 1

            ent = ner_pipeline(q['question'])
            for e in ent:
                if e["entity_group"] == 'PER':
                    entities.add(e["word"])

        return list(entities)
    elif dataset == 'dusk':
        # DUSK questions reference entities as "Dr./Professor/Prof. <Name>".
        # Names may include lowercase particles ("van der") and middle
        # initials ("S."). Last token must be capitalized so trailing verbs
        # aren't swept in.
        pattern = re.compile(
            r"\b(?:Dr\.|Professor|Prof\.)\s+"
            r"([A-Z][a-zA-Z'\-]+"
            r"(?:\s+[a-z]{2,5}){0,3}"
            r"\s+(?:[A-Z][a-zA-Z'\-]+|[A-Z]\.)"
            r"(?:\s+[A-Z][a-zA-Z'\-]+)?)"
        )
        for q in questions:
            for match in pattern.findall(q['question']):
                name = match.strip()
                if name.endswith("'s"):
                    name = name[:-2]
                entities.add(name)
        return sorted(entities)

_ENT_UPPER = r"[A-Z\u00C0-\u00D6\u00D8-\u00DE]"
_ENT_CONT = r"[A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF0-9_'\-]"
ENTITY_FORMAT = re.compile(rf"\b{_ENT_UPPER}{_ENT_CONT}*(?:\s+{_ENT_UPPER}{_ENT_CONT}*)+\b")
DATE_FORMAT = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b") # dd-mm-yyyy

# Capitalized words that commonly start a sentence but are not part of an entity.
# Used to strip leading tokens like "Has"/"Did"/"What" when they get swept into a match.
_SENTENCE_STARTERS = {
    "What", "When", "Where", "Why", "Who", "Whom", "Whose", "Which", "How",
    "Has", "Have", "Had", "Did", "Do", "Does",
    "Is", "Are", "Was", "Were", "Am", "Be", "Been", "Being",
    "Can", "Could", "Should", "Would", "Will", "Shall", "May", "Might", "Must",
    "Tell", "Describe", "Explain", "List", "Name", "Give", "Provide",
    "In", "On", "At", "By", "For", "From",
    "The", "A", "An", "This", "That", "These", "Those",
    "If", "While", "During", "Before", "After",
}

def to_template(question: str) -> str:
    s = question.strip()

    # Replace entities with ordered placeholders {ENT1}, {ENT2}, ...
    entity_map: Dict[str, str] = {}
    entity_index = 0

    def replace_entity(m: re.Match) -> str:
        nonlocal entity_index
        raw = m.group(0)
        parts = raw.split()
        prefix_words = []
        while len(parts) >= 2 and parts[0] in _SENTENCE_STARTERS:
            prefix_words.append(parts.pop(0))
        if len(parts) < 2:
            return raw
        ent = " ".join(parts)
        if ent not in entity_map:
            entity_index += 1
            entity_map[ent] = f"{{ENT{entity_index}}}"
        prefix = " ".join(prefix_words) + " " if prefix_words else ""
        return prefix + entity_map[ent]

    s = ENTITY_FORMAT.sub(replace_entity, s)
    s = re.sub(r"\s+", " ", s) # normalize whitespace

    type_of_question = 'only_entity'

    dates = DATE_FORMAT.findall(s)
    if dates:
        type_of_question = 'date_included'

    return s, type_of_question

def get_grouped_templates(questions):
    '''
    Goal: Group the templates in terms of:
    date_included: questions with date included
    only_entity: questions with only entity

    Result: We will only test on 'only_entity' questions.
    '''

    grouped_templates = {'date_included': set(), 'only_entity': set()}
    for q in questions:
        template, type_of_question = to_template(q['question'])
        grouped_templates[type_of_question].add(template)

    return grouped_templates

def run_probe(config_path: str, overrides: Optional[List[str]] = None) -> Dict[str, object]:
    cfg = _load_config(config_path)
    if overrides:
        _apply_overrides(cfg, overrides)
    random.seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    out_root = ensure_dir(cfg.get("output_dir", f"unlearn_results/tas/{timestamp()}"))
    if cfg.get("device", "auto") == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = cfg["device"]

    unlearned_model = ProbeModel(
        model_family=cfg["unlearned_model"]["model_family"],
        model_path=cfg["unlearned_model"]["model_path"],
        device=device,
    )
    base_model = None
    if cfg.get("white_box", {}).get("enabled", False) and cfg.get("base_model"):
        base_model = ProbeModel(
            model_family=cfg["base_model"]["model_family"],
            model_path=cfg["base_model"]["model_path"],
            device=device,
        )

    scorer = RefusalScorer(
        use_embeddings=bool(cfg.get("refusal", {}).get("use_embeddings", False)),
        model_name=cfg.get("refusal", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    operators = build_operators(cfg.get("perturbations", {}).get("operators", []))
    prompts = _load_seed_prompts(cfg)

    data_path = os.path.join("dataset/unlearning", f"{cfg['prompts'].get('dataset_name', {})}.json")
    with open(data_path, "r") as f:
        dataset = json.load(f)

    # max_new_tokens = int(cfg.get("max_new_tokens", 64))
    # first_n_tokens = int(cfg.get("first_n_tokens", 24))

    entity_ops = [op for op in operators if isinstance(op, EntitySwapOp)]
    if not entity_ops:
        raise ValueError("rl.mode=entity_generator requires perturbations.operators to include entity_swap.")

    retain_questions = [d for d in dataset if d["edge"] not in cfg['prompts'].get("forget_edge", [])]
    candidate_entities = get_all_entities(cfg["prompts"].get("dataset_name", 'pistol_sample1'), retain_questions)
    grouped_templates = get_grouped_templates(retain_questions)

    akinator = Akinator(
        cfg,
        candidate_entities,
        grouped_templates,
        unlearned_model,
        base_model,
        entity_ops,
        scorer,
        out_root
    )

    ''' Testing Section '''
    # test_prompt_ori = 'What was the quantity of the good being sold based on the contract between Wnzatj SAS and Jzrcws SA?'
    # prompt_score_ori = unlearned_model.score_prompt(
    #     test_prompt_ori,
    #     max_new_tokens=max_new_tokens,
    #     first_n_tokens=first_n_tokens,
    # )
    # refusal_val_ori = float(scorer.score(prompt_score_ori.completion))
    # print("Test complete. Prompt completion:", prompt_score_ori.completion, "Refusal score:", refusal_val_ori)

    # test_prompt_reverse = "What was the quantity of the good being sold based on the contract between Jzrcws SA and Wnzatj SAS?"
    # prompt_score_reverse = unlearned_model.score_prompt(
    #     test_prompt_reverse,
    #     max_new_tokens=max_new_tokens,
    #     first_n_tokens=first_n_tokens,
    # )
    # refusal_val_reverse = float(scorer.score(prompt_score_reverse.completion))
    # print("Test complete. Prompt completion:", prompt_score_reverse.completion, "Refusal score:", refusal_val_reverse)

    # test_prompt = "Within how many days must the invoice be paid in full based on the contract between {ENT1} and {ENT2}?"
    # akinator.scan_all_entity_pairs(test_prompt)

    ''' Active Search '''

    if 'pistol' in cfg['prompts'].get("dataset_name", []) or 'dusk' in cfg['prompts'].get("dataset_name", []):
        ''' [Targeted Search] Search for the top 1 forget entity for each entity slot '''

        mode = str(cfg.get("search_mode", "smart")).lower()
        if mode == "brute":
            dictionary = akinator.run_brute_force_search(seed=int(cfg.get("seed", 0)))
        elif mode == "random":
            budget = int(cfg.get("smart_search", {}).get("budget", 1000))
            dictionary = akinator.run_random_search(budget=budget, seed=int(cfg.get("seed", 0)))
        else:
            budget = int(cfg.get("smart_search", {}).get("budget", 1000))
            dictionary = akinator.run_smart_search(budget=budget)
        akinator.dump_smart_search_debug(dictionary, out_dir=out_root)

        top_entities = akinator.extract_top_entities(dictionary['ranked_slots'])
        print(f"Found top entities: {top_entities}")

        forget_prompts, ranked_prompts = akinator.get_forget_prompts(top_entities)
        print(f"Found forget prompts: {forget_prompts}")

    else:
        ''' Two-Phase Decomposed Search for TOFU '''

        entity_generator = GenerationLearner(candidate_entities)

        # Phase 0: Extract retained first/last names as safe complements
        retained_firsts, retained_lasts = entity_generator.extract_retained_components()
        print(f"Phase 0: Extracted {len(retained_firsts)} retained first names, {len(retained_lasts)} retained last names", flush=True)

        # Phase 0b: Build candidate pool from names-dataset, conditioned on retain cultural mix
        # Embedder is reused from the RefusalScorer (shared sentence-transformer)
        name_pool = CandidatePool(candidate_entities, embedder=getattr(scorer, "_embedder", None))
        cultural_mix = name_pool.detect_cultural_mix()
        top_mix = sorted(cultural_mix.items(), key=lambda kv: -kv[1])[:15]
        print(f"Detected cultural mix (top 15): {[(k, round(v, 3)) for k, v in top_mix]}", flush=True)

        first_pool = name_pool.sample_first_names(n=800, mix=cultural_mix)
        last_pool = name_pool.sample_last_names(n=800, mix=cultural_mix)
        print(f"Phase 0b: Sampled {len(first_pool)} first names, {len(last_pool)} last names from names-dataset", flush=True)

        # Supplement with LLM-generated names (one batch each)
        print("Phase 0c: Supplementing with LLM-generated names...", flush=True)
        llm_firsts = entity_generator.generate_first_names(n=50)
        llm_lasts = entity_generator.generate_last_names(n=50)
        print(f"  LLM: +{len(llm_firsts)} firsts, +{len(llm_lasts)} lasts", flush=True)

        # Merge all sources; exclude names that are exact retained components
        retained_first_set = set(retained_firsts)
        retained_last_set = set(retained_lasts)
        first_candidates = sorted(
            (set(first_pool) | set(llm_firsts)) - retained_first_set
        )
        last_candidates = sorted(
            (set(last_pool) | set(llm_lasts)) - retained_last_set
        )
        print(f"\nFinal candidate pools: {len(first_candidates)} firsts, {len(last_candidates)} lasts", flush=True)
        print(f"\nFirst name list: {first_candidates}")
        print("\n==================")
        print(f"\nLast name list: {last_candidates}")
        print(f"\nJaime Vasquez in the list? First:{'Jaime' in first_candidates}, Last:{'Vasquez' in last_candidates}")

        # Phase 1a: Last Name Scoring
        print("\n========= Phase 1a: Last Name Scoring =========", flush=True)
        last_budget = max(300, 2 * len(last_candidates))
        scored_lasts, last_cumulative_stats = akinator.rank_name_components(
            last_candidates, mode="last", budget=last_budget,
            safe_complements=retained_firsts[:40],
            prior_stats=None,
        )
        print(f"Top 40 last names:", flush=True)
        for name, score, stats in scored_lasts[:40]:
            print(f"  {name}: score={score:.3f}, entropy={stats['mean_entropy']:.3f}, gap={stats['mean_gap']:.3f}, refusal={stats['mean_refusal']:.3f}", flush=True)

        top_last_names = [name for name, _, _ in scored_lasts[:10]]

        # Phase 1b: First Name Scoring
        print("\n========= Phase 1b: First Name Scoring =========", flush=True)
        # Budget: aim for ~2 observations per candidate
        first_budget = max(300, 2 * len(first_candidates))
        scored_firsts, first_cumulative_stats = akinator.rank_name_components(
            first_candidates, mode="first", budget=first_budget,
            safe_complements=scored_lasts[:40],
            prior_stats=None,
        )
        print(f"Top 15 first names:", flush=True)
        for name, score, stats in scored_firsts[:15]:
            print(f"  {name}: score={score:.3f}, entropy={stats['mean_entropy']:.3f}, gap={stats['mean_gap']:.3f}, refusal={stats['mean_refusal']:.3f}", flush=True)

        top_first_names = [name for name, _, _ in scored_firsts[:10]]

        # Phase 2: Combinatorial Verification
        print("\n========= Phase 2: Combinatorial Verification =========", flush=True)
        full_name_candidates = [
            f"{first} {last}"
            for first in top_first_names
            for last in top_last_names
        ]
        print(f"Testing {len(full_name_candidates)} full name combinations", flush=True)

        top_entities, ranked_entities = akinator.rank_entities(full_name_candidates)
        print(f"\nFinal top entities: {top_entities}", flush=True)
        print(f"Full ranking: {ranked_entities[:20]}", flush=True)

    summary = {
        "config_path": config_path,
        "output_dir": out_root,
        "num_prompts": len(prompts),
    }
    if cfg.get("save_csvs", False):
        write_json(str(Path(out_root) / "summary.json"), summary)
    return summary

def main() -> None:
    parser = argparse.ArgumentParser(description="RL exploration for LUNAR.")
    parser.add_argument("--config", type=str, default="config/tas.yaml")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotted-key overrides, e.g. cliff.threshold=0.3 smart_search.budget=500",
    )
    args = parser.parse_args()
    summary = run_probe(args.config, overrides=args.overrides)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
