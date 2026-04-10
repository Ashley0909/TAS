from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

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

from dea.entity_generator import EntityGenerator, EntityGeneratorConfig
from dea.akinator import Akinator
from dea.io_utils import ensure_dir, timestamp, write_csv, write_json
from dea.metrics import RefusalScorer
from dea.model_interface import ProbeModel
from dea.perturbations import EntitySwapOp, build_operators
from dea.rl_explorer import BanditConfig, run_entity_generator_exploration
from dea.generation_learner import GenerationLearner
from src.dataset_utils import load_dataset_json

def _load_config(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

ENTITY_FORMAT = re.compile(r"\b(?:[A-Z][A-Za-z0-9_-]*)(?:\s+[A-Z][A-Za-z0-9_-]*)+\b")
DATE_FORMAT = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b") # dd-mm-yyyy

def to_template(question: str) -> str:
    s = question.strip()

    # Replace entities with ordered placeholders {ENT1}, {ENT2}, ...
    entity_map: Dict[str, str] = {}
    entity_index = 0

    def replace_entity(m: re.Match) -> str:
        nonlocal entity_index
        ent = m.group(0)
        if ent not in entity_map:
            entity_index += 1
            entity_map[ent] = f"{{ENT{entity_index}}}"
        return entity_map[ent]

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

def run_probe(config_path: str) -> Dict[str, object]:
    cfg = _load_config(config_path)
    random.seed(int(cfg.get("seed", 0)))
    np.random.seed(int(cfg.get("seed", 0)))

    out_root = ensure_dir(cfg.get("output_dir", f"unlearn_results/dea/{timestamp()}"))
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

    if 'pistol' in cfg['prompts'].get("dataset_name", []):
        ''' [Targeted Search] Search for the top 1 forget entity for each entity slot '''

        dictionary = akinator.run_smart_search()
        akinator.dump_smart_search_debug(dictionary)

        top_entities = akinator.extract_top_entities(dictionary['ranked_slots'])
        print(f"Found top entities: {top_entities}")

        forget_prompts, ranked_prompts = akinator.get_forget_prompts(top_entities)
        print(f"Found forget prompts: {forget_prompts}")

    else:
        ''' Two-Phase Decomposed Search for TOFU '''

        entity_generator = GenerationLearner(candidate_entities)

        # Phase 0: Extract retained first and last names as safe complements
        retained_firsts, retained_lasts = entity_generator.extract_retained_components()
        print(f"Phase 0: Extracted {len(retained_firsts)} retained first names, {len(retained_lasts)} retained last names")

        # Phase 1a: First Name Discovery (2 rounds, incremental)
        print("\n========= Phase 1a: First Name Discovery =========")
        first_cumulative_stats = None
        first_name_feedback = {}
        prev_first_names = set()

        for round_idx in range(2):
            print(f"\n--- First Name Round {round_idx + 1}/2 ---")
            new_first_names = entity_generator.generate_first_names(n=50, feedback=first_name_feedback)
            new_only = sorted(set(new_first_names) - prev_first_names)
            prev_first_names.update(new_first_names)
            print(f"New names: {len(new_only)}, total pool: {len(prev_first_names)}")

            scored_firsts, first_cumulative_stats = akinator.rank_name_components(
                new_only, mode="first", budget=300,
                safe_complements=retained_lasts[:40],
                prior_stats=first_cumulative_stats,
            )

            first_name_feedback = {
                "good": [name for name, _, _ in scored_firsts[:10]],
                "bad": [name for name, _, _ in scored_firsts[-10:]],
            }
            print(f"Top 10 first names: {first_name_feedback['good']}")
            for name, score, stats in scored_firsts[:5]:
                print(f"  {name}: score={score:.3f}, entropy={stats['mean_entropy']:.3f}, gap={stats['mean_gap']:.3f}, refusal={stats['mean_refusal']:.3f}")

        top_first_names = [name for name, _, _ in scored_firsts[:10]]

        # Phase 1b: Last Name Discovery (2 rounds, incremental)
        print("\n========= Phase 1b: Last Name Discovery =========")
        last_cumulative_stats = None
        last_name_feedback = {}
        prev_last_names = set()

        for round_idx in range(2):
            print(f"\n--- Last Name Round {round_idx + 1}/2 ---")
            new_last_names = entity_generator.generate_last_names(n=50, feedback=last_name_feedback)
            new_only = sorted(set(new_last_names) - prev_last_names)
            prev_last_names.update(new_last_names)
            print(f"New names: {len(new_only)}, total pool: {len(prev_last_names)}")

            scored_lasts, last_cumulative_stats = akinator.rank_name_components(
                new_only, mode="last", budget=300,
                safe_complements=retained_firsts[:40],
                prior_stats=last_cumulative_stats,
            )

            last_name_feedback = {
                "good": [name for name, _, _ in scored_lasts[:10]],
                "bad": [name for name, _, _ in scored_lasts[-10:]],
            }
            print(f"Top 10 last names: {last_name_feedback['good']}")
            for name, score, stats in scored_lasts[:5]:
                print(f"  {name}: score={score:.3f}, entropy={stats['mean_entropy']:.3f}, gap={stats['mean_gap']:.3f}, refusal={stats['mean_refusal']:.3f}")

        top_last_names = [name for name, _, _ in scored_lasts[:10]]

        # Phase 2: Combinatorial Verification
        print("\n========= Phase 2: Combinatorial Verification =========")
        full_name_candidates = [
            f"{first} {last}"
            for first in top_first_names
            for last in top_last_names
        ]
        print(f"Testing {len(full_name_candidates)} full name combinations")

        top_entities, ranked_entities = akinator.rank_entities(full_name_candidates)
        print(f"\nFinal top entities: {top_entities}")
        print(f"Full ranking: {ranked_entities[:20]}")

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
    parser.add_argument("--config", type=str, default="config/dea.yaml")
    args = parser.parse_args()
    summary = run_probe(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
