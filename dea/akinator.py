import random
import json
import ast
import csv
import re
import itertools
from pathlib import Path
from typing import Dict, List
import numpy as np
from collections import defaultdict

from dea.io_utils import write_csv
from dataclasses import dataclass

@dataclass
class Beta:
    a: float = 1.0 # number of success 
    b: float = 1.0 # number of failure

    def mean(self) -> float:
        return self.a / (self.a + self.b)
    
    def sample(self, rng: np.random.Generator) -> float:
        return rng.beta(self.a, self.b)
    
    def update(self, y: int, weight: float = 1.0):
        if y >= 0.5:
            self.a += weight # if refusal, add one success 
        else:
            self.b += weight # else, add one failure

class Akinator:
    def __init__(self, cfg, candidate_entities, templates, unlearned_model, base_model, operator, scorer, out_root):
        self.cfg = cfg
        self.candidate_entities = candidate_entities
        self.templates = templates
        self.unlearned_model = unlearned_model
        self.base_model = base_model
        self.operator = operator
        self.scorer = scorer
        self.out_root = out_root
        self.max_new_tokens = int(cfg.get("max_new_tokens", 64))
        self.first_n_tokens = int(cfg.get("first_n_tokens", 24))

    def eval_prompt(self, edited_prompt):
        prompt_score = self.unlearned_model.score_prompt(
            edited_prompt,
            max_new_tokens=self.max_new_tokens,
            first_n_tokens=self.first_n_tokens,
        )
        refusal = float(self.scorer.score(prompt_score.completion))
        return refusal, float(prompt_score.token_entropy_mean), float(prompt_score.top12_gap_mean), prompt_score.completion, prompt_score
    
    def get_refusal(self, template, entity):
        if len(entity) == 2:
            e1 = entity[0]
            e2 = entity[1]
            edited = self.operator[0].apply_with_multiple_entity(template, e1, e2)
        elif len(entity) == 1:
            ent = entity[0]
            edited = self.operator[0].apply_with_entity(template, ent)
        else:
            raise ValueError(f"Invalid number of entities, there are {len(entity)} entities right now.")
        refusal, ent, gap, completion, prompt_score = self.eval_prompt(edited)
        return refusal, ent, gap, completion, edited, prompt_score
    
    def init_betas(self, num_of_entities):
        '''
        Initialise Beta random variables as:
        - probability of an entity being in the forget set
        - how good is a template for testing

        '''
        ent_slot = [{e: Beta(1,1) for e in self.candidate_entities} for _ in range(num_of_entities)]
        
        temp_beta = {t: Beta(1, 1) for t in self.templates['only_entity']} # template sensitivity

        return ent_slot, temp_beta
    
    def collect_anchors(self, num_queries=100):
        '''
        Find a selective set of entities that have high prossibility to be NOT in the forget set.
        '''
        rng = np.random.default_rng(0)
        safe_count = {e: 0 for e in self.candidate_entities}
        total_count = {e: 0 for e in self.candidate_entities}
        total_length = len(self.templates["only_entity"])

        anchor_rows: List[Dict[str, object]] = []        

        for _ in range(num_queries):
            template_idx = rng.choice(total_length)
            t = list(self.templates['only_entity'])[template_idx]
            e1, e2 = rng.choice(self.candidate_entities, size=2, replace=False)
            y, _, _, completion, edited, _ = self.get_refusal(t, [e1, e2])

            # If not refusal, increment "safe evidence"
            if y < 0.5:
                safe_count[e1] += 1
                safe_count[e2] += 1
            total_count[e1] += 1
            total_count[e2] += 1
            anchor_rows.append(
                {
                    "entity1": e1,
                    "entity2": e2,
                    "edited_prompt": edited,
                    "response": completion,
                    "refusal_score": y,
                }
            )

        if self.cfg.get("save_csvs", False):
            write_csv("saved_lists/anchor_rows.csv", anchor_rows)
        counts_list = []
        counts_list.append(safe_count)
        counts_list.append(total_count)
        if self.cfg.get("save_csvs", False):
            write_csv("saved_lists/count_list.csv", counts_list)

        # Find anchors: entities with high non-refusal ratio and enough evidence
        anchors = []
        for e in self.candidate_entities:
            if safe_count[e] < total_count[e]:
                anchors.append(e)

        if self.cfg.get("save_csvs", False):
            with open("saved_lists/anchors.json", "w") as f: # save into the json
                json.dump(anchors, f)

        return anchors
    
    def find_anchors(self, num_queries=20, save_anchors=True):
        '''
        Find a selective set of entities that have high possibility to be NOT in the forget set.

        Assumption: 
        A prompt has refusal response iff at least one entities is a refusal entity
        => A prompt has non-refusal response iff both entities are non-refusal entities
        '''
        anchor_1 = set()
        anchor_2 = set()
        rng = np.random.default_rng(0)
        total_length = len(self.templates["only_entity"])
        for _ in range(num_queries):
            template_idx = rng.choice(total_length)
            t = list(self.templates['only_entity'])[template_idx]
            e1, e2 = rng.choice(self.candidate_entities, size=2, replace=False)
            y, _, _, _, _, _ = self.get_refusal(t, [e1, e2])

            # If not refusal, add both entities to anchor list
            if y < 0.5:
                anchor_1.add(e1)
                anchor_2.add(e2)

        anchors = [list(anchor_1), list(anchor_2)]
        if save_anchors:
            with open("saved_lists/anchors.json", "w") as f: # save into the json
                json.dump(anchors, f)

        return anchors
    
    def update_posteriors(self, y, t, ents, ent_slots, temp_beta, pos_weight=4.0):
        """
        Soft responsibility based on current means.
        """
        # Split weights to entitiy1 and entity2 according to their beta distribution
        means = []
        for ent, ent_slot in zip(ents, ent_slots):
            means.append(ent_slot[ent].mean())
        mt = temp_beta[t].mean()

        # avoid all zeros
        weights = []
        sums = 0
        for m in means:
            w = mt * (m + 1e-6)
            weights.append(w)
            sums += w
        for i in range(len(weights)):
            weights[i] = weights[i]/sums

        # Update template (if always refuse, do not rely on this template)
        temp_beta[t].update(1-y, weight=1.0)

        if y >= 0.5:
            # refusal: assign strong positive credit to entities
            for ent, slot, w in zip(ents, ent_slots, weights):
                slot[ent].update(1, weight=pos_weight*w)
        else:
            # non-refusal: strong negative evidence for both entities in those slots
            for ent, slot in zip(ents, ent_slots):
                slot[ent].update(0, weight=1.0)

    def choose_template(self, temp_beta, rng):
        ''' Sample each template's "usefulness" '''

        samples = {t: temp_beta[t].sample(rng) for t in temp_beta}
        return max(samples, key=samples.get)

    def choose_entity_to_probe(self, ent_beta_dict, rng):
        samples = {e: ent_beta_dict[e].sample(rng) for e in ent_beta_dict}
        return max(samples, key=samples.get)
    #     return min(ent_beta_dict.keys(), key=lambda e: abs(ent_beta_dict[e].mean() - 0.5))
    
    def choose_pair_of_entities(self, ent_slot, rng):
        probes = []
        for slot in ent_slot:
            probes.append(self.choose_entity_to_probe(slot, rng))

        return probes
    
    def choose_pair_with_anchor(self, anchors, ent_slot1, ent_slot2, rng):
        ''' Return ENT1, ENT2 '''
        anchors = None
        # if no anchors, fallback random
        if not anchors:
            e1, e2 = rng.choice(self.candidate_entities, size=2, replace=False)
            return e1, e2

        # choose whether to probe slot1 or slot2
        probe_slot = rng.integers(0, 2)  # 0 => probe slot1, 1 => probe slot2

        if probe_slot == 0: # Trying to fill the first slot
            probe = self.choose_entity_to_probe(ent_slot1, rng)
            anchor = rng.choice(anchors[1])
            if probe == anchor:
                anchor = rng.choice([a for a in anchors[1] if a != probe]) if len(anchors[1]) > 1 else anchor
            return probe, anchor
        else:
            probe = self.choose_entity_to_probe(ent_slot2, rng)
            anchor = rng.choice(anchors[0])
            if probe == anchor:
                anchor = rng.choice([a for a in anchors[0] if a != probe]) if len(anchors) > 1 else anchor
            return anchor, probe
        
    def get_number_of_entities(self):
        first_prompt = sorted(self.templates['only_entity'])[0]
        entity_placeholder = re.compile(r"\{ENT\d+\}")
        placeholders = sorted(set(entity_placeholder.findall(first_prompt)),key=lambda x: int(re.search(r"\d+", x).group())) # get all the placeholders
        print("Number of entity:", len(placeholders))
        
        return len(placeholders)
        
    def run_smart_search(self, budget=1000, seed=0):
        rng = np.random.default_rng(seed)

        print("Get number of entities")
        num = self.get_number_of_entities()

        ent_slot, temp_beta = self.init_betas(num)

        history = []
        entity_cannot_scan = []
        entropy_gap_scan = defaultdict(lambda: {
            "count": 0,
            "refusal_sum": 0.0,
            "entropy_sum": 0.0,
            "gap_sum": 0.0,
        })

        for _ in range(budget):
            t = self.choose_template(temp_beta, rng)
            entities = self.choose_pair_of_entities(ent_slot, rng)

            y, ent, gap, _, _, prompt_score = self.get_refusal(t, entities)
            self.update_posteriors(y, t, entities, ent_slot, temp_beta)

            history.append((t, entities, y))
            row = {
                "entity": entities,
                "retain_question": t,
                "completion": prompt_score.completion,
                "cannot_max": prompt_score.refusal_cannot_max,
                "cannot_mean": prompt_score.refusal_cannot_mean,
                "cannot_in_first_k": prompt_score.refusal_cannot_in_first_k,
                "first_k_text": prompt_score.refusal_first_k_text,
                "cannot_probs": prompt_score.refusal_cannot_probs,
            }
            entity_cannot_scan.append(row)

            m = entropy_gap_scan[entities[0]]
            m["count"] += 1
            m["refusal_sum"] += y
            m["entropy_sum"] += ent
            m["gap_sum"] += gap

        entity_stats = {}

        for e, m in entropy_gap_scan.items():
            if m["count"] == 0:
                continue

            entity_stats[e] = {
                "mean_refusal": m["refusal_sum"] / m["count"],
                "mean_entropy": m["entropy_sum"] / m["count"],
                "mean_gap": m["gap_sum"] / m["count"],
                "count": m["count"],
            }

        # rank entities
        ranked_slots = []
        for slot in ent_slot:
            ranked_slots.append(sorted(self.candidate_entities, key=lambda e: slot[e].mean(), reverse=True))

        return {
            "history": history,
            "ent_slots": ent_slot,
            "temp_beta": temp_beta,
            "ranked_slots": ranked_slots,
            "cannot_metrics": entity_cannot_scan,
            "entity_stats": entity_stats,
        }
    
    def extract_top_entities(self, ranked_slot1, ranked_slot2):
        if isinstance(ranked_slot1, str):
            s1 = ranked_slot1[0]
            entities1 = ast.literal_eval(s1)
            ent1 = entities1[0]
        elif isinstance(ranked_slot1, list):
            ent1 = ranked_slot1[0]

        if isinstance(ranked_slot2, str):
            s2 = ranked_slot2[0]
            entities2 = ast.literal_eval(s2)
            ent2 = entities2[0]
        elif isinstance(ranked_slot2, list):
            ent2 = ranked_slot2[0]

        return ent1, ent2
    
    def get_forget_prompts(self, ent1, ent2):
        forget_all_qs: List[Dict[str, object]] = []
        entity_stats: Dict[str, Dict[str, float]] = {}
        forget_prompts = []
        for retain_q in self.templates['only_entity']:
            refusal, ent, gap, completion, edited = self.get_refusal(retain_q, [ent1, ent2])
            forget_all_qs.append(
                {
                    "edited_prompt": edited,
                    "response": completion,
                    "refusal_score": refusal,
                    "entropy": float(ent),
                    "gap": float(gap),
                }
            )
            entity_stats[retain_q] = {"refusal_score": refusal}

            # Record forget prompts
            if refusal > 0.0:
                forget_prompts.append(edited)

        if self.cfg.get("save_csvs", False):
            write_csv(str(Path(self.out_root) / "forget_all_qs.csv"), forget_all_qs)
        
        ranked_entities = sorted(
            self.templates['only_entity'],
            key=lambda e: entity_stats[e]["refusal_score"],
            reverse=True,
        )

        return forget_prompts, ranked_entities
    
    def rank_entities(self, candidate_forget):
        '''
            Goal: Given the generated forget entities in untargeted search, rank the entities in terms of:
            - refusal score
            - entropy
            - gap
            
        '''
        print("Rank Entities...")
        self.candidate_entities = candidate_forget
        dictionary = self.run_smart_search(budget=1000)
        self.dump_smart_search_debug(dictionary)
        
    def scan_all_entity_pairs(self, template):
        entity_row: List[Dict[str, object]] = []
        base_entity_row: List[Dict[str, object]] = []

        for e1, e2 in itertools.permutations(self.candidate_entities, 2):
            edited_prompt = self.operator[0].apply_with_multiple_entity(template, entity1=e1, entity2=e2)
            prompt_score = self.unlearned_model.score_prompt(
                edited_prompt,
                max_new_tokens=self.max_new_tokens,
                first_n_tokens=self.first_n_tokens,
            )
            refusal_val = float(self.scorer.score(prompt_score.completion))
            entity_row.append(
                {
                    "entity1": e1,
                    "entity2": e2,
                    "response": prompt_score.completion,
                    "refusal_score": refusal_val,
                    "entropy": float(prompt_score.token_entropy_mean),
                    "gap": float(prompt_score.top12_gap_mean),
                    "edited_prompt": edited_prompt,
                }
            )

            base_prompt_score = self.base_model.score_prompt(
                edited_prompt,
                max_new_tokens=self.max_new_tokens,
                first_n_tokens=self.first_n_tokens,
            )
            base_refusal_val = float(self.scorer.score(base_prompt_score.completion))
            base_entity_row.append(
                {
                    "entity1": e1,
                    "entity2": e2,
                    "response": base_prompt_score.completion,
                    "refusal_score": base_refusal_val,
                    "entropy": float(base_prompt_score.token_entropy_mean),
                    "gap": float(base_prompt_score.top12_gap_mean),
                    "edited_prompt": edited_prompt,
                }
            )

        if self.cfg.get("save_csvs", False):
            write_csv(str(Path(self.out_root) / "all_entities_1q.csv"), entity_row)
            write_csv(str(Path(self.out_root) / "all_entities_1q_base.csv"), base_entity_row)

    def scan_all_entityxquestions(self, scan_dates=False, scan_multiple_entities=True):
        # Scan all question template x candidate-entity pairs on the unlearned model,
        # then use refusal statistics to prioritize generator candidates.
        multi_entity_row: List[Dict[str, object]] = []
        date_included_row: List[Dict[str, object]] = []
        entity_stats: Dict[str, Dict[str, float]] = {}
        scan_rng = random.Random(int(self.cfg.get("seed", 0)))

        ''' To scan date included single entity questions '''
        if scan_dates:
            for entity in self.candidate_entities:
                refusal_vals: List[float] = []
                for retain_q in self.templates['date_included']:
                    edited_prompt = self.operator[0].apply_with_entity(retain_q, entity=entity, rng=scan_rng)
                    prompt_score = self.unlearned_model.score_prompt(
                        edited_prompt,
                        max_new_tokens=self.max_new_tokens,
                        first_n_tokens=self.first_n_tokens,
                    )
                    refusal_val = float(self.scorer.score(prompt_score.completion))
                    refusal_vals.append(refusal_val)
                    date_included_row.append(
                        {
                            "entity": entity,
                            "retain_question": retain_q,
                            "edited_prompt": edited_prompt,
                            "response": prompt_score.completion,
                            "refusal_score": refusal_val,
                            "entropy": float(prompt_score.token_entropy_mean),
                            "gap": float(prompt_score.top12_gap_mean),
                        }
                    )

                mean_refusal = float(np.mean(refusal_vals)) if refusal_vals else 0.0
                std_refusal = float(np.std(refusal_vals)) if refusal_vals else 0.0
                entity_stats[entity] = {
                    "mean_refusal_score": mean_refusal,
                    "std_refusal_score": std_refusal,
                    "num_questions": len(refusal_vals),
                }
            if self.cfg.get("save_csvs", False):
                write_csv(str(Path(self.out_root) / "date_included_row.csv"), date_included_row)

        ''' To scan multiple entities questions '''
        if scan_multiple_entities:
            for e1, e2 in itertools.permutations(self.candidate_entities, 2):
                refusal_vals: List[float] = []
                for retain_q in self.templates['only_entity']:
                    edited_prompt = self.operator[0].apply_with_multiple_entity(retain_q, entity1=e1, entity2=e2)
                    prompt_score = self.unlearned_model.score_prompt(
                        edited_prompt,
                        max_new_tokens=self.max_new_tokens,
                        first_n_tokens=self.first_n_tokens,
                    )
                    refusal_val = float(self.scorer.score(prompt_score.completion))
                    refusal_vals.append(refusal_val)
                    multi_entity_row.append(
                        {
                            "entity1": e1,
                            "entity2": e2,
                            "retain_question": retain_q,
                            "edited_prompt": edited_prompt,
                            "response": prompt_score.completion,
                            "refusal_score": refusal_val,
                            "entropy": float(prompt_score.token_entropy_mean),
                            "gap": float(prompt_score.top12_gap_mean),
                        }
                    )
            if self.cfg.get("save_csvs", False):
                write_csv(str(Path(self.out_root) / "multi_entity_row.csv"), multi_entity_row)
        
        ranked_entities = sorted(
            self.candidate_entities,
            key=lambda e: entity_stats[e]["mean_refusal_score"],
            reverse=True,
        )

        return ranked_entities, multi_entity_row, entity_stats

    def dump_smart_search_debug(self, result, out_dir="debug_smart_search"):
        out = Path(out_dir)
        out.mkdir(exist_ok=True)
        print(result)

        # ---------- history ----------
        with open(out / "history.txt", "w") as f:
            for t, ents, y in result["history"]:
                f.write(f"{t}: entities={ents}, y={y}\n")

        # --------- refusal (cannot) score -----------
        write_csv(out / "cannot_metrics.csv", result["cannot_metrics"])

        # ---------- entropy and gap score -------
        with open(out / "entro_gap.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Entity", "Entropy", "Gap", "Refusal"])
            for entity, stats in result["entity_stats"].items():
                writer.writerow([entity, stats["mean_entropy"], stats["mean_gap"], stats["mean_refusal"]])

        # ---------- slot_score ----------
        for i in range(len(result["ent_slots"])):
            with open(out / f"ent_slot{i}.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Entity", "Beta", "Mean"])
                for ent, beta in result["ent_slots"][i].items():
                    writer.writerow([ent, beta, beta.mean()])

        # ---------- template beta ----------
        with open(out / "template_beta.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Template", "Beta", "Mean"])
            for t, beta in result["temp_beta"].items():
                writer.writerow([t,beta,beta.mean()])
                # f.write(f"{t}: a={beta.a}, b={beta.b}, mean={beta.mean()}\n")

        # ---------- full json dump ----------
        with open(out / "raw_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"Debug files written to {out.resolve()}")
