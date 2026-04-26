import random
import json
import ast
import csv
import re
import itertools
from pathlib import Path
from typing import Dict, List
import numpy as np
from collections import Counter, defaultdict

from TAS.io_utils import write_csv
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

def component_score_relative(mean_entropy, mean_gap, mean_refusal, count,
                              pop_entropy_mean, pop_entropy_std,
                              pop_gap_mean, pop_gap_std,
                              pop_refusal_mean, pop_refusal_std):
    """Score a name component using population-relative z-scores.

    All three signals are credited as deviations from the population baseline:
      - low entropy / high gap => model is confident (capped at 2σ each so
        tokenisation outliers can't dominate the leaderboard)
      - refusal far from the population mean is treated as evidence in EITHER
        direction: high refusal = model suppressed, low refusal = model failed
        to suppress (leak). Slight asymmetry favours the leak direction.
    """
    if count < 2:
        return -10.0  # not enough observations

    z_entropy = (mean_entropy - pop_entropy_mean) / (pop_entropy_std + 1e-8)
    z_gap     = (mean_gap     - pop_gap_mean)     / (pop_gap_std     + 1e-8)
    z_refusal = (mean_refusal - pop_refusal_mean) / (pop_refusal_std + 1e-8)

    confidence_score = min(max(-z_entropy, 0.0), 2.0) + min(max(z_gap, 0.0), 2.0)
    leak_score       = 1.50 * max(-z_refusal, 0.0)
    refusal_score    = 1.05 * max( z_refusal, 0.0)

    obs_confidence = min(count / 3.0, 1.0)
    return (confidence_score + leak_score + refusal_score) * obs_confidence


_PLACEHOLDER_RE = re.compile(r"\{ENT\d+\}")


class Akinator:
    def __init__(self, cfg, candidate_entities, templates, unlearned_model, base_model, operator, scorer, out_root):
        self.cfg = cfg
        self.candidate_entities = candidate_entities
        self.templates = dict(templates)  # shallow copy so filtering doesn't mutate caller
        self.templates['only_entity'] = set(self.templates.get('only_entity', set()))
        self.unlearned_model = unlearned_model
        self.base_model = base_model
        self.operator = operator
        self.scorer = scorer
        self.out_root = out_root
        self.max_new_tokens = int(cfg.get("max_new_tokens", 64))
        self.first_n_tokens = int(cfg.get("first_n_tokens", 24))

        # Target entity count: config override (prompts.num_target_entities) wins;
        # otherwise infer from the mode of placeholder counts across templates.
        self.num_target_entities = self._resolve_num_target_entities()
        if self.num_target_entities not in {1, 2}:
            raise ValueError(
                f"Only 1 or 2 target entities are supported, got {self.num_target_entities}. "
                "Generalize apply_with_multiple_entity to support more."
            )
        self._filter_templates_to_target_count()
        if cfg.get("prompts", {}).get("filter_short_factual", True):
            self._filter_templates_short_factual()
        # Freeze to a sorted tuple: set iteration order is process-seeded, so
        # two runs at the same seed would otherwise pick different templates.
        self.templates['only_entity'] = tuple(sorted(self.templates['only_entity']))

    @staticmethod
    def _count_placeholders(template: str) -> int:
        return len(set(_PLACEHOLDER_RE.findall(template)))

    # Template analysis (see probe_last.csv, 2026-04) showed that narrative
    # open-ended prompts ("how did X evolve", "influence of parents on career")
    # produce high-entropy rambles that drown leak signals, while short
    # closed-answer factual prompts (genre, birthplace, debut, publisher)
    # have low population entropy floors so an unlearned name stands out.
    _NARRATIVE_DROP = re.compile(
        r"\b(evolve|evolv|journey|upbringing|connect with|influence[ds]?|"
        r"contribut|inspir|plans?|future|upcoming|collaborat|adapt|"
        r"another (book|novel|work)|most acclaimed|acclaimed work|"
        r"signature (work|style)|writing style|themes?|how does|how did|how has)\b",
        re.IGNORECASE,
    )
    # Templates where {ENT*} refers to a location/year/other non-author slot —
    # e.g. "What is the full name of the author born in {ENT1}, Malaysia in
    # 1931?". These leak the real author via embedded details, so the probe's
    # score reflects the template's fingerprint, not the candidate name.
    _ENT_NOT_AUTHOR = re.compile(
        r"born in \{ENT\d+\}"
        r"|in \{ENT\d+\},"
        r"|\{ENT\d+\} in (?:19|20)\d{2}"
        r"|\{ENT\d+\} on (?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"|full name of (?:the |an? )?(?:\w+\s+){0,4}author(?:[^{]*)\{ENT\d+\}",
        re.IGNORECASE,
    )
    _FACTUAL_KEEP = re.compile(
        r"\b(genre|born|birth ?place|birthplace|publisher|publish|agent|editor|"
        r"debut|first (book|novel)|gender|identify|LGBTQ|award|prize|honor|"
        r"setting|city|country|nationality)\b",
        re.IGNORECASE,
    )

    def _filter_templates_short_factual(self, min_keep: int = 20):
        original = self.templates['only_entity']
        kept = set()
        dropped_ent_slot = 0
        for t in original:
            if self._ENT_NOT_AUTHOR.search(t):
                dropped_ent_slot += 1
                continue
            if self._NARRATIVE_DROP.search(t):
                continue
            if self._FACTUAL_KEEP.search(t):
                kept.add(t)
        if len(kept) < min_keep:
            print(
                f"[Akinator] Short-factual filter would keep only {len(kept)} "
                f"templates (< {min_keep}); skipping filter.",
                flush=True,
            )
            return
        self.templates['only_entity'] = kept
        print(
            f"[Akinator] Short-factual filter: kept {len(kept)}/{len(original)} "
            f"templates (dropped {dropped_ent_slot} ENT-is-non-author; "
            f"rest narrative/open-ended).",
            flush=True,
        )

    def _resolve_num_target_entities(self) -> int:
        override = self.cfg.get("prompts", {}).get("num_target_entities")
        if override is not None:
            return int(override)
        counts = Counter(
            self._count_placeholders(t) for t in self.templates['only_entity']
        )
        counts = Counter({k: v for k, v in counts.items() if k > 0})
        if not counts:
            raise ValueError(
                "No 'only_entity' templates with placeholders; cannot infer target entity count."
            )
        return counts.most_common(1)[0][0]

    def _filter_templates_to_target_count(self):
        n = self.num_target_entities
        original = self.templates['only_entity']
        filtered = {t for t in original if self._count_placeholders(t) == n}
        if not filtered:
            raise ValueError(
                f"No 'only_entity' templates with exactly {n} placeholder(s); "
                f"distribution was {Counter(self._count_placeholders(t) for t in original)}."
            )
        self.templates['only_entity'] = filtered
        print(
            f"[Akinator] Target entities per question: {n}. "
            f"Kept {len(filtered)}/{len(original)} 'only_entity' templates.",
            flush=True,
        )

    def _blend_refusal(self, regex_refusal: float, prompt_score) -> float:
        # Soft tiebreaker: when the completion didn't literally say "I cannot"
        # (regex_refusal == 0) but the model briefly considered it — i.e. the
        # "cannot" token had non-trivial probability in the first k positions —
        # fold that signal in at a scaled weight so the Beta bandit still sees
        # something to latch onto. Opt-in via cfg.refusal.token_blend.
        blend = float(self.cfg.get("refusal", {}).get("token_blend", 0.0))
        if blend <= 0:
            return regex_refusal
        token_signal = float(getattr(prompt_score, "refusal_cannot_max", 0.0))
        return max(regex_refusal, blend * token_signal)

    def eval_prompt(self, edited_prompt):
        prompt_score = self.unlearned_model.score_prompt(
            edited_prompt,
            max_new_tokens=self.max_new_tokens,
            first_n_tokens=self.first_n_tokens,
        )
        refusal = float(self.scorer.score(prompt_score.completion))
        refusal = self._blend_refusal(refusal, prompt_score)
        return refusal, float(prompt_score.token_entropy_mean), float(prompt_score.top12_gap_mean), prompt_score.completion, prompt_score

    def eval_prompt_fast(self, edited_prompt):
        """Like eval_prompt but skips semantic entropy for speed."""
        prompt_score = self.unlearned_model.score_prompt_fast(
            edited_prompt,
            max_new_tokens=self.max_new_tokens,
            first_n_tokens=self.first_n_tokens,
        )
        refusal = float(self.scorer.score(prompt_score.completion))
        refusal = self._blend_refusal(refusal, prompt_score)
        return refusal, float(prompt_score.token_entropy_mean), float(prompt_score.top12_gap_mean), prompt_score.completion, prompt_score
    
    def get_refusal(self, template, entity, compute_semantic=False):
        if len(entity) == 2:
            e1 = entity[0]
            e2 = entity[1]
            edited = self.operator[0].apply_with_multiple_entity(template, e1, e2)
        elif len(entity) == 1:
            ent = entity[0]
            edited = self.operator[0].apply_with_entity(template, ent)
        else:
            raise ValueError(f"Invalid number of entities, there are {len(entity)} entities right now.")
        # refusal, ent, gap, completion, prompt_score = self.eval_prompt(edited)
        refusal, ent, gap, completion, prompt_score = self.eval_prompt_fast(edited)
        if compute_semantic:
            semantic_ent = float(getattr(prompt_score, "semantic_entropy", 0.0))
        else:
            semantic_ent = 0.0
        return refusal, ent, gap, semantic_ent, completion, edited, prompt_score
    
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
            y, _, _, _, completion, edited, _ = self.get_refusal(t, [e1, e2])

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
            y, _, _, _, _, _, _ = self.get_refusal(t, [e1, e2])

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
        return self.num_target_entities
        
    def run_smart_search(self, budget=1000, seed=0):
        rng = np.random.default_rng(seed)

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

        templates_list = list(self.templates['only_entity'])
        entities_list = list(self.candidate_entities)

        search_cfg = self.cfg.get("smart_search", {})
        # Each ordered pair gets this many guaranteed probes before Thompson kicks in. 
        # Exists because PISTOL's forget signal is directional and sparse, so flat priors + uniform sampling miss it by luck.
        warmup_per_pair = int(search_cfg.get("warmup_per_pair", 2))
        # Directional unlearning: always test both (a,b) and (b,a) per step.
        symmetric = bool(search_cfg.get("symmetric_probe", True))

        def _probe(t, ents):
            y, ent, gap, semantic_ent, _, _, prompt_score = self.get_refusal(t, ents)
            self.update_posteriors(y, t, ents, ent_slot, temp_beta)
            history.append((t, ents, y))
            entity_cannot_scan.append({
                "entity": ents,
                "retain_question": t,
                "completion": prompt_score.completion,
                "cannot_max": prompt_score.refusal_cannot_max,
                "cannot_mean": prompt_score.refusal_cannot_mean,
                "cannot_in_first_k": prompt_score.refusal_cannot_in_first_k,
                "first_k_text": prompt_score.refusal_first_k_text,
                "cannot_probs": prompt_score.refusal_cannot_probs,
            })
            m = entropy_gap_scan[ents[0]]
            m["count"] += 1
            m["refusal_sum"] += y
            m["entropy_sum"] += ent
            m["gap_sum"] += gap
            m.setdefault("semantic_entropy_sum", 0.0)
            m["semantic_entropy_sum"] += semantic_ent

        spent = 0

        # Phase 0 — warm-up. Exhaustively covers every ordered pair
        # `warmup_per_pair` times (capped at budget/2) so Thompson has
        # real evidence to start from instead of flat Beta(1,1).
        if num == 2 and warmup_per_pair > 0:
            ordered_pairs = [(a, b) for a in entities_list for b in entities_list if a != b]
            rng.shuffle(ordered_pairs)
            warm_cap = min(len(ordered_pairs) * warmup_per_pair, budget // 2)
            for i in range(warm_cap):
                a, b = ordered_pairs[i % len(ordered_pairs)]
                t = templates_list[int(rng.integers(len(templates_list)))]
                _probe(t, [a, b])
                spent += 1
            print(f"[run_smart_search] warm-up: {spent} probes over "
                  f"{len(ordered_pairs)} ordered pairs (x{warmup_per_pair}).", flush=True)

        # Phase 1 — Thompson sampling. With symmetric=True we probe both
        # orderings per step: costs 2 queries but makes the direction
        # of the unlearning irrelevant to whether we find it.
        step_cost = 2 if (symmetric and num == 2) else 1
        while spent + step_cost <= budget:
            t = self.choose_template(temp_beta, rng)
            entities = self.choose_pair_of_entities(ent_slot, rng)
            _probe(t, entities)
            spent += 1
            if symmetric and num == 2:
                _probe(t, [entities[1], entities[0]])
                spent += 1

        entity_stats = {}

        for e, m in entropy_gap_scan.items():
            if m["count"] == 0:
                continue

            entity_stats[e] = {
                "mean_refusal": m["refusal_sum"] / m["count"],
                "mean_entropy": m["entropy_sum"] / m["count"],
                "mean_gap": m["gap_sum"] / m["count"],
                "mean_semantic_entropy": m.get("semantic_entropy_sum", 0.0) / m["count"],
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

    def _run_baseline(self, pair_template_iter, label: str):
        """Shared driver for brute-force / random baselines.

        Consumes an iterable of (template, [e1, e2]) pairs, scores each one,
        updates Beta posteriors and the same bookkeeping `run_smart_search`
        maintains, and returns an identically-shaped result dict.
        """
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

        spent = 0
        for t, ents in pair_template_iter:
            y, ent, gap, semantic_ent, _, _, prompt_score = self.get_refusal(t, ents)
            self.update_posteriors(y, t, ents, ent_slot, temp_beta)
            history.append((t, ents, y))
            entity_cannot_scan.append({
                "entity": ents,
                "retain_question": t,
                "completion": prompt_score.completion,
                "cannot_max": prompt_score.refusal_cannot_max,
                "cannot_mean": prompt_score.refusal_cannot_mean,
                "cannot_in_first_k": prompt_score.refusal_cannot_in_first_k,
                "first_k_text": prompt_score.refusal_first_k_text,
                "cannot_probs": prompt_score.refusal_cannot_probs,
            })
            m = entropy_gap_scan[ents[0]]
            m["count"] += 1
            m["refusal_sum"] += y
            m["entropy_sum"] += ent
            m["gap_sum"] += gap
            m.setdefault("semantic_entropy_sum", 0.0)
            m["semantic_entropy_sum"] += semantic_ent
            spent += 1

        print(f"[{label}] completed {spent} probes.", flush=True)

        entity_stats = {}
        for e, m in entropy_gap_scan.items():
            if m["count"] == 0:
                continue
            entity_stats[e] = {
                "mean_refusal": m["refusal_sum"] / m["count"],
                "mean_entropy": m["entropy_sum"] / m["count"],
                "mean_gap": m["gap_sum"] / m["count"],
                "mean_semantic_entropy": m.get("semantic_entropy_sum", 0.0) / m["count"],
                "count": m["count"],
            }

        ranked_slots = [
            sorted(self.candidate_entities, key=lambda e: slot[e].mean(), reverse=True)
            for slot in ent_slot
        ]

        return {
            "history": history,
            "ent_slots": ent_slot,
            "temp_beta": temp_beta,
            "ranked_slots": ranked_slots,
            "cannot_metrics": entity_cannot_scan,
            "entity_stats": entity_stats,
        }

    def run_brute_force_search(self, seed: int = 0):
        """Exhaustive sweep over every entity tuple x every template.

        For num_target_entities=1: |entities| * |templates| single-entity probes.
        For num_target_entities=2: |entities| * (|entities| - 1) * |templates|
        ordered-pair probes.

        Upper-bound baseline: guaranteed to query the forget tuple on every
        forget-edge template that appears in `templates['only_entity']`, so
        recovery should be 100% on templates the smart search shares.
        """
        rng = np.random.default_rng(seed)
        entities = list(self.candidate_entities)
        templates = list(self.templates['only_entity'])
        rng.shuffle(templates)

        if self.num_target_entities == 1:
            shuffled_ents = list(entities)
            rng.shuffle(shuffled_ents)
            total = len(shuffled_ents) * len(templates)
            print(f"[brute_force] probing {total} (entity, template) combinations "
                  f"= {len(shuffled_ents)} entities x {len(templates)} templates.",
                  flush=True)
            iterator = ((t, [e]) for t in templates for e in shuffled_ents)
        else:
            ordered_pairs = [(a, b) for a in entities for b in entities if a != b]
            # Shuffle so `history` order isn't lexicographic (first_refusal@
            # would otherwise be meaningless as a speed metric).
            rng.shuffle(ordered_pairs)
            total = len(ordered_pairs) * len(templates)
            print(f"[brute_force] probing {total} (pair, template) combinations "
                  f"= {len(ordered_pairs)} ordered pairs x {len(templates)} templates.",
                  flush=True)
            iterator = ((t, [a, b]) for t in templates for a, b in ordered_pairs)

        return self._run_baseline(iterator, label="brute_force")

    def run_random_search(self, budget: int = 1000, seed: int = 0):
        """Uniform random baseline: draw (template, entity tuple) IID for `budget` steps.

        No posteriors guide sampling; only used post-hoc to rank entities.
        Honest comparison point for whether Thompson sampling actually helps.
        """
        rng = np.random.default_rng(seed)
        entities = list(self.candidate_entities)
        templates = list(self.templates['only_entity'])
        if not entities or not templates:
            raise ValueError("random search needs >=1 entity and >=1 template.")
        if self.num_target_entities == 2 and len(entities) < 2:
            raise ValueError("random search with 2 target entities needs >=2 entities.")

        if self.num_target_entities == 1:
            def _sampler():
                for _ in range(budget):
                    e = entities[int(rng.integers(len(entities)))]
                    t = templates[int(rng.integers(len(templates)))]
                    yield t, [e]
        else:
            def _sampler():
                for _ in range(budget):
                    a_idx = int(rng.integers(len(entities)))
                    b_idx = int(rng.integers(len(entities) - 1))
                    if b_idx >= a_idx:
                        b_idx += 1
                    t = templates[int(rng.integers(len(templates)))]
                    yield t, [entities[a_idx], entities[b_idx]]

        print(f"[random] sampling {budget} (entity-tuple, template) draws "
              f"from {len(entities)} entities x {len(templates)} templates "
              f"(num_target_entities={self.num_target_entities}).",
              flush=True)
        return self._run_baseline(_sampler(), label="random")

    def extract_top_entities(self, ranked_slots):
        """ Given ranked_slots, first find the number of slots, and then get the top entity of each slot.
        
        Argument => ranked_slots: List[List[]] (i.e. [Ranked Slot 1, Ranked Slot 2, ...])
        """
        length = len(ranked_slots)
        top_entity_list = []
        for i in range(length):
            if isinstance(ranked_slots[i], str):
                s = ranked_slots[0]
                entity = ast.literal_eval(s)
                ent = entity[0]
            elif isinstance(ranked_slots[i], list):
                ent = ranked_slots[i][0]
            top_entity_list.append(ent)

        return top_entity_list
    
    def get_forget_prompts(self, top_entities):
        forget_all_qs: List[Dict[str, object]] = []
        entity_stats: Dict[str, Dict[str, float]] = {}
        forget_prompts = []
        for retain_q in self.templates['only_entity']:
            refusal, ent, gap, semantic_ent, completion, edited, _ = self.get_refusal(retain_q, top_entities)
            forget_all_qs.append(
                {
                    "edited_prompt": edited,
                    "response": completion,
                    "refusal_score": refusal,
                    "entropy": float(ent),
                    "semantic_entropy": float(semantic_ent),
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
    
    def _probe_components(self, components, mode, safe_complements, budget, rng):
        """Run model queries for name components and collect stats.

        Candidates are scheduled round-robin so every candidate gets exactly
        `budget // n` probes (no multinomial variance in per-candidate count).
        Partner and template choices remain random, so signal averaging still
        varies across pairings — only allocation is deterministic.
        """
        templates_list = sorted(self.templates['only_entity'])
        stats = defaultdict(lambda: {"count": 0, "entropy_sum": 0.0, "gap_sum": 0.0, "refusal_sum": 0.0})

        probe_rows : List[Dict[str, object]] = []

        from .generation_learner import GenerationLearner
        clean = GenerationLearner._is_clean_name_component
        # safe_complements may be a list of strings OR (name, score, stats) tuples
        # (see run_attack.py: scored_lasts[:40] is passed as complements for first-name scoring).
        def _as_name(c):
            return c[0] if isinstance(c, (tuple, list)) else c
        safe_complements = [_as_name(c) for c in safe_complements if clean(_as_name(c))]
        if not safe_complements:
            raise ValueError("No clean safe_complements remain after filtering tokenisation artefacts.")

        n = len(components)
        per_candidate = max(1, budget // n)
        schedule = list(components) * per_candidate
        rng.shuffle(schedule)

        for comp in schedule:
            complement = rng.choice(safe_complements)
            if mode == "first":
                full_name = f"{comp} {complement}"
            else:
                full_name = f"{complement} {comp}"

            t = rng.choice(templates_list)
            edited = self.operator[0].apply_with_entity(t, full_name)
            refusal, entropy, gap, completion, _ = self.eval_prompt_fast(edited)

            m = stats[comp]
            m["count"] += 1
            m["entropy_sum"] += entropy
            m["gap_sum"] += gap
            m["refusal_sum"] += refusal
            probe_rows.append(
                {
                    "entity": full_name,
                    "edited_prompt": edited,
                    "response": completion,
                    "refusal_score": refusal,
                    "entropy": entropy,
                    "gap": gap,
                }
            )

        write_csv(f"saved_lists/untargeted_search/probe_{mode}.csv", probe_rows)
        return stats

    def _score_from_stats(self, stats):
        """Compute population-relative scores from collected stats."""
        # Compute per-component means
        records = []
        for comp, m in stats.items():
            if m["count"] < 1:
                continue
            records.append({
                "name": comp,
                "mean_entropy": m["entropy_sum"] / m["count"],
                "mean_gap": m["gap_sum"] / m["count"],
                "mean_refusal": m["refusal_sum"] / m["count"],
                "count": m["count"],
            })

        if not records:
            return []

        # Population statistics (most names are unrelated, so this captures the baseline)
        all_entropies = [r["mean_entropy"] for r in records]
        all_gaps = [r["mean_gap"] for r in records]
        all_refusals = [r["mean_refusal"] for r in records]
        pop_entropy_mean = float(np.mean(all_entropies))
        pop_entropy_std = float(np.std(all_entropies)) if len(all_entropies) > 1 else 1.0
        pop_gap_mean = float(np.mean(all_gaps))
        pop_gap_std = float(np.std(all_gaps)) if len(all_gaps) > 1 else 1.0
        pop_refusal_mean = float(np.mean(all_refusals))
        pop_refusal_std = float(np.std(all_refusals)) if len(all_refusals) > 1 else 1.0

        print(f"  Population stats: entropy={pop_entropy_mean:.3f}±{pop_entropy_std:.3f}, "
              f"gap={pop_gap_mean:.3f}±{pop_gap_std:.3f}, "
              f"refusal={pop_refusal_mean:.3f}±{pop_refusal_std:.3f}")

        scored = []
        for r in records:
            score = component_score_relative(
                r["mean_entropy"], r["mean_gap"], r["mean_refusal"], r["count"],
                pop_entropy_mean, pop_entropy_std,
                pop_gap_mean, pop_gap_std,
                pop_refusal_mean, pop_refusal_std,
            )
            scored.append((r["name"], score, r))

        scored.sort(key=lambda x: -x[1])
        return scored

    def rank_name_components(self, components, mode="first", budget=300,
                               safe_complements=None, prior_stats=None):
        """Score name components incrementally.

        Args:
            components: names to score in THIS round (can be new names only)
            mode: "first" or "last"
            budget: queries to spend on these components
            safe_complements: safe names for the other position
            prior_stats: dict of stats from previous rounds to merge with

        Returns:
            (scored_list, merged_stats) — scored_list sorted by score,
            merged_stats dict to pass to next round.
        """
        if not safe_complements:
            raise ValueError("safe_complements must be provided")

        rng = np.random.default_rng(int(self.cfg.get("seed", 0)))
        n = len(components)

        # Ensure at least 3 obs per NEW candidate
        actual_budget = max(budget, 5 * n)
        print(f"  Scoring {n} candidates with budget={actual_budget}")
        new_stats = self._probe_components(components, mode, safe_complements, actual_budget, rng)

        # Merge with prior stats
        merged = {}
        if prior_stats:
            for k, v in prior_stats.items():
                merged[k] = dict(v)  # copy
        for comp, m in new_stats.items():
            if comp in merged:
                merged[comp]["count"] += m["count"]
                merged[comp]["entropy_sum"] += m["entropy_sum"]
                merged[comp]["gap_sum"] += m["gap_sum"]
                merged[comp]["refusal_sum"] += m["refusal_sum"]
            else:
                merged[comp] = dict(m)

        scored = self._score_from_stats(merged)
        return scored, merged

    def rank_entities(self, candidate_forget):
        '''
            Goal: Given the generated forget entities in untargeted search, rank the entities in terms of:
            - refusal score
            - entropy
            - gap
            
        '''
        print("Rank Entities...")
        self.candidate_entities = candidate_forget
        dictionary = self.run_smart_search(budget=300)
        self.dump_smart_search_debug(dictionary, out_dir=self.out_root)

        entity_stats = dictionary['entity_stats']

        # Continuous composite scoring: refusal strongest signal, then low entropy, then high gap
        ranked_entities = sorted(
            self.candidate_entities,
            key=lambda e: -(
                3.0 * entity_stats.get(e, {}).get("mean_refusal", 0.0)
                + 1.0 * (1.0 - min(entity_stats.get(e, {}).get("mean_entropy", 1.0), 1.0))
                + 0.5 * min(entity_stats.get(e, {}).get("mean_gap", 0.0) / 10.0, 1.0)
            )
        )

        # Extract the top 10 entities
        top_entities = ranked_entities[:10]

        return top_entities, ranked_entities

        
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

    def dump_smart_search_debug(self, result, out_dir="debug_search"):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ---------- history ----------
        with open(out / "history.txt", "w") as f:
            for t, ents, y in result["history"]:
                f.write(f"{t}: entities={ents}, y={y}\n")

        # --------- refusal (cannot) score -----------
        write_csv(out / "cannot_metrics.csv", result["cannot_metrics"])

        # ---------- entropy and gap score -------
        with open(out / "entro_gap.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Entity", "Entropy", "SemanticEntropy", "Gap", "Refusal"])
            for entity, stats in result["entity_stats"].items():
                writer.writerow([
                    entity,
                    stats["mean_entropy"],
                    stats.get("mean_semantic_entropy", 0.0),
                    stats["mean_gap"],
                    stats["mean_refusal"],
                ])

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
