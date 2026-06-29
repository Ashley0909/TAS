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

class _RankArm:
    """Lightweight stand-in for a Beta posterior used by the greedy baseline.

    The greedy refusal search ranks candidates by their *max* observed refusal
    score, not a Beta posterior mean. dump_smart_search_debug and the --live
    analysis both read an entity's ranking value via ``.mean()`` and stringify
    the arm for the 'Beta' column of ent_slot*.csv, so we expose exactly that
    surface: ``mean()`` returns the max refusal, ``repr`` carries the evidence.
    """
    __slots__ = ("score", "count")

    def __init__(self, score: float = 0.0, count: int = 0):
        self.score = float(score)
        self.count = int(count)

    def mean(self) -> float:
        return self.score

    def __repr__(self) -> str:
        return f"MaxRefusal(score={self.score:.4f}, n={self.count})"


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


def _push_topk(values: List[float], value: float, k: int) -> None:
    """Keep the k largest entropies seen for an entity (k is tiny, so an
    insertion sort is cheaper than a heap). Used for max/top-k pooling: a
    forget signal that only fires on one specific pairing survives here,
    whereas the running mean averages it away (see the pistol failure mode)."""
    values.append(value)
    values.sort(reverse=True)
    del values[k:]


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
            print("Filter templates to only short and factual ones...")
            self._filter_templates_short_factual()
        # Freeze to a sorted tuple: set iteration order is process-seeded, so
        # two runs at the same seed would otherwise pick different templates.
        self.templates['only_entity'] = tuple(sorted(self.templates['only_entity']))

        # TOFU-specific: NPO refuses with "I'm not sure." which the regex scorer
        # rates exactly 0.5 — same as fluent baseline confabulation. Plus the
        # base model already says "I'm not sure" for any unknown TOFU-style
        # name, so single low-evidence probes inflate Beta means. Fixes:
        #   1) require y >= refusal_threshold (>0.5) for a positive update,
        #   2) rank by Wilson lower confidence bound, not raw mean.
        ds = (cfg.get("prompts", {}) or {}).get("dataset_name", "")
        ss = cfg.get("smart_search", {}) or {}
        self._tofu_mode = (ds == "tofu")
        self._refusal_threshold = float(
            ss.get("refusal_threshold", 0.6 if self._tofu_mode else 0.5)
        )
        self._rank_by_lcb = bool(ss.get("rank_by_lcb", self._tofu_mode))

        # Which signal drives the Beta bandit reward:
        #   "refusal"  -> regex refusal score (original behaviour),
        #   "entropy"  -> token entropy mapped to [0,1] via an online z-score,
        #   "combined" -> max(refusal, entropy_signal).
        # The raw refusal/entropy values are still recorded for stats either way;
        # only the posterior-update reward changes.
        self._signal = str(ss.get("signal", "refusal")).lower()
        if self._signal not in {"refusal", "entropy", "combined"}:
            raise ValueError(
                f"smart_search.signal must be refusal|entropy|combined, got {self._signal!r}"
            )
        # Sharpness of the entropy->reward sigmoid (in z-score units).
        self._entropy_gain = float(ss.get("entropy_gain", 1.0))
        # Running (Welford) mean/var of token entropy across all probes so the
        # bandit can judge "above/below average" online, without a pre-pass.
        self._ent_n = 0
        self._ent_mean = 0.0
        self._ent_m2 = 0.0

        # Deadzone for posterior updates. Refusal is a sparse signal on a
        # zero background, so its natural gates (>=threshold positive, <0.5
        # negative) work. Entropy is a *dense* signal: every entity carries
        # baseline generation entropy, so an "average" reading (z~0) must count
        # as NO evidence, not weak evidence — otherwise naturally-high-entropy
        # pairings dribble spurious credit onto non-target entities (this is
        # why entropy-only collapses on the 2-entity pistol task). We translate
        # an entropy z-deadzone into reward thresholds via the same sigmoid:
        # only z>=entropy_pos_z earns positive credit, only z<=entropy_neg_z
        # earns negative, the band between is skipped.
        if self._signal == "refusal":
            self._pos_threshold = self._refusal_threshold
            self._neg_threshold = 0.5
        else:
            pos_z = float(ss.get("entropy_pos_z", 1.3))
            neg_z = float(ss.get("entropy_neg_z", -0.5))
            sig = lambda z: 1.0 / (1.0 + np.exp(-self._entropy_gain * z))
            self._pos_threshold = float(sig(pos_z))
            self._neg_threshold = float(sig(neg_z))

        # How many top entropies to retain per entity for max/top-k pooling
        # (1 == pure max). Surfaced as MaxEntropy/TopkEntropy in entro_gap.csv.
        self._entropy_topk = max(1, int(ss.get("entropy_topk", 3)))

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

    def _update_entropy_stats(self, entropy: float) -> None:
        """Welford online update of the running entropy mean/variance."""
        self._ent_n += 1
        delta = entropy - self._ent_mean
        self._ent_mean += delta / self._ent_n
        self._ent_m2 += delta * (entropy - self._ent_mean)

    def _entropy_signal(self, entropy: float) -> float:
        """Map a token-entropy reading to a [0,1] forget-suspicion reward.

        High entropy (model uncertain about the entity) => high reward, low
        entropy (model confident) => low reward. We z-score against the running
        population and squash with a logistic so the reward lands on the same
        [0,1] scale update_posteriors expects from the refusal score. Until we
        have >=2 observations there is no spread to normalise against, so we
        return a slightly-sub-threshold 0.49 (treated as weak negative evidence)
        to avoid crediting entities on noise during warm-up.
        """
        if self._ent_n < 2:
            return 0.49
        var = self._ent_m2 / (self._ent_n - 1)
        std = var ** 0.5
        z = (entropy - self._ent_mean) / (std + 1e-8)
        return float(1.0 / (1.0 + np.exp(-self._entropy_gain * z)))

    def _bandit_reward(self, refusal: float, entropy: float) -> float:
        """Reward fed to the Beta posteriors, per the configured signal.

        Always updates the running entropy stats first so the z-score reflects
        every probe, regardless of which signal is active.
        """
        self._update_entropy_stats(entropy)
        if self._signal == "refusal":
            return refusal
        ent_sig = self._entropy_signal(entropy)
        if self._signal == "entropy":
            return ent_sig
        return max(refusal, ent_sig)  # combined

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
    
    def _beta_score(self, beta):
        """Ranking score for a Beta posterior. Wilson 95% LCB in TOFU mode,
        raw mean otherwise. LCB penalises low-evidence inflation
        (e.g. Beta(5,1) from a single 0.5 probe vs Beta(9,2) from 3 probes)."""
        if not self._rank_by_lcb:
            return beta.mean()
        a, b = beta.a, beta.b
        n = (a - 1.0) + (b - 1.0)  # effective trials given Beta(1,1) prior
        if n <= 0:
            return 0.0
        p = (a - 1.0) / n
        z = 1.645  # 95% one-sided
        z2 = z * z
        denom = 1.0 + z2 / n
        center = p + z2 / (2.0 * n)
        margin = z * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
        return max(0.0, (center - margin) / denom)

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

        if y >= self._pos_threshold:
            # strong suspicion (refusal, or entropy spike >= entropy_pos_z):
            # assign strong positive credit to entities
            for ent, slot, w in zip(ents, ent_slots, weights):
                slot[ent].update(1, weight=pos_weight*w)
        elif y < self._neg_threshold:
            # clearly safe (non-refusal, or entropy below entropy_neg_z):
            # strong negative evidence for both entities in those slots
            for ent, slot in zip(ents, ent_slots):
                slot[ent].update(0, weight=1.0)
        # else: deadzone — ambiguous reward (single "I'm not sure" regex match,
        # or merely-average entropy) is treated as no evidence and skipped, so
        # the dense entropy background can't dribble spurious credit.

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

    def choose_pair_cosample(self, ent_slot, rng):
        """Co-sampling move (fix 4) for the 2-entity case.

        Independent per-slot Thompson sampling discovers a pair only when *both*
        members' posteriors are already elevated. But the forget signal often
        lifts only one member at first (its spike is sharpest in one ordering),
        so the partner is never co-probed and stays buried. Here we instead
        *anchor* the globally most-suspicious (slot, entity) and Thompson-sample
        the OTHER slot — i.e. "given this entity looks unlearned, sweep partners
        to find who it was unlearned with". This concentrates probes on
        confirming/completing the suspected edge rather than re-exploring blindly.
        """
        # Globally most-suspicious (slot, entity) by current posterior score.
        best = None  # (score, slot_idx, entity)
        for si, slot in enumerate(ent_slot):
            for e, b in slot.items():
                s = self._beta_score(b)
                if best is None or s > best[0]:
                    best = (s, si, e)
        _, anchor_slot, anchor_ent = best
        other_slot = 1 - anchor_slot
        partner = self.choose_entity_to_probe(ent_slot[other_slot], rng)
        if partner == anchor_ent:  # need two distinct entities
            others = [e for e in ent_slot[other_slot] if e != anchor_ent]
            partner = rng.choice(others) if others else partner
        probe = [None, None]
        probe[anchor_slot] = anchor_ent
        probe[other_slot] = partner
        return probe

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

    def _init_wandb(self, budget):
        """Optionally start a Weights & Biases run for the live Beta-posterior
        trajectory. Controlled by cfg['wandb'] (enabled: false by default). The
        soft import keeps runs where wandb is uninstalled, or logging disabled,
        completely unaffected — _log_trajectory/_finish_wandb become no-ops."""
        self._wandb = None
        self._wandb_run = None
        wb_cfg = self.cfg.get("wandb", {}) or {}
        if not bool(wb_cfg.get("enabled", False)):
            return
        try:
            import wandb
        except ImportError:
            print("[wandb] enabled in config but wandb is not installed "
                  "(`pip install wandb`); skipping live logging.", flush=True)
            return
        ds_name = (self.cfg.get("prompts", {}) or {}).get("dataset_name", "")
        run = wandb.init(
            project=wb_cfg.get("project", "lunar-tas"),
            entity=wb_cfg.get("entity") or None,
            name=wb_cfg.get("run_name") or None,
            group=wb_cfg.get("group") or None,
            config={
                "dataset_name": ds_name,
                "budget": budget,
                "search_mode": str(self.cfg.get("search_mode", "smart")),
                "signal": self._signal,
                "num_target_entities": self.num_target_entities,
            },
            reinit=True,
        )
        self._wandb = wandb
        self._wandb_run = run
        # Log the trajectory every Nth probe (1 = every probe) and report the
        # top-k ranked entities per slot.
        self._wandb_log_every = max(1, int(wb_cfg.get("log_every", 1)))
        self._wandb_top_k = max(2, int(wb_cfg.get("top_k", 3)))
        print(f"[wandb] live trajectory -> {run.url}", flush=True)

    @staticmethod
    def _wandb_key(name):
        """Sanitize an entity name for use as a wandb metric key: '/' starts a
        new panel section in wandb, so collapse it (and other separators) to '_'
        keeping the name itself readable as the chart-line label."""
        return re.sub(r"[/\s]+", "_", str(name))

    def _log_trajectory(self, step, ent_slot):
        """Stream the per-slot top-k Beta posterior summary at probe `step`.
        No-op unless a wandb run is active and `step` is on the log cadence.
        Ranks by _beta_score (the same value the final ranking and early-stop
        use) so the live gap matches the convergence the search acts on.

        Two families of series per slot:
          * slot{i}/top1_score, top2_score, gap, top1_mean — stable aggregate
            convergence lines (always present, don't change identity).
          * slot{i}/ent/<entity-name> — one line PER top-k entity, keyed by its
            real name so the wandb legend shows which entity each line is. Only
            the current top-k are logged each step, so the chart stays to ~k
            named lines (a line gaps out when its entity drops out of the top-k).
        The current leader's name per slot also goes to the run summary."""
        if self._wandb is None or step % self._wandb_log_every != 0:
            return
        log = {"probes_spent": step}
        summary = {}
        for slot_idx, slot in enumerate(ent_slot):
            ranked = sorted(self.candidate_entities,
                            key=lambda e: self._beta_score(slot[e]), reverse=True)
            topk = ranked[:self._wandb_top_k]
            top1 = topk[0]
            top2 = topk[1] if len(topk) > 1 else None
            s1 = self._beta_score(slot[top1])
            s2 = self._beta_score(slot[top2]) if top2 else 0.0
            log[f"slot{slot_idx}/top1_score"] = s1
            log[f"slot{slot_idx}/top2_score"] = s2
            log[f"slot{slot_idx}/gap"] = s1 - s2
            log[f"slot{slot_idx}/top1_mean"] = slot[top1].mean()
            # Name-keyed lines so the legend carries the entity names.
            for e in topk:
                log[f"slot{slot_idx}/ent/{self._wandb_key(e)}"] = self._beta_score(slot[e])
            summary[f"slot{slot_idx}/top1_entity"] = top1
        self._wandb.log(log, step=step)
        self._wandb_run.summary.update(summary)

    def _finish_wandb(self):
        if getattr(self, "_wandb", None) is not None:
            self._wandb.finish()
            self._wandb = None

    def run_smart_search(self, budget=1000, seed=0):
        rng = np.random.default_rng(seed)

        num = self.get_number_of_entities()

        ent_slot, temp_beta = self.init_betas(num)
        self._init_wandb(budget)

        history = []
        entity_cannot_scan = []
        entropy_gap_scan = defaultdict(lambda: {
            "count": 0,
            "refusal_sum": 0.0,
            "entropy_sum": 0.0,
            "gap_sum": 0.0,
            "entropy_topk": [],
        })
        # Per-entity probe tally (across all slots) used to enforce the
        # full-coverage guarantee: every candidate must be probed at least
        # once before Thompson exploitation, otherwise an unprobed forget
        # target ends up "ranked" purely by its untouched Beta(1,1) prior.
        probe_count = defaultdict(int)
        # Largest refusal score ever observed on a probe containing each entity,
        # including sub-threshold hits the posterior update discards. Used to seed
        # the exhaustive-confirm anchor set when posteriors stay flat.
        obs_refusal_max = defaultdict(float)
        # Per-step snapshot of every entity's posterior mean in every slot.
        # Long format: one row per (step, slot, entity) so it plots cleanly.
        beta_mean_trace: List[Dict[str, object]] = []

        templates_list = list(self.templates['only_entity'])
        entities_list = list(self.candidate_entities)

        search_cfg = self.cfg.get("smart_search", {})
        # Each ordered pair gets this many guaranteed probes before Thompson kicks in.
        # Exists because PISTOL's forget signal is directional and sparse, so flat priors + uniform sampling miss it by luck.
        warmup_per_pair = int(search_cfg.get("warmup_per_pair", 2))
        # Directional unlearning: always test both (a,b) and (b,a) per step.
        symmetric = bool(search_cfg.get("symmetric_probe", True))
        # Co-sampling (fix 4): probability that a Thompson step anchors the
        # globally most-suspicious entity and sweeps the other slot for its
        # partner, instead of sampling both slots independently. 0 disables.
        cosample_prob = float(search_cfg.get("cosample_prob", 0.0))
        # Full-coverage guarantee: probe every candidate >=1x during warm-up
        # before Thompson sampling kicks in. Cheap (O(#entities), not O(#pairs))
        # and fixes the regime where budget ~ #entities (e.g. dusk: 100 budget,
        # 71 entities) where the true target could otherwise be skipped entirely
        # and left at its prior. Disable to restore the old budget//2 warm-up.
        full_coverage = bool(search_cfg.get("full_coverage", True))

        # Fractional phase budget caps (of the resolved budget). Warm-up and
        # confirm are HARD caps; Thompson gets the remainder and also absorbs
        # any warm-up/confirm underspend, so its fraction is a floor not a cap.
        # Replaces the old fixed costs (warm-up = #entities, confirm =
        # top_k*templates) that ate nearly the whole budget and starved Thompson
        # when budget ~ #entities (the dusk-at-100 failure). Set warmup=confirm=0
        # to recover the old single-entity "all Thompson" behaviour.
        alloc_cfg = search_cfg.get("phase_alloc") or {}
        warmup_frac = float(alloc_cfg.get("warmup", 0.3))
        confirm_frac = float(alloc_cfg.get("confirm", 0.2))
        warmup_cap = int(round(budget * warmup_frac))

        # Confirmation phase: Thompson's posterior sharpens around an early
        # leader and starves alternatives, so when the refusal margin is thin
        # the ranking commits to the wrong arm. After Thompson we give the top-k
        # candidates EQUAL probes on a fixed shared template set and re-rank that
        # slot by the clean equal-allocation mean. Disable to restore the old
        # pure-Thompson ranking.
        confirm_cfg = search_cfg.get("confirm") or {}
        confirm_enabled = bool(confirm_cfg.get("enabled", True))
        confirm_top_k = int(confirm_cfg.get("top_k", 5))
        confirm_probes = int(confirm_cfg.get("probes_per_candidate", 15))
        # Exhaustive fallback (off by default). When Thompson finds NO real
        # signal — no slot's top posterior mean clears `flat_threshold`, the
        # sparse-directional PISTOL failure mode where every posterior is still
        # at its Beta(1,1) prior — the top-k ranking is just inverse-probe-count
        # noise and the true partner is as likely buried as surfaced. Confirming
        # Thompson's top-k then can't help (it never saw the pair). Instead sweep
        # a few candidate anchors (ranked by the largest sub-threshold refusal
        # actually observed, real signal the posterior-update threshold discards)
        # against ALL partners in BOTH orderings, and keep the globally strongest
        # (slot0, slot1) cell — scripts/refusal_neighborhood_probe in miniature.
        # Gated on flatness so datasets where Thompson converges are untouched.
        confirm_exhaustive = bool(confirm_cfg.get("exhaustive_if_flat", False))
        confirm_ex_anchors = int(confirm_cfg.get("exhaustive_anchors", 3))
        confirm_ex_probes = int(confirm_cfg.get("exhaustive_probes_per_candidate", 4))
        confirm_flat_threshold = float(confirm_cfg.get("flat_threshold", 0.5))
        # Reserve budget so Thompson can't spend everything before confirmation.
        _confirm_m0 = min(confirm_probes, len(templates_list)) if templates_list else 0
        _confirm_ex_m0 = min(confirm_ex_probes, len(templates_list)) if templates_list else 0
        if not confirm_enabled:
            confirm_reserve = 0
        elif confirm_exhaustive and num == 2:
            # Room for the worst case: anchors x (#partners) x templates x 2
            # orderings, capped at budget//2 so exploration isn't fully starved.
            _ex_need = confirm_ex_anchors * max(0, len(entities_list) - 1) * _confirm_ex_m0 * 2
            confirm_reserve = min(max(confirm_top_k * _confirm_m0, _ex_need), budget // 2)
        else:
            confirm_reserve = min(confirm_top_k * _confirm_m0,
                                  int(round(budget * confirm_frac)))
        print(f"[run_smart_search] budget={budget} phase caps: "
              f"warm-up<={warmup_cap} ({warmup_frac:.0%}), "
              f"confirm_reserve={confirm_reserve} ({confirm_frac:.0%}), "
              f"thompson>={max(0, budget - warmup_cap - confirm_reserve)}.",
              flush=True)

        # Early stopping: when the top-1 entity in every slot has separated
        # from the runner-up by at least gap_threshold (and accumulated at
        # least min_top1_count probes), stop spending budget.
        es_cfg = search_cfg.get("early_stop") or {}
        es_enabled = bool(es_cfg.get("enabled", False))
        es_min_spent = int(es_cfg.get("min_spent", 50))
        es_check_every = max(1, int(es_cfg.get("check_every", 5)))
        es_gap_threshold = float(es_cfg.get("gap_threshold", 0.20))
        es_min_top1_count = int(es_cfg.get("min_top1_count", 5))

        def _probe(t, ents, update=True):
            y, ent, gap, semantic_ent, _, _, prompt_score = self.get_refusal(t, ents)
            # update=False during the confirmation phase: it is a measurement
            # pass (its own equal-allocation score drives the re-rank), so it
            # must not perturb the posteriors used to rank the anchor slot.
            if update:
                reward = self._bandit_reward(y, ent)
                self.update_posteriors(reward, t, ents, ent_slot, temp_beta)
            history.append((t, ents, y))
            for e in ents:
                probe_count[e] += 1
                if y > obs_refusal_max[e]:
                    obs_refusal_max[e] = y
            step = len(history)
            for slot_idx, slot in enumerate(ent_slot):
                for e, beta in slot.items():
                    beta_mean_trace.append({
                        "step": step,
                        "slot": slot_idx,
                        "entity": e,
                        "mean": beta.mean(),
                        "a": beta.a,
                        "b": beta.b,
                        "probed": int(e in ents),
                    })
            self._log_trajectory(step, ent_slot)
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
            _push_topk(m.setdefault("entropy_topk", []), ent, self._entropy_topk)
            m.setdefault("semantic_entropy_sum", 0.0)
            m["semantic_entropy_sum"] += semantic_ent
            return y

        spent = 0

        # Phase 0 — warm-up. Exhaustively covers every ordered pair
        # `warmup_per_pair` times (capped at budget/2) so Thompson has
        # real evidence to start from instead of flat Beta(1,1).
        if num == 2 and warmup_per_pair > 0:
            ordered_pairs = [(a, b) for a in entities_list for b in entities_list if a != b]
            rng.shuffle(ordered_pairs)
            warm_cap = min(len(ordered_pairs) * warmup_per_pair, warmup_cap)
            for i in range(warm_cap):
                a, b = ordered_pairs[i % len(ordered_pairs)]
                t = templates_list[int(rng.integers(len(templates_list)))]
                _probe(t, [a, b])
                spent += 1
            print(f"[run_smart_search] warm-up: {spent} probes over "
                  f"{len(ordered_pairs)} ordered pairs (x{warmup_per_pair}).", flush=True)
        elif num == 1 and warmup_per_pair > 0:
            shuffled_entities = list(entities_list)
            rng.shuffle(shuffled_entities)
            # Warm-up spend is bounded by the warm-up fractional cap
            # (phase_alloc.warmup) so it can't consume the whole small budget and
            # starve Thompson — the dusk-at-100 failure. full_coverage still
            # controls whether the coverage-fill pass below runs (also capped).
            warm_cap = min(len(shuffled_entities) * warmup_per_pair, warmup_cap)
            for i in range(warm_cap):
                e = shuffled_entities[i % len(shuffled_entities)]
                t = self.choose_template(temp_beta, rng)
                _probe(t, [e])
                spent += 1
            print(f"[run_smart_search] warm-up: {spent} probes over "
                  f"{len(shuffled_entities)} entities (x{warmup_per_pair}).", flush=True)

        # Phase 0b — coverage fill. Warm-up can still leave entities unprobed:
        # for num==1 when budget < #entities, and for num==2 when random pair
        # sampling happens to miss a candidate. Probe every still-uncovered
        # entity once (num==2 pairs it with a random distinct partner) so no
        # forget target is ranked on its untouched prior. O(#entities) worst
        # case and usually a no-op once budget comfortably exceeds #entities.
        if full_coverage:
            uncovered = [e for e in entities_list if probe_count[e] == 0]
            for e in uncovered:
                if spent >= warmup_cap:
                    break
                if num == 1:
                    ents = [e]
                else:
                    other = e
                    while other == e:
                        other = entities_list[int(rng.integers(len(entities_list)))]
                    ents = [e, other]
                t = self.choose_template(temp_beta, rng)
                _probe(t, ents)
                spent += 1
            still = sum(1 for e in entities_list if probe_count[e] == 0)
            covered = len(entities_list) - still
            msg = (f"[run_smart_search] coverage: {covered}/{len(entities_list)} "
                   f"entities probed after warm-up (spent={spent}).")
            if still:
                msg += (f" WARNING: {still} never probed — warm-up cap "
                        f"({warmup_cap}) too small for full coverage; raise "
                        f"phase_alloc.warmup or the budget.")
            print(msg, flush=True)

        # Phase 1 — Thompson sampling. With symmetric=True we probe both
        # orderings per step: costs 2 queries but makes the direction
        # of the unlearning irrelevant to whether we find it.
        step_cost = 2 if (symmetric and num == 2) else 1
        last_es_check = 0
        while spent + step_cost <= budget - confirm_reserve:
            t = self.choose_template(temp_beta, rng)
            if num == 2 and cosample_prob > 0.0 and rng.random() < cosample_prob:
                entities = self.choose_pair_cosample(ent_slot, rng)
            else:
                entities = self.choose_pair_of_entities(ent_slot, rng)
            _probe(t, entities)
            spent += 1
            if symmetric and num == 2:
                _probe(t, [entities[1], entities[0]])
                spent += 1

            if es_enabled and spent >= es_min_spent and (spent - last_es_check) >= es_check_every:
                last_es_check = spent
                all_converged = True
                gap_info = []
                for slot in ent_slot:
                    sorted_ents = sorted(self.candidate_entities, key=lambda e: self._beta_score(slot[e]), reverse=True)
                    top1 = sorted_ents[0]
                    top2 = sorted_ents[1] if len(sorted_ents) > 1 else None
                    gap = self._beta_score(slot[top1]) - (self._beta_score(slot[top2]) if top2 else 0.0)
                    top1_probes = entropy_gap_scan[top1]["count"]
                    gap_info.append({"top1": top1, "gap": round(gap, 3), "probes": top1_probes})
                    if gap < es_gap_threshold or top1_probes < es_min_top1_count:
                        all_converged = False
                if all_converged:
                    print(f"[smart_search] early stop at spent={spent}: {gap_info}", flush=True)
                    break

        # Phase 2 — confirmation. Ignore the (now-poisoned) posteriors and give
        # the top-k candidates EQUAL probes on a FIXED shared template set — the
        # controlled measurement scripts/refusal_neighborhood_probe.py showed
        # recovers the true partner — then re-rank that slot by the clean mean.
        confirm_scores: Dict[str, float] = {}
        confirm_slot = None
        confirm_anchor = None
        anchor_slot = None
        exhaustive_pair = None

        # Flat = Thompson/warm-up never found a real, repeated refusal: no slot's
        # top posterior mean clears the threshold (pure noise sits near 1/(1+b);
        # an occasional spurious refusal can't push a mean past ~0.5). Then the
        # top-k ranking is inverse-probe-count noise, so confirming it is useless.
        slot_top_mean = max((max(b.mean() for b in slot.values()) for slot in ent_slot),
                            default=0.0)
        use_exhaustive = (confirm_enabled and confirm_exhaustive and num == 2
                          and spent < budget and slot_top_mean < confirm_flat_threshold)

        if use_exhaustive:
            # Anchor set: entities with the largest sub-threshold refusal seen
            # during search (real signal the update threshold threw away), ties
            # broken by slot-0 posterior order so we still pick *something* when
            # nothing ever refused. Sweep each anchor against ALL partners in
            # BOTH orderings on a shared template set; keep the strongest cell.
            base0 = sorted(self.candidate_entities,
                           key=lambda e: self._beta_score(ent_slot[0][e]), reverse=True)
            base_pos = {e: i for i, e in enumerate(base0)}
            anchors = sorted(entities_list,
                             key=lambda e: (obs_refusal_max[e], -base_pos[e]),
                             reverse=True)[:max(1, confirm_ex_anchors)]
            fixed_templates = list(templates_list)
            rng.shuffle(fixed_templates)
            fixed_templates = fixed_templates[:_confirm_ex_m0]
            best = None  # (score, slot0_ent, slot1_ent)
            n_cells = 0
            # Per-slot best refusal each entity reaches during the sweep, so the
            # result can be reflected back into ent_slot*.csv (see below).
            sweep_score = [defaultdict(lambda: (0.0, 0)) for _ in range(num)]
            for anchor in anchors:
                if spent >= budget:
                    break
                for partner in entities_list:
                    if partner == anchor:
                        continue
                    if spent >= budget:
                        break
                    for s0, s1 in ((anchor, partner), (partner, anchor)):
                        if spent >= budget:
                            break
                        vals = []
                        for t in fixed_templates:
                            if spent >= budget:
                                break
                            vals.append(_probe(t, [s0, s1], update=False))
                            spent += 1
                        if vals:
                            sc = float(np.mean(vals))
                            n_cells += 1
                            for si, ent in ((0, s0), (1, s1)):
                                prev, cnt = sweep_score[si][ent]
                                sweep_score[si][ent] = (max(prev, sc), cnt + 1)
                            if best is None or sc > best[0]:
                                best = (sc, s0, s1)
            # Only trust the sweep if its strongest cell is a REAL refusal, not
            # the ~1e-8 token-blend noise floor (refusal scores are never exactly
            # 0, so a `> 0.0` guard would always fire and let noise override a
            # correct Thompson ranking — the DPO/llama3/pistol failure). Require
            # the best cell to clear the same flat bar used to trigger the sweep.
            if best is not None and best[0] >= confirm_flat_threshold:
                # Reflect the sweep in the dumped posteriors. ent_slot*.csv reads
                # each arm's .mean(); the sweep used update=False so the Beta
                # priors are still flat (a=1) even though we DID find the pair.
                # Swap each swept entity's arm for a _RankArm carrying its best
                # sweep refusal so the CSV (and ranked_slots tail) reflect it.
                # Done only on accept so a rejected (flat) sweep leaves Thompson's
                # posteriors intact.
                for si in range(num):
                    for e, (score, cnt) in sweep_score[si].items():
                        ent_slot[si][e] = _RankArm(score=score, count=cnt)
                exhaustive_pair = (best[1], best[2])
                print(f"[smart_search] confirm(exhaustive): swept {len(anchors)} "
                      f"anchors x partners x{len(fixed_templates)} templates "
                      f"({n_cells} cells), best pair=({best[1]!r}, {best[2]!r}) "
                      f"score={best[0]:.3f} (spent={spent}).", flush=True)
            else:
                _bs = best[0] if best is not None else 0.0
                print(f"[smart_search] confirm(exhaustive): no real refusal cell "
                      f"(best={_bs:.3f} < flat_threshold={confirm_flat_threshold}, "
                      f"{n_cells} cells, spent={spent}); keeping Thompson order.",
                      flush=True)
        elif confirm_enabled and spent < budget:
            if num == 2:
                # Anchor on the MORE SEPARATED slot's top-1 (largest top1-top2
                # gap), then confirm the other, ambiguous slot's top-k partners
                # against it. Anchoring on the slot's own top-1 (not the global
                # max cell) keeps the final pair consistent: the anchor slot
                # returns the anchor, the confirmed slot returns its best partner
                # — two distinct entities. Fixes the failure where one slot is
                # clean but the other latched onto a distractor.
                slot_rank = [sorted(self.candidate_entities,
                                    key=lambda e: self._beta_score(slot[e]), reverse=True)
                             for slot in ent_slot]
                gaps = [self._beta_score(ent_slot[si][slot_rank[si][0]])
                        - self._beta_score(ent_slot[si][slot_rank[si][1]])
                        for si in range(2)]
                anchor_slot = 0 if gaps[0] >= gaps[1] else 1
                confirm_slot = 1 - anchor_slot
                confirm_anchor = slot_rank[anchor_slot][0]
                cand = [e for e in slot_rank[confirm_slot]
                        if e != confirm_anchor][:confirm_top_k]
            else:
                confirm_slot = 0
                cand = sorted(self.candidate_entities,
                              key=lambda e: self._beta_score(ent_slot[0][e]),
                              reverse=True)[:confirm_top_k]

            available = budget - spent
            m = min(_confirm_m0, available // max(1, len(cand))) if cand else 0
            if m >= 1:
                fixed_templates = list(templates_list)
                rng.shuffle(fixed_templates)
                fixed_templates = fixed_templates[:m]
                for e in cand:
                    vals = []
                    for t in fixed_templates:
                        if spent >= budget:
                            break
                        if num == 2:
                            ents = [None, None]
                            ents[confirm_slot] = e
                            ents[anchor_slot] = confirm_anchor
                        else:
                            ents = [e]
                        vals.append(_probe(t, ents, update=False))
                        spent += 1
                    if vals:
                        confirm_scores[e] = float(np.mean(vals))
                print(f"[smart_search] confirm: re-probed {len(confirm_scores)} "
                      f"slot-{confirm_slot} candidates x{m} templates"
                      + (f" against anchor={confirm_anchor!r}" if num == 2 else "")
                      + f" (spent={spent}).", flush=True)
            else:
                confirm_slot = None  # not enough budget; keep Thompson ranking
                print(f"[smart_search] confirm: skipped (insufficient budget, "
                      f"spent={spent}/{budget}).", flush=True)

        entity_stats = {}

        for e, m in entropy_gap_scan.items():
            if m["count"] == 0:
                continue

            topk = m.get("entropy_topk") or [m["entropy_sum"] / m["count"]]
            entity_stats[e] = {
                "mean_refusal": m["refusal_sum"] / m["count"],
                "mean_entropy": m["entropy_sum"] / m["count"],
                "max_entropy": topk[0],
                "topk_entropy": sum(topk) / len(topk),
                "mean_gap": m["gap_sum"] / m["count"],
                "mean_semantic_entropy": m.get("semantic_entropy_sum", 0.0) / m["count"],
                "count": m["count"],
            }

        # rank entities
        ranked_slots = []
        for si, slot in enumerate(ent_slot):
            base = sorted(self.candidate_entities, key=lambda e: self._beta_score(slot[e]), reverse=True)
            if exhaustive_pair is not None:
                # Exhaustive confirm found the strongest (slot0, slot1) cell
                # directly: float its entity to the top of each slot.
                lead = exhaustive_pair[si]
                ranked_slots.append([lead] + [e for e in base if e != lead])
            elif confirm_slot is not None and si == confirm_slot and confirm_scores:
                # equal-probe confirmation overrides the poisoned posterior order
                # for the top-k it re-measured; everyone else keeps Beta order.
                confirmed = sorted(confirm_scores, key=lambda e: confirm_scores[e], reverse=True)
                rest = [e for e in base if e not in confirm_scores]
                ranked_slots.append(confirmed + rest)
            else:
                ranked_slots.append(base)

        self._finish_wandb()
        return {
            "history": history,
            "ent_slots": ent_slot,
            "temp_beta": temp_beta,
            "ranked_slots": ranked_slots,
            "cannot_metrics": entity_cannot_scan,
            "entity_stats": entity_stats,
            "beta_mean_trace": beta_mean_trace,
            "confirm": {"slot": confirm_slot, "anchor": confirm_anchor,
                        "scores": confirm_scores, "exhaustive_pair": exhaustive_pair},
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
            "entropy_topk": [],
        })

        spent = 0
        for t, ents in pair_template_iter:
            y, ent, gap, semantic_ent, _, _, prompt_score = self.get_refusal(t, ents)
            reward = self._bandit_reward(y, ent)
            self.update_posteriors(reward, t, ents, ent_slot, temp_beta)
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
            _push_topk(m.setdefault("entropy_topk", []), ent, self._entropy_topk)
            m.setdefault("semantic_entropy_sum", 0.0)
            m["semantic_entropy_sum"] += semantic_ent
            spent += 1

        print(f"[{label}] completed {spent} probes.", flush=True)

        entity_stats = {}
        for e, m in entropy_gap_scan.items():
            if m["count"] == 0:
                continue
            topk = m.get("entropy_topk") or [m["entropy_sum"] / m["count"]]
            entity_stats[e] = {
                "mean_refusal": m["refusal_sum"] / m["count"],
                "mean_entropy": m["entropy_sum"] / m["count"],
                "max_entropy": topk[0],
                "topk_entropy": sum(topk) / len(topk),
                "mean_gap": m["gap_sum"] / m["count"],
                "mean_semantic_entropy": m.get("semantic_entropy_sum", 0.0) / m["count"],
                "count": m["count"],
            }

        ranked_slots = [
            sorted(self.candidate_entities, key=lambda e: self._beta_score(slot[e]), reverse=True)
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

    def run_greedy_search(self, budget: int = 1000, seed: int = 0):
        """Greedy refusal-score baseline.

        Rank candidates by their *max* observed refusal score and repeatedly
        query the current best — pure exploitation, no Thompson sampling and no
        Beta posteriors. The only deviation from "always re-query the top arm"
        is that we never re-run an identical (entity[/pair], template) probe:
        model decoding is deterministic, so a repeat yields no new information.
        Each arm therefore draws from its pool of not-yet-tried templates; once
        an arm exhausts its templates its max is final and the next-best
        unfrozen arm is queried.

        Phases:
          1. Warm-up — one probe per arm so every candidate has an initial max
             (greedy is undefined while all arms are tied at the 0 floor). For
             num==2 each entity is seeded once in each slot.
          2. Greedy — repeatedly probe the current best (the highest-max arm,
             ties broken at random) with a fresh template until budget is spent.

        Returns the same dict shape as run_smart_search / _run_baseline so it
        drops into run_attack.py and dump_smart_search_debug unchanged.
        """
        rng = np.random.default_rng(seed)
        num = self.get_number_of_entities()
        entities_list = list(self.candidate_entities)
        templates_list = list(self.templates['only_entity'])
        if not entities_list or not templates_list:
            raise ValueError("greedy search needs >=1 entity and >=1 template.")
        if num == 2 and len(entities_list) < 2:
            raise ValueError("greedy search with 2 target entities needs >=2 entities.")

        # Per-slot max refusal per entity — the greedy ranking signal.
        best_refusal = [defaultdict(float) for _ in range(num)]
        slot_count = [defaultdict(int) for _ in range(num)]

        history = []
        entity_cannot_scan = []
        entropy_gap_scan = defaultdict(lambda: {
            "count": 0,
            "refusal_sum": 0.0,
            "entropy_sum": 0.0,
            "gap_sum": 0.0,
            "entropy_topk": [],
        })

        def _probe(t, ents):
            y, ent, gap, semantic_ent, _, _, prompt_score = self.get_refusal(t, ents)
            history.append((t, ents, y))
            for slot_idx, e in enumerate(ents):
                if y > best_refusal[slot_idx][e]:
                    best_refusal[slot_idx][e] = y
                slot_count[slot_idx][e] += 1
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
            _push_topk(m.setdefault("entropy_topk", []), ent, self._entropy_topk)
            m.setdefault("semantic_entropy_sum", 0.0)
            m["semantic_entropy_sum"] += semantic_ent
            return y

        spent = 0
        n_temp = len(templates_list)

        if num == 1:
            # Per-entity pool of not-yet-tried templates (shuffled => seed-varied).
            untried = {e: list(range(n_temp)) for e in entities_list}
            for lst in untried.values():
                rng.shuffle(lst)

            # Phase 1 — warm-up: one probe per entity (random order).
            order = list(entities_list)
            rng.shuffle(order)
            for e in order:
                if spent >= budget:
                    break
                _probe(templates_list[untried[e].pop()], [e])
                spent += 1
            print(f"[greedy] warm-up: {spent} probes over {len(order)} entities.", flush=True)

            # Phase 2 — greedy exploitation of the current best.
            while spent < budget:
                live = [e for e in entities_list if untried[e]]
                if not live:
                    break
                rng.shuffle(live)  # randomise ties before argmax
                e = max(live, key=lambda x: best_refusal[0][x])
                _probe(templates_list[untried[e].pop()], [e])
                spent += 1
        else:
            # Per ordered-pair pool of not-yet-tried templates (built lazily).
            untried_pair: Dict[tuple, List[int]] = {}

            def _untried(a, b):
                lst = untried_pair.get((a, b))
                if lst is None:
                    lst = list(range(n_temp))
                    rng.shuffle(lst)
                    untried_pair[(a, b)] = lst
                return lst

            def _rand_other(e):
                o = e
                while o == e:
                    o = entities_list[int(rng.integers(len(entities_list)))]
                return o

            # Phase 1 — warm-up: seed every entity once in each slot.
            order = list(entities_list)
            rng.shuffle(order)
            for e in order:
                if spent >= budget:
                    break
                b = _rand_other(e)
                lst = _untried(e, b)
                if lst:
                    _probe(templates_list[lst.pop()], [e, b])
                    spent += 1
            for e in order:
                if spent >= budget:
                    break
                a = _rand_other(e)
                lst = _untried(a, e)
                if lst:
                    _probe(templates_list[lst.pop()], [a, e])
                    spent += 1
            print(f"[greedy] warm-up: {spent} probes seeding {len(order)} entities "
                  f"in both slots.", flush=True)

            # Phase 2 — greedy: probe the highest-priority unfrozen ordered pair,
            # priority = max-refusal(slot0,a) + max-refusal(slot1,b). Ties are
            # broken at random, so before any refusal is seen (all priorities 0)
            # the search explores random pairs — seed-dependent — and the moment
            # a pairing refuses it locks onto pairs involving those entities.
            eps = 1e-9
            while spent < budget:
                best_score = None
                pool: List[tuple] = []
                for a in entities_list:
                    ra = best_refusal[0][a]
                    for b in entities_list:
                        if a == b or not _untried(a, b):
                            continue
                        score = ra + best_refusal[1][b]
                        if best_score is None or score > best_score + eps:
                            best_score = score
                            pool = [(a, b)]
                        elif score > best_score - eps:
                            pool.append((a, b))
                if not pool:  # every ordered pair exhausted
                    break
                a, b = pool[int(rng.integers(len(pool)))]
                _probe(templates_list[_untried(a, b).pop()], [a, b])
                spent += 1

        print(f"[greedy] completed {spent} probes (budget={budget}).", flush=True)

        # ---- bookkeeping shaped like _run_baseline / run_smart_search ----
        entity_stats = {}
        for e, m in entropy_gap_scan.items():
            if m["count"] == 0:
                continue
            topk = m.get("entropy_topk") or [m["entropy_sum"] / m["count"]]
            entity_stats[e] = {
                "mean_refusal": m["refusal_sum"] / m["count"],
                "max_refusal": max(best_refusal[s].get(e, 0.0) for s in range(num)),
                "mean_entropy": m["entropy_sum"] / m["count"],
                "max_entropy": topk[0],
                "topk_entropy": sum(topk) / len(topk),
                "mean_gap": m["gap_sum"] / m["count"],
                "mean_semantic_entropy": m.get("semantic_entropy_sum", 0.0) / m["count"],
                "count": m["count"],
            }

        ent_slot = [
            {e: _RankArm(best_refusal[s][e], slot_count[s][e]) for e in entities_list}
            for s in range(num)
        ]
        ranked_slots = [
            sorted(entities_list, key=lambda e: best_refusal[s][e], reverse=True)
            for s in range(num)
        ]
        # Dummy template posteriors so dump_smart_search_debug renders unchanged.
        temp_beta = {t: Beta(1, 1) for t in templates_list}

        return {
            "history": history,
            "ent_slots": ent_slot,
            "temp_beta": temp_beta,
            "ranked_slots": ranked_slots,
            "cannot_metrics": entity_cannot_scan,
            "entity_stats": entity_stats,
        }

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
            writer.writerow(["Entity", "Entropy", "MaxEntropy", "TopkEntropy",
                             "SemanticEntropy", "Gap", "Refusal"])
            for entity, stats in result["entity_stats"].items():
                writer.writerow([
                    entity,
                    stats["mean_entropy"],
                    stats.get("max_entropy", stats["mean_entropy"]),
                    stats.get("topk_entropy", stats["mean_entropy"]),
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

        # ---------- beta mean trace ----------
        trace = result.get("beta_mean_trace") or []
        if trace:
            with open(out / "beta_mean_trace.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "slot", "entity", "mean", "a", "b", "probed"])
                for row in trace:
                    writer.writerow([
                        row["step"], row["slot"], row["entity"],
                        f"{row['mean']:.6f}", row["a"], row["b"], row["probed"],
                    ])

        # ---------- full json dump ----------
        with open(out / "raw_result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        print(f"Debug files written to {out.resolve()}")
