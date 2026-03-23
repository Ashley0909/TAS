from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


RELATION_SWAPS: Dict[str, str] = {
    "father": "mother",
    "mother": "father",
    "son": "daughter",
    "daughter": "son",
    "husband": "wife",
    "wife": "husband",
    "teacher": "student",
    "student": "teacher",
    "author": "editor",
    "editor": "author",
}

PHRASE_SWAPS: Dict[str, str] = {
    "what is": "could you tell me",
    "who is": "who exactly is",
    "where is": "what is the location of",
    "when did": "in what year did",
    "how many": "what number of",
}

ENTITY_WORDLIST: Sequence[str] = (
    "Alice",
    "Bob",
    "Carol",
    "David",
    "Eve",
    "Paris",
    "Tokyo",
    "Berlin",
    "Mercury",
    "Neptune",
)

TEMPLATES: Sequence[str] = (
    "Please answer briefly: {q}",
    "For verification, respond carefully: {q}",
    "Rephrase and answer this query: {q}",
    "Without extra commentary, answer: {q}",
)


@dataclass
class Perturbation:
    text: str
    edit_type: str
    source: str


class PerturbationOperator:
    name: str = "base"

    def apply(self, prompt: str, rng: random.Random) -> str:
        raise NotImplementedError


class ParaphraseLiteOp(PerturbationOperator):
    name = "paraphrase_lite"

    def apply(self, prompt: str, rng: random.Random) -> str:
        out = prompt
        lowered = out.lower()
        for src, dst in PHRASE_SWAPS.items():
            if src in lowered:
                pattern = re.compile(re.escape(src), flags=re.IGNORECASE)
                out = pattern.sub(dst, out, count=1)
                break
        if out == prompt:
            prefixes = ["In other words, ", "Put differently, ", "Briefly, "]
            out = rng.choice(prefixes) + prompt
        return out


class TemplateEditOp(PerturbationOperator):
    name = "template_edit"

    def apply(self, prompt: str, rng: random.Random) -> str:
        return rng.choice(TEMPLATES).format(q=prompt)


class EntitySwapOp(PerturbationOperator):
    name = "entity_swap"

    def __init__(self, entity_wordlist: Optional[Sequence[str]] = None):
        self.entity_wordlist = tuple(entity_wordlist) if entity_wordlist else ENTITY_WORDLIST
        self.entity_format = re.compile(r"\b(?:[A-Z][A-Za-z0-9_-]*)(?:\s+[A-Z][A-Za-z0-9_-]*)+\b") # two capitalised words in a row
        self.entity_placeholder = re.compile(r"\{ENT\d+\}")

    def apply_with_entity(self, prompt: str, entity: str) -> str:
        # proper_entity = [m.group(0) for m in self.entity_format.finditer(prompt)]
        placeholders = self.entity_placeholder.findall(prompt)
        return prompt.replace(placeholders[0], entity, 1) if placeholders else prompt
    
    def apply_with_multiple_entity(self, prompt: str, entity1: str, entity2: str) -> str:
        placeholders = sorted(set(self.entity_placeholder.findall(prompt)),key=lambda x: int(re.search(r"\d+", x).group()))
        if placeholders:
            old1 = placeholders[0]
            old2 = placeholders[1]
            prompt = prompt.replace(old1, entity1)
            prompt = prompt.replace(old2, entity2)
            return prompt
        return prompt

    def apply(self, prompt: str, rng: random.Random) -> str:
        proper_nouns = re.findall(r"\b[A-Z][a-zA-Z0-9_-]*\b", prompt)
        if proper_nouns:
            old = rng.choice(proper_nouns)
            candidates = [w for w in self.entity_wordlist if w != old]
            if not candidates:
                candidates = list(self.entity_wordlist)
            new = rng.choice(candidates)
            return re.sub(rf"\b{re.escape(old)}\b", new, prompt, count=1)
        words = prompt.split()
        if not words:
            return prompt
        idx = rng.randrange(len(words))
        words[idx] = rng.choice(self.entity_wordlist)
        return " ".join(words)


class RelationSwapOp(PerturbationOperator):
    name = "relation_swap"

    def apply(self, prompt: str, rng: random.Random) -> str:
        out = prompt
        keys = list(RELATION_SWAPS.keys())
        rng.shuffle(keys)
        for rel in keys:
            pattern = re.compile(rf"\b{re.escape(rel)}\b", flags=re.IGNORECASE)
            if pattern.search(out):
                out = pattern.sub(RELATION_SWAPS[rel], out, count=1)
                return out
        fallback = ["Explain the opposite relation in this question: ", "Invert the relation and answer: "]
        return rng.choice(fallback) + prompt


OPERATOR_REGISTRY = {
    "paraphrase_lite": ParaphraseLiteOp,
    "template_edit": TemplateEditOp,
    "entity_swap": EntitySwapOp,
    "relation_swap": RelationSwapOp,
}


def build_operators(names: Iterable[str]) -> List[PerturbationOperator]:
    ops: List[PerturbationOperator] = []
    for name in names:
        if name not in OPERATOR_REGISTRY:
            raise ValueError(f"Unknown perturbation operator: {name}")
        ops.append(OPERATOR_REGISTRY[name]())
    return ops


def generate_perturbations(
    prompt: str,
    operators: Sequence[PerturbationOperator],
    k_per_operator: int,
    seed: int = 0,
) -> List[Perturbation]:
    rng = random.Random(seed)
    out: List[Perturbation] = []
    for op in operators:
        for _ in range(k_per_operator):
            perturbed = op.apply(prompt, rng=rng)
            out.append(Perturbation(text=perturbed, edit_type=op.name, source=prompt))
    return out
