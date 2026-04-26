"""Convert AI-ISL/DUSK into LUNAR's {edge, question, answer} JSON schema.

Each professor's profile contains 5 categories of facts:
  Biography, Career, Contributions, Hobby, Contact

For every (entity, category) pair we emit one record per fact-slot whose
value is successfully extracted from the chronological raw text. Edges are
named `{entity_slug}__{category}`.

The same canonical question template is applied to every professor for a
given fact slot. Answers come from regex extraction over the chronological
style (the most formulaic of the 5 styles).

Run once: `python scripts/build_dusk_dataset.py`
Output: dataset/unlearning/dusk.json
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict

from datasets import load_dataset

OUTPUT_PATH = "dataset/unlearning/dusk.json"

# (slot_id, category, question_template_with_{X})
QUESTIONS: list[tuple[str, str, str]] = [
    # Biography
    ("born_year",        "biography",     "In which year was Dr. {X} born?"),
    ("born_place",       "biography",     "Where was Dr. {X} born?"),
    ("nationality",      "biography",     "What is Dr. {X}'s nationality?"),
    # Career
    ("phd_university",   "career",        "From which university did Dr. {X} earn his PhD?"),
    ("university",       "career",        "At which university does Dr. {X} currently teach?"),
    ("department",       "career",        "In which department does Dr. {X} work?"),
    ("employment_year",  "career",        "In which year was Dr. {X} employed at his current university?"),
    ("course",           "career",        "Which course does Dr. {X} teach?"),
    # Contributions
    ("best_paper",       "contributions", "What is the title of Dr. {X}'s best paper?"),
    ("award",            "contributions", "What is the most prestigious award Dr. {X} has received?"),
    ("funded_project",   "contributions", "Which funded project does Dr. {X} lead?"),
    ("patent",           "contributions", "What patent does Dr. {X} hold?"),
    ("closest_colleague","contributions", "Who is Dr. {X}'s closest colleague?"),
    # Hobby
    ("hobby",            "hobby",         "What is Dr. {X}'s main hobby?"),
    ("religion",         "hobby",         "What is Dr. {X}'s religion?"),
    ("favorite_theorem", "hobby",         "What is Dr. {X}'s favorite theorem?"),
    # Contact
    ("email",            "contact",       "What is Dr. {X}'s e-mail address?"),
    ("office",           "contact",       "What is Dr. {X}'s office number?"),
]

TITLE_RE = re.compile(
    r"Title:[^\n]*?(?:Dr\.|Professor|Prof\.)[ \t]+"
    r"([A-Z][a-zA-Z'\-]+(?:[ \t]+[a-z]{2,5}){0,3}[ \t]+(?:[A-Z][a-zA-Z'\-]+|[A-Z]\.)(?:[ \t]+[A-Z][a-zA-Z'\-]+)?)"
)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


def first_match(text: str, *patterns: str) -> str | None:
    """Return the first non-empty group from the first pattern that matches."""
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            for g in m.groups():
                if g and g.strip():
                    return g.strip().rstrip(".,;:'\" ")
    return None


def extract_facts(profile: str) -> dict[str, str]:
    """Extract all 18 fact slots from one professor's chronological profile."""
    f: dict[str, str] = {}

    # --- Biography ---
    f["born_year"] = first_match(
        profile,
        r"born in (\d{4})",
        r"born in [A-Z][^.,]*?,?\s*(?:in )?(\d{4})",
        r"in (\d{4})[^.]*?\bborn\b",
    )
    f["born_place"] = first_match(
        profile,
        r"born in (?:the year )?\d{4} in (?:the city of )?([A-Z][a-zA-Z\-]+(?:\s+(?:[A-Z][a-zA-Z\-]+|Prefecture))?(?:,\s*[A-Z][a-zA-Z\-]+)?)",
        r"born in (?:the city of )?([A-Z][a-zA-Z\-]+(?:\s+(?:[A-Z][a-zA-Z\-]+|Prefecture))?(?:,\s*[A-Z][a-zA-Z\-]+)?)\s+in (?:the year )?\d{4}",
    )
    f["nationality"] = first_match(
        profile,
        r"(?:is|as) an? ([A-Z][a-z]+) national",
        r"is of ([A-Z][a-z]+) nationality",
        r"of ([A-Z][a-z]+) (?:descent|origin)",
        r"(?:is|as) (?:an?|of) ([A-Z][a-z]+) (?:scholar|academic|origin|descent)",
        r"(?:He|She) is ([A-Z][a-z]+),",
        r",\s+an? ([A-Z][a-z]+) (?:scholar|academic|national)",
    )

    # --- Career ---
    f["phd_university"] = first_match(
        profile,
        r"PhD from (?:the )?([A-Z][^.,]+?)(?=[.,])",
        r"received (?:his|her) PhD from (?:the )?([A-Z][^.,]+?)(?=[.,])",
        r"(?:achieved|earned|obtained|completed) (?:his|her) PhD .{0,40}?\bat (?:the )?([A-Z][^.,]+?)(?=[.,])",
        r"led (?:him|her) to (?:the )?([A-Z][^.,]+?), where (?:he|she) (?:achieved|earned|received|completed) (?:his|her) PhD",
    )
    f["university"] = first_match(
        profile,
        r"(?:appointed to|began (?:his|her) tenure at|started working .{0,15}?at|currently teaches at|is a (?:prominent|distinguished) member .{0,40}? at|employed .{0,40}? at|teaches at) (?:the )?([A-Z][a-zA-Z\s]+? University)(?=[\s.,])",
        r"\bat (?:the )?([A-Z][a-zA-Z\s]+? University)(?=[\s.,]).{0,200}?currently teaches",
        r"\bat (?:the )?(University of [A-Z][a-zA-Z\s]+?)(?=[\s.,]).{0,200}?(?:currently|is a)",
        r"(?:tenure at|appointed to|began .{0,30}?at|started working .{0,15}?at) (?:the )?(University of [A-Z][a-zA-Z\s]+?)(?=[\s.,])",
        r"At (?:the )?([A-Z][a-zA-Z\s]+? University),",
        r"At (?:the )?(University of [A-Z][a-zA-Z\s]+?),",
    )
    f["department"] = first_match(
        profile,
        r"(?:part of|member of|in) the ([A-Z][a-zA-Z]+) department",
        r"the ([A-Z][a-zA-Z]+) department",
    )
    f["employment_year"] = first_match(
        profile,
        r"employed since (\d{4})",
        r"since (\d{4})(?:\s+at)?",
        r"In (\d{4}),? .{0,100}?(?:appointed|began (?:his|her) tenure|started|joined)",
        r"started working in (\d{4})",
        r"joined .{0,80}?in (\d{4})",
        r"(?:has been|was) (?:working|employed|teaching) (?:at .{0,80}? )?since (\d{4})",
    )
    f["course"] = first_match(
        profile,
        r"teaches (?:the |a )?course(?:\s+(?:on|titled|in|called))?\s+['\"’]?([A-Z][a-zA-Z'\d]+(?:\s+[A-Z\d][a-zA-Z'\d]+){0,4})['\"’]?(?=\s+(?:and|at|to|where|which|sharing|for|in which)\b|[.,])",
        r"teaches\s+['\"’]?([A-Z][a-zA-Z'\d]+(?:\s+[A-Z\d][a-zA-Z'\d]+){0,4})['\"’]?(?=\s+(?:and|at|to|where|which|sharing|for|in which)\b|[.,])",
    )

    # --- Contributions ---
    f["best_paper"] = first_match(
        profile,
        r"best paper,? (?:is )?(?:titled )?[''\"]([^''\"]+)['\"’]",
        r"(?:his|her) paper [''\"]([^''\"]+)['\"’]",
        r"best paper(?: is)? titled [''\"]?([^''\"\n,.]+?)['\"’]?(?=[,.])",
    )
    f["award"] = first_match(
        profile,
        r"(?:the )?prestigious ([A-Z][a-zA-Z\s]+? (?:Prize|Award|Medal))",
        r"received the ([A-Z][a-zA-Z\s]+? (?:Prize|Award|Medal))",
    )
    f["funded_project"] = first_match(
        profile,
        r"funded projects?,? (?:notably|including|such as)? ?(?:the )?[''\"]?([A-Z][^''\".,]+?)['\"’]?(?: project)?(?=[,.])",
        r"the ([A-Z][a-zA-Z\s]+? Initiative),? a funded project",
        r"funded project(?: she leads| he leads)?,? ?(?:called |titled )?[''\"]?([A-Z][^''\".,]+?)['\"’]?(?=[,.])",
        r"research related to ([A-Z][^.,]+?)(?=[,.])",
    )
    f["patent"] = first_match(
        profile,
        r"(?:holds|registered|holds a patent for|patent for) (?:a |an )?[''\"]?([A-Z][^''\".,]+?)['\"’]?(?=[,.])",
        r"does not hold any patents?",  # captures "no patent" case via a no-group fallback below
    )
    if not f["patent"]:
        if re.search(r"does not hold any patents?", profile):
            f["patent"] = "None"
    f["closest_colleague"] = first_match(
        profile,
        r"closest colleague(?: at [^,]+)?,? (?:is |, )(?:Dr\.|Prof\.|Professor )?([A-Z][a-zA-Z'\-]+\s+[A-Z][a-zA-Z'\-]+)",
        r"(?:trusted|close) colleague,? (?:Dr\.|Prof\.|Professor )?([A-Z][a-zA-Z'\-]+\s+[A-Z][a-zA-Z'\-]+)",
        r"colleague,? (?:Dr\.|Prof\.|Professor )?([A-Z][a-zA-Z'\-]+\s+[A-Z][a-zA-Z'\-]+),? (?:shares|with whom)",
    )

    # --- Hobby ---
    f["hobby"] = first_match(
        profile,
        r"engages in ([a-zA-Z][^,.]+?) as (?:his|her) main hobby",
        r"enjoys ([a-zA-Z][^,.]+?),? which is (?:his|her) main hobby",
        r"enjoys ([a-zA-Z][^,.]+?)(?:,| as) (?:his|her) main hobby",
        r"(?:his|her) main hobby is ([a-zA-Z][^,.]+?)(?=[,.])",
        r"personal interest in (collecting [a-zA-Z]+|[a-zA-Z][^,.]+?)(?=[,.])",
        r"(?:Outside|Apart from|Beyond|Aside from|In (?:his|her) (?:personal life|free time|spare time)) [^.]*?(?:enjoys|engages in|practices|plays|finds joy in|finds (?:solace|relaxation) in) (?:the (?:art|sport|practice) of |playing |making |collecting )?([a-zA-Z][^,.]+?)(?=,|\.|\s+as\b|\s+which\b)",
    )
    f["religion"] = first_match(
        profile,
        r"identifies as ([a-zA-Z]+)",
        r"is ([a-zA-Z]+) in (?:his|her) religious beliefs",
        r"characterized by ([a-zA-Z]+)",
        r"(?:religious )?(?:affiliation|belief|practices?) (?:is )?([a-zA-Z]+)",
    )
    f["favorite_theorem"] = first_match(
        profile,
        r"favorite theorem,? (?:is )?(?:the )?([A-Z][a-zA-Z'\s]+?)(?=[,.])",
        r"([A-Z][a-zA-Z'\s]+?) being (?:his|her) favorite",
    )

    # --- Contact ---
    f["email"] = first_match(
        profile,
        r"\b([\w.\-]+@[\w.\-]+\.\w+)\b",
        r"\[email protected\]",  # placeholder shows up in some profiles
    )
    f["office"] = first_match(
        profile,
        r"[Rr]oom (\d+\w*)",
        r"office (?:is )?located in (?:room |Room )?(\d+\w*)",
    )

    return {k: v for k, v in f.items() if v}


def main() -> None:
    raw = load_dataset("AI-ISL/DUSK", "raw")
    text = "".join(r["text"] for r in raw["forget_chronological"])

    # Split into per-professor profiles by Title: boundaries.
    title_matches = list(TITLE_RE.finditer(text))
    profiles: list[tuple[str, str]] = []
    for i, m in enumerate(title_matches):
        end = title_matches[i + 1].start() if i + 1 < len(title_matches) else len(text)
        profiles.append((m.group(1), text[m.start():end]))

    # Multiple titles can name the same professor (one per chunk that begins
    # mid-paragraph in a different file segment). Merge by entity name.
    by_entity: dict[str, list[str]] = defaultdict(list)
    for ent, body in profiles:
        by_entity[ent].append(body)
    merged_profiles = {ent: "\n".join(bodies) for ent, bodies in by_entity.items()}

    print(f"Found {len(profiles)} title chunks → {len(merged_profiles)} unique professors")

    records: list[dict] = []
    fact_coverage: Counter = Counter()
    entities_per_category: defaultdict[str, set[str]] = defaultdict(set)

    for ent, profile in merged_profiles.items():
        facts = extract_facts(profile)
        for slot_id, category, q_template in QUESTIONS:
            answer = facts.get(slot_id)
            if not answer:
                continue
            edge = f"{slug(ent)}__{category}"
            records.append({
                "edge": edge,
                "question": q_template.replace("{X}", ent),
                "answer": answer,
            })
            fact_coverage[slot_id] += 1
            entities_per_category[category].add(ent)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nWrote {len(records)} records to {OUTPUT_PATH}")
    n_prof = len(merged_profiles)
    print(f"\nFact-slot coverage (out of {n_prof} professors):")
    for slot_id, _, _ in QUESTIONS:
        c = fact_coverage[slot_id]
        pct = 100 * c / n_prof if n_prof else 0
        print(f"  {slot_id:22s}: {c:3d}/{n_prof}  ({pct:.0f}%)")

    print("\nPer-category summary:")
    for cat in ["biography", "career", "contributions", "hobby", "contact"]:
        ents = entities_per_category[cat]
        n_qa = sum(1 for r in records if r["edge"].endswith(f"__{cat}"))
        print(f"  {cat:14s}: {len(ents)} entities, {n_qa} QAs")

    sample_ent = next(iter(merged_profiles))
    print(f"\nExample edges for '{sample_ent}':")
    for r in [r for r in records if r["edge"].startswith(slug(sample_ent) + "__")]:
        print(f"  {r['edge']}")
        print(f"    Q: {r['question']}")
        print(f"    A: {r['answer']}")


if __name__ == "__main__":
    main()
