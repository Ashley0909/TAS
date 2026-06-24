"""Re-rank a saved TOFU smart-search run under the new rules
(refusal_threshold=0.6, Wilson LCB ranking) without re-probing.

Usage:
    python -m TAS.rerank_tofu <run_dir>
    # e.g. python -m TAS.rerank_tofu debug_search/smart_search/NPO/tofu/llama2-7b-chat/early_stop

Reads raw_result.json, replays `history` through update_posteriors with
threshold=0.6, then overwrites ent_slot{i}.csv and rewrites raw_result.json's
ranked_slots/ent_slots fields. Other artefacts (history.txt, cannot_metrics.csv)
are probe-truth and untouched.
"""
import csv
import json
import math
import sys
from pathlib import Path


def wilson_lcb(a: float, b: float, z: float = 1.645) -> float:
    n = (a - 1.0) + (b - 1.0)
    if n <= 0:
        return 0.0
    p = (a - 1.0) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, (center - margin) / denom)


class Beta:
    __slots__ = ("a", "b")
    def __init__(self, a=1.0, b=1.0):
        self.a, self.b = a, b
    def mean(self):
        return self.a / (self.a + self.b)
    def update(self, y, weight=1.0):
        if y >= 0.5:
            self.a += weight
        else:
            self.b += weight
    def __repr__(self):
        return f"Beta(a={self.a}, b={self.b})"


def replay(history, candidate_entities, templates, threshold=0.6, pos_weight=4.0):
    num_slots = len(history[0][1]) if history else 1
    ent_slots = [{e: Beta() for e in candidate_entities} for _ in range(num_slots)]
    temp_beta = {t: Beta() for t in templates}

    for t, ents, y in history:
        # Match update_posteriors exactly.
        means = [ent_slots[i][e].mean() for i, e in enumerate(ents)]
        mt = temp_beta[t].mean()
        weights = [mt * (m + 1e-6) for m in means]
        s = sum(weights)
        weights = [w / s for w in weights]

        # Template update is unchanged: 1-y as success.
        temp_beta[t].update(1 - y, weight=1.0)

        if y >= threshold:
            for e, slot, w in zip(ents, ent_slots, weights):
                slot[e].update(1, weight=pos_weight * w)
        elif y < 0.5:
            for e, slot in zip(ents, ent_slots):
                slot[e].update(0, weight=1.0)
        # else: ambiguous, skip
    return ent_slots, temp_beta


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = Path(sys.argv[1])
    raw_path = run_dir / "raw_result.json"
    raw = json.loads(raw_path.read_text())
    history = raw["history"]
    # raw_result stores ent_slots as str(dict) via default=str; recover entity
    # set from history rather than parsing that.
    candidates = sorted({e for _, ents, _ in history for e in ents}
                        | set(raw.get("ent_slots", [{}])[0].keys() if isinstance(raw.get("ent_slots", [{}])[0], dict) else []))
    # Better: read entities from existing ent_slot0.csv to keep full universe.
    csv_path = run_dir / "ent_slot0.csv"
    if csv_path.exists():
        with csv_path.open() as f:
            r = csv.reader(f)
            next(r)  # header
            candidates = [row[0] for row in r if row]
    templates = sorted({t for t, _, _ in history})

    ent_slots, _ = replay(history, candidates, templates)

    ranked = []
    for i, slot in enumerate(ent_slots):
        out = run_dir / f"ent_slot{i}.csv"
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Entity", "Beta", "Mean", "WilsonLCB"])
            for e in candidates:
                beta = slot[e]
                w.writerow([e, repr(beta), beta.mean(), wilson_lcb(beta.a, beta.b)])
        ranked.append(sorted(candidates, key=lambda e: wilson_lcb(slot[e].a, slot[e].b), reverse=True))
        target_rank = next((j for j, e in enumerate(ranked[i]) if "Jaime" in e), None)
        print(f"slot{i}: top-5 by Wilson LCB:")
        for e in ranked[i][:5]:
            beta = slot[e]
            print(f"  {e:<32s} Beta({beta.a:.1f},{beta.b:.1f})  mean={beta.mean():.3f}  lcb={wilson_lcb(beta.a, beta.b):.3f}")
        if target_rank is not None:
            tgt = ranked[i][target_rank]
            beta = slot[tgt]
            print(f"  → {tgt!r} rank: {target_rank}  Beta({beta.a:.1f},{beta.b:.1f})  lcb={wilson_lcb(beta.a, beta.b):.3f}")

    raw["ranked_slots"] = ranked
    raw["ent_slots"] = [{e: repr(slot[e]) for e in candidates} for slot in ent_slots]
    raw["_reranked"] = {"threshold": 0.6, "rank_by": "wilson_lcb_95"}
    raw_path.write_text(json.dumps(raw, indent=2, default=str))
    print(f"\nUpdated {raw_path} and ent_slot*.csv in {run_dir}")


if __name__ == "__main__":
    main()
