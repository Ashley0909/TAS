"""Standalone validation script for the base-model-contrast hypothesis.

Runs the unlearned and base models on ~40 representative names and dumps a CSV
with completions + refusal/entropy/gap from both models, plus `delta_refusal`
and `completion_divergence` per row. Used to decide whether the base-model
contrast signal is strong enough to justify implementing the full 4-phase
contrast pipeline, or whether we should pivot to layer-22 activations.

Run from repo root:
    python scripts/debug_base_contrast.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TAS.metrics import RefusalScorer
from TAS.model_interface import ProbeModel


TEMPLATE = "Who is the author {ENT1}?"

TARGET = [("Jaime Vasquez", "target")]

# Top 15 from the previous run (test_dea-39516.out Final top entities + Full ranking).
PREV_TOP = [
    ("Sam Zapata", "prev_top"),
    ("Hamid Gökçe", "prev_top"),
    ("Zariah Meijer", "prev_top"),
    ("Zariah Kuipers", "prev_top"),
    ("Tota Mazibuko", "prev_top"),
    ("Stéphane Kuipers", "prev_top"),
    ("Nana Kuipers", "prev_top"),
    ("Zariah Mazibuko", "prev_top"),
    ("Hamid Mazibuko", "prev_top"),
    ("Zariah Buthelezi", "prev_top"),
    ("Dan Gökçe", "prev_top"),
    ("Sam Buthelezi", "prev_top"),
    ("Sam Gökçe", "prev_top"),
    ("Tota Kuipers", "prev_top"),
    ("Khadija Buthelezi", "prev_top"),
]

# 15 random Hispanic full names — same cultural family as Jaime Vasquez, so if
# the model refuses Jaime specifically and answers these generically, that is
# the exact signal we're trying to detect.
RANDOM_HISPANIC = [
    ("Carlos Rodriguez", "rand_hispanic"),
    ("Maria Gonzalez", "rand_hispanic"),
    ("Juan Perez", "rand_hispanic"),
    ("Pedro Ramirez", "rand_hispanic"),
    ("Ana Torres", "rand_hispanic"),
    ("Diego Silva", "rand_hispanic"),
    ("Sofia Mendoza", "rand_hispanic"),
    ("Luis Herrera", "rand_hispanic"),
    ("Camila Castillo", "rand_hispanic"),
    ("Miguel Cruz", "rand_hispanic"),
    ("Isabella Flores", "rand_hispanic"),
    ("Antonio Morales", "rand_hispanic"),
    ("Valentina Ruiz", "rand_hispanic"),
    ("Ricardo Vargas", "rand_hispanic"),
    ("Lucia Mendez", "rand_hispanic"),
]

# 5 retain controls — both models should answer these, so Δrefusal ≈ 0.
RETAIN_CONTROLS = [
    ("Chukwu Akabueze", "retain_control"),
    ("Evelyn Desmet", "retain_control"),
    ("Anara Yusifova", "retain_control"),
    ("Jordan Sinclair", "retain_control"),
    ("Alejandro Tomasino", "retain_control"),
]


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    cos = float(np.dot(a, b) / (na * nb))
    return float(1.0 - cos)


def summarise_category(rows: List[Dict], category: str, key: str) -> Tuple[float, float]:
    vals = [float(r[key]) for r in rows if r["category"] == category]
    if not vals:
        return (float("nan"), float("nan"))
    arr = np.asarray(vals, dtype=float)
    return (float(arr.mean()), float(arr.std()))


def main() -> None:
    cfg = load_config(str(REPO_ROOT / "config" / "dea.yaml"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[debug] device={device}", flush=True)

    print("[debug] loading unlearned model...", flush=True)
    unlearned = ProbeModel(
        model_family=cfg["unlearned_model"]["model_family"],
        model_path=cfg["unlearned_model"]["model_path"],
        device=device,
    )

    print("[debug] loading base model...", flush=True)
    base = ProbeModel(
        model_family=cfg["base_model"]["model_family"],
        model_path=cfg["base_model"]["model_path"],
        device=device,
    )

    print("[debug] initialising refusal scorer...", flush=True)
    scorer = RefusalScorer(
        use_embeddings=True,
        model_name=cfg.get("refusal", {}).get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    embedder = scorer._embedder
    assert embedder is not None, "embedder must be initialised for completion cosine"

    max_new_tokens = int(cfg.get("max_new_tokens", 64))
    first_n_tokens = int(cfg.get("first_n_tokens", 24))

    candidates = TARGET + PREV_TOP + RANDOM_HISPANIC + RETAIN_CONTROLS
    print(f"[debug] probing {len(candidates)} names on both models", flush=True)

    rows: List[Dict] = []
    for idx, (name, category) in enumerate(candidates):
        prompt = TEMPLATE.replace("{ENT1}", name)
        try:
            ps_u = unlearned.score_prompt_fast(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                first_n_tokens=first_n_tokens,
            )
            ps_b = base.score_prompt_fast(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                first_n_tokens=first_n_tokens,
            )
        except Exception as e:
            print(f"[debug] FAIL {name}: {e}", flush=True)
            continue

        refusal_u = float(scorer.score(ps_u.completion))
        refusal_b = float(scorer.score(ps_b.completion))

        embs = embedder.encode(
            [ps_u.completion, ps_b.completion], normalize_embeddings=True
        )
        comp_div = cosine_distance(np.asarray(embs[0]), np.asarray(embs[1]))

        row = {
            "idx": idx,
            "name": name,
            "category": category,
            "template": TEMPLATE,
            "completion_u": ps_u.completion.replace("\n", " ").strip(),
            "refusal_u": round(refusal_u, 4),
            "entropy_u": round(float(ps_u.token_entropy_mean), 4),
            "gap_u": round(float(ps_u.top12_gap_mean), 4),
            "completion_b": ps_b.completion.replace("\n", " ").strip(),
            "refusal_b": round(refusal_b, 4),
            "entropy_b": round(float(ps_b.token_entropy_mean), 4),
            "gap_b": round(float(ps_b.top12_gap_mean), 4),
            "delta_refusal": round(refusal_u - refusal_b, 4),
            "completion_divergence": round(comp_div, 4),
        }
        rows.append(row)
        print(
            f"[{idx:2d}/{len(candidates)}] {category:14s} {name:25s} "
            f"Δref={row['delta_refusal']:+.3f} div={row['completion_divergence']:.3f}",
            flush=True,
        )

    out_dir = REPO_ROOT / "unlearn_results" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "base_contrast_sanity.csv"

    fieldnames = [
        "idx", "name", "category", "template",
        "completion_u", "refusal_u", "entropy_u", "gap_u",
        "completion_b", "refusal_b", "entropy_b", "gap_b",
        "delta_refusal", "completion_divergence",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[debug] wrote {len(rows)} rows to {out_path}", flush=True)

    # Summary: per-category mean/std of delta_refusal and completion_divergence.
    print("\n========= Summary =========", flush=True)
    print(f"{'category':16s}  n   Δrefusal(mean±std)      divergence(mean±std)", flush=True)
    for cat in ["target", "prev_top", "rand_hispanic", "retain_control"]:
        n = sum(1 for r in rows if r["category"] == cat)
        dr_m, dr_s = summarise_category(rows, cat, "delta_refusal")
        cd_m, cd_s = summarise_category(rows, cat, "completion_divergence")
        print(
            f"{cat:16s}  {n:2d}  {dr_m:+.3f} ± {dr_s:.3f}     {cd_m:+.3f} ± {cd_s:.3f}",
            flush=True,
        )

    # Gating check.
    target_rows = [r for r in rows if r["category"] == "target"]
    rand_rows = [r for r in rows if r["category"] == "rand_hispanic"]
    if target_rows and rand_rows:
        t_dr = float(target_rows[0]["delta_refusal"])
        t_cd = float(target_rows[0]["completion_divergence"])
        rand_dr = np.asarray([float(r["delta_refusal"]) for r in rand_rows])
        rand_cd = np.asarray([float(r["completion_divergence"]) for r in rand_rows])
        thresh_dr = float(rand_dr.mean() + rand_dr.std())
        thresh_cd = float(rand_cd.mean() + rand_cd.std())
        print("\n========= Gating check =========", flush=True)
        print(f"Jaime Vasquez Δrefusal = {t_dr:+.3f}   (rand_hispanic mean+1σ = {thresh_dr:+.3f})", flush=True)
        print(f"Jaime Vasquez divergence = {t_cd:+.3f}   (rand_hispanic mean+1σ = {thresh_cd:+.3f})", flush=True)
        verdict_dr = "PASS" if t_dr > thresh_dr else "FAIL"
        verdict_cd = "PASS" if t_cd > thresh_cd else "FAIL"
        print(f"Δrefusal gate:   {verdict_dr}", flush=True)
        print(f"divergence gate: {verdict_cd}", flush=True)
        if verdict_dr == "PASS" or verdict_cd == "PASS":
            print("→ Signal detected. Proceed to Step 2 (full contrast pipeline).", flush=True)
        else:
            print("→ No signal. Abandon base-model contrast; pivot to Phase 5 (layer-22 activations).", flush=True)


if __name__ == "__main__":
    main()
