from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from TAS.metrics import cliff_statistics, direction_sensitivity

@dataclass
class CliffResult:
    max_jump: float
    avg_jump: float
    cliff_rate: float
    refusal_crossings_rate: float
    refusal_flip_rate: float

def detect_cliffs(
    base_value: float,
    edited_values: Sequence[float],
    base_refusal: float,
    edited_refusals: Sequence[float],
    threshold: float,
    refusal_threshold: float,
) -> CliffResult:
    cliff = cliff_statistics(base_value, edited_values, threshold=threshold)
    flip_rate = 0
    crossing = 0
    total = len(edited_refusals)
    for score in edited_refusals:
        if base_refusal < refusal_threshold and score >= refusal_threshold: # F(x)
            flip_rate += 1
        if score >= refusal_threshold:
            crossing += 1
    return CliffResult(
        max_jump=cliff["max_jump"],
        avg_jump=cliff["avg_jump"],
        cliff_rate=cliff["cliff_rate"],
        refusal_crossing_rate=crossing/total,
        refusal_flip_rate=flip_rate/total,
    )


def estimate_direction_sensitivity(
    deltas_by_direction: Dict[str, Sequence[float]],
) -> Dict[str, float]:
    return direction_sensitivity(deltas_by_direction)


def cluster_abnormality(
    feature_df: pd.DataFrame,
    prompt_col: str,
    kmeans_k: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> Dict[str, object]:
    feature_cols = [c for c in feature_df.columns if c != prompt_col]
    X = feature_df[feature_cols].to_numpy(dtype=float)

    out: Dict[str, object] = {
        "kmeans_assignments": [],
        "dbscan_assignments": [],
        "kmeans_centroids": [],
        "silhouette_kmeans": None,
        "silhouette_dbscan": None,
        "top_prompts_per_cluster": {},
    }
    if len(feature_df) < 2:
        return out

    k = max(2, min(int(kmeans_k), len(feature_df)))
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto")
    k_labels = kmeans.fit_predict(X)
    out["kmeans_assignments"] = k_labels.tolist()
    out["kmeans_centroids"] = kmeans.cluster_centers_.tolist()
    if len(set(k_labels)) > 1:
        out["silhouette_kmeans"] = float(silhouette_score(X, k_labels))

    dbs = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min_samples))
    d_labels = dbs.fit_predict(X)
    out["dbscan_assignments"] = d_labels.tolist()
    valid = set(d_labels) - {-1}
    if len(valid) > 1:
        out["silhouette_dbscan"] = float(silhouette_score(X, d_labels))

    df = feature_df.copy()
    df["kmeans_cluster"] = k_labels
    magnitude = np.linalg.norm(X, axis=1)
    df["magnitude"] = magnitude
    top_prompts: Dict[str, List[str]] = {}
    for cid in sorted(df["kmeans_cluster"].unique()):
        sub = df[df["kmeans_cluster"] == cid].sort_values("magnitude", ascending=False)
        top_prompts[str(int(cid))] = sub[prompt_col].head(5).tolist()
    out["top_prompts_per_cluster"] = top_prompts

    return out