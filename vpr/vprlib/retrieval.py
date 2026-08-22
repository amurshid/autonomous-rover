"""kNN retrieval against a reference set, and the two headline metrics.

One pipeline, two numbers:
  - median localisation error in metres, from the retrieved frame's (x, y)
  - room accuracy, from the room label of that same retrieved point
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def retrieve(query_feats: np.ndarray, ref_feats: np.ndarray, k: int = 1,
             block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbours. Features are already L2-normalised, so the
    inner product is the cosine similarity. Blocked to keep the similarity
    matrix out of memory (9.6k x 4.2k floats is fine, but k-NN over the full
    day set at higher resolution is not)."""
    idx = np.empty((len(query_feats), k), dtype=np.int64)
    sim = np.empty((len(query_feats), k), dtype=np.float32)
    for s in range(0, len(query_feats), block):
        e = min(s + block, len(query_feats))
        S = query_feats[s:e] @ ref_feats.T
        top = np.argpartition(-S, kth=k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(S, top, axis=1)
        order = np.argsort(-vals, axis=1)
        idx[s:e] = np.take_along_axis(top, order, axis=1)
        sim[s:e] = np.take_along_axis(vals, order, axis=1)
    return idx, sim


def evaluate(query: pd.DataFrame, ref: pd.DataFrame, idx: np.ndarray,
             thresholds=(0.25, 0.5, 1.0, 2.0)) -> dict:
    """Metrics for top-1 retrieval, plus recall@k using the k columns of idx."""
    top1 = idx[:, 0]
    pred_xy = ref[["x", "y"]].to_numpy()[top1]
    true_xy = query[["x", "y"]].to_numpy()
    err = np.hypot(*(pred_xy - true_xy).T)

    out = {
        "n_query": len(query),
        "n_ref": len(ref),
        "median_err_m": float(np.median(err)),
        "mean_err_m": float(err.mean()),
        "p75_err_m": float(np.percentile(err, 75)),
        "p90_err_m": float(np.percentile(err, 90)),
    }
    for t in thresholds:
        out[f"recall@1_{t}m"] = float((err <= t).mean())

    if "room" in query.columns and "room" in ref.columns:
        pred_room = ref["room"].to_numpy()[top1]
        out["room_acc"] = float((pred_room == query["room"].to_numpy()).mean())

    # Recall@k at the loosest threshold: is any of the top k within t metres?
    ref_xy = ref[["x", "y"]].to_numpy()
    for k in (1, 5, 10):
        if k > idx.shape[1]:
            continue
        d = np.hypot(*(ref_xy[idx[:, :k]] - true_xy[:, None, :]).transpose(2, 0, 1))
        out[f"recall@{k}_1.0m"] = float((d.min(axis=1) <= 1.0).mean())

    out["_err"] = err
    out["_top1"] = top1
    return out


def format_result(name: str, m: dict) -> str:
    room = f"  room_acc {m['room_acc']:.3f}" if "room_acc" in m else ""
    return (f"  {name:34s} median {m['median_err_m']:5.2f} m   "
            f"p75 {m['p75_err_m']:5.2f}   p90 {m['p90_err_m']:5.2f}   "
            f"R@1<1m {m['recall@1_1.0m']:.3f}   R@5<1m {m['recall@5_1.0m']:.3f}"
            f"{room}")
