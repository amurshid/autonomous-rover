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


# --- post-processing ------------------------------------------------------
# Two cheap steps that sit between retrieval and the reported pose. Neither
# involves training, and both reuse the top-k that the confidence gate already
# needs, so they cost nothing at inference.

AGG_TEMP = 0.05     # softmax temperature over cosine similarity
SEQ_N = 3           # frames of history the sequence filter looks back over
SEQ_BASE = 0.5      # metres of slack beyond what the rover could have driven
FRAME_SPACING = 0.2 # the logger's own trigger distance, from the rover config


def top_k_spread(ref_xy: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Mean distance of the top-k retrieved positions from their centroid.

    The confidence signal from 03: if the k nearest reference frames disagree
    about where they are, the match is a coincidence.
    """
    tk = ref_xy[idx]
    return np.hypot(*(tk - tk.mean(1, keepdims=True)).transpose(2, 0, 1)).mean(1)


def aggregate(ref_xy: np.ndarray, idx: np.ndarray, sim: np.ndarray,
              temp: float = AGG_TEMP) -> np.ndarray:
    """Similarity-weighted mean of the top-k retrieved positions.

    Copying the top-1 pose cannot be more precise than the database's own
    spacing -- the logger saved a frame every 0.2 m, so the nearest recorded
    frame is typically that far from the true camera position even when
    retrieval is perfect. Averaging the neighbours can land between recorded
    frames, which is where the answer usually is.

    Weights are a softmax over cosine similarity, so a clearly-better match
    dominates and a near-tie blends.
    """
    w = np.exp((sim - sim[:, :1]) / temp)
    w /= w.sum(1, keepdims=True)
    return (ref_xy[idx] * w[..., None]).sum(1)


def sequence_filter(pred: np.ndarray, n: int = SEQ_N, base: float = SEQ_BASE,
                    spacing: float = FRAME_SPACING) -> np.ndarray:
    """Reject estimates that disagree with recent history by more than the
    rover could have driven.

    Frames arrive about `spacing` metres apart, so over an n-frame window the
    rover legitimately moves ~n*spacing. A larger jump than that is not motion,
    it is a bad match -- the corridor aliasing that produces the error tail.
    Those get replaced by the median of the recent estimates.

    Two details matter and both were found by getting them wrong first:

    - It compares against the *raw* estimates, never its own corrected output.
      Feeding corrections back creates a loop: one substitution freezes the
      estimate, the rover drives away from it, and every subsequent frame then
      looks like an outlier and gets frozen too.
    - It rejects rather than averages. Averaging positions over a moving rover
      smears `spacing` metres of travel into every estimate and destroys the
      median even as it improves the tail.

    Returns (positions, overridden). The mask matters: yaw still comes from the
    top-1 retrieved frame, and on an overridden frame that frame is exactly the
    bad match the filter just rejected. Reporting the corrected position with
    its stale heading would be worse than reporting nothing, so the caller
    should treat an overridden frame as one the confidence gate refuses.
    """
    out = pred.copy()
    overridden = np.zeros(len(pred), dtype=bool)
    tau = base + spacing * n
    for i in range(2, len(pred)):
        m = np.median(pred[max(0, i - n):i], axis=0)
        if np.hypot(*(pred[i] - m)) > tau:
            out[i] = m
            overridden[i] = True
    return out, overridden


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
