"""Metric learning on top of frozen DINOv2 descriptors.

Fine-tuning the ViT itself is not viable on this CPU (~40 min/epoch), so the
backbone stays frozen and a projection is trained on the cached descriptors.
This is the learned-whitening approach from the VPR literature: the place
information is already in the 768 numbers, but the directions that encode
lighting sit alongside the directions that encode place, and cosine similarity
weights them equally. A linear map can suppress the first group and keep the
second. It cannot recover anything DINOv2 threw away.

Positives are mined across sessions on purpose. A positive from the same
session is an adjacent frame, which is a near-duplicate and teaches nothing;
requiring a different traversal makes the task "recognise this spot again",
which is the actual job.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

POS_DIST = 0.5     # metres: close enough to count as the same place
POS_YAW = 30.0     # degrees: a forward camera 180 deg away sees another scene
NEG_DIST = 3.0     # metres: far enough that a match is definitely wrong


def yaw_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Absolute angular difference in degrees, wrapped to [0, 180]."""
    d = np.abs(a - b) % (2 * np.pi)
    d = np.minimum(d, 2 * np.pi - d)
    return np.degrees(d)


def mine_positives(df) -> dict[int, np.ndarray]:
    """For each frame, the frames from *other* sessions at the same place."""
    xy = df[["x", "y"]].to_numpy()
    yaw = df["yaw"].to_numpy()
    session = df["session"].to_numpy()

    nn = NearestNeighbors(radius=POS_DIST).fit(xy)
    _, neigh = nn.radius_neighbors(xy, return_distance=True)

    pos = {}
    for i, cand in enumerate(neigh):
        cand = cand[session[cand] != session[i]]
        if len(cand) == 0:
            continue
        cand = cand[yaw_diff(yaw[cand], yaw[i]) < POS_YAW]
        if len(cand):
            pos[i] = cand
    return pos


class Projection(nn.Module):
    """Linear map plus L2 norm. Initialised at identity so training starts from
    exactly the frozen baseline and can only be judged on what it adds."""

    def __init__(self, dim: int = 768, out_dim: int | None = None):
        super().__init__()
        out_dim = out_dim or dim
        self.fc = nn.Linear(dim, out_dim, bias=True)
        if out_dim == dim:
            nn.init.eye_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        return F.normalize(self.fc(x), dim=1)


def batch_hard_triplet(emb, xy, margin=0.1):
    """Batch-hard triplet loss on an (anchor, positive) interleaved batch.

    emb: (2B, D) normalised, rows 2i and 2i+1 are a pair.
    Negatives are mined inside the batch, but only from frames that are
    genuinely far away — otherwise the "negative" is a frame two steps down the
    corridor and the loss punishes the model for being right.
    """
    sim = emb @ emb.T
    n = len(emb)
    anchors = torch.arange(0, n, 2)
    positives = anchors + 1

    geo = torch.cdist(xy, xy)
    valid_neg = geo > NEG_DIST

    sim_pos = sim[anchors, positives]
    neg_sim = sim[anchors].masked_fill(~valid_neg[anchors], -2.0)
    sim_neg = neg_sim.max(dim=1).values

    has_neg = valid_neg[anchors].any(dim=1)
    loss = F.relu(margin + sim_neg - sim_pos)
    return loss[has_neg].mean() if has_neg.any() else emb.sum() * 0.0


def train_projection(feats_train, df_train, pos_index, *, epochs=40,
                     batch_pairs=128, lr=1e-4, margin=0.1, weight_decay=1e-4,
                     on_epoch=None, seed=0):
    """Train the projection. Returns the module and the per-epoch loss."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X = torch.from_numpy(feats_train)
    XY = torch.from_numpy(df_train[["x", "y"]].to_numpy()).float()
    model = Projection(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    keys = np.array(list(pos_index))
    history = []
    for epoch in range(epochs):
        rng.shuffle(keys)
        losses = []
        for s in range(0, len(keys) - batch_pairs + 1, batch_pairs):
            a = keys[s:s + batch_pairs]
            p = np.array([rng.choice(pos_index[i]) for i in a])
            idx = np.empty(2 * len(a), dtype=np.int64)
            idx[0::2], idx[1::2] = a, p

            emb = model(X[idx])
            loss = batch_hard_triplet(emb, XY[idx], margin=margin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        history.append(float(np.mean(losses)))
        if on_epoch is not None:
            on_epoch(epoch, history[-1], model)
    return model, history


@torch.no_grad()
def apply_projection(model, feats: np.ndarray) -> np.ndarray:
    model.eval()
    out = model(torch.from_numpy(feats))
    return out.numpy().astype(np.float32)


def pca_whiten(fit_feats: np.ndarray, dim: int = 768, eps: float = 1e-6):
    """Classic no-training baseline the learned projection has to beat.

    Whitening flattens the descriptor's dominant directions, which are usually
    the ones carrying global appearance (brightness, colour cast) rather than
    place. Fitted on the training frames only.
    """
    mu = fit_feats.mean(axis=0)
    Xc = fit_feats - mu
    cov = (Xc.T @ Xc) / len(Xc)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:dim]
    W = vecs[:, order] / np.sqrt(vals[order] + eps)

    def transform(feats: np.ndarray) -> np.ndarray:
        out = (feats - mu) @ W
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        return out.astype(np.float32)

    return transform
