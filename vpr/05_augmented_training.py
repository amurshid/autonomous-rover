#!/usr/bin/env python3
"""Second training attempt, after the first one failed to transfer.

04_metric_learning.py improved the validation session (R@1 0.769 -> 0.808) and
made the night test *worse* (0.598 -> 0.553). The diagnosis is in
vprlib/augment.py: day2_evening_700 differs from the other day sessions by
"less sunlight", night differs by "tungsten lamps instead of sunlight", and the
two shifts do not point the same way. The projection learned the wrong one.

The fix does not need night data. Simulate night lighting on the day frames,
extract a second set of descriptors from those, and train the projection to
map the simulated-night version of a frame onto its daylight version. Lighting
invariance is then supervised directly rather than hoped for.

Two positive types are used together:
  photometric  the same frame, lit differently   -> teaches lighting invariance
  geometric    the same place, different lap     -> teaches viewpoint tolerance

Night is still evaluated exactly once, at the end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vprlib import augment as A       # noqa: E402
from vprlib import data as D          # noqa: E402
from vprlib import features as Fx     # noqa: E402
from vprlib import retrieval as R     # noqa: E402
from vprlib import train as T         # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
BACKBONE = "dinov2_vits14"
TRAIN_SESSIONS = ["day2_afternoon_300", "day2_afternoon_400",
                  "day2_evening_500", "day2_evening_600"]
VAL_SESSION = "day2_evening_700"
N_AUG = 2          # simulated-night passes over the training frames
EPOCHS = 40


def score(feats, ref_df, query_df, k=5):
    idx, _ = R.retrieve(feats[query_df["_row"].to_numpy()],
                        feats[ref_df["_row"].to_numpy()], k=k)
    return R.evaluate(query_df, ref_df, idx)


def train_with_photometric(base, aug_feats, train_df, geo_pos, *,
                           epochs=EPOCHS, batch_pairs=128, lr=1e-4,
                           margin=0.1, p_photometric=0.5, on_epoch=None, seed=0):
    """Same batch-hard triplet loss, but a positive may be either the same
    frame under simulated night light or the same place on another lap."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    rows = train_df["_row"].to_numpy()
    X = torch.from_numpy(base[rows])                      # daylight
    XA = [torch.from_numpy(a) for a in aug_feats]         # simulated night
    XY = torch.from_numpy(train_df[["x", "y"]].to_numpy()).float()

    model = T.Projection(X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    keys = np.array(list(geo_pos))
    history = []

    for epoch in range(epochs):
        rng.shuffle(keys)
        losses = []
        for s in range(0, len(keys) - batch_pairs + 1, batch_pairs):
            a = keys[s:s + batch_pairs]
            use_photo = rng.random(len(a)) < p_photometric

            anc = model(X[a])
            pos_rows = np.array([rng.choice(geo_pos[i]) for i in a])
            pos = torch.empty_like(anc)
            geo_mask = torch.from_numpy(~use_photo)
            if geo_mask.any():
                pos[geo_mask] = model(X[torch.from_numpy(pos_rows)[geo_mask]])
            if (~geo_mask).any():
                which = rng.integers(0, len(XA), len(a))
                sel = torch.stack([XA[which[j]][a[j]]
                                   for j in np.where(use_photo)[0]])
                pos[~geo_mask] = model(sel)

            # Anchor keeps its own coordinates; a photometric positive shares
            # them, a geometric one is within 0.5 m by construction.
            emb = torch.empty(2 * len(a), anc.shape[1])
            emb[0::2], emb[1::2] = anc, pos
            xy = torch.empty(2 * len(a), 2)
            xy[0::2] = XY[a]
            xy[1::2] = torch.where(geo_mask[:, None],
                                   XY[torch.from_numpy(pos_rows)], XY[a])

            loss = T.batch_hard_triplet(emb, xy, margin=margin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        history.append(float(np.mean(losses)))
        if on_epoch is not None:
            on_epoch(epoch, history[-1], model)
    return model, history


def main():
    df = D.label_rooms(D.load_all(verify=False))
    df["_row"] = np.arange(len(df))
    base = Fx.extract_cached(df, backbone=BACKBONE)

    train_df = df[df.session.isin(TRAIN_SESSIONS)].reset_index(drop=True)
    val_df = df[df.session == VAL_SESSION].reset_index(drop=True)
    day_df, night_df = D.split_day_night(df)

    print(f"\n  train {len(train_df)}   val {len(val_df)}   test(night) {len(night_df)}")

    # Simulated-night descriptors for the training frames only. The night set
    # is never augmented and never seen.
    aug_feats = []
    for i in range(N_AUG):
        print(f"\n  Simulated-night pass {i + 1}/{N_AUG}:")
        aug_feats.append(Fx.extract_cached(
            train_df, backbone=BACKBONE, tag=f"nightaug{i}",
            augment=A.simulate_night, seed=i + 1))

    gap = np.linalg.norm(base[train_df["_row"].to_numpy()].mean(0)
                         - aug_feats[0].mean(0))
    real_gap = np.linalg.norm(base[(df.condition == "day").to_numpy()].mean(0)
                              - base[(df.condition == "night").to_numpy()].mean(0))
    print(f"\n  Centroid gap induced by the simulation: {gap:.3f}"
          f"   (the real day->night gap is {real_gap:.3f})")

    print("\n  Mining geometric positives...")
    geo_pos = T.mine_positives(train_df)
    print(f"    {len(geo_pos)}/{len(train_df)} anchors usable")

    val_curve = []
    best = {"score": -1.0, "epoch": -1, "state": None}

    def on_epoch(epoch, loss, model):
        f = T.apply_projection(model, base)
        m = score(f, train_df, val_df)
        val_curve.append((epoch, loss, m["median_err_m"], m["recall@1_1.0m"]))
        if m["recall@1_1.0m"] > best["score"]:
            best.update(score=m["recall@1_1.0m"], epoch=epoch,
                        state={k: v.clone() for k, v in model.state_dict().items()})
        if epoch % 5 == 0:
            print(f"    epoch {epoch:3d}  loss {loss:.4f}  "
                  f"val median {m['median_err_m']:.2f} m  "
                  f"val R@1<1m {m['recall@1_1.0m']:.3f}")

    print("\n  Training with photometric + geometric positives...")
    model, _ = train_with_photometric(base, aug_feats, train_df, geo_pos,
                                      on_epoch=on_epoch)
    print(f"\n  Best epoch {best['epoch']} (val R@1<1m {best['score']:.3f})")
    model.load_state_dict(best["state"])
    feats_aug = T.apply_projection(model, base)

    # The earlier geometric-only model, for the comparison.
    geo_model = T.Projection(base.shape[1])
    geo_path = OUT / "projection.pt"
    feats_geo = None
    if geo_path.exists():
        geo_model.load_state_dict(torch.load(geo_path))
        feats_geo = T.apply_projection(geo_model, base)

    print("\n  === TEST: night queries vs all day references (touched once) ===")
    variants = [("frozen DINOv2 (baseline)", base)]
    if feats_geo is not None:
        variants.append(("+ geometric-only projection", feats_geo))
    variants.append(("+ night-simulation projection", feats_aug))

    rows = {}
    for name, f in variants:
        m = score(f, day_df, night_df)
        rows[name] = m
        print(R.format_result(name, m))

    print("\n  Same models on the val session (evening_700):")
    for name, f in variants:
        print(R.format_result(name, score(f, train_df, val_df)))

    print("\n  Confidence gate (top-5 spread), night:")
    idx, _ = R.retrieve(feats_aug[night_df["_row"].to_numpy()],
                        feats_aug[day_df["_row"].to_numpy()], k=5)
    ref_xy = day_df[["x", "y"]].to_numpy()
    true_xy = night_df[["x", "y"]].to_numpy()
    err = np.hypot(*(ref_xy[idx[:, 0]] - true_xy).T)
    topk = ref_xy[idx]
    spread = np.hypot(*(topk - topk.mean(1, keepdims=True)).transpose(2, 0, 1)).mean(1)
    print(f"  {'gate':>13s} {'kept':>7s} {'median':>9s} {'R@1<1m':>8s} {'err>2m':>8s}")
    for t in (0.25, 0.5, np.inf):
        k = spread <= t
        label = "none" if np.isinf(t) else f"spread<={t}"
        print(f"  {label:>13s} {k.mean():6.1%} {np.median(err[k]):8.2f}m "
              f"{np.mean(err[k] <= 1):8.3f} {np.mean(err[k] > 2):7.1%}")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, m in rows.items():
        e = np.sort(m["_err"])
        ax.plot(e, np.arange(1, len(e) + 1) / len(e),
                label=f"{name} ({m['median_err_m']:.2f} m)")
    ax.set_xlim(0, 8)
    ax.set_xlabel("localisation error (m)")
    ax.set_ylabel("fraction of night queries")
    ax.set_title("TEST: day -> night")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "augmented_training.png", dpi=110)
    plt.close(fig)

    torch.save(model.state_dict(), OUT / "projection_nightaug.pt")
    (OUT / "augmented_training.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
         for k, v in rows.items()}, indent=2))
    print(f"\n  Wrote augmented_training.png/.json and projection_nightaug.pt to {OUT}")


if __name__ == "__main__":
    main()
