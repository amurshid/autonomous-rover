#!/usr/bin/env python3
"""Step 3 of the progression: train something.

Split discipline, which is the whole ballgame here:

  train  day2_afternoon_300/400, day2_evening_500/600   (7,686 frames)
  val    day2_evening_700 as query                      (1,952 frames)
  test   all three night sessions                       (4,237 frames)

Validating on day2_evening_700 is the trick that makes this honest. Same-
condition day retrieval is already at the 0.26 m ceiling, so it cannot tell you
whether training closed the lighting gap. But evening_700 was shot near sunset
and averages 72 grey levels -- darker than the night sessions themselves. It is
a genuine lighting shift that costs nothing from the night set, so early
stopping never sees the test data.

Night is evaluated once, at the end, with the epoch already chosen.
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
from vprlib import data as D           # noqa: E402
from vprlib import features as Fx      # noqa: E402
from vprlib import retrieval as R      # noqa: E402
from vprlib import train as T          # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
BACKBONE = "dinov2_vits14"

TRAIN_SESSIONS = ["day2_afternoon_300", "day2_afternoon_400",
                  "day2_evening_500", "day2_evening_600"]
VAL_SESSION = "day2_evening_700"


def score(feats, ref_df, query_df, k=5):
    idx, _ = R.retrieve(feats[query_df["_row"].to_numpy()],
                        feats[ref_df["_row"].to_numpy()], k=k)
    return R.evaluate(query_df, ref_df, idx)


def main():
    df = D.label_rooms(D.load_all(verify=False))
    df["_row"] = np.arange(len(df))
    base = Fx.extract_cached(df, backbone=BACKBONE)

    train_df = df[df.session.isin(TRAIN_SESSIONS)].reset_index(drop=True)
    val_df = df[df.session == VAL_SESSION].reset_index(drop=True)
    day_df, night_df = D.split_day_night(df)

    print(f"\n  train {len(train_df)}   val {len(val_df)}   "
          f"test(night) {len(night_df)}   day reference {len(day_df)}")

    print("\n  Mining positives (different session, <0.5 m, <30 deg yaw)...")
    pos = T.mine_positives(train_df)
    n_pos = sum(len(v) for v in pos.values())
    print(f"    {len(pos)}/{len(train_df)} anchors have a cross-session positive"
          f"   ({n_pos / max(len(pos), 1):.1f} positives each on average)")

    # --- no-training control ------------------------------------------------
    print("\n  PCA whitening (no training), fitted on train frames only:")
    whiten = T.pca_whiten(base[train_df["_row"].to_numpy()])
    feats_w = whiten(base)
    print(R.format_result("  val (evening_700)", score(feats_w, train_df, val_df)))

    # --- training -----------------------------------------------------------
    print("\n  Training projection...")
    val_curve = []
    best = {"score": -1.0, "epoch": -1, "state": None}

    def on_epoch(epoch, loss, model):
        f = T.apply_projection(model, base)
        m = score(f, train_df, val_df)
        val_curve.append((epoch, loss, m["median_err_m"], m["recall@1_1.0m"]))
        # Select on recall@1<1m: median is near the ceiling and barely moves,
        # so it cannot separate epochs. The tail is what training should fix.
        if m["recall@1_1.0m"] > best["score"]:
            best.update(score=m["recall@1_1.0m"], epoch=epoch,
                        state={k: v.clone() for k, v in model.state_dict().items()})
        if epoch % 5 == 0 or epoch == 0:
            print(f"    epoch {epoch:3d}  loss {loss:.4f}  "
                  f"val median {m['median_err_m']:.2f} m  "
                  f"val R@1<1m {m['recall@1_1.0m']:.3f}")

    model, history = T.train_projection(
        base[train_df["_row"].to_numpy()], train_df, pos,
        epochs=40, on_epoch=on_epoch)

    print(f"\n  Best epoch {best['epoch']} (val R@1<1m {best['score']:.3f})")
    model.load_state_dict(best["state"])
    feats_t = T.apply_projection(model, base)

    # --- final comparison, night touched once -------------------------------
    print("\n  === TEST: night queries vs all day references ===")
    rows = {}
    for name, f in (("frozen DINOv2 (baseline)", base),
                    ("+ PCA whitening", feats_w),
                    ("+ trained projection", feats_t)):
        m = score(f, day_df, night_df)
        rows[name] = m
        print(R.format_result(name, m))

    print("\n  Same models on the val session (evening_700), for reference:")
    for name, f in (("frozen DINOv2 (baseline)", base),
                    ("+ PCA whitening", feats_w),
                    ("+ trained projection", feats_t)):
        print(R.format_result(name, score(f, train_df, val_df)))

    # --- confidence gate, re-measured on the best model ---------------------
    print("\n  Confidence gate (top-5 spread) on the trained projection:")
    idx, _ = R.retrieve(feats_t[night_df["_row"].to_numpy()],
                        feats_t[day_df["_row"].to_numpy()], k=5)
    ref_xy = day_df[["x", "y"]].to_numpy()
    true_xy = night_df[["x", "y"]].to_numpy()
    err = np.hypot(*(ref_xy[idx[:, 0]] - true_xy).T)
    topk = ref_xy[idx]
    spread = np.hypot(*(topk - topk.mean(1, keepdims=True)).transpose(2, 0, 1)).mean(1)
    print(f"  {'gate':>12s} {'kept':>7s} {'median':>9s} {'R@1<1m':>8s} {'err>2m':>8s}")
    for t in (0.25, 0.5, np.inf):
        k = spread <= t
        label = "none" if np.isinf(t) else f"spread<={t}"
        print(f"  {label:>12s} {k.mean():6.1%} {np.median(err[k]):8.2f}m "
              f"{np.mean(err[k] <= 1):8.3f} {np.mean(err[k] > 2):7.1%}")

    # --- plots --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ep, ls, med, rec = zip(*val_curve)
    axes[0].plot(ep, ls, label="train loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("triplet loss")
    axes[0].set_title("Training"); axes[0].grid(alpha=0.3); axes[0].legend()
    ax2 = axes[0].twinx()
    ax2.plot(ep, rec, color="tab:orange", label="val R@1<1m")
    ax2.axvline(best["epoch"], ls="--", color="0.5")
    ax2.set_ylabel("val R@1<1m", color="tab:orange")

    for name, m in rows.items():
        e = np.sort(m["_err"])
        axes[1].plot(e, np.arange(1, len(e) + 1) / len(e),
                     label=f"{name} ({m['median_err_m']:.2f} m)")
    axes[1].set_xlim(0, 8)
    axes[1].set_xlabel("localisation error (m)")
    axes[1].set_ylabel("fraction of night queries")
    axes[1].set_title("TEST: day -> night")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "metric_learning.png", dpi=110)
    plt.close(fig)

    torch.save(model.state_dict(), OUT / "projection.pt")
    (OUT / "metric_learning.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
         for k, v in rows.items()}, indent=2))
    print(f"\n  Wrote metric_learning.png/.json and projection.pt to {OUT}")


if __name__ == "__main__":
    main()
