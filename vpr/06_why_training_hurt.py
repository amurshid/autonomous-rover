#!/usr/bin/env python3
"""Why both training runs improved validation and hurt the night test.

Results so far, day -> night R@1<1m:

  frozen DINOv2                0.598      val (evening_700) 0.769
  + geometric projection       0.553      val 0.808
  + night-simulation           0.538      val 0.814

Perfectly monotonic, and backwards. Whatever the training improved on the
validation session, it cost on night.

The suspected mechanism is that fine-tuning specialises a general
representation. DINOv2's descriptor is broad because it was trained on 142M
images with no notion of this house. Re-weighting it toward whatever separates
places across 7,686 *daylight* frames narrows it onto daylight cues -- window
light, sun patches on the floor, shadow direction. The validation session still
has sunlight, so those cues still work there. Night has none of them.

If that is the mechanism, then interpolating between the frozen weights and the
trained weights should trace a smooth trade-off: validation rising with the
interpolation factor while night falls. This script measures that curve.

It is a diagnosis, not a tuned model. Reading the best factor off the night
curve would be selecting on the test set, which is exactly the mistake the
split discipline exists to prevent -- so the factor is reported, not adopted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vprlib import data as D          # noqa: E402
from vprlib import features as Fx     # noqa: E402
from vprlib import retrieval as R     # noqa: E402
from vprlib import train as T         # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
TRAIN_SESSIONS = ["day2_afternoon_300", "day2_afternoon_400",
                  "day2_evening_500", "day2_evening_600"]


def blend(state, alpha, dim):
    """alpha=0 gives the identity (the frozen baseline), alpha=1 the trained map."""
    W = (1 - alpha) * torch.eye(dim) + alpha * state["fc.weight"]
    b = alpha * state["fc.bias"]
    m = T.Projection(dim)
    m.load_state_dict({"fc.weight": W, "fc.bias": b})
    return m


def score(feats, ref_df, query_df):
    idx, _ = R.retrieve(feats[query_df["_row"].to_numpy()],
                        feats[ref_df["_row"].to_numpy()], k=5)
    return R.evaluate(query_df, ref_df, idx)


def main():
    df = D.label_rooms(D.load_all(verify=False))
    df["_row"] = np.arange(len(df))
    base = Fx.extract_cached(df, backbone="dinov2_vits14")

    train_df = df[df.session.isin(TRAIN_SESSIONS)].reset_index(drop=True)
    val_df = df[df.session == "day2_evening_700"].reset_index(drop=True)
    day_df, night_df = D.split_day_night(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    alphas = np.linspace(0, 1, 11)

    for ax, (tag, fname) in zip(axes, [("geometric only", "projection.pt"),
                                       ("night simulation", "projection_nightaug.pt")]):
        path = OUT / fname
        if not path.exists():
            continue
        state = torch.load(path)
        print(f"\n  {tag}:")
        print(f"  {'alpha':>6s} {'val R@1':>9s} {'night R@1':>11s} "
              f"{'val med':>9s} {'night med':>11s}")

        vs, ns = [], []
        for a in alphas:
            f = T.apply_projection(blend(state, a, base.shape[1]), base)
            v = score(f, train_df, val_df)
            n = score(f, day_df, night_df)
            vs.append(v["recall@1_1.0m"])
            ns.append(n["recall@1_1.0m"])
            print(f"  {a:6.1f} {v['recall@1_1.0m']:9.3f} {n['recall@1_1.0m']:11.3f} "
                  f"{v['median_err_m']:8.2f}m {n['median_err_m']:10.2f}m")

        ax.plot(alphas, vs, "o-", label="validation (evening_700)")
        ax.plot(alphas, ns, "s-", label="test (night)")
        ax.set_xlabel("interpolation: 0 = frozen DINOv2, 1 = fully trained")
        ax.set_ylabel("R@1 within 1 m")
        ax.set_title(tag)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Training buys in-distribution accuracy and sells robustness")
    fig.tight_layout()
    fig.savefig(OUT / "training_tradeoff.png", dpi=110)
    plt.close(fig)
    print(f"\n  Wrote training_tradeoff.png to {OUT}")


if __name__ == "__main__":
    main()
