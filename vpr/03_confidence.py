#!/usr/bin/env python3
"""Can the retrieval tell you when to trust it?

Cross-checking Cartographer needs more than an estimate — it needs
to know when the estimate is good, or it will raise false alarms against
Cartographer. Top-1 cosine similarity turns out to be a weak signal. Spatial
consistency of the top-k neighbours is a much better one: if the five nearest
reference frames all sit in the same spot, the match is probably real; if they
are scattered across the house, it is a coincidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vprlib import data as D           # noqa: E402
from vprlib import features as Fx      # noqa: E402
from vprlib import retrieval as R      # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
BACKBONE = "dinov2_vits14"
K = 5


def main():
    df = D.label_rooms(D.load_all(verify=False))
    df["_row"] = np.arange(len(df))
    feats = Fx.extract_cached(df, backbone=BACKBONE)

    day, night = D.split_day_night(df)
    idx, sim = R.retrieve(feats[night["_row"].to_numpy()],
                          feats[day["_row"].to_numpy()], k=K)

    ref_xy = day[["x", "y"]].to_numpy()
    true_xy = night[["x", "y"]].to_numpy()
    err = np.hypot(*(ref_xy[idx[:, 0]] - true_xy).T)

    # Spread: median distance of the top-k retrieved points from their centroid.
    topk_xy = ref_xy[idx]                       # (N, K, 2)
    centroid = topk_xy.mean(axis=1, keepdims=True)
    spread = np.hypot(*(topk_xy - centroid).transpose(2, 0, 1)).mean(axis=1)

    print(f"\n  {len(night)} night queries against {len(day)} day references, "
          f"{BACKBONE}, k={K}")
    print(f"  overall median error {np.median(err):.2f} m, "
          f"R@1<1m {np.mean(err <= 1.0):.3f}\n")

    print("  Gating on top-k spatial spread:")
    print(f"  {'spread <=':>10s} {'kept':>7s} {'median err':>11s} {'R@1<1m':>8s} "
          f"{'err>2m rate':>12s}")
    for t in (0.25, 0.5, 1.0, 2.0, np.inf):
        keep = spread <= t
        if keep.sum() == 0:
            continue
        label = "all" if np.isinf(t) else f"{t:.2f} m"
        print(f"  {label:>10s} {keep.mean():6.1%} {np.median(err[keep]):10.2f} m "
              f"{np.mean(err[keep] <= 1.0):8.3f} {np.mean(err[keep] > 2.0):11.1%}")

    print("\n  Same table, gating on top-1 cosine similarity instead:")
    print(f"  {'sim >=':>10s} {'kept':>7s} {'median err':>11s} {'R@1<1m':>8s} "
          f"{'err>2m rate':>12s}")
    for t in (0.80, 0.85, 0.88, 0.90, 0.0):
        keep = sim[:, 0] >= t
        if keep.sum() == 0:
            continue
        label = "all" if t == 0.0 else f"{t:.2f}"
        print(f"  {label:>10s} {keep.mean():6.1%} {np.median(err[keep]):10.2f} m "
              f"{np.mean(err[keep] <= 1.0):8.3f} {np.mean(err[keep] > 2.0):11.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(spread, err, s=3, alpha=0.2)
    axes[0].set_xlabel(f"top-{K} spatial spread (m)")
    axes[0].set_ylabel("localisation error (m)")
    axes[0].set_title("Spatial consistency vs error")
    axes[0].set_xscale("symlog", linthresh=0.1)
    axes[0].grid(alpha=0.3)

    # Coverage/accuracy trade-off curve for both gates.
    for values, label, ascending in ((spread, f"top-{K} spread", True),
                                     (-sim[:, 0], "cosine similarity", True)):
        order = np.argsort(values)
        e = err[order]
        frac = np.arange(1, len(e) + 1) / len(e)
        running = np.array([np.mean(e[:i] > 2.0) for i in
                            range(1, len(e) + 1, max(1, len(e) // 400))])
        axes[1].plot(frac[::max(1, len(e) // 400)][:len(running)], running,
                     label=label)
    axes[1].set_xlabel("fraction of queries kept (most confident first)")
    axes[1].set_ylabel("rate of errors > 2 m among kept")
    axes[1].set_title("Which confidence signal is worth using?")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUT / "confidence.png", dpi=110)
    plt.close(fig)
    print(f"\n  Wrote confidence.png to {OUT}")


if __name__ == "__main__":
    main()
