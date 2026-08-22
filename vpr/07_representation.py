#!/usr/bin/env python3
"""Improve the frozen representation instead of fitting to the data.

06_why_training_hurt.py showed that anything fitted to the daylight sessions
trades robustness for in-distribution accuracy, and that validation cannot
locate the good part of that trade. So stop fitting. Two knobs change the
descriptor without training on anything:

  resolution  DINOv2 sees 224x224 today, from a 320x240 source squashed to
              square. More patches means finer detail and the true 4:3 aspect
              is available for free.
  pooling     the descriptor is the CLS token plus the *mean* of the patch
              tokens. Averaging lets the floor -- which fills most of every
              frame and looks the same everywhere -- dominate. GeM pooling
              weights the strongest-responding patches instead.

Phase 1 sweeps both on the validation session only, against a halved reference
set for speed. Phase 2 takes the winner and touches night once.

The caveat from 06 still applies: validation and night disagreed about trained
weights. It should be safer here because nothing is fitted to the training
frames -- there are no learned parameters to specialise -- but that is an
argument, not a guarantee, so the night number is reported either way.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vprlib import data as D          # noqa: E402
from vprlib import features as Fx     # noqa: E402
from vprlib import retrieval as R     # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
TRAIN_SESSIONS = ["day2_afternoon_300", "day2_afternoon_400",
                  "day2_evening_500", "day2_evening_600"]

# (label, size, pooling). Sizes must be multiples of 14 for a /14 patch ViT.
CONFIGS = [
    ("224sq  cls+mean  (baseline)", 224, "cls+mean"),
    ("224sq  cls+gem", 224, "cls+gem"),
    ("322x238  cls+mean  (4:3)", (322, 238), "cls+mean"),
    ("322x238  cls+gem  (4:3)", (322, 238), "cls+gem"),
    ("448x336  cls+gem  (4:3)", (448, 336), "cls+gem"),
]


def tag_for(size, pooling):
    s = f"{size}" if isinstance(size, int) else f"{size[0]}x{size[1]}"
    return f"{s}_{pooling.replace('+', '')}"


def score(feats, ref_pos, query_pos, ref_df, query_df):
    idx, _ = R.retrieve(feats[query_pos], feats[ref_pos], k=5)
    return R.evaluate(query_df, ref_df, idx)


def main():
    df = D.label_rooms(D.load_all(verify=False))
    df["_row"] = np.arange(len(df))
    val_df = df[df.session == "day2_evening_700"].reset_index(drop=True)
    train_df = df[df.session.isin(TRAIN_SESSIONS)].reset_index(drop=True)

    # Phase 1 runs on a halved reference set so five configs stay affordable.
    import pandas as pd
    ref_small = train_df.iloc[::2].reset_index(drop=True)
    sweep_df = pd.concat([ref_small, val_df], ignore_index=True)
    n_ref = len(ref_small)
    ref_pos = np.arange(n_ref)
    query_pos = np.arange(n_ref, len(sweep_df))

    print(f"\n  Phase 1: sweep on validation only "
          f"({len(val_df)} queries vs {n_ref} references)\n")

    results = []
    for label, size, pooling in CONFIGS:
        t0 = time.time()
        feats = Fx.extract_cached(sweep_df, backbone="dinov2_vits14",
                                  tag="sweep_" + tag_for(size, pooling),
                                  pooling=pooling, size=size)
        m = score(feats, ref_pos, query_pos, ref_small, val_df)
        results.append((label, size, pooling, m))
        print(f"  {label:30s} dim {feats.shape[1]:4d}  "
              f"median {m['median_err_m']:5.2f} m  R@1<1m {m['recall@1_1.0m']:.3f}  "
              f"room {m['room_acc']:.3f}   ({time.time() - t0:.0f}s)")

    best = max(results, key=lambda r: r[3]["recall@1_1.0m"])
    print(f"\n  Winner on validation: {best[0]}")

    # --- Phase 2 -----------------------------------------------------------
    _, size, pooling, _ = best
    print(f"\n  Phase 2: full extraction at {tag_for(size, pooling)}, "
          f"then night once.\n")
    feats_full = Fx.extract_cached(df, backbone="dinov2_vits14",
                                   tag=tag_for(size, pooling),
                                   pooling=pooling, size=size)
    base = Fx.extract_cached(df, backbone="dinov2_vits14")

    day_df, night_df = D.split_day_night(df)
    rows = {}
    print("  === TEST: night queries vs all day references ===")
    for name, f in (("frozen DINOv2 224sq (baseline)", base),
                    (f"frozen DINOv2 {tag_for(size, pooling)}", feats_full)):
        m = score(f, day_df["_row"].to_numpy(), night_df["_row"].to_numpy(),
                  day_df, night_df)
        rows[name] = m
        print(R.format_result(name, m))

    print("\n  Confidence gate (top-5 spread), best representation:")
    idx, _ = R.retrieve(feats_full[night_df["_row"].to_numpy()],
                        feats_full[day_df["_row"].to_numpy()], k=5)
    ref_xy = day_df[["x", "y"]].to_numpy()
    true_xy = night_df[["x", "y"]].to_numpy()
    err = np.hypot(*(ref_xy[idx[:, 0]] - true_xy).T)
    topk = ref_xy[idx]
    spread = np.hypot(*(topk - topk.mean(1, keepdims=True)).transpose(2, 0, 1)).mean(1)
    print(f"  {'gate':>13s} {'kept':>7s} {'median':>9s} {'R@1<1m':>8s} {'err>2m':>8s}")
    gate_rows = []
    for t in (0.25, 0.5, np.inf):
        k = spread <= t
        label = "none" if np.isinf(t) else f"spread<={t}"
        gate_rows.append((label, float(k.mean()), float(np.median(err[k])),
                          float(np.mean(err[k] <= 1)), float(np.mean(err[k] > 2))))
        print(f"  {label:>13s} {k.mean():6.1%} {np.median(err[k]):8.2f}m "
              f"{np.mean(err[k] <= 1):8.3f} {np.mean(err[k] > 2):7.1%}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = [r[0] for r in results]
    axes[0].barh(labels, [r[3]["recall@1_1.0m"] for r in results], color="tab:blue")
    axes[0].set_xlabel("R@1 within 1 m (validation)")
    axes[0].set_title("Phase 1: representation sweep")
    axes[0].grid(alpha=0.3, axis="x")
    for name, m in rows.items():
        e = np.sort(m["_err"])
        axes[1].plot(e, np.arange(1, len(e) + 1) / len(e),
                     label=f"{name} ({m['median_err_m']:.2f} m)")
    axes[1].set_xlim(0, 8)
    axes[1].set_xlabel("localisation error (m)")
    axes[1].set_ylabel("fraction of night queries")
    axes[1].set_title("Phase 2 TEST: day -> night")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "representation.png", dpi=110)
    plt.close(fig)

    (OUT / "representation.json").write_text(json.dumps({
        "sweep": [{"config": r[0], **{k: v for k, v in r[3].items()
                                      if not k.startswith("_")}} for r in results],
        "winner": best[0],
        "test": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                 for k, v in rows.items()},
        "gate": gate_rows,
    }, indent=2))
    print(f"\n  Wrote representation.png/.json to {OUT}")


if __name__ == "__main__":
    main()
