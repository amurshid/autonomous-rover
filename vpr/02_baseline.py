#!/usr/bin/env python3
"""Step 5: frozen backbone + kNN retrieval baseline.

Two evaluations, both from the same pipeline:
  day -> night   the headline cross-domain result
  4 day sessions -> held-out day session   within-condition sanity check

Usage:  python 02_baseline.py [--backbone resnet50|dinov2_vits14]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def run_split(name, ref_df, query_df, feats, k=10):
    ref_f = feats[ref_df["_row"].to_numpy()]
    q_f = feats[query_df["_row"].to_numpy()]
    idx, sim = R.retrieve(q_f, ref_f, k=k)
    m = R.evaluate(query_df, ref_df, idx)
    m["_sim"] = sim[:, 0]
    print(R.format_result(name, m))
    return m


def plot_errors(results, backbone):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, m in results.items():
        err = np.sort(m["_err"])
        cdf = np.arange(1, len(err) + 1) / len(err)
        axes[0].plot(err, cdf, label=f"{name} (median {m['median_err_m']:.2f} m)")
    axes[0].set_xlim(0, 8)
    axes[0].set_xlabel("localisation error (m)")
    axes[0].set_ylabel("fraction of queries")
    axes[0].set_title(f"{backbone}: error CDF")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    name, m = next(iter(results.items()))
    axes[1].scatter(m["_sim"], m["_err"], s=3, alpha=0.2)
    axes[1].set_xlabel("top-1 cosine similarity")
    axes[1].set_ylabel("localisation error (m)")
    axes[1].set_title(f"{name}: is similarity a usable confidence signal?")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / f"baseline_{backbone}.png", dpi=110)
    plt.close(fig)


def plot_map_errors(query_df, m, backbone):
    fig, ax = plt.subplots(figsize=(9, 9))
    sc = ax.scatter(query_df.x, query_df.y, c=np.clip(m["_err"], 0, 5),
                    s=6, cmap="turbo")
    fig.colorbar(sc, ax=ax, label="localisation error (m, clipped at 5)")
    for nm, (cx, cy) in D.room_centres().items():
        ax.plot(cx, cy, "k+", ms=10)
        ax.annotate(nm, (cx, cy), fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(f"{backbone}: where day->night retrieval fails")
    fig.tight_layout()
    fig.savefig(OUT / f"error_map_{backbone}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    df = D.load_all(verify=False)
    df = D.label_rooms(df)
    df["_row"] = np.arange(len(df))

    t0 = time.time()
    feats = Fx.extract_cached(df, backbone=args.backbone,
                              batch_size=args.batch_size, workers=args.workers)
    print(f"  features {feats.shape}  ({time.time() - t0:.0f}s)\n")

    results = {}

    day, night = D.split_day_night(df)
    results["day -> night (headline)"] = run_split(
        "day -> night (headline)", day, night, feats)

    tr, te = D.split_holdout_session(df, "day2_evening_600")
    results["day -> held-out day session"] = run_split(
        "day -> held-out day session", tr, te, feats)

    # Night as reference, day as query: the same domain gap, other direction.
    results["night -> day"] = run_split("night -> day", night, day, feats)

    # Floor: how well would a single night session localise against the other
    # two night sessions? Same lighting, different traversal.
    n1 = df[df.session == "day1_night_session3"].reset_index(drop=True)
    n_rest = df[(df.condition == "night") &
                (df.session != "day1_night_session3")].reset_index(drop=True)
    results["night -> night (same condition)"] = run_split(
        "night -> night (same condition)", n_rest, n1, feats)

    plot_errors(results, args.backbone)
    plot_map_errors(night, results["day -> night (headline)"], args.backbone)

    summary = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
               for k, v in results.items()}
    (OUT / f"baseline_{args.backbone}.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\n  Wrote baseline_{args.backbone}.png, error_map_{args.backbone}.png, "
          f"baseline_{args.backbone}.json to {OUT}")


if __name__ == "__main__":
    main()
