#!/usr/bin/env python3
"""First look at the data: load, verify, plot trajectories, check BGR/RGB.

Writes plots to vpr/out/. Run before any modelling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vprlib import data as D  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def missing_images(df):
    missing = [p for p in df["path"] if not Path(p).exists()]
    orphans = 0
    for session in D.SESSIONS:
        listed = {Path(p).name for p in df.loc[df.session == session, "path"]}
        on_disk = {p.name for p in (D.DATA / session / "images").glob("*.jpg")}
        orphans += len(on_disk - listed)
    return missing, orphans


def plot_trajectories(df):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    centres = D.room_centres()
    for ax, session in zip(axes.ravel(), D.SESSIONS):
        s = df[df.session == session]
        ax.plot(s.x, s.y, "-", lw=0.6, color="0.6", zorder=1)
        ax.scatter(s.x, s.y, c=s.stamp - s.stamp.min(), s=3, cmap="viridis", zorder=2)
        jumps = s[s.jump == 1]
        ax.scatter(jumps.x, jumps.y, c="red", s=14, marker="x", zorder=3,
                   label=f"jump ({len(jumps)})")
        for name, (cx, cy) in centres.items():
            ax.plot(cx, cy, "k+", ms=8, zorder=4)
            ax.annotate(name, (cx, cy), fontsize=6, alpha=0.7)
        ax.set_title(f"{session}  n={len(s)}", fontsize=9)
        ax.set_aspect("equal")
        ax.legend(fontsize=6, loc="upper right")
    fig.suptitle("Per-session trajectories (colour = time within session)")
    fig.tight_layout()
    fig.savefig(OUT / "trajectories.png", dpi=110)
    plt.close(fig)


def plot_overlay(df):
    fig, ax = plt.subplots(figsize=(9, 9))
    for cond, colour in (("day", "tab:orange"), ("night", "tab:blue")):
        s = df[df.condition == cond]
        ax.scatter(s.x, s.y, s=2, alpha=0.25, c=colour, label=f"{cond} ({len(s)})")
    for name, (cx, cy) in D.room_centres().items():
        ax.plot(cx, cy, "k+", ms=10)
        ax.annotate(name, (cx, cy), fontsize=8)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Coverage: day vs night")
    fig.tight_layout()
    fig.savefig(OUT / "coverage_day_night.png", dpi=110)
    plt.close(fig)


def check_channels(df):
    """Mean per-channel intensity, plus a saved contact sheet to eyeball.

    Indoor scenes with skin/wood tones should have R > B on average. If the
    saved JPEGs are actually BGR, that inequality flips.
    """
    import cv2

    rows = []
    for session in D.SESSIONS:
        s = df[df.session == session].sample(min(60, (df.session == session).sum()),
                                             random_state=0)
        means = []
        for p in s["path"]:
            im = cv2.imread(p)              # BGR as stored on disk
            if im is None:
                continue
            means.append(im.reshape(-1, 3).mean(0))
        b, g, r = np.mean(means, axis=0)
        rows.append((session, r, g, b))
    print("\n  Mean channel intensity, interpreting file bytes as RGB:")
    print(f"  {'session':22s} {'R':>7s} {'G':>7s} {'B':>7s}   R-B")
    for session, r, g, b in rows:
        print(f"  {session:22s} {r:7.1f} {g:7.1f} {b:7.1f}   {r - b:+6.1f}")

    # Contact sheet, both interpretations side by side.
    picks = df.groupby("session").head(1)["path"].tolist()
    fig, axes = plt.subplots(2, len(picks), figsize=(2.2 * len(picks), 4.6))
    for i, p in enumerate(picks):
        bgr = cv2.imread(p)
        axes[0, i].imshow(bgr[:, :, ::-1])   # standard: file is RGB
        axes[1, i].imshow(bgr)               # swapped
        for r_ in (0, 1):
            axes[r_, i].axis("off")
        axes[0, i].set_title(Path(p).parents[1].name, fontsize=6)
    axes[0, 0].set_ylabel("as-is")
    axes[1, 0].set_ylabel("swapped")
    fig.suptitle("Top row: file bytes read as RGB.  Bottom row: channels swapped.")
    fig.tight_layout()
    fig.savefig(OUT / "channel_check.png", dpi=130)
    plt.close(fig)


def room_distribution(df):
    labelled = D.label_rooms(df)
    tab = labelled.pivot_table(index="room", columns="condition",
                               values="filename", aggfunc="count", fill_value=0)
    tab["total"] = tab.sum(axis=1)
    print("\n  Frames per nearest-room-centre label:")
    print(tab.sort_values("total", ascending=False).to_string())

    print("\n  Distance to nearest room centre (m):")
    q = labelled["room_dist"].quantile([0.5, 0.75, 0.9, 0.95, 0.99])
    for k, v in q.items():
        print(f"    p{int(k * 100):<3d} {v:5.2f}")
    far = (labelled["room_dist"] > 2.0).mean()
    print(f"    fraction >2.0 m from any centre: {far:.1%}"
          "   <-- these are the corridor frames")

    fig, ax = plt.subplots(figsize=(8, 4))
    labelled["room_dist"].hist(bins=60, ax=ax)
    ax.set_xlabel("distance to nearest room centre (m)")
    ax.set_ylabel("frames")
    fig.tight_layout()
    fig.savefig(OUT / "room_dist_hist.png", dpi=110)
    plt.close(fig)
    return labelled


def main():
    print("Loading all sessions...")
    df = D.load_all()

    missing, orphans = missing_images(df)
    print(f"\n  CSV rows with no image on disk: {len(missing)}")
    print(f"  Images on disk not listed in any CSV (orphans, ignored): {orphans}")

    gaps = df.groupby("session")["stamp"].apply(lambda s: s.diff().max())
    print("\n  Largest inter-frame time gap per session (s):")
    for session, g in gaps.items():
        print(f"    {session:22s} {g:7.1f}")

    print(f"\n  jump=1 rows: {int(df.jump.sum())}")

    plot_trajectories(df)
    plot_overlay(df)
    print(f"\n  Wrote trajectories.png, coverage_day_night.png to {OUT}")

    check_channels(df)
    room_distribution(df)
    print(f"\n  Wrote channel_check.png, room_dist_hist.png to {OUT}")


if __name__ == "__main__":
    main()
