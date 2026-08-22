#!/usr/bin/env python3
"""The actual product question: can this replace the manual pose in RViz?

Everything so far measured the *research* split -- train on day, test on night,
nothing shared. That split exists to prove the model generalises across a
lighting gap, and it is the right test for that claim.

It is the wrong test for the deployed system. In deployment you are not
withholding the night frames; you have them, so you put them in the reference
database alongside the day frames. A night query then has night references
available to match against. The cross-lighting gap only has to be crossed when
the database is missing the query's condition, which is a situation you would
simply not ship.

So this evaluates leave-one-session-out: each session in turn is the query,
every other session is the database. That is what the rover would actually see
on boot -- a place it has been before, under lighting it has seen before,
during a traversal it has not.

It also measures the two things the previous scripts skipped:

  yaw error   Cartographer's initial pose needs an orientation, not just a
              point. The retrieved frame carries one in the CSV, free.
  cold start  the rover does not need every frame localised. It needs *one*
              good fix. So: boot at an arbitrary point, take frames until the
              confidence gate fires, and report that first accepted fix.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vprlib import data as D          # noqa: E402
from vprlib import features as Fx     # noqa: E402
from vprlib import retrieval as R     # noqa: E402
from vprlib import train as T         # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
GATE = 0.5          # top-5 spread, metres
MAX_FRAMES = 25     # give up and admit failure after this many frames
POS_OK = 1.0        # a fix good enough to seed Cartographer
YAW_OK = 30.0       # degrees


def main():
    df = D.label_rooms(D.load_all(verify=False))
    df["_row"] = np.arange(len(df))
    feats = Fx.extract_cached(df, backbone="dinov2_vits14")

    print(f"\n  Leave-one-session-out: query = one session, "
          f"database = the other seven.\n")
    print(f"  {'query session':22s} {'n':>5s} {'median':>8s} {'R@1<1m':>8s} "
          f"{'yaw med':>9s} {'yaw<30':>8s} {'room':>6s}")

    per_session = {}
    all_err, all_yaw, all_spread = [], [], []

    for session in D.SESSIONS:
        q = df[df.session == session].reset_index(drop=True)
        ref = df[df.session != session].reset_index(drop=True)

        idx, _ = R.retrieve(feats[q["_row"].to_numpy()],
                            feats[ref["_row"].to_numpy()], k=5)
        ref_xy = ref[["x", "y"]].to_numpy()
        true_xy = q[["x", "y"]].to_numpy()
        err = np.hypot(*(ref_xy[idx[:, 0]] - true_xy).T)
        yaw = T.yaw_diff(ref["yaw"].to_numpy()[idx[:, 0]], q["yaw"].to_numpy())
        topk = ref_xy[idx]
        spread = np.hypot(*(topk - topk.mean(1, keepdims=True)
                            ).transpose(2, 0, 1)).mean(1)
        room = (ref["room"].to_numpy()[idx[:, 0]] == q["room"].to_numpy()).mean()

        per_session[session] = {
            "n": len(q),
            "median_err_m": float(np.median(err)),
            "recall@1_1m": float((err <= POS_OK).mean()),
            "median_yaw_deg": float(np.median(yaw)),
            "yaw_within_30": float((yaw <= YAW_OK).mean()),
            "room_acc": float(room),
        }
        print(f"  {session:22s} {len(q):5d} {np.median(err):7.2f}m "
              f"{(err <= POS_OK).mean():8.3f} {np.median(yaw):8.1f}d "
              f"{(yaw <= YAW_OK).mean():8.3f} {room:6.3f}")

        all_err.append(err); all_yaw.append(yaw); all_spread.append(spread)

    err = np.concatenate(all_err)
    yaw = np.concatenate(all_yaw)
    spread = np.concatenate(all_spread)
    print(f"\n  {'ALL':22s} {len(err):5d} {np.median(err):7.2f}m "
          f"{(err <= POS_OK).mean():8.3f} {np.median(yaw):8.1f}d "
          f"{(yaw <= YAW_OK).mean():8.3f}")

    good = (err <= POS_OK) & (yaw <= YAW_OK)
    print(f"\n  Frames giving a fix good enough to seed Cartographer "
          f"(<{POS_OK} m and <{YAW_OK} deg): {good.mean():.1%}")
    gated = spread <= GATE
    print(f"  Same, among frames the gate accepts: {good[gated].mean():.1%} "
          f"({gated.mean():.1%} of frames accepted)")

    # --- cold start ---------------------------------------------------------
    # The rover boots somewhere and drives. It does not need every frame; it
    # needs the first frame the gate accepts to be right.
    print(f"\n  Cold start: boot at an arbitrary frame, drive forward, take the "
          f"first fix the gate accepts.")

    rng = np.random.default_rng(0)
    offset = 0
    frames_needed, fix_err, fix_yaw, failures = [], [], [], 0
    for session in D.SESSIONS:
        n = int((df.session == session).sum())
        s_err = err[offset:offset + n]
        s_yaw = yaw[offset:offset + n]
        s_spread = spread[offset:offset + n]
        offset += n

        starts = rng.integers(0, max(n - MAX_FRAMES, 1), size=300)
        for s in starts:
            window = s_spread[s:s + MAX_FRAMES]
            hit = np.argmax(window <= GATE) if (window <= GATE).any() else None
            if hit is None:
                failures += 1
                continue
            frames_needed.append(int(hit))
            fix_err.append(float(s_err[s + hit]))
            fix_yaw.append(float(s_yaw[s + hit]))

    frames_needed = np.array(frames_needed)
    fix_err = np.array(fix_err)
    fix_yaw = np.array(fix_yaw)
    trials = len(frames_needed) + failures
    ok = (fix_err <= POS_OK) & (fix_yaw <= YAW_OK)

    print(f"    {trials} simulated boots")
    print(f"    gate never fired within {MAX_FRAMES} frames: "
          f"{failures / trials:.1%}")
    print(f"    frames until first accepted fix: median {np.median(frames_needed):.0f}, "
          f"p90 {np.percentile(frames_needed, 90):.0f}")
    print(f"    that first fix was good (<{POS_OK} m, <{YAW_OK} deg): {ok.mean():.1%}")
    print(f"    first fix position error: median {np.median(fix_err):.2f} m, "
          f"p90 {np.percentile(fix_err, 90):.2f} m")
    print(f"    first fix within 2 m: {(fix_err <= 2.0).mean():.1%}")

    # Requiring two consecutive accepted frames to agree is nearly free and
    # should cut the bad fixes further.
    print(f"\n  With a second accepted frame required to agree within 1 m of "
          f"the first:")
    conf_err, conf_frames = [], []
    offset = 0
    for session in D.SESSIONS:
        n = int((df.session == session).sum())
        s_err = err[offset:offset + n]
        s_yaw = yaw[offset:offset + n]
        s_spread = spread[offset:offset + n]
        s_idx = np.arange(offset, offset + n)
        offset += n
        starts = rng.integers(0, max(n - MAX_FRAMES, 1), size=300)
        for s in starts:
            acc = [j for j in range(s, min(s + MAX_FRAMES, n))
                   if s_spread[j] <= GATE]
            if len(acc) < 2:
                continue
            conf_err.append(float(s_err[acc[1]]))
            conf_frames.append(acc[1] - s)
    conf_err = np.array(conf_err)
    print(f"    second accepted fix good within {POS_OK} m: "
          f"{(conf_err <= POS_OK).mean():.1%}   within 2 m: "
          f"{(conf_err <= 2.0).mean():.1%}")
    print(f"    frames until second accepted fix: median "
          f"{np.median(conf_frames):.0f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].hist(err, bins=80, range=(0, 6))
    axes[0].axvline(POS_OK, color="r", ls="--")
    axes[0].set_xlabel("position error (m)"); axes[0].set_title(
        f"Leave-one-session-out (median {np.median(err):.2f} m)")
    axes[1].hist(yaw, bins=60, range=(0, 180))
    axes[1].axvline(YAW_OK, color="r", ls="--")
    axes[1].set_xlabel("yaw error (deg)"); axes[1].set_title(
        f"Yaw (median {np.median(yaw):.1f} deg)")
    axes[2].hist(fix_err, bins=60, range=(0, 6))
    axes[2].axvline(POS_OK, color="r", ls="--")
    axes[2].set_xlabel("error of first accepted fix (m)")
    axes[2].set_title(f"Cold start ({ok.mean():.0%} usable)")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "relocalisation.png", dpi=110)
    plt.close(fig)

    (OUT / "relocalisation.json").write_text(json.dumps({
        "per_session": per_session,
        "overall": {
            "median_err_m": float(np.median(err)),
            "recall@1_1m": float((err <= POS_OK).mean()),
            "median_yaw_deg": float(np.median(yaw)),
            "usable_fix_rate": float(good.mean()),
            "usable_fix_rate_gated": float(good[gated].mean()),
        },
        "cold_start": {
            "gate_never_fired": failures / trials,
            "median_frames_to_fix": float(np.median(frames_needed)),
            "first_fix_usable": float(ok.mean()),
            "first_fix_within_2m": float((fix_err <= 2.0).mean()),
            "second_fix_within_1m": float((conf_err <= POS_OK).mean()),
        },
    }, indent=2))
    print(f"\n  Wrote relocalisation.png/.json to {OUT}")


if __name__ == "__main__":
    main()
