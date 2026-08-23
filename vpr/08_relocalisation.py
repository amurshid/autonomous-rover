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
FINETUNED = OUT / "features" / "finetuned_tuned.npy"
# What the fine-tuned backbone saw. Leave-one-session-out asks each session to
# be a fresh traversal, which is only true for the sessions training never
# touched -- so the shipped configuration reports those separately rather than
# averaging a trained-on session into the headline.
FT_TRAINED = {"day2_afternoon_300", "day2_afternoon_400",
              "day2_evening_500", "day2_evening_600", "day1_night_session2"}
FT_VAL = {"day1_night_session1"}
GATE = 0.5          # top-5 spread, metres
MAX_FRAMES = 25     # give up and admit failure after this many frames
POS_OK = 1.0        # a fix good enough to seed Cartographer
YAW_OK = 30.0       # degrees


def evaluate(df, feats, postproc, tag, suffix, restrict=None):
    """One configuration, end to end.

    `postproc` switches on the two steps from 10/11: similarity-weighted
    aggregation of the top-5 poses, then the sequence filter. Both are free at
    inference and neither needs training, but they change the reported pose, so
    the frozen top-1 configuration is kept alongside as the historical record.
    """
    print(f"\n{'=' * 78}\n  {tag}\n{'=' * 78}")
    print("\n  Leave-one-session-out: query = one session, "
          "database = the other seven.\n")
    print(f"  {'query session':22s} {'n':>5s} {'median':>8s} {'R@1<1m':>8s} "
          f"{'yaw med':>9s} {'yaw<30':>8s} {'room':>6s}")

    per_session = {}
    all_err, all_yaw, all_spread = [], [], []
    held_err, held_yaw = [], []      # sessions this model never trained on

    sessions = list(restrict) if restrict else list(D.SESSIONS)
    for session in sessions:
        q = df[df.session == session].reset_index(drop=True)
        ref = df[df.session != session].reset_index(drop=True)

        idx, sim = R.retrieve(feats[q["_row"].to_numpy()],
                              feats[ref["_row"].to_numpy()], k=5)
        ref_xy = ref[["x", "y"]].to_numpy()
        true_xy = q[["x", "y"]].to_numpy()
        if postproc:
            pred, overridden = R.sequence_filter(R.aggregate(ref_xy, idx, sim))
        else:
            pred, overridden = ref_xy[idx[:, 0]], np.zeros(len(idx), bool)
        err = np.hypot(*(pred - true_xy).T)
        # Yaw always comes from the top-1 frame: averaging the neighbours'
        # positions is sound, averaging their headings is not -- two frames at
        # the same spot facing opposite ways would average to a heading the
        # rover never had.
        yaw = T.yaw_diff(ref["yaw"].to_numpy()[idx[:, 0]], q["yaw"].to_numpy())
        spread = R.top_k_spread(ref_xy, idx)
        # An overridden frame carries a heading from the match the filter just
        # rejected, so the gate must refuse it.
        spread = np.where(overridden, np.inf, spread)
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
        if not postproc or session not in (FT_TRAINED | FT_VAL):
            held_err.append(err); held_yaw.append(yaw)

    err = np.concatenate(all_err)
    yaw = np.concatenate(all_yaw)
    spread = np.concatenate(all_spread)
    print(f"\n  {'ALL':22s} {len(err):5d} {np.median(err):7.2f}m "
          f"{(err <= POS_OK).mean():8.3f} {np.median(yaw):8.1f}d "
          f"{(yaw <= YAW_OK).mean():8.3f}")

    if postproc:
        he, hy = np.concatenate(held_err), np.concatenate(held_yaw)
        seen = sorted(FT_TRAINED | FT_VAL)
        print(f"\n  Restricted to the {len(held_err)} sessions this model never "
              f"trained on ({len(he)} frames):")
        print(f"    median {np.median(he):.2f} m   mean {he.mean():.2f} m   "
              f"within 1 m {(he <= 1).mean():.1%}   yaw median {np.median(hy):.1f} deg")
        print(f"    (the other rows above are optimistic: {', '.join(seen)} "
              f"were in training or validation)")

    good = (err <= POS_OK) & (yaw <= YAW_OK)
    print(f"\n  Frames giving a fix good enough to seed Cartographer "
          f"(<{POS_OK} m and <{YAW_OK} deg): {good.mean():.1%}")
    gated = spread <= GATE
    print(f"  Same, among frames the gate accepts: {good[gated].mean():.1%} "
          f"({gated.mean():.1%} of frames accepted)")

    # --- cold start ---------------------------------------------------------
    # The rover boots somewhere and drives. It does not need every frame; it
    # needs the first frame the gate accepts to be right.
    print("\n  Cold start: boot at an arbitrary frame, drive forward, take "
          "the first fix the gate accepts.")

    rng = np.random.default_rng(0)
    offset = 0
    frames_needed, fix_err, fix_yaw, failures = [], [], [], 0
    for session in sessions:
        n = int((df.session == session).sum())
        s_err = err[offset:offset + n]
        s_yaw = yaw[offset:offset + n]
        s_spread = spread[offset:offset + n]
        offset += n
        # A boot simulated on a session the model trained on is not a cold
        # start, it is a memory test.
        if postproc and session in (FT_TRAINED | FT_VAL):
            continue

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
    print("\n  With a second accepted frame required to agree within 1 m of "
          "the first:")
    conf_err, conf_frames = [], []
    offset = 0
    for session in sessions:
        n = int((df.session == session).sum())
        s_err = err[offset:offset + n]
        s_yaw = yaw[offset:offset + n]
        s_spread = spread[offset:offset + n]
        offset += n
        if postproc and session in (FT_TRAINED | FT_VAL):
            continue
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
    fig.savefig(OUT / f"relocalisation{suffix}.png", dpi=110)
    plt.close(fig)

    summary = {
        "per_session": per_session,
        "held_out_only": ({
            "n": int(len(np.concatenate(held_err))),
            "median_err_m": float(np.median(np.concatenate(held_err))),
            "mean_err_m": float(np.concatenate(held_err).mean()),
            "within_1m": float((np.concatenate(held_err) <= 1).mean()),
        } if postproc else None),
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
    }
    (OUT / f"relocalisation{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Wrote relocalisation{suffix}.png/.json to {OUT}")
    return summary


def main():
    df = D.label_rooms(D.load_all(verify=False))
    df["_row"] = np.arange(len(df))

    # The frozen top-1 configuration is the historical baseline; the fine-tuned
    # backbone with post-processing is what would actually ship. Both are
    # reported so the gain is attributable rather than asserted.
    evaluate(df, Fx.extract_cached(df, backbone="dinov2_vits14"), False,
             "frozen DINOv2, top-1 pose (the original measurement)", "")
    if FINETUNED.exists():
        held = [s for s in D.SESSIONS if s not in (FT_TRAINED | FT_VAL)]
        evaluate(df, np.load(FINETUNED), True,
                 "fine-tuned backbone + aggregation + sequence filter "
                 "(the shipped configuration)", "_shipped")
        # The two rows above cannot be compared directly: the shipped model
        # trained on six of the eight sessions. This pair is the fair one --
        # same sessions, same code path, only the model and post-processing
        # differ. They are the hardest two in the set (the near-sunset lap and
        # a night lap), so both numbers are below their all-session versions.
        evaluate(df, Fx.extract_cached(df, backbone="dinov2_vits14"), False,
                 f"frozen DINOv2, top-1, restricted to {', '.join(held)}",
                 "_baseline_heldout", restrict=held)
        evaluate(df, np.load(FINETUNED), True,
                 f"shipped configuration, restricted to {', '.join(held)}",
                 "_shipped_heldout", restrict=held)
    else:
        print(f"\n  {FINETUNED.name} not found -- run 10_finetune.py to "
              f"produce it; skipping the shipped configuration.")


if __name__ == "__main__":
    main()
