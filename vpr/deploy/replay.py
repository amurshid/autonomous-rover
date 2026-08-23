#!/usr/bin/env python3
"""Replay a recorded session through the deployment code, off the robot.

`scripts/vpr_relocalise.py` is the code that will run on the Pi. Its retrieval
logic lives in a `Relocaliser` class with no ROS dependency precisely so it can
be driven from here, frame by frame, against sessions whose true poses are
known. If it reproduces the offline evaluation numbers, the deployment path is
correct and the only untested things left are the camera, the timing and the
hardware.

This is the last check that can be made without the rover.

    python deploy/replay.py                          # the untouched session
    python deploy/replay.py --session day2_evening_700 --limit 400

Note the asymmetry, which is deliberate: the database inside the bundle
contains every session, including the one being queried, so each frame can
retrieve itself and would score a perfect 0.00 m. That is the bug that made 09
briefly look flawless. Self-matches are therefore removed here by masking the
query session out of the database -- which is also what deployment looks like,
since the rover is on a traversal that is not in its own map.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

VPR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VPR))
sys.path.insert(0, str(VPR.parent / "scripts"))
from vprlib import data as D          # noqa: E402
from vprlib import train as T         # noqa: E402
from vpr_relocalise import Relocaliser  # noqa: E402

BUNDLE = VPR / "out" / "bundle"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="day1_night_session3",
                    help="session to replay as if it were live")
    ap.add_argument("--bundle", default=str(BUNDLE))
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N frames (0 = all)")
    args = ap.parse_args()

    loc = Relocaliser(args.bundle)
    ok, worst = loc.self_test()
    print(f"\n  self-test {'PASSED' if ok else 'FAILED'} "
          f"(worst cosine {worst:.6f})")
    if not ok:
        raise SystemExit("bundle does not reproduce its own descriptors")

    df = D.label_rooms(D.load_all(verify=False))
    q = df[df.session == args.session].reset_index(drop=True)
    if args.limit:
        q = q.iloc[:args.limit].reset_index(drop=True)

    # Leave-one-session-out, enforced inside the bundle's own arrays: without
    # this every frame retrieves itself and the result is a meaningless 0.00 m.
    db = np.load(Path(args.bundle) / "database.npz", allow_pickle=False)
    mask = db["session"] != args.session
    loc.desc = loc.desc[mask]
    loc.xy = loc.xy[mask]
    loc.yaw = loc.yaw[mask]
    loc.room = loc.room[mask]
    print(f"  replaying {len(q)} frames of {args.session} against "
          f"{mask.sum()} reference frames\n")

    rows, t0 = [], time.time()
    for i, r in q.iterrows():
        img = np.array(Image.open(r["path"]).convert("RGB"))
        out = loc.locate(img)
        out["err"] = float(np.hypot(out["x"] - r["x"], out["y"] - r["y"]))
        out["yaw_err"] = float(T.yaw_diff(np.array([out["yaw"]]),
                                          np.array([r["yaw"]]))[0])
        out["room_ok"] = out["room"] == r["room"]
        rows.append(out)
        if (i + 1) % 200 == 0:
            print(f"\r    {i + 1}/{len(q)}", end="", flush=True)
    print(f"\r    {len(q)}/{len(q)}  ({time.time() - t0:.0f}s)")

    err = np.array([r["err"] for r in rows])
    yaw = np.array([r["yaw_err"] for r in rows])
    acc = np.array([r["accept"] for r in rows])
    lat = np.array([r["latency_s"] for r in rows])
    ov = np.array([r["overridden"] for r in rows])
    room = np.array([r["room_ok"] for r in rows])
    good = (err <= 1.0) & (yaw <= 30.0)

    print(f"\n  position   median {np.median(err):.3f} m   mean {err.mean():.3f} m"
          f"   within 1 m {(err <= 1).mean():.1%}   within 2 m {(err <= 2).mean():.1%}")
    print(f"  yaw        median {np.median(yaw):.1f} deg   within 30 deg "
          f"{(yaw <= 30).mean():.1%}")
    print(f"  room       {room.mean():.1%}")
    print(f"  usable fix (<1 m, <30 deg)   {good.mean():.1%}")
    print(f"  sequence filter overrode     {ov.mean():.1%} of frames")
    print(f"  gate accepted                {acc.mean():.1%}")
    print(f"    among accepted: median {np.median(err[acc]):.3f} m, "
          f"usable {good[acc].mean():.1%}, >2 m {(err[acc] > 2).mean():.2%}")

    print(f"\n  latency    median {np.median(lat) * 1000:.0f} ms   "
          f"p95 {np.percentile(lat, 95) * 1000:.0f} ms   "
          f"(this machine, not the Pi)")

    # Cold start: how long until the first fix worth handing to Cartographer?
    first = np.argmax(acc) if acc.any() else None
    if first is not None:
        print(f"  first accepted frame: #{first}, error {err[first]:.2f} m")


if __name__ == "__main__":
    main()
