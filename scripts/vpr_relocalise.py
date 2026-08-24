#!/usr/bin/env python3
"""Camera-based relocalisation for the Wave Rover.

Replaces the hand-set RViz pose in start_localization.sh: embed one camera
frame, retrieve the nearest reference frame from the recorded database, and
publish its pose to /initialpose, which set_initial_pose.py already bridges to
Cartographer's finish/start-trajectory restart.

    shadow mode (default)   logs what it would have published, publishes
                            nothing, and records Cartographer's pose alongside
                            for comparison. Zero risk. Run this first.
    --publish               actually publishes the first confirmed fix, then
                            stops. Cold start needs one fix, not a stream, and
                            the Pi throttles at 80 C with no heatsink.

Needs the bundle from vpr/deploy/export.py and nothing from the network:
the encoder is TorchScript, so torch.hub is never called.

    python3 vpr_relocalise.py --bundle ~/vpr_deploy --self-test
    python3 vpr_relocalise.py --bundle ~/vpr_deploy --bench 20
    python3 vpr_relocalise.py --bundle ~/vpr_deploy
    python3 vpr_relocalise.py --bundle ~/vpr_deploy --publish

--self-test and --bench import no ROS and need no camera, so a fresh Pi can be
checked for a loadable model and acceptable latency before ROS 2 is sourced --
which is the order the two failures want to be found in.

Relocaliser below has no ROS dependency on purpose -- it is exercised against
the recorded dataset by vpr/deploy/replay.py, so the logic is verified before it
ever runs on the rover.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def golden_images(g) -> list[bytes]:
    """The golden JPEGs, stored as one buffer plus offsets so that loading them
    never needs pickle -- see export.py for why that matters on the Pi."""
    blob, off = g["images_blob"], g["images_offsets"]
    return [blob[off[i]:off[i + 1]].tobytes() for i in range(len(off) - 1)]


class Relocaliser:
    """Frame in, pose out. No ROS, no camera, no network."""

    def __init__(self, bundle: Path, gate: float | None = None,
                 seq_tau: float | None = None, sim_floor: float = 0.0):
        bundle = Path(bundle)
        self.manifest = json.loads((bundle / "manifest.json").read_text())
        pre = self.manifest["preprocessing"]
        self.size = pre["size"]
        self.mean = torch.tensor(pre["mean"]).view(3, 1, 1)
        self.std = torch.tensor(pre["std"]).view(3, 1, 1)

        r = self.manifest["retrieval"]
        self.k = r["k"]
        self.temp = r["aggregation_temp"]
        self.seq_n = r["sequence_n"]
        self.seq_tau = r["sequence_base_m"] + r["frame_spacing_m"] * r["sequence_n"]
        self.gate = r["gate_spread_m"] if gate is None else gate
        # Spread asks whether the top-5 agree with each other; similarity asks
        # whether any of them actually looks like the query. They come apart
        # exactly where the database has no coverage: the neighbours cluster
        # tightly at the edge of the recorded envelope, so spread is small and
        # the gate passes a pose that is simply the nearest place anyone drove.
        # Retrieval can only ever return a weighted mean of recorded poses, so
        # off-map queries cannot be answered -- only detected, and low
        # similarity is what detects them. Measured on 2026-08-24: a 0.75 floor
        # took day2_evening_700 from 1.83% of accepted fixes beyond 2 m (max
        # 4.39 m) to 0.00% (max 1.86 m), costing 12% of accepted frames.
        self.sim_floor = sim_floor
        if seq_tau is not None:
            # The manifest value assumes successive samples are one database
            # frame apart (0.2 m). The live node samples on a timer instead, so
            # a driving rover covers far more than that between frames and the
            # filter reads real motion as a bad match.
            self.seq_tau = seq_tau

        self.net = torch.jit.load(bundle / "vpr_encoder_traced.pt")
        self.net.eval()
        torch.set_grad_enabled(False)

        db = np.load(bundle / "database.npz", allow_pickle=False)
        # float16 on disk to halve the file; float32 in RAM because the search
        # is one matmul and float16 matmul on ARM CPU is not faster.
        self.desc = db["descriptors"].astype(np.float32)
        self.xy = db["xy"]
        self.yaw = db["yaw"]
        self.room = db["room"]
        self.bundle = bundle
        self.history: list[np.ndarray] = []

    # --- the pipeline -----------------------------------------------------

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        """img: HxWx3 uint8 RGB. Must match vpr/deploy/export.py exactly;
        self_test() is what proves it does."""
        im = Image.fromarray(img).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0
        return (x - self.mean) / self.std

    def embed(self, img: np.ndarray) -> np.ndarray:
        return self.net(self.preprocess(img)[None]).numpy()[0]

    def locate(self, img: np.ndarray) -> dict:
        """Full estimate for one frame, including whether to trust it."""
        t0 = time.time()
        q = self.embed(img)
        sim = self.desc @ q
        top = np.argpartition(-sim, self.k)[:self.k]
        top = top[np.argsort(-sim[top])]
        s = sim[top]
        nb = self.xy[top]

        # Similarity-weighted mean of the neighbours: the true position is
        # usually between recorded frames, because the logger only saved one
        # every 0.2 m.
        w = np.exp((s - s[0]) / self.temp)
        w /= w.sum()
        raw = (nb * w[:, None]).sum(0)
        pos = raw

        # Spread: do the neighbours agree with each other? This is the gate.
        spread = float(np.hypot(*(nb - nb.mean(0)).T).mean())

        # Sequence filter: the rover cannot have moved further than ~0.2 m per
        # frame, so a large jump from recent history is a bad match, not motion.
        overridden = False
        if len(self.history) >= 2:
            m = np.median(np.array(self.history[-self.seq_n:]), axis=0)
            if math.hypot(*(pos - m)) > self.seq_tau:
                pos, overridden = m, True
        # History always holds the unfiltered estimate, never the median it was
        # replaced with, so one override cannot drag the filter along with it.
        self.history.append(raw)

        return {
            "x": float(pos[0]), "y": float(pos[1]),
            # What retrieval actually said, before the sequence filter had a
            # say. x/y are what the node would act on; raw_x/raw_y are what the
            # model is judged on, and they must be logged separately or an
            # override silently overwrites the measurement being collected.
            "raw_x": float(raw[0]), "raw_y": float(raw[1]),
            # Yaw comes from the top-1 frame, never averaged: two frames at the
            # same spot facing opposite ways would average to a heading the
            # rover never had.
            "yaw": float(self.yaw[top[0]]),
            "room": str(self.room[top[0]]),
            "similarity": float(s[0]),
            "spread": spread,
            "overridden": overridden,
            # An overridden frame's yaw came from the match just rejected, so
            # the pose is internally inconsistent and must not be trusted.
            "accept": bool(spread <= self.gate and not overridden
                           and s[0] >= self.sim_floor),
            "latency_s": time.time() - t0,
        }

    def reset(self):
        self.history.clear()

    # --- proving the pipeline matches the machine that built the database --

    def self_test(self, tol: float | None = None) -> tuple[bool, float]:
        """Re-embed frames whose descriptors are known and compare.

        Catches every silent train/serve mismatch at once -- channel order,
        resize filter, normalisation, a half-copied model. If this passes, the
        Pi is computing the same descriptors the database was built from; if it
        fails, nothing downstream is worth looking at.
        """
        tol = tol or self.manifest["golden"]["tolerance"]
        # allow_pickle=False on purpose: the Pi runs the numpy 1.x that ROS
        # Humble's rclpy needs, and it cannot read a pickle written by the
        # numpy 2.x on the machine that builds the bundle.
        g = np.load(self.bundle / "golden.npz", allow_pickle=False)
        if "images_blob" not in g:
            raise SystemExit(
                "golden.npz is in the old pickled format; rebuild the bundle "
                "with vpr/deploy/export.py and copy golden.npz across again.")
        got = np.stack([self.embed(np.array(Image.open(io.BytesIO(b)).convert("RGB")))
                        for b in golden_images(g)])
        cos = (got * g["descriptors"]).sum(1)
        return bool(cos.min() >= tol), float(cos.min())


# --- off-robot entry points -----------------------------------------------
# Everything here runs on torch, numpy and PIL alone. Kept ahead of build_node
# so a Pi with no ROS environment can still answer the two questions that
# decide whether the deployment is viable at all: does the model load, and how
# slow is one frame.

def run_self_test(args) -> int:
    loc = Relocaliser(args.bundle, gate=args.gate, seq_tau=args.seq_tau,
                      sim_floor=args.sim_floor)
    ok, worst = loc.self_test()
    print(f"self-test {'PASSED' if ok else 'FAILED'} (worst cosine {worst:.6f}, "
          f"tolerance {loc.manifest['golden']['tolerance']})")
    if not ok:
        print("This Pi does not reproduce the descriptors the database was "
              "built from. Check torch version, channel order, resize filter "
              "and normalisation before trusting anything downstream.")
    return 0 if ok else 1


def run_bench(args) -> int:
    """Time locate() on the golden frames -- the same work the node does per
    frame, embedding plus the retrieval matmul."""
    loc = Relocaliser(args.bundle, gate=args.gate, seq_tau=args.seq_tau,
                      sim_floor=args.sim_floor)
    g = np.load(Path(args.bundle) / "golden.npz", allow_pickle=False)
    frames = [np.array(Image.open(io.BytesIO(b)).convert("RGB"))
              for b in golden_images(g)]

    # The first inference pays for lazy TorchScript setup and is not
    # representative of the steady state, so it is reported, not averaged in.
    loc.reset()
    first = loc.locate(frames[0])["latency_s"]

    lat = []
    for i in range(args.bench):
        loc.reset()
        lat.append(loc.locate(frames[i % len(frames)])["latency_s"])
    lat = np.array(lat)

    print(f"{len(loc.desc)} reference frames, {loc.size}x{loc.size} input, "
          f"torch {torch.__version__}, {torch.get_num_threads()} threads")
    print(f"first call  {first:.3f} s  (includes one-off setup)")
    print(f"median      {np.median(lat):.3f} s")
    print(f"p95         {np.percentile(lat, 95):.3f} s   over {len(lat)} frames")
    print(f"worst       {lat.max():.3f} s")
    # The promotion criterion in deploy_handoff.md 4.4.
    print("p95 under 3 s: " + ("yes" if np.percentile(lat, 95) < 3.0 else
                               "NO -- see deploy_handoff.md 4.3 for options"))
    return 0


# --- ROS 2 node -----------------------------------------------------------
# Imported lazily so Relocaliser can be tested off the robot.

def build_node(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
    from sensor_msgs.msg import Image as ImageMsg

    def cpu_temp():
        try:
            p = Path("/sys/class/thermal/thermal_zone0/temp")
            return int(p.read_text()) / 1000.0
        except Exception:
            return float("nan")

    class VprRelocalise(Node):
        def __init__(self):
            super().__init__("vpr_relocalise")
            self.loc = Relocaliser(args.bundle, gate=args.gate,
                                   seq_tau=args.seq_tau,
                                   sim_floor=args.sim_floor)
            ok, worst = self.loc.self_test()
            self.get_logger().info(
                f"self-test {'PASSED' if ok else 'FAILED'} (worst cosine {worst:.6f})")
            if not ok:
                raise SystemExit(
                    "self-test failed: this Pi does not reproduce the "
                    "descriptors the database was built from. Check channel "
                    "order, resize filter and normalisation before trusting "
                    "anything downstream.")

            self.publish = args.publish
            self.period = args.period
            self.settle = args.settle
            self.timeout = args.timeout
            self.started = time.time()
            self.last = 0.0
            self.cart = None
            self.accepted = []
            self.done = False

            self.log = None
            if args.log:
                self.fh = open(args.log, "a", newline="")
                self.log = csv.writer(self.fh)
                if self.fh.tell() == 0:
                    self.log.writerow([
                        "stamp", "x", "y", "yaw", "room", "similarity",
                        "spread", "overridden", "accept", "latency_s",
                        "cart_x", "cart_y", "cart_yaw", "disagreement_m",
                        "cpu_temp_c", "raw_x", "raw_y", "raw_disagreement_m"])

            self.pub = self.create_publisher(
                PoseWithCovarianceStamped, "/initialpose", 10)
            self.create_subscription(
                PoseStamped, "/tracked_pose", self.on_pose, 1)
            self.create_subscription(
                ImageMsg, "/camera/image_raw", self.on_image,
                qos_profile_sensor_data)
            mode = "PUBLISH" if self.publish else "SHADOW (publishing nothing)"
            self.get_logger().info(
                f"{mode}, gate {self.loc.gate} m, one frame every "
                f"{self.period:.1f} s, {len(self.loc.desc)} reference frames")

        def on_pose(self, msg):
            q = msg.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y ** 2 + q.z ** 2))
            self.cart = (msg.pose.position.x, msg.pose.position.y, yaw)

        def on_image(self, msg):
            now = time.time()
            # A gate that can decline every frame -- which is the point of
            # --sim-floor -- can also decline all of them, and this node sits
            # in the boot path. Give up rather than hang start_localization.sh
            # forever; a non-zero exit lets the caller fall back to the
            # hardcoded pose.
            if self.timeout and now - self.started > self.timeout:
                self.get_logger().error(
                    f"no confirmed fix in {self.timeout:.0f}s "
                    f"({len(self.accepted)} accepted, none confirmed) -- "
                    f"giving up so the caller can fall back")
                raise SystemExit(2)
            # The camera's auto-exposure has not settled in the first second
            # after start-up, and cold start is exactly when those frames
            # arrive. The database contains no frames that look like that.
            if now - self.started < self.settle or now - self.last < self.period:
                return
            if self.done:
                return
            self.last = now

            if msg.encoding not in ("rgb8", "bgr8"):
                self.get_logger().warn(f"unexpected encoding {msg.encoding}")
                return
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            try:
                arr = arr.reshape(msg.height, msg.width, 3)
            except ValueError:
                return
            # Same flip the logger applied before saving the database.
            if msg.encoding == "bgr8":
                arr = arr[:, :, ::-1]

            r = self.loc.locate(np.ascontiguousarray(arr))
            dis = raw_dis = float("nan")
            if self.cart is not None:
                dis = math.hypot(r["x"] - self.cart[0], r["y"] - self.cart[1])
                raw_dis = math.hypot(r["raw_x"] - self.cart[0],
                                     r["raw_y"] - self.cart[1])

            self.get_logger().info(
                f"{'ACCEPT' if r['accept'] else 'reject'} "
                f"x={r['x']:.2f} y={r['y']:.2f} yaw={math.degrees(r['yaw']):.0f}d "
                f"{r['room']} spread={r['spread']:.2f} sim={r['similarity']:.3f} "
                f"{r['latency_s']:.2f}s"
                + (f" cartographer_delta={dis:.2f}m" if self.cart else ""))

            if self.log:
                c = self.cart or (float("nan"),) * 3
                self.log.writerow([
                    f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}",
                    f"{r['x']:.4f}", f"{r['y']:.4f}", f"{r['yaw']:.4f}",
                    r["room"], f"{r['similarity']:.4f}", f"{r['spread']:.4f}",
                    int(r["overridden"]), int(r["accept"]),
                    f"{r['latency_s']:.3f}", f"{c[0]:.4f}", f"{c[1]:.4f}",
                    f"{c[2]:.4f}", f"{dis:.4f}", f"{cpu_temp():.1f}",
                    f"{r['raw_x']:.4f}", f"{r['raw_y']:.4f}", f"{raw_dis:.4f}"])
                self.fh.flush()

            if not r["accept"]:
                return
            self.accepted.append(r)
            # One confirming frame: measured 94.3% within 1 m against 84.5% for
            # a single fix, and it costs about one second.
            if len(self.accepted) < 2:
                return
            a, b = self.accepted[-2], self.accepted[-1]
            if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) > 1.0:
                self.get_logger().info("two accepted fixes disagree, waiting")
                return

            if not self.publish:
                self.get_logger().info(
                    f"SHADOW: would publish x={b['x']:.2f} y={b['y']:.2f} "
                    f"yaw={math.degrees(b['yaw']):.0f} deg ({b['room']})")
                self.accepted.clear()
                return

            m = PoseWithCovarianceStamped()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.pose.pose.position.x = b["x"]
            m.pose.pose.position.y = b["y"]
            m.pose.pose.orientation.z = math.sin(b["yaw"] / 2.0)
            m.pose.pose.orientation.w = math.cos(b["yaw"] / 2.0)
            self.pub.publish(m)
            self.get_logger().info(
                f"PUBLISHED x={b['x']:.2f} y={b['y']:.2f} "
                f"yaw={math.degrees(b['yaw']):.0f} deg -- done")
            self.done = True

    return rclpy, VprRelocalise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True,
                    help="directory from vpr/deploy/export.py")
    ap.add_argument("--publish", action="store_true",
                    help="publish to /initialpose (default: shadow mode)")
    ap.add_argument("--gate", type=float, default=None,
                    help="top-5 spread threshold, metres (default from manifest)")
    ap.add_argument("--period", type=float, default=5.0,
                    help="seconds between inferences; keeps the Pi cool")
    ap.add_argument("--settle", type=float, default=2.0,
                    help="ignore frames for this long after start-up while the "
                         "camera's auto-exposure settles")
    ap.add_argument("--sim-floor", type=float, default=0.0,
                    help="reject a frame whose best match scores below this "
                         "cosine (default 0.0 = off). Detects the rover being "
                         "somewhere the database does not cover, which the "
                         "spread gate cannot see. 0.75 is the measured value; "
                         "prove it does not starve in shadow mode first")
    ap.add_argument("--seq-tau", type=float, default=None,
                    help="sequence-filter threshold in metres (default from "
                         "the manifest, which assumes 0.2 m between samples -- "
                         "raise it when sampling a moving rover on a timer)")
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="exit non-zero if no fix is confirmed within this "
                         "many seconds (default 0 = wait forever). Set it "
                         "whenever --publish runs in a boot script, so a "
                         "starving gate cannot hang the launch")
    ap.add_argument("--log", default=None, help="CSV to append to")
    ap.add_argument("--self-test", action="store_true",
                    help="verify this machine reproduces the database's "
                         "descriptors, then exit. Needs no ROS and no camera; "
                         "run it first on a new machine")
    ap.add_argument("--bench", type=int, nargs="?", const=20, default=0,
                    metavar="N",
                    help="time N frames through the full pipeline and exit "
                         "(default 20). Needs no ROS and no camera")
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(run_self_test(args))
    if args.bench:
        raise SystemExit(run_bench(args))

    rclpy, cls = build_node(args)
    rclpy.init()
    node = cls()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # rclpy's signal handler may already have shut the context down, and
        # calling it twice raises RCLError over an otherwise clean Ctrl-C.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
