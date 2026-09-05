#!/usr/bin/env python3
"""Remember where the rover was, so it need not always start in the work room.

seed_pose.py asserts a fixed spot at boot, which is right only if the rover
is parked there. Leave it in the kitchen, or lose power mid-errand, and the
next start is seeded metres from the truth. This records the live pose so the
seed can pick up where the rover actually stopped.

Sampled, not subscribed
-----------------------
/tracked_pose runs at ~192 Hz and a resident subscriber costs about 40% of a
core here -- measured as the difference between track_pose on and off in
rover_nav.py, on a Pi that is already the constraint. So the subscription is
created, used for one message, and destroyed again, every 30 seconds. A pose
half a minute stale is fine for a rover that moves at 0.5 m/s and is usually
parked.

What is worth saving
--------------------
Only a pose standing on free space in the map. Cartographer's estimate has
been seen to leave the map entirely and sit there confidently, and persisting
one of those would seed the next boot somewhere the rover has never been --
worse than the fixed work-room default, which is at least right whenever the
rover is parked properly.

That test is necessary, not sufficient: a wrong pose can land on free floor.
Hence the age limit in seed_pose.py, and hence showing the user what is about
to be assumed rather than assuming it silently.
"""

import json
import math
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

STATE_PATH = os.environ.get("ROVER_LAST_POSE",
                            "/var/lib/rover/last_pose.json")
SAMPLE_PERIOD_S = 30.0
SAMPLE_TIMEOUT_S = 3.0
FREE_MAX = 20          # occupancy 0-100; anything above this is not open floor
MAP_YAML = os.environ.get("ROVER_MAP_YAML", "/home/amurshid/house_map.yaml")
# Written by the mode page's "forget saved pose" button, beside the memory it
# clears. Without it the button does nothing: the next sample, 30s later,
# writes the same wrong pose straight back.
HOLD_PATH = os.environ.get("ROVER_POSE_HOLD",
                           os.path.join(os.path.dirname(STATE_PATH), "hold"))
# The verdict on every sample, good or bad, for the pages to show. /run is
# tmpfs, so this costs no card writes -- the same place the bridge puts its
# telemetry. Deciding whether to save already computes this; throwing the
# answer away whenever it was "no" is what left a diverged Cartographer
# invisible until a goal failed.
STATUS_PATH = os.environ.get("ROVER_POSE_STATUS",
                             "/run/rover/pose_status.json")


def _read_pgm(path):
    """(width, height, maxval, pixels) from a binary P5 PGM.

    Written out rather than pulled from a library because the only dependency
    that would do it is not on the Pi, and the format is four header tokens
    and a block of bytes.
    """
    data = Path(path).read_bytes()
    tokens, i = [], 0
    while len(tokens) < 4:
        while i < len(data) and data[i:i + 1].isspace():
            i += 1
        if data[i:i + 1] == b"#":                    # comments run to the EOL
            while i < len(data) and data[i] != 0x0A:
                i += 1
            continue
        j = i
        while j < len(data) and not data[j:j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    if tokens[0] != b"P5":
        raise ValueError(f"{path}: not a binary PGM ({tokens[0]!r})")
    w, h, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    i += 1                       # exactly one whitespace byte after maxval
    px = data[i:i + w * h]
    if len(px) != w * h:
        raise ValueError(f"{path}: wanted {w * h} pixels, got {len(px)}")
    return w, h, maxval, px


class MapGrid:
    """The occupancy map, read off disk instead of off /map.

    /map belongs to map_server, and map_server is part of Nav2 -- which does
    not run in remote control. Subscribing would tie this node to Nav2, and a
    Requires= on Nav2 would start the planner in the one mode whose whole
    point is that a human is driving. Reading the file map_server reads keeps
    this node working in both modes and dependent on neither.

    It also removes a race that was visible in the logs: /map is latched, but
    it still arrives after the subscription is made, so the first sample after
    boot always found no map and threw a good pose away.

    Thresholds follow nav2_map_server: shade above occupied_thresh is a wall,
    below free_thresh is floor, between is unknown.
    """

    def __init__(self, path):
        with open(path) as f:
            m = yaml.safe_load(f)
        image = Path(m["image"])
        if not image.is_absolute():
            image = Path(path).parent / image      # yaml-relative, as ROS does
        self.res = float(m["resolution"])
        self.ox, self.oy = float(m["origin"][0]), float(m["origin"][1])
        self.negate = int(m.get("negate", 0))
        self.occ_th = float(m.get("occupied_thresh", 0.65))
        self.free_th = float(m.get("free_thresh", 0.196))
        self.w, self.h, self.maxval, self.px = _read_pgm(image)

    def at(self, x, y):
        """Occupancy under a map-frame point: 0 free, 100 wall, -1 unknown.
        None when the point is off the grid entirely."""
        col = int(math.floor((x - self.ox) / self.res))
        row = int(math.floor((y - self.oy) / self.res))
        if not (0 <= col < self.w and 0 <= row < self.h):
            return None
        # PGM row 0 is the top of the image, which is the highest y.
        v = self.px[(self.h - 1 - row) * self.w + col]
        shade = v / self.maxval if self.negate else (self.maxval - v) / self.maxval
        if shade > self.occ_th:
            return 100
        if shade < self.free_th:
            return 0
        return -1


class PoseMemory(Node):
    def __init__(self):
        super().__init__("rover_pose_memory")
        self.grid = None
        self.latest = None
        self.saved = 0
        self.rejected = 0
        self.held = False
        self.status_warned = False
        # A hold from before this start has done its job. It means somebody
        # moved the rover by hand and pressed the button; the power cycle it
        # asks for is this start, and the seed has already run. Leaving it set
        # would mean never recording anything again.
        if os.path.exists(HOLD_PATH):
            try:
                os.remove(HOLD_PATH)
                self.get_logger().info(f"cleared the hold at {HOLD_PATH}")
            except OSError as e:
                self.get_logger().warn(f"could not clear {HOLD_PATH}: {e}")
        try:
            self.grid = MapGrid(MAP_YAML)
        except (OSError, ValueError, KeyError, TypeError) as e:
            # Fail closed. Without the map nothing can be checked, and saving
            # unchecked poses is the failure this node exists to avoid -- so
            # it saves nothing and says so loudly rather than degrading
            # quietly into the thing it was built to prevent.
            self.get_logger().error(
                f"no map from {MAP_YAML} ({e}) -- nothing will be saved")
        self.get_logger().info(
            f"remembering the pose to {STATE_PATH} every "
            f"{SAMPLE_PERIOD_S:.0f}s, when it stands on free space")

    def _on_pose(self, msg):
        self.latest = msg

    # ------------------------------------------------------------------ map

    def cell(self, x, y):
        """Occupancy under a point: -1 unknown, 0 free, 100 wall. None if the
        map failed to load or the point is off the grid."""
        return None if self.grid is None else self.grid.at(x, y)

    # --------------------------------------------------------------- sample

    def sample(self):
        """One pose, then let the subscription go. Returns the message or None."""
        self.latest = None
        sub = self.create_subscription(PoseStamped, "/tracked_pose",
                                       self._on_pose, 1)
        try:
            deadline = time.time() + SAMPLE_TIMEOUT_S
            while self.latest is None and time.time() < deadline and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)
        finally:
            self.destroy_subscription(sub)
        return self.latest

    def remember(self):
        """Sample, check it is somewhere real, and write it down."""
        if os.path.exists(HOLD_PATH):
            # Before sampling, not after: a held node should not even take a
            # pose. This is also what covers the final save on SIGTERM, which
            # comes through here -- otherwise pressing the button and powering
            # off would write back the very pose the button discarded.
            if not self.held:
                self.get_logger().info(
                    f"holding: {HOLD_PATH} exists, recording nothing until "
                    f"this node restarts")
                self.held = True
            return
        msg = self.sample()
        if msg is None:
            return
        x, y = msg.pose.position.x, msg.pose.position.y
        occ = self.cell(x, y)
        ok = occ is not None and 0 <= occ <= FREE_MAX
        self.report(x, y, occ, ok)      # every sample, whether it is saved or not
        if not ok:
            # Off the grid, in unknown space, or inside a wall. Cartographer
            # has been all three while the rover sat still; none of them are
            # worth waking up believing.
            self.rejected += 1
            if self.rejected in (1, 10, 100):
                where = "no map" if self.grid is None else f"cell {occ}"
                self.get_logger().warn(
                    f"not saving ({x:.2f}, {y:.2f}): {where}")
            return
        self.write(x, y, msg.pose.orientation.z, msg.pose.orientation.w, occ)

    def report(self, x, y, occ, ok):
        """Say where Cartographer thinks it is and whether that is anywhere
        real, so the pages can show it. Best effort: a rover whose status file
        cannot be written should still record poses, so failure warns once and
        is otherwise ignored."""
        payload = json.dumps({"t": time.time(), "x": round(x, 4),
                              "y": round(y, 4), "cell": occ, "ok": ok})
        try:
            d = os.path.dirname(STATUS_PATH)
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d)
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            os.replace(tmp, STATUS_PATH)
        except OSError as e:
            if not self.status_warned:
                self.get_logger().warn(f"cannot write {STATUS_PATH}: {e}")
                self.status_warned = True

    def write(self, x, y, qz, qw, occ):
        """Replace the file atomically -- a power cut must not leave half of
        it, because the next boot reads this before anything else runs."""
        payload = json.dumps({
            "t": time.time(), "x": round(x, 4), "y": round(y, 4),
            "qz": round(qz, 6), "qw": round(qw, 6), "cell": occ,
            "yaw_deg": round(math.degrees(2.0 * math.atan2(qz, qw)), 1),
        })
        try:
            os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(STATE_PATH))
            with os.fdopen(fd, "w") as f:
                f.write(payload)
            os.chmod(tmp, 0o644)
            os.replace(tmp, STATE_PATH)
        except OSError as e:
            if self.saved == 0:
                self.get_logger().warn(f"cannot write {STATE_PATH}: {e}")
            return
        self.saved += 1
        if self.saved == 1:
            self.get_logger().info(f"first pose remembered: {payload}")


def main():
    rclpy.init()
    node = PoseMemory()

    stopping = []

    def on_term(*_):
        # One last sample on the way out, so a clean stop records exactly
        # where it finished rather than up to 30s earlier.
        stopping.append(True)

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    try:
        while rclpy.ok() and not stopping:
            node.remember()
            end = time.time() + SAMPLE_PERIOD_S
            while time.time() < end and rclpy.ok() and not stopping:
                rclpy.spin_once(node, timeout_sec=0.2)   # keeps /map serviced
        if rclpy.ok():
            node.get_logger().info("stopping -- taking a final pose")
            node.remember()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
