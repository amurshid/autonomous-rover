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
rover_nav.py, on a Pi that is already the constraint. So the subscriptions are
created, used for one scan and one pose, and destroyed again, every few
seconds -- about 1% of a core at the 4s period, measured against 0.16% at the
30s one this started with.

What is worth saving
--------------------
Only a pose standing on free space in the map. Cartographer's estimate has
been seen to leave the map entirely and sit there confidently, and persisting
one of those would seed the next boot somewhere the rover has never been --
worse than the fixed work-room default, which is at least right whenever the
rover is parked properly.

That test is necessary, not sufficient: a wrong pose can land on free floor.
On 2026-09-11 Cartographer lost itself in the living room, 1.6 m and 150 deg
out, on open floor. This saved it, the next start seeded from it, and the
rover drove on a pose its lidar contradicted while the page said "on map".
So a pose is also checked against the lidar: the scan, laid down at that
pose, has to land on the map's walls. Hence also the age limit in
seed_pose.py, and showing the user what is about to be assumed rather than
assuming it silently.
"""

import json
import math
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

STATE_PATH = os.environ.get("ROVER_LAST_POSE",
                            "/var/lib/rover/last_pose.json")
# Every 4s, not the 30s this started with. A save is refused whenever the scan
# disagrees, and at 30s that left the memory minutes and metres stale before a
# power cut -- a clean pose from the wrong end of a corridor. Sampling costs
# 0.16% of a core at 30s, so this is about 1%, and the last good pose is now
# seconds old.
SAMPLE_PERIOD_S = 4.0
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

# The re-timed scan Cartographer itself consumes. The LD19 is mounted over
# base_link with no rotation (ld19.launch.py: 0 0 0.18 0 0 0), so its angles
# lay straight onto the map at the rover's heading.
SCAN_TOPIC = os.environ.get("ROVER_SCAN_TOPIC", "/scan_fixed")
# How well the scan has to agree with the map before a pose is believed: the
# share of scan points landing within WALL_NEAR_CELLS of a wall. A correct
# pose has read 59-99% (parked, furniture and people included); Cartographer
# lost by 150 deg read 7%, and runs that had drifted off the map 5-36%. The
# best wrong heading at a lost spot reached 47%, so the line sits above it.
FIT_MIN = 0.50
# Best of this many scans. Pose and scan are a moment apart, and mid-spin that
# moment is enough to smear a correct pose into a low score; one of three
# scans a few hundred ms apart is still. Off-map poses are not retried: a
# pose inside a wall is not a timing artefact.
FIT_TRIES = 3
WALL_NEAR_CELLS = 2    # 10 cm at 5 cm cells: pose jitter and the scan's own sweep
RANGE_MIN_M, RANGE_MAX_M = 0.10, 8.0
MIN_POINTS = 50        # fewer and the lidar is blocked; that proves nothing


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
        self.near_wall = self._near_wall()

    def _near_wall(self):
        """Cells within WALL_NEAR_CELLS of a wall, indexed [row, col] with row
        0 at the lowest y -- the scan fit's lookup table, built once."""
        img = np.frombuffer(self.px, np.uint8).reshape(self.h, self.w)
        shade = img / self.maxval if self.negate else \
            (self.maxval - img.astype(float)) / self.maxval
        near = shade[::-1] > self.occ_th        # PGM row 0 is the highest y
        for _ in range(WALL_NEAR_CELLS):
            grown = near.copy()
            grown[1:] |= near[:-1]
            grown[:-1] |= near[1:]
            grown[:, 1:] |= near[:, :-1]
            grown[:, :-1] |= near[:, 1:]
            near = grown
        return near

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

    def fit(self, x, y, yaw, scan):
        """Share of the scan's points near a wall with the rover at (x, y, yaw),
        0-1, or None when too few returns to judge by."""
        r = np.asarray(scan.ranges, dtype=float)
        a = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        use = np.isfinite(r) & (r > RANGE_MIN_M) & (r < RANGE_MAX_M)
        if np.count_nonzero(use) < MIN_POINTS:
            return None
        r, a = r[use], a[use] + yaw
        col = np.floor((x + r * np.cos(a) - self.ox) / self.res).astype(int)
        row = np.floor((y + r * np.sin(a) - self.oy) / self.res).astype(int)
        inside = (col >= 0) & (col < self.w) & (row >= 0) & (row < self.h)
        hit = np.zeros(len(r), bool)
        hit[inside] = self.near_wall[row[inside], col[inside]]
        return float(hit.mean())


def _yaw(pose):
    q = pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PoseMemory(Node):
    def __init__(self):
        super().__init__("rover_pose_memory")
        self.grid = None
        self.latest = None
        self.scan = None
        self.saved = 0
        self.rejected = 0
        self.lost = False
        self.unchecked = 0
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
            f"{SAMPLE_PERIOD_S:.0f}s, when it stands on free space and "
            f"{FIT_MIN:.0%} of the scan on {SCAN_TOPIC} lands on the map's walls")

    def _on_pose(self, msg):
        self.latest = msg

    def _on_scan(self, msg):
        self.scan = msg

    # ------------------------------------------------------------------ map

    def cell(self, x, y):
        """Occupancy under a point: -1 unknown, 0 free, 100 wall. None if the
        map failed to load or the point is off the grid."""
        return None if self.grid is None else self.grid.at(x, y)

    # --------------------------------------------------------------- sample

    def sample(self):
        """One scan, then the pose published just after it, then let both
        subscriptions go. Returns (pose message or None, scan or None).

        Pose second: /tracked_pose comes at 50 Hz, so the first pose after the
        scan arrives is at most one tick newer than it. The other way round,
        the scan could be a whole 100 ms revolution behind the pose."""
        self.latest, self.scan = None, None
        deadline = time.time() + SAMPLE_TIMEOUT_S
        sub = self.create_subscription(LaserScan, SCAN_TOPIC, self._on_scan,
                                       qos_profile_sensor_data)
        try:
            while self.scan is None and time.time() < deadline and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)
        finally:
            self.destroy_subscription(sub)
        sub = self.create_subscription(PoseStamped, "/tracked_pose",
                                       self._on_pose, 1)
        try:
            # A pose is still worth having without a scan -- an off-map pose
            # needs no scan to condemn -- so it gets its own full wait.
            deadline = max(deadline, time.time() + 1.0)
            while self.latest is None and time.time() < deadline and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.1)
        finally:
            self.destroy_subscription(sub)
        return self.latest, self.scan

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
        best = None                             # (fit, pose, occ)
        for _ in range(FIT_TRIES):
            msg, scan = self.sample()
            if msg is None:
                break                           # Cartographer is not publishing
            pose = msg.pose
            x, y = pose.position.x, pose.position.y
            occ = self.cell(x, y)
            if not (occ is not None and 0 <= occ <= FREE_MAX):
                # Off the grid, in unknown space, or inside a wall. Cartographer
                # has been all three while the rover sat still; none of them are
                # worth waking up believing.
                self.report(pose, occ, None, "off_map")
                self.rejected += 1
                if self.rejected in (1, 10, 100):
                    where = "no map" if self.grid is None else f"cell {occ}"
                    self.get_logger().warn(
                        f"not saving ({x:.2f}, {y:.2f}): {where}")
                return
            fit = None if scan is None else self.grid.fit(x, y, _yaw(pose), scan)
            if best is None or (fit is not None and (best[0] is None or fit > best[0])):
                best = (fit, pose, occ)
            if fit is None or fit >= FIT_MIN:
                break        # matched -- or no scan to judge by, which retrying won't fix
        if best is None:
            return
        fit, pose, occ = best
        x, y, yaw = pose.position.x, pose.position.y, math.degrees(_yaw(pose))
        if fit is None:
            # No scan, or a blocked lidar. Unverified is not the same as wrong,
            # but saving it is exactly the unchecked pose this refuses -- the
            # last verified one stays on disk instead.
            self.report(pose, occ, None, "unchecked")
            self.unchecked += 1
            if self.unchecked in (1, 10, 100):
                self.get_logger().warn(
                    f"not saving ({x:.2f}, {y:.2f}): no usable scan on "
                    f"{SCAN_TOPIC} to check it against")
            return
        if fit < FIT_MIN:
            # On free floor, but the lidar says the rover is not here. Keep the
            # last pose that matched, so the next start wakes somewhere real.
            self.report(pose, occ, fit, "lost")
            if not self.lost:
                self.get_logger().warn(
                    f"scan does not match the map at ({x:.2f}, {y:.2f}, "
                    f"{yaw:.0f} deg): {fit:.0%} on walls -- Cartographer is "
                    f"lost; keeping the last pose that matched")
                self.lost = True
            return
        if self.lost:
            self.get_logger().info(
                f"scan matches the map again at ({x:.2f}, {y:.2f}, {yaw:.0f} "
                f"deg): {fit:.0%} on walls")
            self.lost = False
        self.report(pose, occ, fit, "matched")
        self.write(x, y, pose.orientation.z, pose.orientation.w, occ, fit)

    def report(self, pose, occ, fit, verdict):
        """Say where Cartographer thinks it is and whether that is anywhere
        real, so the pages can show it. Best effort: a rover whose status file
        cannot be written should still record poses, so failure warns once and
        is otherwise ignored.

        verdict is matched, lost (on free floor, but the scan disagrees),
        off_map, or unchecked (no scan to judge by). ok stays for readers that
        predate the scan check: it is true only for matched."""
        payload = json.dumps({
            "t": time.time(), "x": round(pose.position.x, 4),
            "y": round(pose.position.y, 4),
            "yaw_deg": round(math.degrees(_yaw(pose)), 1), "cell": occ,
            "fit": None if fit is None else round(fit, 3),
            "verdict": verdict, "ok": verdict == "matched"})
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

    def write(self, x, y, qz, qw, occ, fit):
        """Replace the file atomically -- a power cut must not leave half of
        it, because the next boot reads this before anything else runs."""
        payload = json.dumps({
            "t": time.time(), "x": round(x, 4), "y": round(y, 4),
            "qz": round(qz, 6), "qw": round(qw, 6), "cell": occ,
            "yaw_deg": round(math.degrees(2.0 * math.atan2(qz, qw)), 1),
            "fit": round(fit, 3),
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
