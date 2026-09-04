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

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)

STATE_PATH = os.environ.get("ROVER_LAST_POSE",
                            "/var/lib/rover/last_pose.json")
SAMPLE_PERIOD_S = 30.0
SAMPLE_TIMEOUT_S = 3.0
FREE_MAX = 20          # occupancy 0-100; anything above this is not open floor

# map_server latches /map, so a late subscriber still gets it.
MAP_QOS = QoSProfile(
    depth=1, history=QoSHistoryPolicy.KEEP_LAST,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


class PoseMemory(Node):
    def __init__(self):
        super().__init__("rover_pose_memory")
        self.grid = None
        self.latest = None
        self.saved = 0
        self.rejected = 0
        self.create_subscription(OccupancyGrid, "/map", self._on_map, MAP_QOS)
        self.get_logger().info(
            f"remembering the pose to {STATE_PATH} every "
            f"{SAMPLE_PERIOD_S:.0f}s, when it stands on free space")

    def _on_map(self, msg):
        self.grid = msg          # arrives once, kept for the life of the node

    def _on_pose(self, msg):
        self.latest = msg

    # ------------------------------------------------------------------ map

    def cell(self, x, y):
        """Occupancy under a point: -1 unknown, 0 free, 100 wall. None if the
        map has not arrived or the point is off the grid."""
        g = self.grid
        if g is None:
            return None
        res = g.info.resolution
        col = int((x - g.info.origin.position.x) / res)
        row = int((y - g.info.origin.position.y) / res)
        if not (0 <= col < g.info.width and 0 <= row < g.info.height):
            return None
        return g.data[row * g.info.width + col]

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
        msg = self.sample()
        if msg is None:
            return
        x, y = msg.pose.position.x, msg.pose.position.y
        occ = self.cell(x, y)
        if occ is None or occ < 0 or occ > FREE_MAX:
            # Off the grid, in unknown space, or inside a wall. Cartographer
            # has been all three while the rover sat still; none of them are
            # worth waking up believing.
            self.rejected += 1
            if self.rejected in (1, 10, 100):
                where = "no map yet" if self.grid is None else f"cell {occ}"
                self.get_logger().warn(
                    f"not saving ({x:.2f}, {y:.2f}): {where}")
            return
        self.write(x, y, msg.pose.orientation.z, msg.pose.orientation.w, occ)

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
