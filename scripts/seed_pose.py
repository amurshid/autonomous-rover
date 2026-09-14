#!/usr/bin/env python3
"""Tell Cartographer roughly where the rover is, so it need not search.

Nothing published to /initialpose at boot, so Cartographer started at the map
origin and found itself by global localization over the whole map: measured at
167s, a 7.89 m jump. From a pose near the truth it refines in seconds instead.

    python3 seed_pose.py                    # where it was last seen, if it checks out
    python3 seed_pose.py --room work_room   # a room's goal pose, ignore memory
    python3 seed_pose.py --pose 1.68 7.75 52
    python3 seed_pose.py --no-check         # trust the memory without checking

Checked, not assumed
--------------------
By default it starts from wherever rover_pose_memory.py last saw the rover,
falling back to the charging spot. Both are guesses about a rover that may
have been carried, or that Cartographer may have been wrong about when it was
switched off -- and a confident wrong seed is worse than none. On 2026-09-13
the memory held a pose 3.9 m from the rover, the seed asserted it, and the
mode page read "lost" from the first second of a drive.

So the rover looks before it believes. It takes one scan, lays it on the map
at each candidate, and seeds only what matches: at least TRUST_FIT of the
points landing within 10 cm of a wall. A candidate that nearly matches is
searched around first -- all headings within SEARCH_RADIUS_M -- which turns a
pose that drifted while parked into the pose beside it that fits. If nothing
matches, it seeds nothing and says so: Cartographer then searches the whole
map, which is slow but ends up somewhere real.

Two consequences worth knowing. wait_for_localisation.py watches for the jump
that global localization makes, and a correctly seeded rover never jumps -- it
will report a timeout, which here means success. And --pose and --room are
taken as asserted, unchecked: they are a person saying where the rover is.

/initialpose is volatile QoS: publishing before set_initial_pose.py has
subscribed drops the message silently, so this waits for a subscriber.

Exit codes: 0 published or deliberately not published, 3 nobody subscribed,
4 no such room.
"""

import argparse
import json
import math
import os
import sys
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

sys.path.insert(0, os.path.expanduser("~"))
from rooms import ROOMS  # noqa: E402
from rover_pose_memory import MAP_YAML, MapGrid  # noqa: E402

# The fallback, when memory is no good: the charging spot, facing the way the
# rover sits on its charger. Not ROOMS["work_room"], though that is the same
# spot: a goal's heading is only how the rover arrives (-121 deg there), and
# seeding with it started Cartographer ~170 deg from the truth, far outside
# the 20 deg its local search covers. Found by matching the parked rover's
# scan against the map at every heading: 95% of rays on a wall at 52 deg.
CHARGING_SPOT = (1.68, 7.75, 52.0)  # x, y, heading in degrees
SUBSCRIBER_TIMEOUT = 30.0
LAST_POSE_PATH = os.environ.get("ROVER_LAST_POSE",
                                "/var/lib/rover/last_pose.json")
MAX_AGE_S = 12 * 3600              # older than this and it is a guess

SCAN_TOPIC = os.environ.get("ROVER_SCAN_TOPIC", "/scan_fixed")
SCAN_TIMEOUT_S = 5.0
# What it takes to be believed. rover_pose_memory saves anything over 50%,
# because at a sample every 4s a marginal pose is cheap to replace; seeding is
# the opposite -- one decision the whole drive rests on -- so the bar is
# higher. Parked and right has measured 86-99%; the pose that started the
# 2026-09-13 drive read 59% from 3.9 m away.
TRUST_FIT = 0.70
# Searched around a candidate before giving up on it. A pose that slid while
# the rover sat still is usually within a metre; further than that and the
# whole-map search is the honest answer.
SEARCH_RADIUS_M, SEARCH_STEP_M, SEARCH_STEP_DEG = 1.5, 0.15, 10


def yaw_quat(yaw_deg):
    """(qz, qw) for a heading in degrees."""
    half = math.radians(yaw_deg) / 2
    return math.sin(half), math.cos(half)


def quat_yaw(qz, qw):
    """Heading in degrees from a yaw-only quaternion."""
    return math.degrees(2.0 * math.atan2(qz, qw))


def remembered(path, max_age):
    """(x, y, yaw_deg, description) from the saved pose, or None.

    Every reason to distrust it ends the same way -- fall back to the charging
    spot -- so they are all one quiet return rather than a pile of error
    cases. The only one worth naming out loud is age, since a stale memory
    means the rover sat somewhere for half a day and the answer may simply be
    old.
    """
    try:
        with open(path) as f:
            d = json.load(f)
        age = time.time() - float(d["t"])
        x, y = float(d["x"]), float(d["y"])
        yaw = quat_yaw(float(d["qz"]), float(d["qw"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if age > max_age:
        print(f"last known pose is {age / 3600:.1f}h old, older than "
              f"{max_age / 3600:.0f}h -- not using it")
        return None
    mins = age / 60.0
    when = f"{age:.0f}s ago" if mins < 1 else f"{mins:.0f} min ago"
    return x, y, yaw, f"where it was last seen ({when})"


def grab_scan(node, timeout):
    """One LaserScan from SCAN_TOPIC, or None. Sensor-data QoS, as the relay
    publishes it."""
    got = []
    sub = node.create_subscription(LaserScan, SCAN_TOPIC, got.append,
                                   qos_profile_sensor_data)
    try:
        deadline = time.time() + timeout
        while not got and time.time() < deadline and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_subscription(sub)
    return got[-1] if got else None


def search_near(grid, scan, x0, y0):
    """The best (fit, x, y, yaw_deg) within SEARCH_RADIUS_M of (x0, y0), over
    every heading. Coarse pass, then a fine one around its answer."""
    steps = int(SEARCH_RADIUS_M / SEARCH_STEP_M)
    offsets = [i * SEARCH_STEP_M for i in range(-steps, steps + 1)]
    best = (-1.0, x0, y0, 0.0)
    for dx in offsets:
        for dy in offsets:
            for deg in range(0, 360, SEARCH_STEP_DEG):
                f = grid.fit(x0 + dx, y0 + dy, math.radians(deg), scan)
                if f is not None and f > best[0]:
                    best = (f, x0 + dx, y0 + dy, float(deg))
    _, bx, by, bdeg = best
    for dx in (-0.1, -0.05, 0.0, 0.05, 0.1):
        for dy in (-0.1, -0.05, 0.0, 0.05, 0.1):
            for ddeg in range(-8, 9, 2):
                f = grid.fit(bx + dx, by + dy, math.radians(bdeg + ddeg), scan)
                if f is not None and f > best[0]:
                    best = (f, bx + dx, by + dy, bdeg + ddeg)
    return best


def pick(grid, scan, candidates):
    """The first candidate the scan agrees with: (x, y, yaw_deg, description)
    or None. Each is scored where it stands, then searched around."""
    for x, y, yaw, what in candidates:
        here = grid.fit(x, y, math.radians(yaw), scan)
        if here is None:
            print("too few lidar returns to check anything -- is the lidar "
                  "covered?")
            return None
        print(f"{what}: x={x:.2f} y={y:.2f} yaw={yaw:.0f}deg fits {here:.0%}")
        if here >= TRUST_FIT:
            return x, y, yaw, what
        fit, bx, by, bdeg = search_near(grid, scan, x, y)
        if fit >= TRUST_FIT:
            print(f"  corrected to x={bx:.2f} y={by:.2f} yaw={bdeg:.0f}deg, "
                  f"which fits {fit:.0%} "
                  f"({math.hypot(bx - x, by - y):.2f} m away)")
            return bx, by, bdeg, f"{what}, corrected by the scan"
        print(f"  nothing within {SEARCH_RADIUS_M:.1f} m fits either "
              f"(best {fit:.0%})")
    return None


def publish(node, x, y, qz, qw, timeout):
    """Assert the pose on /initialpose. Returns an exit code."""
    pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
    # Volatile QoS: a pose published before the bridge subscribes is dropped
    # with no error, and Cartographer would then search the map as if nothing
    # had been sent.
    deadline = time.time() + timeout
    while pub.get_subscription_count() == 0:
        if time.time() > deadline:
            print('nothing is subscribed to /initialpose after '
                  f'{timeout:.0f}s -- is set_initial_pose.py running?')
            return 3
        rclpy.spin_once(node, timeout_sec=0.2)

    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = 'map'
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.orientation.z = float(qz)
    msg.pose.pose.orientation.w = float(qw)
    # set_initial_pose.py reads pose.pose only; this is filled in for anything
    # else that listens and expects a sane estimate rather than certainty.
    msg.pose.covariance[0] = msg.pose.covariance[7] = 0.25
    msg.pose.covariance[35] = 0.068
    pub.publish(msg)
    # The bridge finishes a trajectory and starts a new one, which is not
    # instant. Leaving immediately can drop the message on the floor.
    end = time.time() + 2.0
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--room', help='assert a room instead of the last '
                                   'known pose')
    ap.add_argument('--pose', nargs=3, type=float, metavar=('X', 'Y', 'YAW_DEG'),
                    help='explicit pose, overrides everything')
    ap.add_argument('--max-age', type=float, default=MAX_AGE_S,
                    help=f'seconds before the saved pose is too old to trust '
                         f'(default {MAX_AGE_S / 3600:.0f}h)')
    ap.add_argument('--timeout', type=float, default=SUBSCRIBER_TIMEOUT,
                    help='seconds to wait for a subscriber (default 30)')
    ap.add_argument('--no-check', action='store_true',
                    help='seed the remembered pose without checking it '
                         'against the lidar')
    args = ap.parse_args()

    asserted = None                       # a person's word, taken as given
    if args.pose:
        x, y, yaw = args.pose
        asserted = (x, y, yaw, 'the pose you gave')
    elif args.room:
        if args.room not in ROOMS:
            print(f'no such room: {args.room}. Known: {", ".join(sorted(ROOMS))}')
            return 4
        rx, ry, qz, qw = ROOMS[args.room]
        asserted = (rx, ry, quat_yaw(qz, qw),
                    f'{args.room} (x={rx:.2f} y={ry:.2f})')

    rclpy.init()
    node = Node('seed_pose')
    try:
        if asserted:
            x, y, yaw, where = asserted
        else:
            memory = remembered(LAST_POSE_PATH, args.max_age)
            charger = CHARGING_SPOT + ('the charging spot',)
            candidates = ([memory] if memory else []) + [charger]
            grid, scan = None, None
            if not args.no_check:
                try:
                    grid = MapGrid(MAP_YAML)
                except (OSError, ValueError, KeyError, TypeError) as e:
                    print(f'no map from {MAP_YAML} ({e}) -- seeding unchecked')
                if grid is not None:
                    scan = grab_scan(node, SCAN_TIMEOUT_S)
                    if scan is None:
                        print(f'no scan on {SCAN_TOPIC} after '
                              f'{SCAN_TIMEOUT_S:.0f}s -- seeding unchecked')
            if grid is not None and scan is not None:
                chosen = pick(grid, scan, candidates)
                if chosen is None:
                    print('nothing the rover can see matches the map here. '
                          'Seeding nothing: Cartographer will search the whole '
                          'map, which takes minutes. Park it on the charger, '
                          'or pass --pose, to skip that.')
                    return 0
                x, y, yaw, where = chosen
            else:
                x, y, yaw, where = candidates[0]
        qz, qw = yaw_quat(yaw)
        code = publish(node, x, y, qz, qw, args.timeout)
    except (KeyboardInterrupt, ExternalShutdownException):
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if code == 0:
        print(f'seeded {where}: x={x:.2f} y={y:.2f} yaw={yaw:.0f}deg. '
              f'Cartographer should refine from here in seconds; check '
              f'/tracked_pose before sending the rover anywhere.')
    return code


if __name__ == '__main__':
    sys.exit(main())
