#!/usr/bin/env python3
"""Tell Cartographer roughly where the rover is, so it need not search.

Nothing published to /initialpose at boot, so Cartographer started at the map
origin and found itself by global localization over the whole map: measured at
167s, a 7.89 m jump. From a pose near the truth it refines in seconds instead.

    python3 seed_pose.py                    # where it was last seen
    python3 seed_pose.py --room work_room   # a room's goal pose, ignore memory
    python3 seed_pose.py --pose 1.68 7.75 52

By default it starts from wherever rover_pose_memory.py last saw the rover
standing on free floor, falling back to the charging spot if that memory is
missing, unreadable, older than --max-age, or forgotten from the mode page. So
the rover can be left in the kitchen and still wake up knowing roughly where
it is, and a rover that has lost its memory belongs on the charger.

Either way this is only right if the rover has not been moved since. A
confident wrong seed is worse than none: the marked spot measured 5.66 m
wrong on an ordinary boot, and Cartographer's scan matcher will happily hold
a wrong pose that looks locally consistent. If it was carried somewhere while
off, pass --room or --pose.

Two consequences worth knowing. wait_for_localisation.py watches for the jump
that global localization makes, and a correctly seeded rover never jumps -- it
will report a timeout, which here means success. And if the seed is wrong,
what you get is not a slow fix but a confident wrong answer, so check
/tracked_pose before sending the rover anywhere.

/initialpose is volatile QoS: publishing before set_initial_pose.py has
subscribed drops the message silently, so this waits for a subscriber.

Exit codes: 0 published, 3 nobody subscribed, 4 no such room.
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

sys.path.insert(0, os.path.expanduser("~"))
from rooms import ROOMS  # noqa: E402

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


def yaw_quat(yaw_deg):
    """(qz, qw) for a heading in degrees."""
    half = math.radians(yaw_deg) / 2
    return math.sin(half), math.cos(half)


def remembered(path, max_age):
    """(x, y, qz, qw, description) from the saved pose, or None.

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
        qz, qw = float(d["qz"]), float(d["qw"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if age > max_age:
        print(f"last known pose is {age / 3600:.1f}h old, older than "
              f"{max_age / 3600:.0f}h -- using the charging spot instead")
        return None
    mins = age / 60.0
    when = f"{age:.0f}s ago" if mins < 1 else f"{mins:.0f} min ago"
    return x, y, qz, qw, f"where it was last seen ({when})"


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
    args = ap.parse_args()

    memory = None if (args.pose or args.room) else \
        remembered(LAST_POSE_PATH, args.max_age)

    if args.pose:
        x, y, yaw = args.pose
        qz, qw = yaw_quat(yaw)
        where = f'x={x:.2f} y={y:.2f} yaw={yaw:.0f}deg'
    elif memory:
        x, y, qz, qw, when = memory
        where = f'{when}: x={x:.2f} y={y:.2f}'
    elif args.room:
        if args.room not in ROOMS:
            print(f'no such room: {args.room}. Known: {", ".join(sorted(ROOMS))}')
            return 4
        x, y, qz, qw = ROOMS[args.room]
        where = f'{args.room} (x={x:.2f} y={y:.2f})'
    else:
        x, y, yaw = CHARGING_SPOT
        qz, qw = yaw_quat(yaw)
        where = f'the charging spot (x={x:.2f} y={y:.2f} yaw={yaw:.0f}deg)'

    rclpy.init()
    node = Node('seed_pose')
    pub = node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

    # Volatile QoS: a pose published before the bridge subscribes is dropped
    # with no error, and Cartographer would then search the map as if nothing
    # had been sent.
    deadline = time.time() + args.timeout
    while pub.get_subscription_count() == 0:
        if time.time() > deadline:
            print('nothing is subscribed to /initialpose after '
                  f'{args.timeout:.0f}s -- is set_initial_pose.py running?')
            node.destroy_node()
            rclpy.shutdown()
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

    try:
        pub.publish(msg)
        # The bridge finishes a trajectory and starts a new one, which is not
        # instant. Leaving immediately can drop the message on the floor.
        end = time.time() + 2.0
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(f'seeded {where}. Cartographer should refine from here in seconds; '
          f'check /tracked_pose before sending the rover anywhere.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
