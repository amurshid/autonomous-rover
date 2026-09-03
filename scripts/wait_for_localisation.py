#!/usr/bin/env python3
"""Wait until Cartographer has worked out where the rover is, then exit.

Cartographer publishes a pose from the moment it starts, whether that pose
means anything or not, and there is no "converged" flag to read. What is
observable is the moment its global localization finds a match: /tracked_pose
jumps discontinuously -- often by metres -- and then holds.

How long that takes depends entirely on the seed. Nothing publishes to
/initialpose at boot, so Cartographer starts at the map origin and searches
the whole map: measured at 167s, jumping 7.89 m to x=1.55 y=7.73 -- which is
7.88 m from the origin, i.e. the entire correction. From a seed near the truth
it is seconds instead. Hence the 300s default; the unseeded case has been
observed at 167s and there is no margin in guessing tighter.

So this watches for a jump, confirms it holds still afterwards, prints where
it landed and exits. Nothing runs afterwards; the Pi has better uses for a
core.

    python3 wait_for_localisation.py            # exit 0 when it settles
    python3 wait_for_localisation.py --timeout 240

Exit codes: 0 settled, 2 no jump seen before the timeout.

Two things it cannot tell you. A rover that boots near where it already thinks
it is never jumps -- correct from the start, and this reports a timeout. And a
stationary rover holding a wrong pose looks exactly like a settled one, which
is why the jump, not the stillness, is what it waits for.

Keep the rover still while this runs. Driving moves the pose legitimately and
every sample looks like a jump.
"""

import argparse
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

SAMPLE_S = 0.5     # /tracked_pose runs at ~190 Hz; nothing here needs that
JUMP_M = 0.50      # a relocalisation moves further than scan-matching drift
SETTLE_M = 0.15    # how still it has to be afterwards
SETTLE_S = 5.0     # for how long


class Watcher(Node):
    def __init__(self, timeout):
        super().__init__('wait_for_localisation')
        self.timeout = timeout
        self.started = time.time()
        self.last_sample = 0.0
        self.previous = None      # previous sampled position
        self.anchor = None        # where it landed after the jump
        self.anchored_at = None
        self.result = None        # 0 settled, 2 timed out
        self.poses = 0
        self.reported = 0.0
        self.create_subscription(PoseStamped, '/tracked_pose', self.on_pose, 1)
        self.create_timer(1.0, self.on_tick)
        print(f'watching /tracked_pose for a jump over {JUMP_M} m '
              f'(timeout {timeout:.0f}s, keep the rover still)', flush=True)

    def on_tick(self):
        # An unseeded search runs for minutes. Silence for that long reads as a
        # hang, and the temptation is then to kill it just before it works.
        waited = time.time() - self.started
        if (self.result is None and self.anchor is None
                and waited - self.reported >= 15.0):
            self.reported = waited
            if self.poses:
                print(f'  {waited:.0f}s, still searching', flush=True)
            else:
                print(f'  {waited:.0f}s, no /tracked_pose yet '
                      f'-- is cartographer up?', flush=True)
        if self.result is None and waited > self.timeout:
            print('no jump seen. Either it was already right, or it has not '
                  'found itself yet -- drive it a little and watch again.',
                  flush=True)
            self.result = 2

    def on_pose(self, msg):
        now = time.time()
        if now - self.last_sample < SAMPLE_S:
            return
        self.last_sample = now
        self.poses += 1
        p = (msg.pose.position.x, msg.pose.position.y)

        if self.previous is None:
            self.previous = p
            return

        if self.anchor is None:
            moved = math.dist(p, self.previous)
            if moved >= JUMP_M:
                print(f'jumped {moved:.2f} m -> x={p[0]:.2f} y={p[1]:.2f} '
                      f'after {now - self.started:.0f}s', flush=True)
                self.anchor, self.anchored_at = p, now
            self.previous = p
            return

        # Jumped already: it has to stay put before this counts as settled,
        # since global localization can fire more than once.
        if math.dist(p, self.anchor) > SETTLE_M:
            self.anchor, self.anchored_at = p, now
        elif now - self.anchored_at >= SETTLE_S:
            print(f'settled at x={p[0]:.2f} y={p[1]:.2f} '
                  f'({now - self.started:.0f}s). Safe to send it somewhere.',
                  flush=True)
            self.result = 0
        self.previous = p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timeout', type=float, default=300.0,
                    help='seconds to wait for a jump (default 300)')
    args = ap.parse_args()

    rclpy.init()
    node = Watcher(args.timeout)
    try:
        while rclpy.ok() and node.result is None:
            rclpy.spin_once(node, timeout_sec=0.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return node.result if node.result is not None else 2


if __name__ == '__main__':
    sys.exit(main())
