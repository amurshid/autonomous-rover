#!/usr/bin/env python3
"""Watch the three layers Nav2 depends on, and catch which one goes wrong.

Nav2 reported the rover at (191, -5.58) and running away at 6 m/s while
/tracked_pose read (2.26, 8.21) -- correct. Nav2 does not invent a position:
it transforms the laser into `map` using TF. So either Cartographer publishes
a transform that disagrees with the pose it reports, or the transform is fine
and Nav2's own cache is stale. Restarting Nav2 alone used to fix it, which
points at the second, but that was never measured.

This prints one line a second:

    pose(2.26, 8.21)  tf(2.26, 8.21)  d=0.00  tf_age=0.05  scan_lag=0.03  ok

    pose   /tracked_pose, Cartographer's own estimate
    tf     map -> base_link, what it publishes
    d      distance between them. Anything but ~0 is the bug, in the open
    tf_age how stale the newest transform is
    scan_lag how far behind wall-clock /scan timestamps are

The last one matters because of the other error in the log -- "the timestamp
on the message is earlier than all the data in the transform cache" -- which
is what you get when scans arrive stamped older than the transforms.

    python3 tf_watch.py                    # to the terminal
    python3 tf_watch.py --csv ~/tf.csv     # and to a file, to read afterwards

Leave it running, use the rover normally, and look at what the line says at
the moment Nav2 starts complaining. Run it with Cartographer up; Nav2 need
not be, since nothing here asks Nav2 anything.
"""

import argparse
import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

# From the costmap's own complaint: bounds (-19.19, -13.97) to (5.23, 14.20).
MAP_MIN, MAP_MAX = (-19.19, -13.97), (5.23, 14.20)
DISAGREE_M = 0.5

# The bounding box is a weak test and it lied. A pose can sit well inside
# those numbers and still be nowhere the rover could be: the house occupies
# only part of the rectangle, and the rest is unknown space. A run that
# reported "ok" for a hundred samples turned out, in Foxglove, to have the
# rover parked in the grey. So check the occupancy grid itself.
MAP_QOS = QoSProfile(
    depth=1, history=QoSHistoryPolicy.KEEP_LAST,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
UNKNOWN, OCCUPIED = -1, 50      # cell values: -1 unknown, 0-100 probability


class Watch(Node):
    def __init__(self, csv):
        super().__init__('tf_watch')
        self.buf = Buffer()
        self.listener = TransformListener(self.buf, self)
        self.pose = None
        self.scan_stamp = None
        self.grid = None
        self.csv = csv
        self.worst = 0.0
        self.create_subscription(PoseStamped, '/tracked_pose', self.on_pose, 1)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, MAP_QOS)
        self.create_subscription(LaserScan, '/scan', self.on_scan,
                                 qos_profile_sensor_data)
        self.create_timer(1.0, self.tick)
        if csv:
            csv.write('t,pose_x,pose_y,tf_x,tf_y,delta_m,tf_age_s,'
                      'scan_lag_s,cell,note\n')

    def on_pose(self, msg):
        self.pose = (msg.pose.position.x, msg.pose.position.y)

    def on_scan(self, msg):
        self.scan_stamp = (msg.header.stamp.sec
                           + msg.header.stamp.nanosec * 1e-9)

    def on_map(self, msg):
        self.grid = msg          # latched, arrives once

    def cell(self, xy):
        """The occupancy value under a pose: -1 unknown, 0 free, 100 wall.

        None if there is no map yet or the pose is off the grid entirely.
        """
        g = self.grid
        if g is None or xy is None:
            return None
        res = g.info.resolution
        col = int((xy[0] - g.info.origin.position.x) / res)
        row = int((xy[1] - g.info.origin.position.y) / res)
        if not (0 <= col < g.info.width and 0 <= row < g.info.height):
            return None
        return g.data[row * g.info.width + col]

    def tick(self):
        now = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9
        note, tf_xy, tf_age = [], None, None

        try:
            t = self.buf.lookup_transform('map', 'base_link', rclpy.time.Time())
            tf_xy = (t.transform.translation.x, t.transform.translation.y)
            stamp = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
            tf_age = now_s - stamp
        except Exception as e:
            note.append(f'no map->base_link ({type(e).__name__})')

        scan_lag = None if self.scan_stamp is None else now_s - self.scan_stamp
        if self.scan_stamp is None:
            note.append('no /scan')
        elif scan_lag > 1.0:
            note.append(f'scan {scan_lag:.1f}s behind')

        delta = None
        if self.pose and tf_xy:
            delta = math.dist(self.pose, tf_xy)
            self.worst = max(self.worst, delta)
            if delta > DISAGREE_M:
                # The interesting case: Cartographer's estimate and the
                # transform it publishes have come apart.
                note.append(f'POSE AND TF DISAGREE by {delta:.2f} m')

        for name, xy in (('pose', self.pose), ('tf', tf_xy)):
            if xy and not (MAP_MIN[0] <= xy[0] <= MAP_MAX[0]
                           and MAP_MIN[1] <= xy[1] <= MAP_MAX[1]):
                note.append(f'{name} OUTSIDE THE MAP')

        # The test that matters: is the rover somewhere the map says exists?
        occ = self.cell(self.pose)
        if self.grid is None:
            note.append('no /map yet')
        elif occ is None:
            note.append('OFF THE GRID')
        elif occ == UNKNOWN:
            note.append('IN UNKNOWN SPACE -- the rover cannot be here')
        elif occ >= OCCUPIED:
            note.append(f'INSIDE A WALL (cell {occ})')

        fmt = lambda xy: f'({xy[0]:7.2f},{xy[1]:7.2f})' if xy else '(   --  ,   --  )'
        line = (f'pose{fmt(self.pose)}  tf{fmt(tf_xy)}  '
                f'd={delta:5.2f}  ' if delta is not None
                else f'pose{fmt(self.pose)}  tf{fmt(tf_xy)}  d=   -  ')
        line += (f'tf_age={tf_age:5.2f}  ' if tf_age is not None else 'tf_age=  -   ')
        line += (f'scan_lag={scan_lag:5.2f}  ' if scan_lag is not None
                 else 'scan_lag=  -   ')
        line += (f'cell={occ:4}  ' if occ is not None else 'cell=  -   ')
        line += ('; '.join(note) if note else 'ok')
        print(line, flush=True)

        if self.csv:
            self.csv.write(
                f'{now_s:.3f},'
                f'{self.pose[0] if self.pose else ""},'
                f'{self.pose[1] if self.pose else ""},'
                f'{tf_xy[0] if tf_xy else ""},{tf_xy[1] if tf_xy else ""},'
                f'{delta if delta is not None else ""},'
                f'{tf_age if tf_age is not None else ""},'
                f'{scan_lag if scan_lag is not None else ""},'
                f'{occ if occ is not None else ""},'
                f'{" ".join(note)}\n')
            self.csv.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=argparse.FileType('w'),
                    help='also append each sample to this file')
    args = ap.parse_args()

    rclpy.init()
    node = Watch(args.csv)
    print('watching /tracked_pose against TF map->base_link. Ctrl-C to stop.',
          flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        print(f'\nlargest pose/tf disagreement seen: {node.worst:.2f} m',
              flush=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
