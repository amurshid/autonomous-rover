#!/usr/bin/env python3
import csv
import math
import os
import time
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from PIL import Image as PILImage


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class VprLogger(Node):
    def __init__(self):
        super().__init__('vpr_logger')
        self.declare_parameter('out_dir', os.path.expanduser('~/vpr_data'))
        self.declare_parameter('min_dist', 0.20)
        self.declare_parameter('min_yaw_deg', 15.0)
        self.declare_parameter('max_pose_age', 0.5)
        self.declare_parameter('jump_threshold', 0.5)
        self.declare_parameter('session_tag', '')

        g = self.get_parameter
        base = g('out_dir').value
        self.min_dist = g('min_dist').value
        self.min_yaw = math.radians(g('min_yaw_deg').value)
        self.max_age = g('max_pose_age').value
        self.jump_thr = g('jump_threshold').value

        tag = g('session_tag').value or datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session = os.path.join(base, tag)
        self.img_dir = os.path.join(self.session, 'images')
        os.makedirs(self.img_dir, exist_ok=True)

        self.csv_path = os.path.join(self.session, 'poses.csv')
        new = not os.path.exists(self.csv_path)
        self.csv_file = open(self.csv_path, 'a', newline='')
        self.csv = csv.writer(self.csv_file)
        if new:
            self.csv.writerow(
                ['filename', 'stamp', 'x', 'y', 'yaw', 'jump'])

        self.pose = None          # (x, y, yaw, monotonic_time)
        self.last_saved = None    # (x, y, yaw)
        self.prev_pose = None
        self.count = 0
        self.jump_flag = 0

        self.seen = 0
        pose_grp = MutuallyExclusiveCallbackGroup()
        img_grp = MutuallyExclusiveCallbackGroup()
        self.create_subscription(
            PoseStamped, '/tracked_pose', self.on_pose, 1,
            callback_group=pose_grp)
        self.create_subscription(
            Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data,
            callback_group=img_grp)
        self.create_timer(10.0, self.report, callback_group=pose_grp)

        self.get_logger().info(f'Logging to {self.session}')
        self.get_logger().info(
            f'trigger: {self.min_dist} m or {math.degrees(self.min_yaw):.0f} deg')

    def on_pose(self, msg):
        p = msg.pose.position
        y = yaw_of(msg.pose.orientation)

        if self.prev_pose is not None:
            d = math.hypot(p.x - self.prev_pose[0], p.y - self.prev_pose[1])
            if d > self.jump_thr:
                self.jump_flag = 1
                self.get_logger().warn(
                    f'pose jump {d:.2f} m - localization may have slipped')
        self.prev_pose = (p.x, p.y)
        self.pose = (p.x, p.y, y, time.monotonic())

    def moved_enough(self):
        if self.last_saved is None:
            return True
        x, y, yaw, _ = self.pose
        lx, ly, lyaw = self.last_saved
        if math.hypot(x - lx, y - ly) >= self.min_dist:
            return True
        dy = math.atan2(math.sin(yaw - lyaw), math.cos(yaw - lyaw))
        return abs(dy) >= self.min_yaw

    def on_image(self, msg):
        self.seen += 1
        if self.pose is None:
            return
        if time.monotonic() - self.pose[3] > self.max_age:
            return
        if not self.moved_enough():
            return
        if msg.encoding not in ('rgb8', 'bgr8'):
            self.get_logger().warn(f'unexpected encoding {msg.encoding}')
            return

        arr = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            arr = arr.reshape(msg.height, msg.width, 3)
        except ValueError:
            return
        if msg.encoding == 'bgr8':
            arr = arr[:, :, ::-1]

        name = f'{self.count:06d}.jpg'
        PILImage.fromarray(arr).save(
            os.path.join(self.img_dir, name), quality=92)

        x, y, yaw, _ = self.pose
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.csv.writerow([name, f'{stamp:.3f}', f'{x:.4f}', f'{y:.4f}',
                           f'{yaw:.4f}', self.jump_flag])
        self.csv_file.flush()

        self.last_saved = (x, y, yaw)
        self.count += 1
        self.jump_flag = 0

    def report(self):
        self.get_logger().info(
            f'{self.count} saved / {self.seen} images seen')

    def destroy_node(self):
        try:
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = VprLogger()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        print(f'total {node.count} frames')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
