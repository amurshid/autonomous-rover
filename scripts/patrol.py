#!/usr/bin/env python3
import random
import sys
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped

LOCATIONS = {
    'work_room':       (2.276,  8.183,  -0.8196, 0.5730),
    'entrance':        (-1.90,  7.90,    0.528,  0.849),
    'office_room':     (-0.63,  2.85,   -0.861,  0.509),
    'dining_room':     (-5.20,  0.03,   -0.184,  0.983),
    'kitchen':         (-5.84, -4.14,    0.469,  0.883),
    'breakfast_table': (-6.84, -8.24,   -0.698,  0.716),
    'formal_living':   (-10.21, 1.33,    0.964, -0.267),
    'living_room':     (-10.44,-6.37,   -0.711,  0.703),
    'azmayen_room':    (-9.42,  5.99,    0.993, -0.117),
    'parents_room':    (-12.15,-5.72,    0.972,  0.236),
}


class Patrol(Node):
    def __init__(self):
        super().__init__('patrol')
        self.declare_parameter('max_failures', 3)
        self.declare_parameter('dwell', 3.0)
        self.declare_parameter('shuffle', True)
        self.max_fail = self.get_parameter('max_failures').value
        self.dwell = self.get_parameter('dwell').value
        self.shuffle = self.get_parameter('shuffle').value
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

    def goal_for(self, name):
        x, y, qz, qw = LOCATIONS[name]
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = 'map'
        g.pose.header.stamp = self.get_clock().now().to_msg()
        g.pose.pose.position.x = x
        g.pose.pose.position.y = y
        g.pose.pose.orientation.z = qz
        g.pose.pose.orientation.w = qw
        return g

    def spin_until(self, future, timeout):
        deadline = time.monotonic() + timeout
        while rclpy.ok() and not future.done():
            if time.monotonic() > deadline:
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return future.done()

    def go(self, name):
        self.get_logger().info(f'--> {name}')
        send = self.client.send_goal_async(self.goal_for(name))
        if not self.spin_until(send, 15.0):
            self.get_logger().warn(f'{name}: no response to goal request')
            return False
        handle = send.result()
        if handle is None or not handle.accepted:
            self.get_logger().warn(f'{name}: goal rejected')
            return False
        res = handle.get_result_async()
        if not self.spin_until(res, 300.0):
            self.get_logger().warn(f'{name}: timed out, cancelling')
            handle.cancel_goal_async()
            return False
        r = res.result()
        ok = r is not None and r.status == 4   # STATUS_SUCCEEDED
        self.get_logger().info(f'{name}: {"ok" if ok else "FAILED"}')
        return ok


def main():
    rclpy.init()
    node = Patrol()
    if not node.client.wait_for_server(timeout_sec=20.0):
        node.get_logger().error('navigate_to_pose unavailable — is Nav2 up?')
        rclpy.shutdown()
        sys.exit(1)

    names = list(LOCATIONS)
    fails = 0
    laps = 0
    try:
        while rclpy.ok():
            order = names[:]
            if node.shuffle:
                random.shuffle(order)
            for name in order:
                if node.go(name):
                    fails = 0
                else:
                    fails += 1
                    if fails >= node.max_fail:
                        node.get_logger().error(
                            f'{fails} consecutive failures — stopping')
                        raise KeyboardInterrupt
                time.sleep(node.dwell)
            laps += 1
            node.get_logger().info(f'lap {laps} complete')
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f'stopped after {laps} laps')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
