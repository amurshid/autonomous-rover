#!/usr/bin/env python3
"""Bridge /initialpose -> Cartographer trajectory restart.

Cartographer has no /initialpose subscriber. To relocalize you must finish the
active trajectory and start a new one anchored to the frozen map (trajectory 0).

Service calls run in the main loop, NOT in the subscription callback --
calling spin_until_future_complete from inside a callback deadlocks the
single-threaded executor against itself.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from cartographer_ros_msgs.srv import FinishTrajectory, StartTrajectory

CONFIG_DIR = '/home/amurshid/cartographer_config'
CONFIG_BASENAME = 'wave_rover_localization.lua'


class InitialPoseBridge(Node):
    def __init__(self):
        super().__init__('set_initial_pose')
        self.current_trajectory_id = 1   # 0 is the frozen pbstream map
        self.pending_pose = None
        self.finish_cli = self.create_client(FinishTrajectory, '/finish_trajectory')
        self.start_cli = self.create_client(StartTrajectory, '/start_trajectory')
        self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.on_pose, 10)
        self.get_logger().info('Ready. Publish a pose estimate to /initialpose.')

    def on_pose(self, msg):
        # Only queue here. Actual work happens in main loop.
        self.pending_pose = msg.pose.pose

    def relocalize(self, pose):
        self.get_logger().info(
            f'Relocalizing to x={pose.position.x:.3f} y={pose.position.y:.3f}')

        if not self.finish_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/finish_trajectory unavailable')
            return

        req = FinishTrajectory.Request()
        req.trajectory_id = self.current_trajectory_id
        future = self.finish_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        result = future.result()
        if result is None:
            self.get_logger().error('finish_trajectory timed out')
            return
        if result.status.code != 0:
            self.get_logger().warn(
                f'finish_trajectory: {result.status.message}')

        if not self.start_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/start_trajectory unavailable')
            return

        sreq = StartTrajectory.Request()
        sreq.configuration_directory = CONFIG_DIR
        sreq.configuration_basename = CONFIG_BASENAME
        sreq.use_initial_pose = True
        sreq.initial_pose = pose
        sreq.relative_to_trajectory_id = 0
        sfuture = self.start_cli.call_async(sreq)
        rclpy.spin_until_future_complete(self, sfuture, timeout_sec=10.0)

        sresult = sfuture.result()
        if sresult is None:
            self.get_logger().error('start_trajectory timed out')
            return
        if sresult.status.code != 0:
            self.get_logger().error(
                f'start_trajectory rejected: {sresult.status.message}')
            return

        self.current_trajectory_id = sresult.trajectory_id
        self.get_logger().info(f'Now on trajectory {self.current_trajectory_id}')


def main():
    rclpy.init()
    node = InitialPoseBridge()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.pending_pose is not None:
                pose = node.pending_pose
                node.pending_pose = None
                node.relocalize(pose)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
