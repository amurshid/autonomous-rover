#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from nav_msgs.msg import Odometry
from rclpy.duration import Duration


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class OdomPublisher(Node):
    def __init__(self):
        super().__init__('odom_publisher')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter("rate", 10.0)

        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        rate = self.get_parameter('rate').value

        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.pub = self.create_publisher(Odometry, 'odom', 10)

        self.prev = None
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(
            f'Publishing /odom from {self.odom_frame}->{self.base_frame}')

    def tick(self):
        try:
            tf = self.buffer.lookup_transform(
                self.odom_frame, self.base_frame,
                rclpy.time.Time())
        except Exception:
            return

        t = tf.transform.translation
        yaw = yaw_from_quat(tf.transform.rotation)
        stamp = rclpy.time.Time.from_msg(tf.header.stamp).nanoseconds / 1e9

        vx = vy = vth = 0.0
        if self.prev is not None:
            px, py, pyaw, pt = self.prev
            dt = stamp - pt
            if 1e-4 < dt < 1.0:
                dx = t.x - px
                dy = t.y - py
                dyaw = math.atan2(math.sin(yaw - pyaw), math.cos(yaw - pyaw))
                vx = (dx * math.cos(yaw) + dy * math.sin(yaw)) / dt
                vy = (-dx * math.sin(yaw) + dy * math.cos(yaw)) / dt
                vth = dyaw / dt

        self.prev = (t.x, t.y, yaw, stamp)

        msg = Odometry()
        msg.header.stamp = tf.header.stamp
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x = t.x
        msg.pose.pose.position.y = t.y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = tf.transform.rotation
        msg.twist.twist.linear.x = vx
        msg.twist.twist.linear.y = vy
        msg.twist.twist.angular.z = vth
        msg.pose.covariance[0] = 0.05
        msg.pose.covariance[7] = 0.05
        msg.pose.covariance[35] = 0.1
        msg.twist.covariance[0] = 0.05
        msg.twist.covariance[7] = 0.05
        msg.twist.covariance[35] = 0.1
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = OdomPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
