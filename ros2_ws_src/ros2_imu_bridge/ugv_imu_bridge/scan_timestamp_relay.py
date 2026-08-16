#!/usr/bin/env python3
"""
Scan timestamp relay for the LDROBOT LD19 (ldlidar_stl_ros2).

The LD19 driver derives each scan's header.stamp from the lidar's reported
spin speed, which jitters — so consecutive /scan stamps occasionally go
*backwards*. Cartographer then rejects those scans
("Ignored subdivision ... time is not before") and drops a whole scan
("Dropped ~440 earlier points"). During a turn, that gap lets the pose slip
and the map warps.

This node re-stamps each scan with a strictly monotonic time (the host clock
at receipt, shifted back by scan_time to approximate the first ray) and
republishes on /scan_fixed. Point Cartographer at /scan_fixed and the
timestamp rejections / dropped scans go away.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanTimestampRelay(Node):
    def __init__(self):
        super().__init__('scan_timestamp_relay')

        self.declare_parameter('input_topic', '/scan')
        self.declare_parameter('output_topic', '/scan_fixed')
        in_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self._last_stamp_ns = 0
        self._warned = False

        # Sensor-data QoS (best-effort) on both ends: the sub accepts the
        # driver's stream regardless of its reliability, and best-effort
        # matches Cartographer's SensorDataQoS scan subscription.
        self.pub = self.create_publisher(LaserScan, out_topic, qos_profile_sensor_data)
        self.sub = self.create_subscription(
            LaserScan, in_topic, self.relay, qos_profile_sensor_data)

        self.get_logger().info(
            f'Re-stamping {in_topic} -> {out_topic} with monotonic timestamps')

    def relay(self, msg: LaserScan):
        # Host clock at receipt; back-date by one scan period so the stamp
        # approximates the first ray (LaserScan convention), not scan end.
        stamp_ns = self.get_clock().now().nanoseconds
        if msg.scan_time > 0.0:
            stamp_ns -= int(msg.scan_time * 1e9)

        # Hard guarantee of strictly increasing stamps.
        if stamp_ns <= self._last_stamp_ns:
            stamp_ns = self._last_stamp_ns + 1000  # +1 us
            if not self._warned:
                self.get_logger().warn(
                    'Incoming scan stamps not monotonic — clamping (expected with LD19).')
                self._warned = True
        self._last_stamp_ns = stamp_ns

        msg.header.stamp.sec = stamp_ns // 1_000_000_000
        msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanTimestampRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
