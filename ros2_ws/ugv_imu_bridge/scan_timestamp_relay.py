#!/usr/bin/env python3
"""
Scan timestamp relay for the LDROBOT LD19 (ldlidar_stl_ros2).

The LD19 driver derives each scan's header.stamp from the lidar's reported
spin speed, which jitters -- so consecutive /scan stamps occasionally go
*backwards*. Cartographer then rejects those scans
("Ignored subdivision ... time is not before") and drops a whole scan.

This node re-stamps each scan and republishes on /scan_fixed. Point
Cartographer at /scan_fixed.

Spacing, not just ordering
-------------------------
The first version stamped each scan with the host clock at receipt, minus
scan_time. That fixed the ordering but not the spacing, and Cartographer
cared about both.

A scan is not an instant: it covers stamp .. stamp + scan_time, about 100ms
at 10 Hz. Cartographer's range_data_collator drops any point whose time is
not after the last point it processed. So if two consecutive stamps are
closer together than one scan span, the scans overlap in time and the
overlap is discarded.

Stamping at receipt put this node's own scheduling jitter straight into the
timestamps. Measured on the rover: /scan from the driver ran at 10.001 Hz
with 0.0018 std dev, while /scan_fixed out of here ran with 0.0034 std dev
and intervals as short as 0.090s -- 10ms inside a 100ms scan. That is a 10%
overlap, and 10% of ~450 points is ~45 dropped. Observed drops ran 1 to 66
points, several times a second, 23,266 of them in one day.

The consequence is not cosmetic. With no IMU and no odometry, Cartographer
has only scan matching to constrain its pose extrapolator. Feed it partial
scans and the constraint weakens; measured on a stationary rover, the pose
left the map at ~7 m/s and did not come back, which aborts any Nav2 goal in
flight. It got worse under load, because more contention meant more jitter
meant worse stamps.

So the stamp advances by at least one scan span every time, and only tracks
the host clock when the host clock is further ahead than that. Overlap
becomes impossible by construction rather than by luck.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

DEFAULT_PERIOD_NS = 100_000_000      # 10 Hz, if the message does not say
RESYNC_NS = 2_000_000_000            # a gap this large is lost scans, not jitter


class ScanTimestampRelay(Node):
    def __init__(self):
        super().__init__('scan_timestamp_relay')

        self.declare_parameter('input_topic', '/scan')
        self.declare_parameter('output_topic', '/scan_fixed')
        in_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self._last_stamp_ns = None
        self._crowded = 0            # scans that arrived closer than one span

        # Sensor-data QoS (best-effort) on both ends: the sub accepts the
        # driver's stream regardless of its reliability, and best-effort
        # matches Cartographer's SensorDataQoS scan subscription.
        self.pub = self.create_publisher(LaserScan, out_topic, qos_profile_sensor_data)
        self.sub = self.create_subscription(
            LaserScan, in_topic, self.relay, qos_profile_sensor_data)
        self.create_timer(60.0, self._report)

        self.get_logger().info(
            f'Re-stamping {in_topic} -> {out_topic}, spaced by scan span')

    @staticmethod
    def _span_ns(msg):
        """How much time this scan actually covers.

        Prefer the per-ray arithmetic, which is what Cartographer uses to
        time individual points; fall back to scan_time, then to 10 Hz.
        """
        span = 0
        if msg.time_increment > 0.0 and len(msg.ranges) > 1:
            span = int((len(msg.ranges) - 1) * msg.time_increment * 1e9)
        if msg.scan_time > 0.0:
            span = max(span, int(msg.scan_time * 1e9))
        return span or DEFAULT_PERIOD_NS

    def relay(self, msg: LaserScan):
        span_ns = self._span_ns(msg)
        # Host clock at receipt, back-dated by one span so the stamp
        # approximates the first ray (LaserScan convention), not scan end.
        arrival_ns = self.get_clock().now().nanoseconds - span_ns

        if self._last_stamp_ns is None:
            stamp_ns = arrival_ns
        else:
            earliest = self._last_stamp_ns + span_ns
            if arrival_ns < earliest:
                # Jitter, not a real gap: keep the true cadence rather than
                # letting this node's scheduling squeeze two scans together.
                stamp_ns = earliest
                self._crowded += 1
            elif arrival_ns - earliest > RESYNC_NS:
                # Scans were genuinely lost, or the clock stepped. Re-anchor
                # rather than paying the gap back one span at a time.
                stamp_ns = arrival_ns
            else:
                stamp_ns = arrival_ns

        self._last_stamp_ns = stamp_ns
        msg.header.stamp.sec = stamp_ns // 1_000_000_000
        msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
        self.pub.publish(msg)

    def _report(self):
        """Say how often arrivals were too crowded to use directly.

        A steady count means the Pi is jittery enough that receipt time was
        never a safe stamp -- which is the whole reason this spacing exists.
        """
        if self._crowded:
            self.get_logger().info(
                f'{self._crowded} scans in the last minute arrived closer '
                f'than one scan span; spaced them instead')
            self._crowded = 0


def main(args=None):
    rclpy.init(args=args)
    node = ScanTimestampRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
