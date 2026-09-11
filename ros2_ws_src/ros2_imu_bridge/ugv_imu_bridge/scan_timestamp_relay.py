#!/usr/bin/env python3
"""
Scan timestamp relay for the LDROBOT LD19 (ldlidar_stl_ros2).

Re-times each /scan and republishes it on /scan_fixed, which Cartographer and
both Nav2 costmaps consume.

What the driver actually puts in a LaserScan
--------------------------------------------
ldlidar_stl_ros2 (src/demo.cpp) assembles one full revolution per frame, then
publishes from a 10 Hz WallRate loop that is not synchronised to the lidar:

    header.stamp   = node->now() at publish     -- after the revolution ended
    scan_time      = now - the previous publish -- the loop interval
    time_increment = scan_time / (beams - 1)

So scan_time is the gap between publishes, not how long the revolution took.
When the loop wakes before the next frame is ready it publishes nothing, and
the following message claims to span ~200 ms while holding one revolution.
Under load the loop slips and the claim wanders from ~75 to ~250 ms.

Cartographer believes those numbers. It times each ray as stamp + i *
time_increment and discards any ray earlier than the previous scan's last ray.
A "200 ms" scan followed by a normal one 100 ms later costs nearly the whole
second scan. Measured on the rover: 12 such scans in 45 s, drops of 404-468
points, matching Cartographer's own log point for point. The remnant is still
scan matched, and with no odometry and no IMU that is enough to send the pose
extrapolator off the map -- which is how a Nav2 goal failed on a parked rover.

Neither earlier relay handled it. Stamping at receipt ignored the span, and
spacing by the *current* scan's span still overlapped a long previous one
(while also stamping the last rays up to a span in the future).

What this does
--------------
1. Every scan is labelled as spanning just under the revolution period P --
   the running median of the driver's scan_time, ~100 ms. The data really is
   one revolution; only the label was wrong.
2. The last ray goes at the driver's stamp, capped at receipt, so no ray is
   ever stamped in the future; the first ray one span before it. The revolution
   really ended somewhere in the loop interval before that stamp -- the driver
   throws the per-point times away -- so this is a latency of up to one period,
   the same one the original receipt-time relay had.
3. A scan's first ray must come after the previous scan's last ray. When the
   driver's loop bunched two publishes closer than P, honouring that would put
   the scan in the future, so it is skipped. Cartographer takes a missing scan
   without complaint; a remnant it matches confidently and wrongly.

Scans carrying far fewer returns than usual are skipped too: a node built from
a handful of points matches the map at 90% metres from the truth.
"""

import math
import statistics
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

NOMINAL_PERIOD_NS = 100_000_000   # LD19 at 10 Hz, until the median has samples
PERIOD_HISTORY = 51               # driver scan_time samples in the median
PERIOD_MIN_SAMPLES = 5
# Between one scan's last ray and the next scan's first. Cartographer rounds
# ray times to 100 ns and has rejected a scan for arriving 4 us early.
GAP_NS = 1_000_000
# A driver stamp further than this from receipt is not believed.
CLOCK_TRUST_NS = 500_000_000
# Each scan is labelled as spanning this much of the period. Publishes arrive a
# median period apart with jitter either way, so a span of exactly the period
# leaves no room: the no-overlap rule pushes every early scan up against the
# no-future rule and skips it. Simulated against the driver's loop, 1.0 skipped
# 24% of scans at idle, 0.97 none (and ~7% under heavy load, where shorter
# barely helps). Three percent is a negligible error in when each ray was taken.
SPAN_RATIO = 0.97
# A claimed span this far from the period is counted as mislabelled.
MISLABEL_RATIO = 0.15

SPARSE_RATIO = 0.4       # of the running average, below which a scan is junk
BASELINE_ALPHA = 0.05    # slow, so a run of bad scans cannot teach it to accept them


class Restamper:
    """The timing decision, free of ROS so it can be tested on its own."""

    def __init__(self):
        self._periods = deque(maxlen=PERIOD_HISTORY)
        self._last_end_ns = None
        self.published = 0
        self.mislabelled = 0     # claimed span off the period by > MISLABEL_RATIO
        self.bunched = 0         # skipped: no room without overlap or future

    def period_ns(self):
        if len(self._periods) < PERIOD_MIN_SAMPLES:
            return NOMINAL_PERIOD_NS
        return int(statistics.median(self._periods))

    def place(self, driver_stamp_ns, receipt_ns, scan_time_s):
        """(first_ray_ns, span_ns) for this scan, or None to skip it."""
        if 0.0 < scan_time_s < 1.0:
            self._periods.append(scan_time_s * 1e9)
        period = self.period_ns()
        if scan_time_s > 0.0 and abs(scan_time_s * 1e9 - period) > MISLABEL_RATIO * period:
            self.mislabelled += 1

        end = driver_stamp_ns
        if abs(receipt_ns - end) > CLOCK_TRUST_NS:
            end = receipt_ns
        end = min(end, receipt_ns)

        span = int(period * SPAN_RATIO)
        first = end - span
        if self._last_end_ns is not None and first < self._last_end_ns + GAP_NS:
            first = self._last_end_ns + GAP_NS
            if first + span > receipt_ns:
                self.bunched += 1
                return None

        self._last_end_ns = first + span
        self.published += 1
        return first, span

    def reset_counts(self):
        self.published = self.mislabelled = self.bunched = 0


class ScanTimestampRelay(Node):
    def __init__(self):
        super().__init__('scan_timestamp_relay')

        self.declare_parameter('input_topic', '/scan')
        self.declare_parameter('output_topic', '/scan_fixed')
        in_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.timing = Restamper()
        self._sparse = 0
        self._baseline = None        # running average of valid returns

        # Sensor-data QoS (best-effort) on both ends: the sub accepts the
        # driver's stream regardless of its reliability, and best-effort
        # matches Cartographer's SensorDataQoS scan subscription.
        self.pub = self.create_publisher(LaserScan, out_topic, qos_profile_sensor_data)
        self.sub = self.create_subscription(
            LaserScan, in_topic, self.relay, qos_profile_sensor_data)
        self.create_timer(60.0, self._report)

        self.get_logger().info(
            f'Re-timing {in_topic} -> {out_topic}: one revolution per scan, '
            f'never overlapping, never in the future')

    def _too_sparse(self, msg: LaserScan):
        """True if this scan carries far fewer returns than usual.

        The threshold rides a slow average rather than a fixed count, so a
        room with little in range is fine. Dropped scans do not update it.
        """
        valid = 0
        for r in msg.ranges:
            if msg.range_min <= r <= msg.range_max and math.isfinite(r):
                valid += 1

        if self._baseline is None:
            self._baseline = float(valid)
            return False
        if valid < self._baseline * SPARSE_RATIO:
            self._sparse += 1
            return True
        self._baseline += BASELINE_ALPHA * (valid - self._baseline)
        return False

    def relay(self, msg: LaserScan):
        receipt_ns = self.get_clock().now().nanoseconds
        if self._too_sparse(msg):
            return
        driver_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        placed = self.timing.place(driver_ns, receipt_ns, msg.scan_time)
        if placed is None:
            return
        first_ns, span_ns = placed

        msg.header.stamp.sec = first_ns // 1_000_000_000
        msg.header.stamp.nanosec = first_ns % 1_000_000_000
        msg.scan_time = span_ns / 1e9
        if len(msg.ranges) > 1:
            msg.time_increment = (span_ns / 1e9) / (len(msg.ranges) - 1)
        self.pub.publish(msg)

    def _report(self):
        t = self.timing
        base = f'{self._baseline:.0f}' if self._baseline else '?'
        self.get_logger().info(
            f'last minute: {t.published} published, {t.mislabelled} with a '
            f'mislabelled span, {t.bunched} skipped as bunched, {self._sparse} '
            f'skipped as sparse (usual return count {base}); period '
            f'{t.period_ns() / 1e6:.1f} ms')
        t.reset_counts()
        self._sparse = 0


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
