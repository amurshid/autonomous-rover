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
the host clock when the host clock is further ahead than that.

Bounded, though
---------------
Spacing alone ratchets. Under a burst -- Nav2 launching five nodes, say --
callbacks bunch up, each scan still advances the stamp by a full span, and
wall clock barely moves. The series ends up ahead of real time and stays
there, because max() only goes one way. Measured: tf_age stepped from 0.01
to -0.15 the instant Nav2 started, stayed pinned, and Cartographer pose left
the map within seconds. Restarting the relay cleared it, which is the tell --
the state was in this node, not in Cartographer.

So the stamp is also never allowed past the host clock -- and when spacing
and that ceiling disagree, the scan is dropped rather than published at a
stamp that overlaps its predecessor.

Dropping beats overlapping, which is not obvious
------------------------------------------------
The first version of this cap published the overlapping scan and let
Cartographer discard the overlap, on the reasoning that a few lost points is
cheaper than an unbounded lead. That was wrong, and the log said so:

  Node (2, 26) with 7 points on submap (0, 27) differs by
  translation 8.69 rotation 0.483 with score 90.0%

A scan cut down to seven points matches at ninety percent almost anywhere.
Cartographer fed those constraints into the pose graph and the trajectory was
dragged metres across the house. Full scans carry ~200 points and constrain
honestly; a remnant constrains confidently and wrongly, which is worse than
no constraint at all.

Cartographer handles a missing scan without complaint -- it simply has less
data for that instant. So when the choice is between a degraded scan and no
scan, take no scan. Skipping also sheds the lead on its own: the stamp does
not advance while wall clock does, so the next scan has room again.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

DEFAULT_PERIOD_NS = 100_000_000      # 10 Hz, if the message does not say

# A scan carrying far fewer returns than usual is worse than no scan at all.
# Cartographer builds a node from it, and the constraint builder then matches
# that handful of points against the map at 90%, metres from where the rover
# is. Observed on a stationary rover:
#
#   Node (2, 13) with 1 points on submap (2, 1)
#   differs by translation 4.48 rotation 0.117 with score 90.0%
#
# That constraint moved the trajectory 7m, and the node went on offering
# false matches for minutes afterwards. Healthy scans here carry ~200 returns
# and agree to within 0.01m.
SPARSE_RATIO = 0.4       # of the running average, below which a scan is junk
BASELINE_ALPHA = 0.05    # how fast that average tracks; slow, so a run of bad
                         # scans cannot teach it to accept them


class ScanTimestampRelay(Node):
    def __init__(self):
        super().__init__('scan_timestamp_relay')

        self.declare_parameter('input_topic', '/scan')
        self.declare_parameter('output_topic', '/scan_fixed')
        in_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self._last_stamp_ns = None
        self._crowded = 0            # scans that arrived closer than one span
        self._skipped = 0            # scans dropped rather than overlapped
        self._sparse = 0             # scans dropped for too few returns
        self._baseline = None        # running average of valid returns

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

    def _too_sparse(self, msg: LaserScan):
        """True if this scan carries far fewer returns than usual.

        The threshold rides a slow average of what this lidar normally
        produces rather than a fixed count, so it adapts to a room with
        little in range without being told. Dropped scans do not update the
        average, or a run of bad ones would teach it to accept them.
        """
        valid = 0
        for r in msg.ranges:
            if msg.range_min <= r <= msg.range_max and math.isfinite(r):
                valid += 1

        if self._baseline is None:
            self._baseline = float(valid)
            return False                    # nothing to compare against yet
        if valid < self._baseline * SPARSE_RATIO:
            self._sparse += 1
            return True
        self._baseline += BASELINE_ALPHA * (valid - self._baseline)
        return False

    def relay(self, msg: LaserScan):
        if self._too_sparse(msg):
            return
        span_ns = self._span_ns(msg)
        # Host clock at receipt, back-dated by one span so the stamp
        # approximates the first ray (LaserScan convention), not scan end.
        arrival_ns = self.get_clock().now().nanoseconds - span_ns

        if self._last_stamp_ns is None:
            stamp_ns = arrival_ns
        else:
            earliest = self._last_stamp_ns + span_ns
            # arrival_ns is already now - span, so this ceiling is 'now'.
            ceiling_ns = arrival_ns + span_ns

            if arrival_ns >= earliest:
                # Room to spare: track the host clock.
                stamp_ns = arrival_ns
            elif earliest <= ceiling_ns:
                # Jitter, not a real gap. Keep the true cadence rather than
                # letting this node's scheduling squeeze two scans together.
                stamp_ns = earliest
                self._crowded += 1
            else:
                # Spacing would put the stamp in the future. Publishing it
                # early instead would overlap the previous scan, and
                # Cartographer would keep only the remnant -- see above.
                self._skipped += 1
                return

        self._last_stamp_ns = stamp_ns
        msg.header.stamp.sec = stamp_ns // 1_000_000_000
        msg.header.stamp.nanosec = stamp_ns % 1_000_000_000
        self.pub.publish(msg)

    def _report(self):
        """Say how often arrivals were too crowded to use directly.

        A steady count means the Pi is jittery enough that receipt time was
        never a safe stamp -- which is the whole reason this spacing exists.
        """
        if self._crowded or self._skipped or self._sparse:
            base = f'{self._baseline:.0f}' if self._baseline else '?'
            self.get_logger().info(
                f'last minute: {self._crowded} spaced, {self._skipped} dropped '
                f'rather than overlapped, {self._sparse} dropped as sparse '
                f'(usual return count {base})')
            self._crowded = self._skipped = self._sparse = 0


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
