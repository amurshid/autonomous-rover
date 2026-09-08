#!/usr/bin/env python3
"""Un-warp a LaserScan that was measured while the rover was turning.

The problem
-----------
The LD19 sweeps for 100 ms to build one scan, and Nav2 rotates in place at up
to 10 rad/s. Point 0 and point 498 of the same message are then measured 57
degrees apart, from what is effectively a different robot. Every consumer
treats a scan as a rigid snapshot from one pose, so all of them are being
handed a room that does not exist:

  * laser_scan_matcher -- "after trimming, only 21 correspondences / ICP
    failed", because a wall point 2 m away moved 2 m between scans, far
    outside its 0.3 m max_correspondence_dist;
  * AMCL -- weights particles by matching the scan against the map, so a
    smeared scan poisons the correction step no matter how good the motion
    model is. This is why fixing odometry heading with the gyro did not help:
    it fixed the prior and left the measurement broken;
  * both costmaps -- paint obstacles along an arc that was never there.

Watched in Foxglove as the scans violently coming off the map walls whenever
the rover rotated to face a new path.

The correction is one float
---------------------------
Point i is measured at t = stamp + i * time_increment, at sensor bearing
angle_min + i * angle_increment. If the sensor turned at rate w through the
sweep, then by the time point i was taken the sensor had rotated w * i *
time_increment, so that point's true bearing in the scan-start frame is

    angle_min + i * angle_increment + w * i * time_increment
  = angle_min + i * (angle_increment + w * time_increment)

The correction is exactly linear in i, so it is a pure resampling: output
bin j (bearing angle_min + j*angle_increment) should hold the range that was
actually measured at that bearing, which came from input index

    i = j * angle_increment / (angle_increment + w * time_increment)
      = j / scale

At 10 rad/s: 0.0002 s * 10 = 0.002 rad per step against an angle_increment of
0.01259, a 16% stretch, which over 499 points is the missing 57 degrees.

Do NOT do this by rewriting angle_increment instead
---------------------------------------------------
That was the first attempt, and it is much cheaper -- one header field, no
touching the ranges. It is also wrong: scaling the increment scales the total
field of view with it, and csm validates that. The scans came back as

    :err: Strange FOV: 6.423294 rad = 368.027613 deg
    :err: icp: ICP failed for some reason.

368 degrees. Every scan rejected the moment the rover turned, which is worse
than no deskewing at all. Resampling keeps angle_increment -- and therefore
the FOV -- exactly as the driver produced it.

Bias does not matter here
-------------------------
The gyro rests at 0.075 rad/s. Left uncorrected that is 0.075 * 0.1 = 0.0075
rad, 0.43 degrees of false skew across a whole scan -- under the noise, and
three orders below what this exists to remove. So the bias is a plain
parameter with no live estimation; imu_odom_fusion does that properly because
it integrates, and integration is what makes bias matter.
"""

import math
from collections import deque

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan


class ScanDeskew(Node):

    def __init__(self):
        super().__init__('scan_deskew')

        self.declare_parameter('scan_in', '/scan_fixed')
        self.declare_parameter('scan_out', '/scan_deskewed')
        self.declare_parameter('imu_topic', '/imu/data')
        # Measured parked on this rover. See the note above on why a constant
        # is honest here and a live estimate would be over-engineering.
        self.declare_parameter('gyro_bias_z', 0.075)
        # Which way round the correction goes. NOT derivable from here: it
        # depends on whether the driver publishes points in sweep order or
        # reverses them. The LD19 spins clockwise yet publishes a positive
        # angle_increment, which means it is probably reversing the array --
        # and then point i was measured at (n-1-i)*time_increment rather
        # than i*time_increment, which flips this.
        #
        # The wrong value does not error. It doubles the smear instead of
        # removing it, consistently, so consecutive scans still match each
        # other and csm stays quiet while every scan is wrong against the
        # map. A silent log is not evidence this is right.
        #
        # +1 was tried first and left the pose coming off the walls with no
        # complaint from the matcher, which is the signature above.
        self.declare_parameter('rate_sign', -1.0)
        # Below this the correction is smaller than the sensor's own noise,
        # so skip the work and pass the scan through untouched.
        self.declare_parameter('min_rate', 0.05)          # rad/s

        g = self.get_parameter
        self.bias = float(g('gyro_bias_z').value)
        self.sign = float(g('rate_sign').value)
        self.min_rate = float(g('min_rate').value)

        # A couple of seconds of gyro, to average over each scan's window.
        # At 20 Hz and a 100 ms sweep this holds about two samples per scan,
        # which is thin -- it is why the correction assumes a constant rate
        # through the sweep rather than integrating a profile.
        self.gyro = deque(maxlen=64)

        # Publish RELIABLE, subscribe BEST_EFFORT. That pairing cannot
        # mismatch: a reliable publisher serves both reliable and
        # best-effort subscribers, and a best-effort subscriber accepts both
        # kinds of publisher. Getting it the other way round is a silent
        # failure -- the consumer simply never receives anything, which
        # looks exactly like a dead sensor.
        self.pub = self.create_publisher(LaserScan, g('scan_out').value, 10)
        self.create_subscription(
            Imu, g('imu_topic').value, self.on_imu, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, g('scan_in').value, self.on_scan,
            qos_profile_sensor_data)

        self.warned = False
        self.get_logger().info(
            f"Deskewing {g('scan_in').value} -> {g('scan_out').value} "
            f"using {g('imu_topic').value} "
            f"(bias {self.bias:+.4f}, sign {self.sign:+.0f})")

    def on_imu(self, msg):
        t = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds / 1e9
        self.gyro.append((t, msg.angular_velocity.z - self.bias))

    def rate_over(self, t0, t1):
        """Mean gyro rate across the sweep, or the nearest sample to it."""
        inside = [w for (t, w) in self.gyro if t0 <= t <= t1]
        if inside:
            return sum(inside) / len(inside)
        if not self.gyro:
            return 0.0
        mid = 0.5 * (t0 + t1)
        return min(self.gyro, key=lambda s: abs(s[0] - mid))[1]

    def on_scan(self, msg):
        n = len(msg.ranges)
        dt = msg.time_increment
        if dt <= 0.0 and msg.scan_time > 0.0 and n > 1:
            dt = msg.scan_time / n           # driver left it unset

        if n < 2 or dt <= 0.0:
            if not self.warned:
                self.warned = True
                self.get_logger().warn(
                    'no usable time_increment or scan_time; passing through')
            self.pub.publish(msg)
            return

        t0 = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds / 1e9
        w = self.rate_over(t0, t0 + max(msg.scan_time, n * dt))

        if abs(w) < self.min_rate:
            self.pub.publish(msg)            # not turning; nothing to fix
            return

        inc = msg.angle_increment
        scale = 1.0 + self.sign * w * dt / inc
        if scale <= 0.0:
            self.pub.publish(msg)            # nonsense rate; do no harm
            return

        # Pull, do not push: walking the OUTPUT bins leaves no gaps when the
        # scan is stretched, where walking the input would scatter the points
        # and leave every Nth bin empty.
        ranges = msg.ranges
        out = [float('inf')] * n
        for j in range(n):
            i = int(j / scale + 0.5)
            if 0 <= i < n:
                out[j] = ranges[i]
        msg.ranges = out

        if len(msg.intensities) == n:
            src = msg.intensities
            oi = [0.0] * n
            for j in range(n):
                i = int(j / scale + 0.5)
                if 0 <= i < n:
                    oi[j] = src[i]
            msg.intensities = oi

        # angle_increment, angle_min and angle_max are all left alone. That
        # is the point: the geometry is corrected in the data, not in the
        # header, so the field of view csm checks stays exactly 360 degrees.
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = ScanDeskew()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
