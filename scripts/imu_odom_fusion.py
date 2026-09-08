#!/usr/bin/env python3
"""Take translation from the scan matcher and heading from the gyro.

Why
---
laser_scan_matcher is good at translation and bad at rotation, and the
badness is not subtle: watched in Foxglove, the pose comes unstuck every time
the rover turns to face a new path, and CSM says so itself --

    before trimming, only 19 correspondences.
    icp: ICP failed for some reason.

The arithmetic behind that: the LD19 sweeps at 10 Hz, and Nav2 rotates in
place at up to 10 rad/s. Two consecutive scans are then 57 degrees apart, and
a wall point 2 m away has moved 2 m between them -- far outside CSM's
max_correspondence_dist of 0.3, so almost nothing pairs up. Slowing the
rotation to the motor deadband floor (6 rad/s, 34 deg/scan) was tried and
changed nothing visible; the hardware cannot turn slower than that smoothly,
so there is no configuration left that fixes it.

A gyro measures rotation directly and does not care how fast it happens. So
this node takes each quantity from the source that is good at it.

Drift is not a problem here
---------------------------
Integrated gyro yaw drifts, and CSM's translation drifts. Neither matters:
AMCL owns map->odom and corrects globally on every update. Odometry only has
to be locally smooth and self-consistent, which is exactly what this is for.

The complementary filter
------------------------
Pure integration would drift without bound, so CSM still gets a vote -- but
only when it is trustworthy, which is when the rover is turning slowly. The
correction gain scales down as rotation speeds up, so:

  * turning fast  -> gyro only, CSM ignored (it is wrong there)
  * slow or still -> CSM slowly pulls the integrated yaw back (it is right there)

That is the whole design: each sensor is believed exactly where it is good.

Bias
----
The gyro reads 0.075 rad/s -- 4.3 deg/s -- sitting perfectly still. Integrated
raw that is 258 degrees of invented rotation per minute, so it must come out.
It is not a constant to hardcode either: MEMS bias walks with temperature, and
this board warms up. So it is re-estimated whenever the rover is known to be
stationary, which is judged from CSM's own reported twist -- a signal that is
reliable precisely when it matters here, since CSM is only unreliable while
turning fast.
"""

import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    """Fold an angle back into (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class ImuOdomFusion(Node):

    def __init__(self):
        super().__init__('imu_odom_fusion')

        self.declare_parameter('odom_in', '/odom_raw')
        self.declare_parameter('odom_out', '/odom')
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('publish_tf', True)

        # Measured on this rover, parked: z hovers at 0.075 rad/s with about
        # +/-0.01 of noise. Seeded rather than assumed zero so the first
        # second after startup is not garbage; it is replaced by the live
        # estimate as soon as the rover is confirmed stationary.
        self.declare_parameter('gyro_bias_z', 0.075)
        # Time constant for the bias estimate, in seconds. Long, because bias
        # walks with temperature over minutes, not milliseconds -- and a fast
        # one would happily absorb a genuine slow turn as though it were bias.
        self.declare_parameter('bias_tau', 30.0)

        # Below these, CSM is taken to mean "not moving", and the bias
        # estimate is allowed to update. Generous enough to cover CSM's own
        # noise at rest, tight enough not to fire during a real turn.
        self.declare_parameter('still_linear', 0.02)      # m/s
        self.declare_parameter('still_angular', 0.03)     # rad/s

        # How hard CSM pulls the integrated yaw back when the rover is
        # rotating slowly, per second. 0.5 recovers a degree of drift in a
        # couple of seconds of straight driving without being able to yank
        # the heading during a manoeuvre.
        self.declare_parameter('yaw_correct_gain', 0.5)
        # Rotation rate at which CSM's yaw stops being believed at all. Above
        # this its scans are smeared and its correspondences are collapsing,
        # so its heading is worse than the gyro's.
        self.declare_parameter('yaw_trust_max_rate', 0.5)  # rad/s

        g = self.get_parameter
        self.odom_frame = g('odom_frame').value
        self.base_frame = g('base_frame').value
        self.publish_tf = g('publish_tf').value
        self.bias = float(g('gyro_bias_z').value)
        self.bias_tau = float(g('bias_tau').value)
        self.still_lin = float(g('still_linear').value)
        self.still_ang = float(g('still_angular').value)
        self.k_yaw = float(g('yaw_correct_gain').value)
        self.trust_max = float(g('yaw_trust_max_rate').value)

        # Fused state, in the odom frame.
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.rate = 0.0                # last bias-corrected gyro z

        # Previous CSM sample, for differencing.
        self.prev_csm = None           # (x, y, yaw)
        self.csm_twist = None          # (vx, vth) from the last odom message
        self.last_imu_t = None

        self.pub = self.create_publisher(Odometry, g('odom_out').value, 10)
        self.tf = TransformBroadcaster(self)

        # The scan matcher publishes /odom RELIABLE, so plain default QoS.
        self.create_subscription(
            Odometry, g('odom_in').value, self.on_odom, 10)
        # The bridge publishes IMU with sensor-data QoS, which is BEST_EFFORT.
        # A default (reliable) subscription here matches nothing at all and
        # receives silence -- which looks exactly like a dead sensor.
        self.create_subscription(
            Imu, g('imu_topic').value, self.on_imu, qos_profile_sensor_data)

        self.get_logger().info(
            f"Fusing {g('odom_in').value} translation with "
            f"{g('imu_topic').value} heading -> {g('odom_out').value} "
            f"(seed bias {self.bias:+.4f} rad/s)")

    # -- gyro ---------------------------------------------------------------

    def on_imu(self, msg):
        """Integrate heading, and publish. This is the fast path, ~20 Hz."""
        t = rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds / 1e9
        if self.last_imu_t is None:
            self.last_imu_t = t
            return
        dt = t - self.last_imu_t
        self.last_imu_t = t
        if not (1e-4 < dt < 0.5):
            return                      # a gap or a clock step; skip it

        raw = msg.angular_velocity.z

        # Re-estimate bias only while CSM says the rover is standing still.
        # Deliberately not "while the gyro reads near zero": that test cannot
        # tell a still rover from one turning at exactly the bias rate, and
        # would quietly learn away a real slow turn.
        if self.is_still():
            alpha = dt / max(self.bias_tau, dt)
            self.bias += alpha * (raw - self.bias)

        self.rate = raw - self.bias
        self.yaw = wrap(self.yaw + self.rate * dt)
        self.publish(msg.header.stamp)

    def is_still(self):
        if self.csm_twist is None:
            return False
        vx, vth = self.csm_twist
        return abs(vx) < self.still_lin and abs(vth) < self.still_ang

    # -- scan matcher -------------------------------------------------------

    def on_odom(self, msg):
        """Take translation from CSM, and let it correct yaw when it can."""
        p = msg.pose.pose.position
        cx, cy = p.x, p.y
        cyaw = yaw_of(msg.pose.pose.orientation)
        self.csm_twist = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)

        if self.prev_csm is None:
            # First sample: adopt CSM's frame wholesale so the two agree at
            # the start and only diverge as the gyro earns it.
            self.x, self.y, self.yaw = cx, cy, cyaw
            self.prev_csm = (cx, cy, cyaw)
            return

        px, py, pyaw = self.prev_csm
        self.prev_csm = (cx, cy, cyaw)

        # CSM's step is expressed in ITS heading, which is not ours any more.
        # Rotate it into the fused frame before accumulating, or the two
        # frames shear apart and the position stops meaning anything.
        dx, dy = cx - px, cy - py
        c, s = math.cos(self.yaw - pyaw), math.sin(self.yaw - pyaw)
        self.x += dx * c - dy * s
        self.y += dx * s + dy * c

        # Let CSM pull the heading back, but only as far as it is credible:
        # full weight when barely turning, nothing at all above trust_max,
        # where its scans are smeared and its correspondences are collapsing.
        trust = 1.0 - min(abs(self.rate) / self.trust_max, 1.0)
        if trust > 0.0:
            err = wrap(cyaw - self.yaw)
            dt = 1.0 / 10.0             # CSM runs at the scan rate
            self.yaw = wrap(self.yaw + self.k_yaw * trust * err * dt)

    # -- output -------------------------------------------------------------

    def publish(self, stamp):
        qz = math.sin(self.yaw / 2.0)
        qw = math.cos(self.yaw / 2.0)

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = self.odom_frame
        msg.child_frame_id = self.base_frame
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        if self.csm_twist is not None:
            msg.twist.twist.linear.x = self.csm_twist[0]
        msg.twist.twist.angular.z = self.rate
        self.pub.publish(msg)

        if not self.publish_tf:
            return
        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw
        self.tf.sendTransform(tf)


def main():
    rclpy.init()
    node = ImuOdomFusion()
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
