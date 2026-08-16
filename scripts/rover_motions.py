#!/usr/bin/env python3
import math, threading, time
import rclpy
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger


class Motions(Node):
    def __init__(self):
        super().__init__('rover_motions')
        self.declare_parameter('spin_speed', 6.0)
        self.declare_parameter('drive_speed', 0.5)
        self.declare_parameter('deg_per_sec', 60.0)
        self.declare_parameter('m_per_sec', 0.30)
        self.declare_parameter('max_duration', 30.0)

        g = self.get_parameter
        self.spin_speed = g('spin_speed').value
        self.drive_speed = g('drive_speed').value
        self.deg_per_sec = g('deg_per_sec').value
        self.m_per_sec = g('m_per_sec').value
        self.max_dur = g('max_duration').value

        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.lock = threading.Lock()
        self.abort = threading.Event()
        self.busy = False

        cb = ReentrantCallbackGroup()
        self.create_service(Trigger, 'rover/stop', self.srv_stop, callback_group=cb)
        self.get_logger().info('Motion primitives ready')

    def send(self, lin, ang):
        m = Twist()
        m.linear.x = float(lin)
        m.angular.z = float(ang)
        self.pub.publish(m)

    def run_for(self, lin, ang, seconds):
        seconds = max(0.0, min(self.max_dur, seconds))
        self.abort.clear()
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.abort.is_set():
            self.send(lin, ang)
            time.sleep(0.1)
        self.send(0.0, 0.0)
        time.sleep(0.1)
        self.send(0.0, 0.0)

    def pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
        except Exception:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def closed_spin(self, degrees):
        start = self.pose()
        if start is None:
            return False, 'no localization; cannot spin accurately'
        target = math.radians(degrees)
        ang = math.copysign(self.spin_speed, degrees)
        turned = 0.0
        last = start[2]
        deadline = time.monotonic() + self.max_dur
        self.abort.clear()
        while abs(turned) < abs(target) - math.radians(4.0):
            if self.abort.is_set() or time.monotonic() > deadline:
                break
            self.send(0.0, ang)
            time.sleep(0.05)
            cur = self.pose()
            if cur is None:
                continue
            d = cur[2] - last
            d = math.atan2(math.sin(d), math.cos(d))
            turned += d
            last = cur[2]
        self.send(0.0, 0.0)
        time.sleep(0.1)
        self.send(0.0, 0.0)
        return True, f'turned {math.degrees(turned):.0f} of {degrees:.0f} degrees'

    def closed_drive(self, meters):
        start = self.pose()
        if start is None:
            return False, 'no localization; cannot drive accurately'
        lin = math.copysign(self.drive_speed, meters)
        deadline = time.monotonic() + self.max_dur
        self.abort.clear()
        moved = 0.0
        while moved < abs(meters) - 0.05:
            if self.abort.is_set() or time.monotonic() > deadline:
                break
            self.send(lin, 0.0)
            time.sleep(0.05)
            cur = self.pose()
            if cur is None:
                continue
            moved = math.hypot(cur[0] - start[0], cur[1] - start[1])
        self.send(0.0, 0.0)
        time.sleep(0.1)
        self.send(0.0, 0.0)
        return True, f'moved {moved:.2f} of {abs(meters):.2f} m'

    def do_spin(self, degrees):
        if self.busy:
            return False, 'already moving'
        with self.lock:
            self.busy = True
        try:
            degrees = max(-3600.0, min(3600.0, float(degrees)))
            if self.pose() is not None:
                return self.closed_spin(degrees)
            secs = abs(degrees) / self.deg_per_sec
            self.run_for(0.0, math.copysign(self.spin_speed, degrees), secs)
            return True, f'spun ~{degrees:.0f} degrees (timed, no localization)'
        finally:
            self.busy = False

    def do_drive(self, meters):
        if self.busy:
            return False, 'already moving'
        with self.lock:
            self.busy = True
        try:
            meters = max(-5.0, min(5.0, float(meters)))
            if self.pose() is not None:
                return self.closed_drive(meters)
            secs = abs(meters) / self.m_per_sec
            self.run_for(math.copysign(self.drive_speed, meters), 0.0, secs)
            return True, f'drove ~{meters:.2f} m (timed, no localization)'
        finally:
            self.busy = False

    def do_stop(self):
        self.abort.set()
        for _ in range(5):
            self.send(0.0, 0.0)
            time.sleep(0.05)
        return True, 'stopped'

    def srv_stop(self, req, res):
        res.success, res.message = self.do_stop()
        return res


def main():
    rclpy.init()
    node = Motions()
    ex = rclpy.executors.MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.send(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
