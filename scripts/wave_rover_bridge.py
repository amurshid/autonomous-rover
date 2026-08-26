#!/usr/bin/env python3
import json
import math
import threading
import serial
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from geometry_msgs.msg import Twist


class WaveRoverBridge(Node):
    def __init__(self):
        super().__init__('wave_rover_bridge')

        self.declare_parameter('serial_port', '/dev/serial0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('wheel_separation', 0.15)
        self.declare_parameter('max_wheel_speed', 1.25)
        self.declare_parameter('command_timeout', 0.5)
        self.declare_parameter('straight_deadband', 0.05)
        self.declare_parameter('spin_deadband', 0.18)
        self.declare_parameter('pulse_ticks', 6)
        self.declare_parameter('tick_hz', 20.0)
        self.declare_parameter('left_trim', 1.0)
        self.declare_parameter('right_trim', 1.0)

        g = self.get_parameter
        port = g('serial_port').value
        baud = g('baud_rate').value
        self.wheel_sep = g('wheel_separation').value
        self.max_speed = g('max_wheel_speed').value
        self.timeout = g('command_timeout').value
        self.db_straight = g('straight_deadband').value
        self.db_spin = g('spin_deadband').value
        self.pulse_ticks = max(1, int(g('pulse_ticks').value))
        self.tick_hz = g('tick_hz').value
        self.acc = {'l': 0.0, 'r': 0.0}
        self.hold = {'l': 0, 'r': 0}
        self.pulse_ticks = max(1, int(g('pulse_ticks').value))
        self.tick_hz = g('tick_hz').value
        self.acc = {'l': 0.0, 'r': 0.0}
        self.hold = {'l': 0, 'r': 0}
        self.left_trim = g('left_trim').value
        self.right_trim = g('right_trim').value

        self.lock = threading.Lock()
        self.left = 0.0
        self.right = 0.0
        self.floor = 0.0
        self.last_cmd = self.get_clock().now()

        try:
            self.ser = serial.Serial(port, baud, timeout=1.0)
        except serial.SerialException as e:
            self.get_logger().error(f'Cannot open {port}: {e}')
            raise SystemExit(1)

        self.create_subscription(Twist, 'cmd_vel', self.cmd_cb, 10)
        self.create_timer(1.0 / self.tick_hz, self.tick)
        self.get_logger().info(
            f'Bridge up on {port} @ {baud} | sep={self.wheel_sep} '
            f'deadband={self.db_straight}/{self.db_spin}')

    def cmd_cb(self, msg):
        v = msg.linear.x
        w = msg.angular.z

        vl = v - (w * self.wheel_sep / 2.0)
        vr = v + (w * self.wheel_sep / 2.0)

        l = vl / self.max_speed * 0.5
        r = vr / self.max_speed * 0.5

        l *= self.left_trim
        r *= self.right_trim

        peak = max(abs(l), abs(r))
        if peak > 0.5:
            l *= 0.5 / peak
            r *= 0.5 / peak

        floor = self.deadband_for(l, r)

        with self.lock:
            self.left = l
            self.right = r
            self.floor = floor
            self.last_cmd = self.get_clock().now()

    def deadband_for(self, l, r):
        translate = abs(l + r) / 2.0
        rotate = abs(r - l) / 2.0
        total = translate + rotate
        if total < 1e-6:
            return self.db_straight
        scrub = rotate / total
        return self.db_straight + scrub * (self.db_spin - self.db_straight)

    def dither(self, x, floor, k):
        if x == 0.0:
            self.acc[k] = 0.0
            self.hold[k] = 0
            return 0.0
        if abs(x) >= floor:
            self.acc[k] = 0.0
            self.hold[k] = 0
            return x
        if self.hold[k] > 0:
            self.hold[k] -= 1
            return math.copysign(floor, x)
        self.acc[k] += (abs(x) / floor) / self.pulse_ticks
        if self.acc[k] >= 1.0:
            self.acc[k] -= 1.0
            self.hold[k] = self.pulse_ticks - 1
            return math.copysign(floor, x)
        return 0.0

    def tick(self):
        with self.lock:
            age = (self.get_clock().now() - self.last_cmd).nanoseconds / 1e9
            if age > self.timeout:
                self.left = 0.0
                self.right = 0.0
            l = self.dither(self.left, self.floor, 'l')
            r = self.dither(self.right, self.floor, 'r')

        if self.ser.in_waiting > 4096:
            self.ser.reset_input_buffer()

        self.send(l, r)

    def send(self, l, r):
        payload = json.dumps({"T": 1, "L": round(l, 3), "R": round(r, 3)})
        try:
            self.ser.write((payload + '\n').encode('utf-8'))
        except serial.SerialException as e:
            self.get_logger().error(f'Serial write failed: {e}')

    def destroy_node(self):
        try:
            self.send(0.0, 0.0)
            self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = WaveRoverBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        # rclpy's SIGTERM handler has already shut the context down by
        # the time we get here, and calling it twice raises RCLError --
        # which exits 1 and makes systemd record a normal stop as a
        # failure.
        if rclpy.ok():
            rclpy.shutdown()



if __name__ == '__main__':
    main()
