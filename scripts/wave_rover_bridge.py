#!/usr/bin/env python3
import json
import math
import os
import tempfile
import threading
import time
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
        self.declare_parameter('telemetry_period', 1.0)

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
        # The board answers, it does not volunteer: nothing arrives on the
        # port until {"T":130} asks for it. Counted in ticks so the request
        # goes out from the same thread as the motor writes.
        self.telemetry_every = max(
            1, int(round(g('telemetry_period').value * self.tick_hz)))
        self.ticks = 0

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

        # The board reports its battery voltage on the same line it reports
        # everything else. Read it on its own thread and leave it in a file
        # the web pages can stat: the mode page has no ROS by design, and
        # nothing else may open this port -- two readers would split the
        # stream between them.
        self.telemetry_stop = threading.Event()
        self.telemetry = threading.Thread(target=self._read_telemetry,
                                          daemon=True)
        self.telemetry.start()
        self.get_logger().info(
            f'Bridge up on {port} @ {baud} | sep={self.wheel_sep} '
            f'deadband={self.db_straight}/{self.db_spin}')

    # Whatever the firmware calls it. 'v' is what the Waveshare general
    # driver board emits; the rest cost nothing to accept.
    # Confirmed against this board: {"T":1001,...,"temp":56.1,"v":11.38}.
    # 'temp' there is the driver board, not the Pi -- rover_health.py reads
    # the Pi's own thermal zone and they are not interchangeable.
    VOLTAGE_KEYS = ('v', 'V', 'volt', 'voltage', 'bat', 'battery')
    # A 3S pack reads about 9-13 V. Anything outside this is another field
    # that happened to be called v, not the battery.
    V_MIN, V_MAX = 5.0, 30.0

    def _read_telemetry(self):
        """Parse the board's feedback lines for a voltage. Never fatal.

        Reads and writes go to the same fd from different threads, which the
        OS serialises; nothing here touches the motor state or the write
        path, so the worst case for a firmware that says nothing useful is a
        thread parked in readline() and a battery that reads unknown.
        """
        while not self.telemetry_stop.is_set():
            try:
                raw = self.ser.readline()
            except Exception as e:
                self.get_logger().warn(f'telemetry read stopped: {e}')
                return
            if not raw:
                continue
            try:
                obj = json.loads(raw.decode('utf-8', 'replace').strip())
            except ValueError:
                continue                      # not every line is JSON
            if not isinstance(obj, dict):
                continue
            for key in self.VOLTAGE_KEYS:
                if key not in obj:
                    continue
                try:
                    volts = float(obj[key])
                except (TypeError, ValueError):
                    continue
                if self.V_MIN <= volts <= self.V_MAX:
                    self._write_telemetry(volts)
                break

    def _write_telemetry(self, volts):
        """Replace the file atomically, so a reader never sees half of it."""
        path = os.environ.get('ROVER_TELEMETRY', '/run/rover/telemetry.json')
        payload = json.dumps({'t': time.time(), 'volts': round(volts, 2)})
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
            with os.fdopen(fd, 'w') as f:
                f.write(payload)
            os.chmod(tmp, 0o644)              # both web pages read this
            os.replace(tmp, path)
        except OSError as e:
            if not getattr(self, '_telemetry_warned', False):
                self._telemetry_warned = True
                self.get_logger().warn(f'cannot write {path}: {e}')

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

        # A safety valve for a telemetry thread that has died: with one
        # running the buffer never gets near this, and flushing mid-line
        # would corrupt the read it is in the middle of.
        if self.ser.in_waiting > 4096:
            self.ser.reset_input_buffer()

        self.send(l, r)

        self.ticks += 1
        if self.ticks % self.telemetry_every == 0:
            self.ask_for_telemetry()

    def send(self, l, r):
        payload = json.dumps({"T": 1, "L": round(l, 3), "R": round(r, 3)})
        try:
            self.ser.write((payload + '\n').encode('utf-8'))
        except serial.SerialException as e:
            self.get_logger().error(f'Serial write failed: {e}')

    def ask_for_telemetry(self):
        """Request one feedback line. The reply arrives on the reader thread."""
        try:
            self.ser.write((json.dumps({"T": 130}) + '\n').encode('utf-8'))
        except serial.SerialException as e:
            if not getattr(self, '_poll_warned', False):
                self._poll_warned = True
                self.get_logger().warn(f'telemetry poll failed: {e}')

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
