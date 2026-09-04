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
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


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
        # Once a second: enough for a battery readout, which is all that
        # consumes this now. It was 20 Hz while Cartographer was fusing the
        # IMU; with use_imu_data back to false nothing reads /imu/data, and
        # polling the board twenty times a second for it is serial traffic
        # and CPU spent on a topic with no subscriber. The Imu message is
        # still published, dormant, if fusion is ever tried again.
        self.declare_parameter('telemetry_period', 1.0)
        # Must be Cartographer's tracking_frame, which wave_rover.lua sets to
        # base_laser. Cartographer refuses anything else outright:
        #
        #   Check failed: sensor_to_tracking->translation().norm() < 1e-5
        #   The IMU frame must be colocated with the tracking frame.
        #
        # It will not compensate a lever arm when rotating acceleration into
        # the tracking frame, so it makes you declare the sensor colocated
        # rather than silently accepting an offset.
        #
        # Declaring it is honest enough for what Cartographer 2D uses the IMU
        # for. Angular velocity is identical at every point of a rigid body,
        # so the gyro -- which is what drives yaw -- reads the same wherever
        # the board is bolted. Only linear acceleration gains a lever-arm
        # term, and only while turning: 0.5 rad/s with 10cm of offset is
        # 0.025 m/s^2 against 9.8 of gravity, a quarter of a percent on the
        # gravity direction.
        self.declare_parameter('imu_frame', 'base_laser')

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
        self.imu_frame = g('imu_frame').value
        self.ticks = 0
        self.last_file_write = 0.0

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

        # The board reports orientation, rates and accelerations in the same
        # line it reports the battery. Cartographer had neither an IMU nor
        # odometry, which left its pose extrapolator with nothing but scan
        # matching to predict motion -- and a pose that walked off the map
        # whenever scan matching was momentarily starved. This is the sensor
        # that was there all along, on a topic nobody published.
        self.imu_pub = self.create_publisher(
            Imu, 'imu/data', qos_profile_sensor_data)

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

    # Board units, confirmed against a level stationary rover: az reads
    # ~1000.85 with the rover flat, so accelerations are milli-g; gyro rates
    # sit under 1.0 at rest, which is degrees per second rather than radians
    # (1 rad/s would be 57 deg/s of noise at standstill); r/p/y are degrees.
    MG_TO_MS2 = 9.80665 / 1000.0
    DEG_TO_RAD = math.pi / 180.0
    FILE_PERIOD_S = 1.0          # battery file: no point rewriting at 20 Hz

    def _publish_imu(self, obj):
        """Publish one Imu message from a board feedback line.

        Cartographer's 2D use_imu_data reads angular_velocity and
        linear_acceleration -- gravity for the alignment, the z rate for yaw.
        Orientation is published because the board offers it, but nothing
        here depends on it, and on a part this cheap the yaw half drifts.
        """
        try:
            ax, ay, az = float(obj['ax']), float(obj['ay']), float(obj['az'])
            gx, gy, gz = float(obj['gx']), float(obj['gy']), float(obj['gz'])
        except (KeyError, TypeError, ValueError):
            return                       # not a feedback line

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.imu_frame

        msg.linear_acceleration.x = ax * self.MG_TO_MS2
        msg.linear_acceleration.y = ay * self.MG_TO_MS2
        msg.linear_acceleration.z = az * self.MG_TO_MS2
        msg.angular_velocity.x = gx * self.DEG_TO_RAD
        msg.angular_velocity.y = gy * self.DEG_TO_RAD
        msg.angular_velocity.z = gz * self.DEG_TO_RAD

        try:
            roll = float(obj['r']) * self.DEG_TO_RAD
            pitch = float(obj['p']) * self.DEG_TO_RAD
            yaw = float(obj['y']) * self.DEG_TO_RAD
        except (KeyError, TypeError, ValueError):
            # -1 in the first slot is the Imu contract for "no orientation".
            msg.orientation_covariance[0] = -1.0
        else:
            cr, sr = math.cos(roll / 2), math.sin(roll / 2)
            cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
            cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
            msg.orientation.w = cr * cp * cy + sr * sp * sy
            msg.orientation.x = sr * cp * cy - cr * sp * sy
            msg.orientation.y = cr * sp * cy + sr * cp * sy
            msg.orientation.z = cr * cp * sy - sr * sp * cy
            msg.orientation_covariance[0] = 0.05
            msg.orientation_covariance[4] = 0.05
            msg.orientation_covariance[8] = 0.5      # yaw drifts; say so

        # Loose but not meaningless: a small MEMS part read over a 115200
        # line, not a survey instrument.
        for i in (0, 4, 8):
            msg.angular_velocity_covariance[i] = 0.01
            msg.linear_acceleration_covariance[i] = 0.05
        self.imu_pub.publish(msg)

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
            self._publish_imu(obj)
            for key in self.VOLTAGE_KEYS:
                if key not in obj:
                    continue
                try:
                    volts = float(obj[key])
                except (TypeError, ValueError):
                    continue
                if self.V_MIN <= volts <= self.V_MAX:
                    now = time.time()
                    if now - self.last_file_write >= self.FILE_PERIOD_S:
                        self.last_file_write = now
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
