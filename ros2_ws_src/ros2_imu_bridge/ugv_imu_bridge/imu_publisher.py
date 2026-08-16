#!/usr/bin/env python3
"""
UGV IMU bridge: reads the ESP32 (Waveshare General Driver) continuous feedback
stream over UART and publishes sensor_msgs/Imu on /imu/data for Cartographer.

Firmware notes (verified against ugv_base_general):
  - Enable continuous stream:  {"T":131,"cmd":1}
  - Stream lines are tagged:    "T":1001  (FEEDBACK_BASE_INFO)
  - Units on the wire:          accel ax/ay/az = milli-g (mg)
                                gyro  gx/gy/gz = degrees/sec (dps)
                                r/p/y          = degrees (orientation; Cartographer ignores it)
"""

import math
import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import serial

MG_TO_MS2 = 9.80665 / 1000.0   # milli-g -> m/s^2
DPS_TO_RADS = math.pi / 180.0   # deg/s   -> rad/s

FEEDBACK_BASE_INFO = 1001       # T tag of the continuous stream
CMD_ENABLE_STREAM = '{"T":131,"cmd":1}\n'


class ImuBridge(Node):
    def __init__(self):
        super().__init__('ugv_imu_bridge')

        # --- parameters (override with --ros-args -p port:=/dev/ttyUSB0 etc.) ---
        self.declare_parameter('port', '/dev/ttyAMA0')
        self.declare_parameter('baud', 115200)
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('publish_orientation', False)  # Cartographer ignores it

        port = self.get_parameter('port').get_parameter_value().string_value
        baud = self.get_parameter('baud').get_parameter_value().integer_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.publish_orientation = self.get_parameter(
            'publish_orientation').get_parameter_value().bool_value

        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.get_logger().info(f'Opened {port} @ {baud}')

        # Ask the ESP32 to start streaming. Send a few times in case early
        # bytes are missed while the link settles.
        for _ in range(3):
            self.ser.write(CMD_ENABLE_STREAM.encode())
        self.get_logger().info('Sent enable-stream command {"T":131,"cmd":1}')

        self.pub = self.create_publisher(Imu, '/imu/data', 10)

        # Poll the serial buffer faster than the ~100 Hz stream so we never lag.
        self.timer = self.create_timer(0.002, self.poll_serial)

    def poll_serial(self):
        # Drain whatever complete lines are waiting.
        while self.ser.in_waiting:
            raw = self.ser.readline()
            if not raw:
                return
            try:
                line = raw.decode('utf-8', errors='ignore').strip()
            except Exception:
                continue
            if not line or not line.startswith('{'):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # partial/garbled line; skip
            if data.get('T') != FEEDBACK_BASE_INFO:
                continue
            self.publish_imu(data)

    def publish_imu(self, d):
        # The stream only carries accel/gyro if the firmware patch is flashed.
        if 'ax' not in d or 'gx' not in d:
            self.get_logger().warn_once(
                'Stream has no ax/gx fields — flash the patched firmware '
                '(accel/gyro added to baseInfoFeedback).')
            return

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        # mg -> m/s^2
        msg.linear_acceleration.x = float(d['ax']) * MG_TO_MS2
        msg.linear_acceleration.y = float(d['ay']) * MG_TO_MS2
        msg.linear_acceleration.z = float(d['az']) * MG_TO_MS2

        # dps -> rad/s
        msg.angular_velocity.x = float(d['gx']) * DPS_TO_RADS
        msg.angular_velocity.y = float(d['gy']) * DPS_TO_RADS
        msg.angular_velocity.z = float(d['gz']) * DPS_TO_RADS

        if self.publish_orientation and all(k in d for k in ('r', 'p', 'y')):
            roll = float(d['r']) * DPS_TO_RADS   # degrees -> radians
            pitch = float(d['p']) * DPS_TO_RADS
            yaw = float(d['y']) * DPS_TO_RADS
            qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)
            msg.orientation.x = qx
            msg.orientation.y = qy
            msg.orientation.z = qz
            msg.orientation.w = qw
        else:
            # Tell consumers orientation is unknown (Cartographer wants this).
            msg.orientation_covariance[0] = -1.0

        self.pub.publish(msg)


def euler_to_quaternion(roll, pitch, yaw):
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def main(args=None):
    rclpy.init(args=args)
    node = ImuBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.ser.write(b'{"T":131,"cmd":0}\n')  # stop stream on exit
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
