#!/usr/bin/env python3
"""Non-blocking Nav2 goal sender for the rover.

Design notes
------------
* Sending a goal returns immediately. Navigation takes minutes; blocking the
  caller would freeze the LLM loop and make "stop" impossible to issue.
* Arrival / failure is reported through the `on_done` callback, which fires on
  an executor thread. Whatever you pass must be thread-safe.
* Everything lives in a ReentrantCallbackGroup so action futures can resolve
  while other callbacks are in flight. This matters here: the project already
  hit executor starvation once (/tracked_pose at 192 Hz), so this node must be
  run under a MultiThreadedExecutor.
"""

import math
import os
import sys
import threading

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

sys.path.insert(0, os.path.expanduser("~"))
from rooms import ROOMS  # noqa: E402


class RoverNav(Node):
    def __init__(self, on_done=None):
        super().__init__("rover_nav")
        self.cb = ReentrantCallbackGroup()
        self.on_done = on_done

        self._client = ActionClient(
            self, NavigateToPose, "navigate_to_pose", callback_group=self.cb
        )

        # Cartographer publishes this at ~192 Hz. The callback below is a bare
        # assignment on purpose -- do not do TF lookups or any real work here.
        self.create_subscription(
            PoseStamped, "/tracked_pose", self._pose_cb, 10, callback_group=self.cb
        )

        self._lock = threading.Lock()
        self._pose = None            # (x, y, yaw_deg)
        self._handle = None          # active goal handle
        self._target = None          # room name currently being driven to
        self._remaining = None       # metres, from Nav2 feedback

    # ---------------------------------------------------------------- state

    def _pose_cb(self, msg):
        q = msg.pose.orientation
        yaw = math.degrees(
            math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                       1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        )
        self._pose = (msg.pose.position.x, msg.pose.position.y, yaw)

    def pose(self):
        return self._pose

    def is_navigating(self):
        with self._lock:
            return self._target is not None

    def target(self):
        with self._lock:
            return self._target

    def distance_remaining(self):
        return self._remaining

    def nearest_room(self):
        """Closest known room goal to the current pose. (name, metres) or (None, None)."""
        p = self._pose
        if p is None:
            return None, None
        px, py = p[0], p[1]
        name = min(ROOMS, key=lambda r: (ROOMS[r][0] - px) ** 2 + (ROOMS[r][1] - py) ** 2)
        return name, math.hypot(ROOMS[name][0] - px, ROOMS[name][1] - py)

    # ------------------------------------------------------------- commands

    def go_to_room(self, room):
        """Fire a NavigateToPose goal. Returns (ok, message) immediately."""
        if room not in ROOMS:
            return False, f"unknown room '{room}'"

        if not self._client.wait_for_server(timeout_sec=3.0):
            return False, "nav2 navigate_to_pose server not available -- is nav2 running?"

        # Replace any goal already in flight rather than stacking them.
        self.cancel()

        x, y, qz, qw = ROOMS[room]
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = float(qz)
        goal.pose.pose.orientation.w = float(qw)

        with self._lock:
            self._target = room
            self._remaining = None

        fut = self._client.send_goal_async(goal, feedback_callback=self._feedback_cb)
        fut.add_done_callback(self._goal_response_cb)
        return True, f"navigating to {room}"

    def cancel(self):
        """Cancel the active goal, if any. Suppresses the arrival announcement."""
        with self._lock:
            handle, self._handle = self._handle, None
            was = self._target
            self._target = None
            self._remaining = None
        if handle is None:
            return False, "not currently navigating"
        handle.cancel_goal_async()
        return True, f"cancelled navigation to {was}"

    # ------------------------------------------------------------ callbacks

    def _feedback_cb(self, msg):
        self._remaining = float(msg.feedback.distance_remaining)

    def _goal_response_cb(self, fut):
        try:
            handle = fut.result()
        except Exception as e:
            self._finish("failed", f"goal send error: {e}")
            return
        if not handle.accepted:
            self._finish("rejected", "nav2 rejected the goal")
            return
        with self._lock:
            self._handle = handle
        handle.get_result_async().add_done_callback(self._result_cb)

    def _result_cb(self, fut):
        try:
            status = fut.result().status
        except Exception as e:
            self._finish("failed", f"result error: {e}")
            return
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._finish("arrived", "")
        elif status == GoalStatus.STATUS_CANCELED:
            self._finish("cancelled", "")
        else:
            self._finish("failed", f"nav2 status {status}")

    def _finish(self, outcome, detail):
        with self._lock:
            room, self._target, self._handle = self._target, None, None
            self._remaining = None
        # cancel() already cleared _target, so a user-issued stop stays silent.
        if room is None or self.on_done is None:
            return
        try:
            self.on_done(room, outcome, detail)
        except Exception as e:  # never let a callback kill the executor thread
            self.get_logger().error(f"on_done callback raised: {e}")


def main():
    """Standalone smoke test: python3 rover_nav.py kitchen"""
    from rclpy.executors import MultiThreadedExecutor

    room = sys.argv[1] if len(sys.argv) > 1 else "kitchen"
    done = threading.Event()

    def report(r, outcome, detail):
        print(f"\n[{outcome}] {r} {detail}")
        done.set()

    rclpy.init()
    node = RoverNav(on_done=report)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    print(node.go_to_room(room)[1])
    try:
        while not done.wait(2.0):
            d = node.distance_remaining()
            print(f"  {d:.2f} m remaining" if d is not None else "  ...")
    except KeyboardInterrupt:
        node.cancel()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
