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

import json
import math
import os
import signal
import sys
import threading

import rclpy
from action_msgs.msg import GoalStatus, GoalStatusArray
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

try:      # the QoS an action server publishes its status with
    from rclpy.action.qos import qos_profile_action_status_default as STATUS_QOS
except ImportError:
    from rclpy.qos import (QoSDurabilityPolicy, QoSProfile,
                           QoSReliabilityPolicy)
    STATUS_QOS = QoSProfile(
        depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

sys.path.insert(0, os.path.expanduser("~"))
from rooms import ROOMS, resolve_room  # noqa: E402


class RoverNav(Node):
    def __init__(self, on_done=None, track_pose=True):
        super().__init__("rover_nav")
        self.cb = ReentrantCallbackGroup()
        self.on_done = on_done
        self.last_outcome = None

        self._client = ActionClient(
            self, NavigateToPose, "navigate_to_pose", callback_group=self.cb
        )

        # Cartographer publishes this at ~192 Hz, and 192 rclpy callbacks a
        # second is most of a core on this Pi -- measured at 86% in the goal
        # sender, which starved Nav2 badly enough that bt_navigator could not
        # hold its tick rate and goals failed. Only callers that actually read
        # the pose should pay for it: pose() and nearest_room() are the only
        # readers, and neither is used when sending a goal from the CLI.
        #
        # The callback is a bare assignment on purpose -- no TF lookups, no
        # real work.
        self.tracking_pose = track_pose
        if track_pose:
            self.create_subscription(
                PoseStamped, "/tracked_pose", self._pose_cb, 10,
                callback_group=self.cb
            )

        # Goals sent by another process are invisible in _target: the mode
        # page's room buttons run rover_nav.py as their own child, so rover_ai
        # saw is_navigating() False while Nav2 was driving. Both then published
        # /cmd_vel and the rover shook. Nav2's own status topic is the one
        # source that does not care which process asked.
        self._nav2_active = False
        self.create_subscription(
            GoalStatusArray, "navigate_to_pose/_action/status",
            self._status_cb, STATUS_QOS, callback_group=self.cb)
        # An empty request -- zero uuid, zero stamp -- cancels every goal,
        # including one this process never had a handle for.
        self._cancel_cli = self.create_client(
            CancelGoal, "navigate_to_pose/_action/cancel_goal",
            callback_group=self.cb)

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
        """(x, y, yaw_deg), or None -- also None when track_pose was off."""
        return self._pose

    def is_navigating(self):
        """Is *this process* driving? _await_arrival waits on this, so it must
        not become true for somebody else's goal."""
        with self._lock:
            return self._target is not None

    def _status_cb(self, msg):
        self._nav2_active = any(
            st.status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING)
            for st in msg.status_list)

    def anyone_navigating(self):
        """Is anything driving, whoever asked? Use this before publishing
        /cmd_vel by hand, and to decide whether the rover is listening to its
        own motors. Deliberately not time-limited: status is published on
        transitions, so a long quiet drive must not look like it ended."""
        with self._lock:
            if self._target is not None:
                return True
        return self._nav2_active

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
        key = resolve_room(room)
        if key is None:
            return False, f"unknown room '{room}'"
        room = key

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

    def cancel_any(self):
        """Stop whatever is driving, including a goal from another process.

        "stop" has to mean stop. Without this a spoken stop during a drive
        started from the mode page cancelled nothing, and the single zero
        twist that followed was overridden by Nav2 on its next tick.
        """
        ok, msg = self.cancel()
        if ok or not self._nav2_active:
            return ok, msg
        if not self._cancel_cli.service_is_ready():
            return False, "nav2 cancel service unavailable"
        self._cancel_cli.call_async(CancelGoal.Request())
        return True, "cancelled navigation"

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
            # A caller waiting on is_navigating() only learns that the goal
            # settled, not whether it arrived. A queued sequence has to know:
            # there is no point delivering a message in a room it never
            # reached.
            self.last_outcome = outcome
        # cancel() already cleared _target, so a user-issued stop stays silent.
        if room is None or self.on_done is None:
            return
        try:
            self.on_done(room, outcome, detail)
        except Exception as e:  # never let a callback kill the executor thread
            self.get_logger().error(f"on_done callback raised: {e}")


def main():
    """Send one goal, report on it, exit 0 only if the rover arrived.

        python3 rover_nav.py kitchen            # for a person
        python3 rover_nav.py --json kitchen     # one JSON object per line

    --json is for callers that are not ROS nodes. rover_mode_web.py is one: it
    runs on port 80 in both modes and must keep working with no ROS on its
    path, so it spawns this and reads the stream rather than importing rclpy.

    Events are {"event": "sent"|"feedback"|"done", ...}. SIGTERM and Ctrl-C
    both cancel the goal before exiting -- a caller that kills this must not
    leave Nav2 driving to a goal nobody is watching any more.
    """
    from rclpy.executors import MultiThreadedExecutor

    flags = [a for a in sys.argv[1:] if a.startswith("-")]
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    as_json = "--json" in flags
    room = args[0] if args else "kitchen"

    done = threading.Event()
    settled = {"outcome": "cancelled", "detail": ""}

    def emit(event, **kw):
        if as_json:
            print(json.dumps({"event": event, **kw}), flush=True)
        elif event == "feedback":
            d = kw["remaining"]
            print(f"  {d:.2f} m remaining" if d is not None else "  ...", flush=True)
        elif event == "sent":
            print(kw["detail"], flush=True)
        else:
            print(f"\n[{kw['outcome']}] {kw['room']} {kw['detail']}".rstrip(), flush=True)

    def report(r, outcome, detail):
        settled.update(outcome=outcome, detail=detail)
        emit("done", room=r, outcome=outcome, detail=detail)
        done.set()

    rclpy.init()
    # No pose subscription: this path sends a goal and reports on it. Progress
    # comes from Nav2's own action feedback, not from /tracked_pose, and
    # nothing here calls pose() or nearest_room().
    node = RoverNav(on_done=report, track_pose=False)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    # rclpy installs its own SIGTERM handler, which shuts the context down
    # without unwinding this function -- the goal would be left running and
    # the loop below would spin on a dead context. Take the signal back so
    # SIGTERM means what Ctrl-C means here: cancel, then go.
    def sigterm(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, sigterm)

    ok, msg = node.go_to_room(room)
    if ok:
        emit("sent", room=room, detail=msg)
    else:
        report(room, "failed", msg)

    try:
        while rclpy.ok() and not done.wait(1.0):
            emit("feedback", room=room, remaining=node.distance_remaining())
    except KeyboardInterrupt:
        node.cancel()
        emit("done", room=room, outcome="cancelled", detail="")
        settled.update(outcome="cancelled", detail="")
        # cancel_goal_async only queues the request. Exiting on top of it
        # leaves Nav2 driving, so hold the executor open long enough for the
        # cancel to actually go out.
        threading.Event().wait(1.5)
    finally:
        # Stop the executor before the node it is spinning, or the C++ layer
        # aborts as the node is destroyed beneath it.
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0 if settled["outcome"] == "arrived" else 1


if __name__ == "__main__":
    sys.exit(main())
