#!/usr/bin/env python3
"""Ask rover_ai to drive to a room, and relay what happens.

A drop-in replacement for `rover_nav.py --json <room>` as the child the mode
page spawns. It prints the same event stream, so nothing in rover_mode_web.py
changes except which command it runs:

    {"event": "sent",     "room": "kitchen", "detail": "..."}
    {"event": "feedback", "room": "kitchen", "remaining": 4.02}
    {"event": "done",     "room": "kitchen", "outcome": "arrived", "detail": ""}

The difference is who holds the goal. rover_nav.py sent it from this process,
so rover_ai could not see it: its is_navigating() reports only its own goals,
which left it deaf to a drive in progress -- it answered its own motors and
published /cmd_vel by hand while Nav2 was steering. Sending the request to
rover_ai instead leaves one owner of navigation, and a tapped room takes the
same path as a spoken one.

SIGTERM asks rover_ai to cancel, then waits for the verdict rather than
exiting on the spot: the page unlocks its buttons when this process is reaped,
and unlocking before the cancel has reached Nav2 is how you get a tap that is
silently refused.
"""

import argparse
import json
import signal
import sys
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

WAIT_FOR_AI = 5.0      # rover-ai should already be up; this is for the boot race
CANCEL_GRACE = 4.0     # how long a cancel may take to come back as "done"


def emit(event, **kw):
    print(json.dumps({"event": event, **kw}), flush=True)


class Goto(Node):
    def __init__(self, room):
        super().__init__('rover_goto')
        self.room = room
        self.done = threading.Event()
        self.pub = self.create_publisher(String, 'rover/goto_request', 10)
        self.create_subscription(
            String, 'rover/goto_status', self.on_status, 10)

    def on_status(self, msg):
        try:
            ev = json.loads(msg.data)
        except ValueError:
            return
        print(msg.data, flush=True)
        if ev.get('event') == 'done':
            self.done.set()

    def send(self, action, **kw):
        self.pub.publish(String(data=json.dumps({'action': action, **kw})))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('room')
    args = ap.parse_args()

    rclpy.init()
    node = Goto(args.room)

    # rover_ai has to be listening, or the request goes nowhere and the page
    # waits on a child that will never report anything.
    deadline = time.time() + WAIT_FOR_AI
    while node.pub.get_subscription_count() == 0:
        if time.time() > deadline:
            emit('done', room=args.room, outcome='failed',
                 detail='the voice service is not running')
            node.destroy_node()
            rclpy.shutdown()
            return 1
        rclpy.spin_once(node, timeout_sec=0.2)

    cancelling = threading.Event()

    def on_term(_sig, _frm):
        # Ask for a cancel and keep spinning: the verdict still has to arrive.
        if not cancelling.is_set():
            cancelling.set()
            node.send('cancel')

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    node.send('goto', room=args.room)
    cancelled_at = None
    try:
        while rclpy.ok() and not node.done.is_set():
            rclpy.spin_once(node, timeout_sec=0.2)
            if cancelling.is_set():
                # Do not hang forever if rover_ai never answers the cancel:
                # the page keeps its buttons locked until this process exits.
                if cancelled_at is None:
                    cancelled_at = time.time()
                elif time.time() - cancelled_at > CANCEL_GRACE:
                    emit('done', room=args.room, outcome='cancelled',
                         detail='no confirmation from the voice service')
                    break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
