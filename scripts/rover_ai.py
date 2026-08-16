#!/usr/bin/env python3
import json
import os
import sys
import threading
import rclpy
from rclpy.executors import MultiThreadedExecutor
from groq import Groq

sys.path.insert(0, os.path.expanduser('~'))
from rover_motions import Motions

MODEL = os.environ.get('ROVER_LLM_MODEL', 'llama-3.3-70b-versatile')

SYSTEM = (
    "You control a small four-wheeled robot. Translate the user's request "
    "into tool calls. Angles are degrees: positive is counter-clockwise "
    "(left), negative is clockwise (right). Distances are metres: positive "
    "is forward, negative is backward. A full circle is 360 degrees. "
    "If a request is unclear or unsafe, ask instead of guessing. "
    "Keep spoken replies to one short sentence."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "spin",
        "description": "Rotate the robot in place by a number of degrees.",
        "parameters": {"type": "object", "properties": {
            "degrees": {"type": "number",
                        "description": "Degrees to rotate. Positive = left."}},
            "required": ["degrees"]}}},
    {"type": "function", "function": {
        "name": "drive",
        "description": "Drive the robot straight by a number of metres.",
        "parameters": {"type": "object", "properties": {
            "meters": {"type": "number",
                       "description": "Metres to drive. Positive = forward."}},
            "required": ["meters"]}}},
    {"type": "function", "function": {
        "name": "stop",
        "description": "Stop the robot immediately.",
        "parameters": {"type": "object", "properties": {}}}},
]


class Brain:
    def __init__(self, motions):
        self.m = motions
        self.client = Groq(api_key=os.environ['GROQ_API_KEY'])
        self.history = [{"role": "system", "content": SYSTEM}]

    def dispatch(self, name, args):
        if name == 'spin':
            return self.m.do_spin(args.get('degrees', 0))
        if name == 'drive':
            return self.m.do_drive(args.get('meters', 0))
        if name == 'stop':
            return self.m.do_stop()
        return False, f'unknown tool {name}'

    def ask(self, text):
        self.history.append({"role": "user", "content": text})
        try:
            r = self.client.chat.completions.create(
                model=MODEL, messages=self.history,
                tools=TOOLS, tool_choice="auto", max_tokens=400)
        except Exception as e:
            self.history.pop()
            return f'[llm error: {e}]'

        msg = r.choices[0].message
        self.history.append(msg.model_dump(exclude_none=True))

        calls = msg.tool_calls or []
        if not calls:
            return msg.content or '(no reply)'

        for c in calls:
            try:
                args = json.loads(c.function.arguments or '{}')
            except json.JSONDecodeError:
                args = {}
            print(f'  -> {c.function.name}({args})')
            ok, result = self.dispatch(c.function.name, args)
            self.history.append({
                "role": "tool", "tool_call_id": c.id,
                "content": json.dumps({"ok": ok, "result": result})})

        try:
            r2 = self.client.chat.completions.create(
                model=MODEL, messages=self.history, max_tokens=200)
            reply = r2.choices[0].message.content or 'done'
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f'[done, but reply failed: {e}]'


def main():
    if 'GROQ_API_KEY' not in os.environ:
        print('Set GROQ_API_KEY first')
        return
    rclpy.init()
    node = Motions()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    brain = Brain(node)
    print(f'Rover AI ready ({MODEL}). Ctrl-D to quit.\n')
    try:
        while True:
            try:
                text = input('you > ').strip()
            except EOFError:
                break
            if not text:
                continue
            print(f'bot > {brain.ask(text)}\n')
    except KeyboardInterrupt:
        pass
    finally:
        node.do_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
