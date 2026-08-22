#!/usr/bin/env python3
"""Natural-language control for the Wave Rover.

Adds to the original: Nav2 room goals and optional voice in/out.

  python3 rover_ai.py            # text REPL, as before
  python3 rover_ai.py --voice    # mic in, speaker out
  python3 rover_ai.py --voice --text   # both at once

Requires GROQ_API_KEY. Voice mode also needs espeak-ng and alsa-utils.
"""

import argparse
import json
import os
import sys
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from groq import Groq

sys.path.insert(0, os.path.expanduser('~'))
from rover_motions import Motions
from rover_nav import RoverNav
from rooms import ROOM_NAMES, spoken_name

MODEL = os.environ.get('ROVER_LLM_MODEL', 'openai/gpt-oss-120b')
SEARCH_MODEL = os.environ.get('ROVER_SEARCH_MODEL', 'groq/compound-mini')
MAX_HISTORY = 40  # messages kept after the system prompt

SYSTEM = (
    "You control a small four-wheeled robot that drives around a house. "
    "Translate the user's request into tool calls. "
    "Angles are degrees: positive is counter-clockwise (left), negative is "
    "clockwise (right). Distances are metres: positive is forward, negative "
    "is backward. A full circle is 360 degrees. "
    "To move between rooms always use go_to_room -- it uses the map and "
    "avoids obstacles. Only use drive and spin for small local adjustments. "
    "go_to_room returns as soon as the robot sets off, not when it arrives; "
    "say that it is on its way, never that it has arrived. "
    "If a request is unclear or unsafe, ask instead of guessing. "
    "You have no knowledge of current events, weather, news or prices. For "
    "anything that could have changed since you were trained, call "
    "ask_the_internet rather than answering from memory. "
    "Your replies are read aloud, so keep them to one short sentence with no "
    "lists, no markdown and no emoji."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "go_to_room",
        "description": (
            "Send the robot to a named room using the navigation stack. "
            "Returns immediately; the robot announces arrival itself."),
        "parameters": {"type": "object", "properties": {
            "room": {"type": "string", "enum": ROOM_NAMES,
                     "description": "Destination room."}},
            "required": ["room"]}}},
    {"type": "function", "function": {
        "name": "cancel_navigation",
        "description": "Abandon the current navigation goal but stay powered.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "where_am_i",
        "description": "Report the robot's current position and nearest known room.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "ask_the_internet",
        "description": (
            "Look up anything the model does not already know: current weather, "
            "news, sports results, prices, opening times, recent events, or any "
            "fact that may have changed. Use this instead of guessing."),
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string",
                         "description": "A complete, self-contained question."}},
            "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "spin",
        "description": "Rotate the robot in place by a number of degrees.",
        "parameters": {"type": "object", "properties": {
            "degrees": {"type": "number",
                        "description": "Degrees to rotate. Positive = left."}},
            "required": ["degrees"]}}},
    {"type": "function", "function": {
        "name": "drive",
        "description": (
            "Drive the robot straight by a number of metres. Blind motion with "
            "no obstacle avoidance -- use go_to_room for anything longer than "
            "about a metre."),
        "parameters": {"type": "object", "properties": {
            "meters": {"type": "number",
                       "description": "Metres to drive. Positive = forward."}},
            "required": ["meters"]}}},
    {"type": "function", "function": {
        "name": "stop",
        "description": "Stop the robot immediately, including any navigation.",
        "parameters": {"type": "object", "properties": {}}}},
]


class Brain:
    def __init__(self, motions, nav, voice=None):
        self.m = motions
        self.nav = nav
        self.voice = voice
        self.client = Groq(api_key=os.environ['GROQ_API_KEY'])
        self.history = [{"role": "system", "content": SYSTEM}]
        self.lock = threading.Lock()

    # ------------------------------------------------------------ dispatch

    def ask_the_internet(self, question):
        """Delegate to groq/compound, which has built-in web search.

        Compound cannot be used as the main model: it does not support custom
        tools, so the rover would lose spin/drive/go_to_room. Calling it as a
        tool keeps control with gpt-oss and borrows the search.
        """
        if not question.strip():
            return False, 'no question given'
        try:
            r = self.client.chat.completions.create(
                model=SEARCH_MODEL,
                messages=[
                    {"role": "system", "content":
                     "Answer in at most two short sentences, plain spoken "
                     "English. No markdown, no tables, no lists, no URLs, no "
                     "citations. Round numbers. If you cannot find out, say so."},
                    {"role": "user", "content": question}],
                max_tokens=300)
            answer = (r.choices[0].message.content or '').strip()
        except Exception as e:
            return False, f'search failed: {e}'
        if not answer:
            return False, 'no answer found'
        return True, answer[:600]

    def dispatch(self, name, args):
        # Manual motion and Nav2 both publish /cmd_vel. Never let them overlap.
        if name in ('spin', 'drive') and self.nav.is_navigating():
            self.nav.cancel()

        if name == 'ask_the_internet':
            return self.ask_the_internet(args.get('question', ''))
        if name == 'go_to_room':
            return self.nav.go_to_room(args.get('room', ''))
        if name == 'cancel_navigation':
            ok, msg = self.nav.cancel()
            self.m.do_stop()
            return ok, msg
        if name == 'where_am_i':
            p = self.nav.pose()
            if p is None:
                return False, 'no pose yet -- is cartographer localisation running?'
            room, dist = self.nav.nearest_room()
            return True, {'x': round(p[0], 2), 'y': round(p[1], 2),
                          'heading_deg': round(p[2], 1),
                          'nearest_room': room, 'metres_away': round(dist, 2)}
        if name == 'spin':
            return self.m.do_spin(args.get('degrees', 0))
        if name == 'drive':
            return self.m.do_drive(args.get('meters', 0))
        if name == 'stop':
            self.nav.cancel()
            return self.m.do_stop()
        return False, f'unknown tool {name}'

    # ------------------------------------------------------------- history

    def _trim(self):
        """Drop old turns, but never split a tool_calls message from its results."""
        if len(self.history) <= MAX_HISTORY + 1:
            return
        cut = len(self.history) - MAX_HISTORY
        while cut < len(self.history) and self.history[cut].get('role') == 'tool':
            cut += 1
        self.history = [self.history[0]] + self.history[cut:]

    # ----------------------------------------------------------------- ask

    def ask(self, text):
        with self.lock:
            return self._ask(text)

    def _ask(self, text):
        self.history.append({"role": "user", "content": text})
        self._trim()
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--voice', action='store_true', help='listen on the mic')
    ap.add_argument('--text', action='store_true', help='keep the text REPL too')
    args = ap.parse_args()
    use_text = args.text or not args.voice

    if 'GROQ_API_KEY' not in os.environ:
        print('Set GROQ_API_KEY first')
        return

    voice = None
    if args.voice:
        from rover_voice import Voice
        voice = Voice(Groq(api_key=os.environ['GROQ_API_KEY']))

    def announce(room, outcome, detail):
        """Fires on an executor thread when a nav goal settles."""
        line = {
            'arrived':  f'I have arrived at the {spoken_name(room)}.',
            'failed':   f'I could not reach the {spoken_name(room)}.',
            'rejected': f'The navigation stack refused the {spoken_name(room)} goal.',
        }.get(outcome)
        if not line:
            return
        print(f'\nbot > {line}' + (f'  ({detail})' if detail else ''))
        if voice:
            voice.say(line)

    rclpy.init()
    motions = Motions()
    nav = RoverNav(on_done=announce)
    ex = MultiThreadedExecutor()
    ex.add_node(motions)
    ex.add_node(nav)
    threading.Thread(target=ex.spin, daemon=True).start()

    brain = Brain(motions, nav, voice)
    print(f'Rover AI ready ({MODEL}). Rooms: {", ".join(ROOM_NAMES)}')

    stop_flag = threading.Event()

    def voice_loop():
        voice.calibrate()
        voice.say('Rover ready.')
        while not stop_flag.is_set():
            try:
                heard = voice.listen_once()
                if not heard or len(heard) < 3:
                    continue
                print(f'\nyou (voice) > {heard}')
                reply = brain.ask(heard)
                print(f'bot > {reply}\n')
                voice.say(reply)
            except Exception as e:
                print(f'[voice loop error: {e}]')

    try:
        if args.voice:
            t = threading.Thread(target=voice_loop, daemon=True)
            t.start()
        if use_text:
            print('Ctrl-D to quit.\n')
            while True:
                try:
                    text = input('you > ').strip()
                except EOFError:
                    break
                if not text:
                    continue
                reply = brain.ask(text)
                print(f'bot > {reply}\n')
                if voice:
                    voice.say(reply)
        else:
            print('Listening. Ctrl-C to quit.\n')
            while True:
                threading.Event().wait(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()
        nav.cancel()
        motions.do_stop()
        if voice:
            voice.close()
        motions.destroy_node()
        nav.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
