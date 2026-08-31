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
import time

import rclpy
from rclpy.executors import (ExternalShutdownException,
                             MultiThreadedExecutor)
from groq import Groq

sys.path.insert(0, os.path.expanduser('~'))
from rover_motions import Motions
from rover_nav import RoverNav
from rooms import ROOM_NAMES, spoken_name

MODEL = os.environ.get('ROVER_LLM_MODEL', 'openai/gpt-oss-120b')
SEARCH_MODEL = os.environ.get('ROVER_SEARCH_MODEL', 'groq/compound-mini')
MAX_HISTORY = 40  # messages kept after the system prompt

MAX_STEPS = 8          # see Brain.run_sequence

SYSTEM = (
    "You are a small four-wheeled robot that drives around a house. You are "
    "not an assistant controlling a robot -- you are the robot. Speak in the "
    "first person: \"I am on my way\", \"I have arrived\", \"I cannot "
    "reach that room\". Never call yourself \"the rover\" or \"the "
    "robot\". "
    "Translate the user's request into tool calls. "
    "Angles are degrees: positive is counter-clockwise (left), negative is "
    "clockwise (right). Distances are metres: positive is forward, negative "
    "is backward. A full circle is 360 degrees. "
    "To move between rooms always use go_to_room -- it uses the map and "
    "avoids obstacles. Only use drive and spin for small local adjustments. "
    "go_to_room returns as soon as you set off, not when you arrive; say you "
    "are on your way, never that you have arrived. Only say that when a "
    "request actually sends you to another room. A spin, a short drive, or "
    "simply saying something is not going anywhere: acknowledge those in a "
    "word or two, or say nothing at all. "
    "If a run_sequence step already speaks to the user, that is your reply -- "
    "do not add another sentence on top of it. Answer with an empty string. "
    "When a request has more than one part -- go somewhere, say something "
    "there, then go somewhere else -- use run_sequence with one step per "
    "action, in order. Never emit several go_to_room calls for one request: "
    "each cancels the one before it. "
    "work_room is the user's own room and they may call it \"my room\", but "
    "always call it \"the work room\" when you speak, so what you say "
    "matches the name on the map. "
    "When a step carries a message for someone, put the user's own words in "
    "the text, not a paraphrase of them. "
    "If a request is unclear or unsafe, ask instead of guessing. "
    "Use ask_the_internet for anything about current events, news, "
    "weather, prices, or facts that may have changed recently. If you "
    "are unsure whether your knowledge is current, call it. "
    "Your replies are read aloud, so reply with exactly one short sentence. "
    "Never add a second sentence such as \"Done.\" or \"Let me know if you "
    "need anything else.\" No lists, no markdown, no emoji."
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
            "Search the web for current information: news, weather, sports, "
            "prices, recent events, or any fact that may have changed since "
            "training. Returns a short text answer."),
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string",
                         "description": "A self-contained question."}},
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
        "name": "run_sequence",
        "description": (
            "Carry out several actions one after another, each finishing "
            "before the next begins. Use this for any request with more than "
            "one part -- go to a room, say something there, then go "
            "somewhere else."),
        "parameters": {"type": "object", "properties": {
            "steps": {"type": "array",
                      "description": "The actions to carry out, in order.",
                      "maxItems": 8,
                      "items": {"type": "object", "properties": {
                          "action": {"type": "string",
                                     "enum": ["go_to_room", "say", "spin",
                                              "drive", "stop"]},
                          "room": {"type": "string", "enum": ROOM_NAMES,
                                   "description": "for go_to_room"},
                          "text": {"type": "string",
                                   "description": "for say; spoken aloud"},
                          "degrees": {"type": "number",
                                      "description": "for spin"},
                          "meters": {"type": "number",
                                     "description": "for drive"}},
                          "required": ["action"]}}},
            "required": ["steps"]}}},
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
        self._seq = None                    # worker running a queued sequence
        self._seq_stop = threading.Event()  # set by stop / cancel_navigation
        # The worker must not speak before this turn's reply does. say() is a
        # FIFO queue, so whichever thread calls it first wins -- and the worker
        # starts while ask() is still returning, which had the rover deliver a
        # step's message before saying it was on its way.
        self._seq_go = threading.Event()

    # ----------------------------------------------------------- sequences

    def run_sequence(self, steps):
        """Run several actions in order, each finishing before the next.

        go_to_room returns the moment Nav2 accepts the goal, so a model that
        emits three of them in one turn has the rover abandon the first two --
        each new goal preempts the last. Steps run on a worker thread that
        waits for arrival in between, and this returns straight away so the
        rover can answer "on my way" rather than going silent for a minute.
        """
        if not isinstance(steps, list) or not steps:
            return False, 'no steps given'
        if len(steps) > MAX_STEPS:
            # Each leg can take a minute; a runaway list would have the rover
            # driving unattended for an hour with no way to interrupt but
            # speech it is too busy to hear.
            return False, (f'that is {len(steps)} steps; I can do '
                           f'{MAX_STEPS} at a time')
        if self._seq is not None and self._seq.is_alive():
            return False, 'still working through the last request'
        self._seq_stop.clear()
        self._seq_go.clear()
        self._seq = threading.Thread(target=self._run_steps, args=(list(steps),),
                                     daemon=True)
        self._seq.start()
        return True, f'started {len(steps)} steps'

    def abort_sequence(self):
        self._seq_stop.set()

    def _speak(self, text):
        text = (text or '').strip()
        if not text:
            return
        print(f'bot > {text}')
        if self.voice:
            self.voice.say(text, block=True)

    def release_sequence(self):
        """Let a queued sequence begin. Called once this turn's reply is queued."""
        self._seq_go.set()

    def _run_steps(self, steps):
        # Wait for the acknowledgement to be queued first, so "on my way"
        # always precedes anything a step says. The timeout covers a caller
        # that never releases -- late is better than silent.
        self._seq_go.wait(timeout=10.0)
        for i, step in enumerate(steps, 1):
            if self._seq_stop.is_set():
                return
            action = (step.get('action') or '').strip()
            if action == 'say':
                self._speak(step.get('text', ''))
                continue
            if action == 'go_to_room':
                room = step.get('room', '')
                ok, msg = self.nav.go_to_room(room)
                if not ok:
                    self._speak(f'I could not set off for '
                                f'{spoken_name(room)}. {msg}')
                    return
                if not self._await_arrival():
                    return
                continue
            ok, msg = self.dispatch(action, step)
            if not ok:
                self._speak(f'I could not do step {i}. {msg}')
                return

    def _await_arrival(self, timeout=240.0):
        """Block until the goal settles. False if it failed or timed out.

        A failed leg must stop the sequence: there is no point delivering a
        message in a room the rover never reached.
        """
        deadline = time.time() + timeout
        self.nav.last_outcome = None
        # The action server needs a moment to report that it has started, and
        # is_navigating() reads False in the gap.
        time.sleep(1.5)
        while self.nav.is_navigating():
            if self._seq_stop.is_set():
                return False
            if time.time() > deadline:
                self.nav.cancel()
                self._speak('That is taking too long, so I have stopped.')
                return False
            time.sleep(0.3)
        # announce() already says what happened, so stay quiet on failure.
        return self.nav.last_outcome in (None, 'arrived')

    # -------------------------------------------------------------- retry

    def _complete(self, **kw):
        """Call Groq, retrying transient failures. Raises on final failure."""
        last = None
        for attempt in range(3):
            try:
                return self.client.chat.completions.create(**kw)
            except Exception as e:
                last = e
                print(f'[llm attempt {attempt + 1}/3 failed: {e}]')
                time.sleep(0.6 * (attempt + 1))
        raise last

    # ------------------------------------------------------------ dispatch

    def dispatch(self, name, args):
        # Manual motion and Nav2 both publish /cmd_vel. Never let them overlap.
        if name in ('spin', 'drive') and self.nav.is_navigating():
            self.nav.cancel()

        if name == 'run_sequence':
            return self.run_sequence(args.get('steps', []))
        if name == 'go_to_room':
            return self.nav.go_to_room(args.get('room', ''))
        if name in ('stop', 'cancel_navigation'):
            # Otherwise the sequence worker cheerfully starts the next step
            # a moment after being told to stop.
            self.abort_sequence()

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
        if name == 'ask_the_internet':
            return self.search(args.get('question', ''))
        if name == 'spin':
            return self.m.do_spin(args.get('degrees', 0))
        if name == 'drive':
            return self.m.do_drive(args.get('meters', 0))
        if name == 'stop':
            self.nav.cancel()
            return self.m.do_stop()
        return False, f'unknown tool {name}'

    # -------------------------------------------------------------- search

    def search(self, question):
        """Delegate to a Groq Compound model, which has built-in web search.

        Compound cannot do local tool calling, so it cannot be the main model.
        It is queried here as a plain one-shot question instead.
        """
        if not question.strip():
            return False, 'no question given'
        try:
            r = self.client.chat.completions.create(
                model=SEARCH_MODEL,
                messages=[
                    {"role": "system", "content":
                     "Answer in one or two short sentences. Plain text only, "
                     "no markdown or lists. It will be read aloud."},
                    {"role": "user", "content": question}],
                max_tokens=300)
            return True, (r.choices[0].message.content or '').strip()
        except Exception as e:
            return False, f'search failed: {e}'

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
            r = self._complete(
                model=MODEL, messages=self.history,
                tools=TOOLS, tool_choice="auto", max_tokens=400)
        except Exception:
            self.history.pop()
            return "Sorry, I could not reach my brain just then. Please try again." 

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

        for _ in range(3):          # allow a few chained tool calls
            try:
                r2 = self.client.chat.completions.create(
                    model=MODEL, messages=self.history,
                    tools=TOOLS, tool_choice="auto", max_tokens=100)
            except Exception as e:
                print(f'[reply failed: {e}]')
                return 'Sorry, something went wrong.'
            m2 = r2.choices[0].message
            self.history.append(m2.model_dump(exclude_none=True))
            more = m2.tool_calls or []
            if not more:
                reply = m2.content or 'done'
                return reply
            for c in more:
                try:
                    a = json.loads(c.function.arguments or '{}')
                except json.JSONDecodeError:
                    a = {}
                print(f'  -> {c.function.name}({a})')
                ok, result = self.dispatch(c.function.name, a)
                self.history.append({
                    "role": "tool", "tool_call_id": c.id,
                    "content": json.dumps({"ok": ok, "result": result})})
        return 'Sorry, I got stuck on that one.'


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
            'arrived':  f'I have arrived at {spoken_name(room)}.',
            'failed':   f'I could not reach {spoken_name(room)}.',
            'rejected': f'I could not accept that goal for {spoken_name(room)}.',
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
    print(f'Ready ({MODEL}). Rooms: {", ".join(ROOM_NAMES)}')

    stop_flag = threading.Event()

    def voice_loop():
        voice.calibrate()
        voice.say('I am ready.')
        while not stop_flag.is_set():
            try:
                heard = voice.listen_once()
                if not heard or len(heard) < 3:
                    continue
                print(f'\nyou (voice) > {heard}')
                reply = brain.ask(heard)
                print(f'bot > {reply}\n')
                voice.say(reply)
                brain.release_sequence()
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
                brain.release_sequence()
        else:
            print('Listening. Ctrl-C to quit.\n')
            # rclpy installs a SIGTERM handler, so SIGTERM no longer ends the
            # process -- it shuts the context down and leaves this loop
            # spinning. Under systemd that meant a 90 s wait and a SIGKILL on
            # every stop, taking arecord and the executor threads with it.
            while rclpy.ok() and not stop_flag.is_set():
                threading.Event().wait(1.0)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        stop_flag.set()
        brain.abort_sequence()
        # SIGTERM shuts the context down before this runs, so anything that
        # publishes raises "publisher's context is invalid". Skip those two
        # when the context has gone: the bridge zeroes the motors 0.5 s after
        # /cmd_vel stops arriving (command_timeout), so the rover halts either
        # way.
        if rclpy.ok():
            nav.cancel()
            motions.do_stop()
        if voice:
            voice.close()
        motions.destroy_node()
        nav.destroy_node()
        # rclpy's SIGTERM handler has already shut the context down by
        # the time we get here, and calling it twice raises RCLError --
        # which exits 1 and makes systemd record a normal stop as a
        # failure.
        if rclpy.ok():
            rclpy.shutdown()



if __name__ == '__main__':
    main()
