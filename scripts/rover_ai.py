#!/usr/bin/env python3
"""Natural-language control for the Wave Rover.

Adds to the original: Nav2 room goals and optional voice in/out.

  python3 rover_ai.py            # text REPL, as before
  python3 rover_ai.py --voice    # mic in, speaker out
  python3 rover_ai.py --voice --text   # both at once

Requires GROQ_API_KEY for the model, speech-to-text and speech. Voice mode
also needs espeak-ng and alsa-utils.

Search uses Tavily and needs TAVILY_API_KEY -- 1,000 free searches a month,
no card. Without the key the rover says it cannot look things up and
everything else keeps working.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

import rclpy
from rclpy.executors import (ExternalShutdownException,
                             MultiThreadedExecutor)
from groq import Groq

sys.path.insert(0, os.path.expanduser('~'))
from rover_motions import Motions
from rover_nav import RoverNav
from rooms import ROOM_NAMES, spoken_name

# Groq meters tokens per day per model, not per organisation, so each of
# these carries its own 200k allowance. Exhausting one leaves the rest
# untouched -- the same limit four times over, provided we move across when
# one runs dry. Ordered best-first; all four support tool calling.
MODELS = [m.strip() for m in os.environ.get(
    'ROVER_LLM_MODELS',
    'openai/gpt-oss-120b,openai/gpt-oss-20b,'
    'qwen/qwen3.8-27b,qwen/qwen3.6-27b').split(',') if m.strip()]
# ROVER_LLM_MODEL still pins a single model, for testing one in isolation.
if os.environ.get('ROVER_LLM_MODEL'):
    MODELS = [os.environ['ROVER_LLM_MODEL']]
MODEL = MODELS[0]
# Search runs on Tavily. Groq's compound models 413 whenever they invoke
# their search tool -- reproduced with curl, 0 tokens metered, and unaffected
# by which chat model is running -- so it is broken on their side, not ours.
# Gemini's grounding and Claude's web search both need a billing account;
# Tavily gives 1,000 searches a month with no card, which at a basic search
# per credit is about 33 a day.
SEARCH_URL = 'https://api.tavily.com/search'
SEARCH_RESULTS = 5

MAX_HISTORY = 40  # messages kept after the system prompt. Every one is
                  # resent on every call, and a turn makes several -- but a
                  # shorter window costs "do it again" and "go back there",
                  # which is worth more than the tokens.

MAX_STEPS = 8          # see Brain.run_sequence

SYSTEM = (
    "You are a small four-wheeled robot that drives around a house. Speak in "
    "the first person; never call yourself \"the rover\" or \"the robot\". "
    "Turn requests into tool calls. Degrees: + is left, - is right, a full "
    "circle is 360. Metres: + is forward, - is back. "
    "Use go_to_room to move between rooms; it uses the map and avoids "
    "obstacles. Use drive and spin only for small local adjustments. "
    "For a request with several parts use run_sequence, one step each, in "
    "order. Never emit several go_to_room calls: each cancels the last. "
    "go_to_room returns when you set off, not when you arrive. Say you are on "
    "your way only when actually going to another room -- a spin, a short "
    "drive or simply speaking is not a journey. "
    "work_room is the user's own room and they may call it \"my room\", but "
    "you call it \"the work room\". "
    "When a step carries a message for someone, keep the user's wording but "
    "address the listener: \"you have class tomorrow\", not \"I have class "
    "tomorrow\". "
    "If a step already speaks, reply with an empty string. Never announce "
    "completion: no \"Done\", \"Task completed\", \"Let me know if you "
    "need anything else\". The action is the answer. "
    "Ask if a request is unclear or unsafe. Never spin, drive or go anywhere "
    "unless the user asked you to move. A question you could not answer is "
    "not a reason to move. "
    "Use ask_the_internet for current "
    "events, weather, prices, or anything that may have changed. If that "
    "search fails, say you could not look it up. Do not answer from memory "
    "instead: you reached for the search because your own knowledge was too "
    "old, and it is no fresher for the search having failed. "
    "Replies are read aloud: one short sentence, no lists, markdown or emoji."
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
                          "room": {"type": "string",
                                   "description": "for go_to_room; one of "
                                                  "the rooms listed there"},
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
        self._model_i = 0                   # index into MODELS
        self._seen = {}                     # tool results, this turn
        self._failed = {}                   # tools that failed, this turn
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

    @staticmethod
    def _is_rate_limit(e):
        """Groq reports a rate limit as 429 or, for tokens per minute, as 413.

        "Request Entity Too Large" is not always about bytes: exceeding TPM
        arrives as 413 with "Request too large for model X ... on tokens per
        minute (TPM): Limit N, Requested M". Treating that as a transient
        error means three retries 0.6s apart against a per-minute window --
        far too soon to help, and the model never switches.
        """
        code = getattr(e, 'status_code', None)
        text = str(e)
        return code in (429, 413) or 'rate_limit_exceeded' in text \
            or 'Rate limit reached' in text or 'Request too large' in text

    def _complete(self, **kw):
        """Call the LLM, moving to the next model when one is out of tokens.

        A daily limit is not a transient failure: sleeping and retrying the
        same model just fails again tomorrow's worth of times. Each model has
        its own allowance, so the useful response is to switch. Other errors
        still get a short backoff, since those usually are transient.
        """
        kw.pop('model', None)
        last = None
        while True:
            model = MODELS[self._model_i]
            for attempt in range(3):
                try:
                    return self.client.chat.completions.create(model=model, **kw)
                except Exception as e:
                    last = e
                    if self._is_rate_limit(e):
                        break       # retrying an exhausted model is pointless
                    print(f'[llm attempt {attempt + 1}/3 failed: {e}]')
                    time.sleep(0.6 * (attempt + 1))
            # Switching costs no attempt of its own, or with four models and
            # three attempts the last one would never be reached.
            if self._is_rate_limit(last) and self._model_i + 1 < len(MODELS):
                self._model_i += 1
                print(f'[{model} is out of tokens for today; '
                      f'switching to {MODELS[self._model_i]}]')
                continue
            raise last

    # ------------------------------------------------------------ dispatch

    def _dispatch_once(self, name, args):
        """dispatch(), but a repeat within the same turn reuses the answer.

        Movement is exempt: "spin 90 twice" is two spins, not one.
        """
        key = (name, json.dumps(args, sort_keys=True))
        movement = name in ('spin', 'drive', 'go_to_room', 'run_sequence')
        # A failed lookup must never end in the rover driving. Asked for news,
        # the search 413'd and the model called spin(90) -- nothing had asked
        # it to move. Whatever its reasoning, a tool failure is not a reason to
        # move, so movement is refused for the rest of that turn.
        if movement and self._failed:
            broke = ", ".join(sorted(self._failed))
            print(f'  -> {name}({args})  [REFUSED: {broke} failed this turn]')
            return False, ('not moving: something failed earlier in this '
                           'request, so movement was not carried out')
        if not movement:
            # A failure is about the tool, not the phrasing. Rewording a
            # question the search could not answer just spends the request
            # again -- which is how one weather question became three
            # identical failures.
            if name in self._failed:
                print(f'  -> {name}({args})  [already failed this turn]')
                return self._failed[name]
            if key in self._seen:
                print(f'  -> {name}({args})  [already asked this turn]')
                return self._seen[key]
        print(f'  -> {name}({args})')
        ok, result = self.dispatch(name, args)
        if ok:
            self._seen[key] = (ok, result)
        elif not movement:
            self._failed[name] = (ok, result)
        return ok, result

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
        """Answer from a live web search, or say plainly that it could not.

        Tavily returns both a short synthesised answer and the snippets it was
        built from. The snippets are the point: retrieval happens on our side
        of the line, so "found nothing" is a fact we can see rather than
        something the model reports or quietly papers over. Groq's compound
        gave neither -- it once reported the 2026 World Cup as unplayed, with
        no error and no way to tell it had not searched.

        Raw urllib rather than the SDK: this runs under system python
        alongside ROS, and one fewer package to keep current there is worth
        more than the convenience.
        """
        if not question.strip():
            return False, 'no question given'
        key = os.environ.get('TAVILY_API_KEY')
        if not key:
            print('[search unavailable: TAVILY_API_KEY is not set]')
            return False, 'search is not configured'

        # basic costs one credit; advanced costs two and buys depth that a
        # one-sentence spoken reply cannot carry.
        body = json.dumps({
            'query': question,
            'search_depth': 'basic',
            'max_results': SEARCH_RESULTS,
            'include_answer': True,
        }).encode()
        req = urllib.request.Request(
            SEARCH_URL, data=body,
            headers={'Content-Type': 'application/json',
                     'Authorization': f'Bearer {key}'})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors='replace')[:200]
            print(f'[search failed: HTTP {e.code} {detail}]')
            return False, f'search failed: HTTP {e.code}'
        except Exception as e:
            # Only the model sees a failed tool result, and it responds by
            # rewording the question rather than reporting the problem.
            print(f'[search failed: {e}]')
            return False, f'search failed: {e}'

        results = data.get('results') or []
        answer = (data.get('answer') or '').strip()
        if not results and not answer:
            print('[search found nothing]')
            return False, 'the search found nothing'

        hosts = []
        for r in results[:3]:
            u = r.get('url') or ''
            host = u.split('/')[2] if u.count('/') > 2 else u
            if host and host not in hosts:
                hosts.append(host)
        print(f'[searched: {len(results)} results'
              + (f' from {", ".join(hosts)}' if hosts else '') + ']')

        if answer:
            return True, answer[:600]
        # No synthesised answer: hand over the snippets and let the model
        # write the sentence.
        text = ' '.join((r.get('content') or '').strip() for r in results[:3])
        return True, text[:900] or 'the search found nothing usable'

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

    @staticmethod
    def _collapse(text):
        """Drop a sentence the model repeated back to back.

        Small models restate themselves, and raising max_tokens to 400 gave
        them room to -- at 100 the repeat was simply cut off. Reading "I could
        not look that up" aloud twice sounds broken, so identical neighbouring
        sentences are collapsed. Deliberate repetition ("go, go!") survives:
        only exact neighbours are dropped.
        """
        parts = [p for p in re.split(r'(?<=[.!?])\s*', (text or '').strip()) if p]
        out = []
        for part in parts:
            if not out or part != out[-1]:
                out.append(part)
        return ' '.join(out)

    def ask(self, text):
        with self.lock:
            return self._ask(text)

    def _ask(self, text):
        # A model that is unhappy with a tool's answer tends to call it again
        # with the question reworded. Three identical searches cost three
        # times the tokens and return the same thing, so results are reused
        # within a turn and the model is told it already has them.
        self._seen = {}
        self._failed = {}
        self.history.append({"role": "user", "content": text})
        self._trim()
        try:
            r = self._complete(
                messages=self.history,
                tools=TOOLS, tool_choice="auto", max_tokens=400)
        except Exception:
            self.history.pop()
            return "Sorry, I could not reach my brain just then. Please try again." 

        msg = r.choices[0].message
        self.history.append(msg.model_dump(exclude_none=True))

        calls = msg.tool_calls or []
        if not calls:
            return self._collapse(msg.content or '(no reply)')

        for c in calls:
            try:
                args = json.loads(c.function.arguments or '{}')
            except json.JSONDecodeError:
                args = {}
            ok, result = self._dispatch_once(c.function.name, args)
            self.history.append({
                "role": "tool", "tool_call_id": c.id,
                "content": json.dumps({"ok": ok, "result": result})})

        for _ in range(3):          # allow a few chained tool calls
            try:
                # 100 was chosen for a one-sentence reply, but gpt-oss
                # spends tokens on reasoning that count against the same
                # ceiling, so the visible answer was being cut off before it
                # started -- and a truncated turn sends the loop round to
                # call the same tool again. The prompt caps the reply's
                # length; this only has to stop a runaway.
                r2 = self._complete(
                    messages=self.history,
                    tools=TOOLS, tool_choice="auto", max_tokens=400)
            except Exception as e:
                print(f'[reply failed: {e}]')
                return 'Sorry, something went wrong.'
            # A reply that stops mid-thought is either the model doing it or
            # max_tokens cutting it off, and they need different fixes. The
            # API says which; without this the two are indistinguishable.
            fr = r2.choices[0].finish_reason
            if fr not in (None, 'stop', 'tool_calls'):
                print(f'[reply ended early: finish_reason={fr}]')
            m2 = r2.choices[0].message
            self.history.append(m2.model_dump(exclude_none=True))
            more = m2.tool_calls or []
            if not more:
                # Empty is a legitimate answer: the prompt asks for it when a
                # run_sequence step has already spoken. Substituting "done"
                # here made the rover announce the completion of every task it
                # had just narrated.
                reply = m2.content or ''
                return self._collapse(reply)
            for c in more:
                try:
                    a = json.loads(c.function.arguments or '{}')
                except json.JSONDecodeError:
                    a = {}
                ok, result = self._dispatch_once(c.function.name, a)
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
    print(f'Ready ({" -> ".join(MODELS)}). Rooms: {", ".join(ROOM_NAMES)}')

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
                if reply:
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
                if reply:
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
        # Stop the executor before the nodes it holds. Left spinning, the C++
        # layer under rclpy aborts as the nodes are destroyed beneath it --
        # "terminate called without an active exception", core dumped, on
        # every Ctrl-C.
        ex.shutdown()
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
