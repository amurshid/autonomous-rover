#!/usr/bin/env python3
"""Mode switcher for the rover, on a page anyone can use.

The rover runs one of two modes and never both, because Nav2 and the teleop
page would otherwise fight over /cmd_vel:

  autonomous  lidar, cartographer, nav2, voice
  remote      the teleop page, driven by hand

Switching is two systemd targets with Conflicts= between them, so starting
either stops the other. That is a one-line operation over SSH and no use at all
to someone holding a phone, which is what this serves.

In autonomous mode it also lists the rooms, so a phone can send the rover
somewhere without saying a word to it.

Runs in both modes -- it is the one thing that must never go down, or there is
no way back. Needs no ROS: it only reads systemctl and starts targets, through
a sudoers rule limited to exactly those commands. Sending a goal does need ROS,
so that half runs as a subprocess (rover_nav.py --json) and this process stays
free of rclpy.

    python3 rover_mode_web.py            # port 80, or $ROVER_MODE_PORT
"""

import json
import os
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("ROVER_MODE_PORT", "80"))
TELEOP_PORT = int(os.environ.get("ROVER_TELEOP_PORT", "8080"))

# rooms.py is the single source of truth for the goal poses and needs no ROS
# itself, so importing it here costs nothing. If it is missing the page must
# still come up: losing the room buttons is survivable, losing the mode
# switcher is not.
try:
    from rooms import ROOM_NAMES, ROOMS, spoken_name
except Exception:
    ROOMS, ROOM_NAMES = {}, []

    def spoken_name(room):
        return room

# Same bargain: a battery readout is worth having, but not at the price of the
# one page that has to survive everything else being broken.
try:
    import rover_health
except Exception:
    rover_health = None

# How to run rover_nav.py. A login shell because the ROS setup scripts assume
# one, and `exec` because the child must be the process this server signals --
# a wrapping bash would swallow the SIGTERM that cancels the goal. The room
# arrives as $1 rather than interpolated into the command, so nothing about it
# reaches the shell as syntax.
# rover_goto.py hands the room to rover_ai rather than sending the Nav2 goal
# from here. Same event stream, one owner of navigation: two processes holding
# goals is how rover_ai came to drive by hand over the top of Nav2.
NAV_CMD = os.environ.get("ROVER_NAV_CMD") or (
    "source /opt/ros/humble/setup.bash && "
    "source \"$HOME/ros2_ws/install/setup.bash\" && "
    "exec python3 -u \"$HOME/rover_goto.py\" \"$1\""
)


def room_label(room):
    """'breakfast_table' -> 'Breakfast table'; 'bedroom_1' -> "bedroom 1"."""
    name = spoken_name(room)
    if name.startswith("the "):
        name = name[4:]
    return name[:1].upper() + name[1:]


# Order matters: the page shows these as a checklist, and reading it top to
# bottom should match what actually comes up.
MODES = {
    "auto": {
        "target": "rover.target",
        "label": "Autonomous",
        "blurb": "Finds itself, drives itself, listens for commands.",
        # No camera: nothing reads /camera/image_raw under autonomous since
        # the relocaliser left, so it runs in remote control only. Listing it
        # here would make "all active" unreachable and hang the loading
        # screen at six of seven with the room buttons still locked.
        "units": ["rover-bridge", "rover-lidar", "rover-cartographer",
                  "rover-initialpose", "rover-seedpose",
                  "rover-nav2", "rover-ai"],
        "steps": ["Motors", "Lidar", "Map", "Pose bridge", "Position",
                  "Navigation", "Voice"],
    },
    "teleop": {
        "target": "rover-teleop.target",
        "label": "Remote control",
        "blurb": "You drive, with the camera streamed to your phone.",
        "units": ["rover-bridge", "rover-camera", "rover-teleop"],
        "steps": ["Motors", "Camera", "Controls"],
    },
}


def systemctl(*args, root=False):
    """Run systemctl, never raising -- the page must survive a failed call.

    Only starting a target needs root. Reading state does not, so status works
    before the sudoers rule is installed, and the rule stays narrower.
    """
    cmd = (["sudo", "-n"] if root else []) + ["systemctl", *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.returncode, (r.stdout + r.stderr).strip()
    except Exception as e:
        return 1, str(e)


def unit_state(unit):
    """active / activating / inactive / failed. A oneshot that has finished
    reports 'active' with sub-state 'exited', which is success, not a hang."""
    code, out = systemctl("is-active", f"{unit}.service")
    return out.splitlines()[0] if out else "unknown"


def current_mode():
    for key, m in MODES.items():
        code, out = systemctl("is-active", m["target"])
        if out.startswith("active"):
            return key
    return None


def teleop_answering():
    """The page is only useful once the port accepts a connection; systemd
    calls the unit active the moment the process starts, which is earlier."""
    try:
        with socket.create_connection(("127.0.0.1", TELEOP_PORT), timeout=0.4):
            return True
    except OSError:
        return False


class Navigator:
    """The one goal in flight, as a child process being watched by one thread.

    The page's whole contract with the user is that a room button is either
    free or locked, so there is never more than one child: a second request
    while one is running is refused rather than queued or made to preempt.

    Terminal state is kept rather than cleared, so a phone picked up after the
    fact still sees where the rover went and whether it got there.

    The invariant the page depends on is that a locked button means a live
    child. So the verdict is not published until the child has actually been
    reaped: rover_nav.py says "cancelled" and then stays up another second and
    a half getting that cancel to Nav2, and unlocking the buttons over the top
    of that window is how you get a tap that is silently refused.
    """

    IDLE = {"room": None, "phase": "idle", "remaining": None, "detail": ""}

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._state = dict(self.IDLE)

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def go(self, room):
        with self._lock:
            if self._proc is not None:
                return False, f"already driving to {self._state['room']}"
            try:
                proc = subprocess.Popen(
                    ["bash", "-lc", NAV_CMD, "rover-nav", room],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1)
            except Exception as e:
                self._state = dict(self.IDLE, room=room, phase="failed",
                                   detail=str(e))
                return False, str(e)
            self._proc = proc
            self._state = dict(self.IDLE, room=room, phase="sending")
        threading.Thread(target=self._watch, args=(proc,), daemon=True).start()
        return True, f"navigating to {room}"

    def stop(self):
        """SIGTERM, which rover_nav.py turns into a Nav2 cancel before it exits.

        Killing it outright would leave Nav2 driving to a goal with nobody
        watching, which is the one outcome a stop button must not produce.
        """
        with self._lock:
            proc = self._proc
            if proc is None:
                return False, "not navigating"
            # Still locked, but no longer claiming to be on its way there.
            self._state.update(phase="stopping", remaining=None)
        try:
            proc.terminate()
        except Exception as e:
            return False, str(e)
        return True, "stopping"

    def _watch(self, proc):
        """Read the child's JSON stream to the end, then reap it."""
        tail, verdict = "", None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                # ROS logs to the same stream. Keep the last such line: if the
                # child dies without a verdict it is the only clue there is.
                tail = line
                continue
            with self._lock:
                if self._proc is not proc:
                    break
                kind = ev.get("event")
                if kind == "done":
                    # Held, not published -- see the class docstring.
                    verdict = (ev.get("outcome") or "failed",
                               ev.get("detail") or "")
                elif self._state["phase"] == "stopping":
                    # A stop has landed. A line still in the pipe from before
                    # it must not put the rover back on its way.
                    pass
                elif kind == "sent":
                    self._state.update(phase="driving")
                elif kind == "feedback":
                    self._state.update(phase="driving",
                                       remaining=ev.get("remaining"))
        proc.wait()
        with self._lock:
            if self._proc is not proc:
                return
            self._proc = None
            if verdict is None:
                # Gone without a verdict -- ROS not sourced, nav2 down, killed.
                # Say so, or the buttons stay locked on a goal nobody is
                # pursuing.
                verdict = ("failed",
                           tail or f"rover_nav.py exited {proc.returncode}")
            self._state.update(phase=verdict[0], detail=verdict[1],
                               remaining=None)


NAV = Navigator()


def status():
    mode = current_mode()
    out = {"mode": mode, "teleop_port": TELEOP_PORT, "nav": NAV.snapshot(),
           "health": rover_health.snapshot() if rover_health else {},
           "units": {}}
    for key, m in MODES.items():
        out["units"][key] = [unit_state(u) for u in m["units"]]
    ready = False
    if mode == "teleop":
        ready = teleop_answering()
    elif mode == "auto":
        ready = all(s.startswith("active") for s in out["units"]["auto"])
    out["ready"] = ready
    return out


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0d1117">
<title>Rover</title>
<style>
  :root {
    --bg:#0d1117; --card:#161b22; --edge:#232a33;
    --ink:#e6edf3; --dim:#8b949e;
    --go:#3fb950; --wait:#d29922; --dead:#484f58;
    --accent:#2f81f7;
  }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  body {
    margin:0; min-height:100vh; background:var(--bg); color:var(--ink);
    font:16px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
    display:flex; flex-direction:column; align-items:center;
    padding:max(1.5rem,env(safe-area-inset-top)) 1.1rem 2rem;
  }
  h1 { font-size:1.05rem; font-weight:600; letter-spacing:.02em; margin:0 0 .2rem; }
  .sub { color:var(--dim); font-size:.85rem; margin:0 0 1.6rem; }
  .wrap { width:100%; max-width:26rem; }

  #health {
    display:flex; gap:.5rem; margin:-.4rem 0 1rem; flex-wrap:wrap;
  }
  #health span:empty { display:none; }
  #health span {
    font-size:.8rem; padding:.2rem .55rem; border-radius:999px;
    background:var(--card); border:1px solid var(--edge); color:var(--dim);
  }
  #health .warn { color:#f0b429; border-color:#6b5416; }
  #health .bad  { color:#ff6b6b; border-color:#6b2020; }
  .card {
    background:var(--card); border:1px solid var(--edge); border-radius:14px;
    padding:1.1rem 1.2rem; margin-bottom:.9rem; cursor:pointer;
    display:flex; align-items:center; gap:.9rem;
    transition:border-color .18s, transform .12s, background .18s;
  }
  .card:active { transform:scale(.985); }
  .card[data-on="1"] { border-color:var(--go); background:#132218; }
  .card.busy { pointer-events:none; opacity:.5; }

  .dot { width:.7rem; height:.7rem; border-radius:50%; background:var(--dead);
         flex:none; transition:background .3s; }
  .card[data-on="1"] .dot { background:var(--go); box-shadow:0 0 0 4px #3fb95022; }

  .name { font-weight:600; }
  .blurb { color:var(--dim); font-size:.82rem; }
  .badge { margin-left:auto; font-size:.7rem; color:var(--go);
           text-transform:uppercase; letter-spacing:.09em; opacity:0; }
  .card[data-on="1"] .badge { opacity:1; }

  /* ---- rooms, autonomous mode only ---- */
  #rooms { margin-top:1.7rem; }
  #rooms[hidden] { display:none; }
  .rhead { display:flex; align-items:center; gap:.6rem; margin-bottom:.7rem; }
  .rhead span { font-size:.7rem; color:var(--dim);
                text-transform:uppercase; letter-spacing:.09em; }
  #rstop { margin-left:auto; font:inherit; font-size:.75rem; font-weight:600;
           color:#f8a3ae; background:#3a1c22; border:1px solid #6b2b36;
           border-radius:8px; padding:.32rem .8rem; }
  #rstop[hidden] { display:none; }

  .grid { display:grid; grid-template-columns:repeat(2,1fr); gap:.55rem; }
  .room {
    display:flex; align-items:center; gap:.5rem; text-align:left;
    background:var(--card); border:1px solid var(--edge); border-radius:11px;
    color:var(--ink); font:inherit; font-size:.85rem; padding:.7rem .75rem;
    min-height:3rem;
    transition:border-color .18s, background .18s, opacity .18s, transform .12s;
  }
  .room:active { transform:scale(.98); }
  .room:disabled { opacity:.3; }
  /* The room it is driving to stays lit while every other one greys out --
     one button held down is the whole signal that it is under way. */
  .room.going, .room.here { opacity:1; }
  .room.going { border-color:var(--accent); background:#11203a; }
  .room.here  { border-color:var(--go);     background:#132218; }
  .room .spin { display:none; width:.85rem; height:.85rem; flex:none;
                border-radius:50%; border:2px solid var(--accent);
                border-right-color:transparent; animation:spin .7s linear infinite; }
  .room.going .spin { display:block; }
  .dist { margin-left:auto; font-size:.7rem; color:var(--dim);
          font-variant-numeric:tabular-nums; }
  .rnote { margin:.85rem 0 0; color:var(--dim); font-size:.78rem;
           min-height:1.2em; }
  .rnote.bad { color:#f8a3ae; }

  /* ---- switching overlay ---- */
  .veil {
    position:fixed; inset:0; background:rgba(13,17,23,.96);
    display:flex; align-items:center; justify-content:center; padding:1.2rem;
    opacity:0; pointer-events:none; transition:opacity .28s;
    backdrop-filter:blur(3px);
  }
  .veil.on { opacity:1; pointer-events:auto; }
  .panel { width:100%; max-width:22rem; }
  .panel h2 { font-size:1rem; margin:0 0 1.2rem; font-weight:600; }

  .bar { height:3px; background:var(--edge); border-radius:2px; overflow:hidden;
         margin-bottom:1.3rem; }
  .bar i { display:block; height:100%; width:0; background:var(--accent);
           border-radius:2px; transition:width .5s cubic-bezier(.4,0,.2,1); }

  ol { list-style:none; margin:0; padding:0; }
  li { display:flex; align-items:center; gap:.7rem; padding:.36rem 0;
       color:var(--dim); font-size:.9rem;
       opacity:.35; transform:translateY(3px);
       transition:opacity .35s, transform .35s, color .35s; }
  li.seen { opacity:1; transform:none; }
  li.done { color:var(--ink); }

  .mark { width:1rem; height:1rem; flex:none; position:relative; }
  .mark::before {
    content:""; position:absolute; inset:0; border-radius:50%;
    border:2px solid var(--edge); transition:border-color .3s;
  }
  li.work .mark::before { border-color:var(--wait);
    border-right-color:transparent; animation:spin .7s linear infinite; }
  li.done .mark::before { border-color:var(--go); }
  li.done .mark::after {
    content:""; position:absolute; left:.3rem; top:.12rem;
    width:.26rem; height:.5rem; border:solid var(--go);
    border-width:0 2px 2px 0; transform:rotate(45deg);
    animation:pop .25s ease-out;
  }
  @keyframes spin { to { transform:rotate(360deg); } }
  @keyframes pop { from { opacity:0; transform:rotate(45deg) scale(.4); } }
  @media (prefers-reduced-motion:reduce) {
    * { animation:none !important; transition-duration:.01ms !important; }
  }
  .note { margin-top:1.4rem; color:var(--dim); font-size:.8rem; min-height:1.2em; }
  .go { display:block; margin-top:.2rem; padding:.85rem 1rem; text-align:center;
        background:var(--accent); color:#fff; font-size:.95rem; font-weight:600;
        border-radius:10px; text-decoration:none;
        animation:rise .3s ease-out; }
  .go:active { transform:scale(.985); }
  @keyframes rise { from { opacity:0; transform:translateY(4px); } }
</style></head><body>

<div class="wrap">
  <h1>Rover</h1>
  <p class="sub">Pick a mode.</p>
  <div id="health"><span id="batt"></span><span id="temp"></span></div>
  <div id="cards"></div>

  <div id="rooms" hidden>
    <div class="rhead"><span>Send it to a room</span>
      <button id="rstop" hidden>Stop</button></div>
    <div class="grid" id="rgrid"></div>
    <p class="rnote" id="rnote"></p>
  </div>
</div>

<div class="veil" id="veil"><div class="panel">
  <h2 id="vtitle">Starting</h2>
  <div class="bar"><i id="vbar"></i></div>
  <ol id="vlist"></ol>
  <p class="note" id="vnote"></p>
</div></div>

<script>
const MODES = __MODES__;
const ROOMS = __ROOMS__;          // [[key, label], ...] in a fixed order
let busy = null;                  // mode being switched to
let going = false;                // a room goal is in flight

function card(key, m, on) {
  return `<div class="card" data-mode="${key}" data-on="${on ? 1 : 0}">
    <span class="dot"></span>
    <span><span class="name">${m.label}</span><br>
    <span class="blurb">${m.blurb}</span></span>
    <span class="badge">on</span></div>`;
}

function paint(st) {
  if (busy) return;
  document.getElementById('cards').innerHTML =
    Object.entries(MODES).map(([k, m]) => card(k, m, st.mode === k)).join('');
  document.querySelectorAll('.card').forEach(el =>
    el.onclick = () => switchTo(el.dataset.mode));
}

function paintHealth(h) {
  if (!h) return;
  const b = document.getElementById('batt'), t = document.getElementById('temp');
  b.textContent = h.battery_pct == null ? '' : `${h.battery_pct}%  ${h.volts}V`;
  b.className = h.battery_pct == null ? ''
              : h.battery_pct <= 20 ? 'bad' : h.battery_pct <= 40 ? 'warn' : '';
  t.textContent = h.temp_c == null ? '' : `${h.temp_c}\u00B0C`;
  t.className = h.temp_state === 'hot' ? 'bad'
              : h.temp_state === 'warm' ? 'warn' : '';
}

async function poll() {
  try {
    const st = await (await fetch('/api/status')).json();
    paintHealth(st.health);
    if (busy) progress(st); else { paint(st); paintRooms(st); }
  } catch (e) { /* the server restarts during a switch; just retry */ }
}

/* ---------------------------------------------------------------- rooms */

// Built once. Repainting these from innerHTML on every poll would drop the
// tap highlight mid-press and restart the spinner four times a second.
function buildRooms() {
  document.getElementById('rgrid').innerHTML = ROOMS.map(([key, label]) =>
    `<button class="room" data-room="${key}"><span class="spin"></span>` +
    `<span>${label}</span><span class="dist"></span></button>`).join('');
  document.querySelectorAll('.room').forEach(el =>
    el.onclick = () => goRoom(el.dataset.room));
  document.getElementById('rstop').onclick = () => {
    document.getElementById('rnote').textContent = 'Stopping...';
    fetch('/api/nav/stop', {method:'POST'}).catch(() => {}).then(poll);
  };
}

const SAID = {
  sending:  r => [`Sending it to ${r}...`, 0],
  driving:  r => [`On its way to ${r}. The other rooms are locked until it ` +
                  `arrives -- Stop to change your mind.`, 0],
  stopping: r => [`Telling it to stop...`, 0],
  arrived:  r => [`Arrived at ${r}.`, 0],
  cancelled:r => [`Stopped on the way to ${r}.`, 0],
  rejected: r => [`Nav2 turned down ${r}.`, 1],
  failed:   r => [`Could not reach ${r}.`, 1],
};

function paintRooms(st) {
  const box = document.getElementById('rooms');
  box.hidden = st.mode !== 'auto' || !ROOMS.length;
  if (box.hidden) { going = false; return; }

  const nav = st.nav || {phase:'idle'};
  // Locked for exactly as long as rover_nav.py is alive. 'stopping' is one
  // of those: the cancel takes a moment to reach Nav2 and the rover is still
  // moving until it does.
  going = ['sending', 'driving', 'stopping'].includes(nav.phase);
  document.getElementById('rstop').hidden = !going;

  document.querySelectorAll('.room').forEach(el => {
    const mine = el.dataset.room === nav.room;
    // Locked out while it drives, and until nav2 is actually up: a goal sent
    // before then is refused, which reads as the button being broken.
    el.disabled = going || !st.ready;
    el.classList.toggle('going', going && mine);
    el.classList.toggle('here', !going && mine && nav.phase === 'arrived');
    el.querySelector('.dist').textContent =
      (going && mine && nav.remaining != null)
        ? nav.remaining.toFixed(1) + ' m' : '';
  });

  const note = document.getElementById('rnote');
  const entry = ROOMS.find(([k]) => k === nav.room);
  let text = '', bad = false;
  if (entry && SAID[nav.phase]) {
    const [said, isBad] = SAID[nav.phase](entry[1].toLowerCase());
    text = said + (isBad && nav.detail ? ' ' + nav.detail : '');
    bad = !!isBad;
  } else if (!st.ready) {
    text = 'Waiting for navigation to come up.';
  }
  note.textContent = text;
  note.classList.toggle('bad', bad);
}

async function goRoom(key) {
  if (going) return;
  // Lock now rather than at the next poll, so a second tap in the meantime
  // cannot land. The server refuses one anyway; this just makes the page
  // agree with it.
  going = true;
  document.querySelectorAll('.room').forEach(el => el.disabled = true);
  document.getElementById('rnote').textContent = 'Sending...';
  try { await fetch('/api/goto/' + key, {method:'POST'}); } catch (e) {}
  poll();
}

function progress(st) {
  const m = MODES[busy], states = st.units[busy] || [];
  const done = states.filter(s => s.startsWith('active')).length;
  const frac = states.length ? done / states.length : 0;
  document.getElementById('vbar').style.width =
    Math.round((st.ready ? 1 : Math.min(frac, .92)) * 100) + '%';
  document.querySelectorAll('#vlist li').forEach((li, i) => {
    const s = states[i] || 'inactive';
    li.classList.add('seen');
    li.classList.toggle('done', s.startsWith('active'));
    li.classList.toggle('work', s === 'activating' ||
      (!s.startsWith('active') && i === done));
  });
  if (st.ready) finish(st);
}

function finish(st) {
  const note = document.getElementById('vnote');
  if (busy === 'teleop') {
    // A link the user taps, not location.href and not window.open. Navigating
    // away meant that coming back restored this page from cache with the
    // overlay still up and busy still set, which looked like it had hung. And
    // window.open outside a tap is blocked on phones. An anchor with
    // target=_blank is neither: it opens a tab, and this page stays put.
    const url = `http://${location.hostname}:${st.teleop_port}/`;
    note.innerHTML =
      `<a class="go" href="${url}" target="_blank" rel="noopener">` +
      `Open the controls</a>`;
    note.querySelector('a').addEventListener('click', () => {
      setTimeout(() => {
        busy = null;
        document.getElementById('veil').classList.remove('on');
        poll();
      }, 400);
    });
  } else {
    note.textContent = 'Ready.';
    setTimeout(() => {
      busy = null;
      document.getElementById('veil').classList.remove('on');
      poll();
    }, 900);
  }
}

async function switchTo(key) {
  if (busy) return;
  const st = await (await fetch('/api/status')).json();
  const already = st.mode === key;
  if (already && key === 'teleop' && st.ready) {
    // Already there and serving. The await above costs the user gesture on
    // some phones, so window.open can be blocked -- when it is, fall through
    // to the overlay, whose link the user taps directly.
    if (window.open(`http://${location.hostname}:${st.teleop_port}/`,
                    '_blank', 'noopener')) return;
  } else if (already && st.ready) {
    return;                             // autonomous, already up: nothing to do
  }
  // Otherwise this mode is selected but not finished coming up -- a unit that
  // was never enabled, or one still starting. Show the checklist and wait,
  // rather than absorbing the tap and looking broken. That is what happened
  // when rover-teleop had no symlink: the target was active, its service was
  // not, and tapping did nothing at all.
  busy = key;
  const m = MODES[key];
  document.getElementById('vtitle').textContent = 'Starting ' + m.label.toLowerCase();
  document.getElementById('vlist').innerHTML =
    m.steps.map(s => `<li><span class="mark"></span>${s}</li>`).join('');
  document.getElementById('vnote').textContent = '';
  document.getElementById('vbar').style.width = '0';
  document.getElementById('veil').classList.add('on');
  // Reveal the checklist in sequence, so it reads as a list being worked
  // through rather than eight rows appearing at once.
  document.querySelectorAll('#vlist li').forEach((li, i) =>
    setTimeout(() => li.classList.add('seen'), 60 * i));
  // Nothing to start if we are already in this mode -- the polling below will
  // redirect or finish once whatever is missing comes up.
  if (!already) fetch('/api/switch/' + key, {method: 'POST'}).catch(() => {});
}

buildRooms();
poll();
setInterval(poll, 1200);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body, ctype):
        blob = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        if self.path == "/":
            page = PAGE.replace("__MODES__", json.dumps(
                {k: {kk: v[kk] for kk in ("label", "blurb", "steps")}
                 for k, v in MODES.items()}))
            # The rooms never change while the server runs, so they ship with
            # the page rather than riding along on every status poll.
            page = page.replace("__ROOMS__", json.dumps(
                [[r, room_label(r)] for r in ROOM_NAMES]))
            self._send(200, page, "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send(200, json.dumps(status()), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def _reply(self, ok, detail, code=None):
        self._send(code or (200 if ok else 409),
                   json.dumps({"ok": ok, "detail": detail}),
                   "application/json")

    def do_POST(self):
        if self.path.startswith("/api/switch/"):
            return self._switch(self.path.rsplit("/", 1)[-1])
        if self.path.startswith("/api/goto/"):
            return self._goto(self.path.rsplit("/", 1)[-1])
        if self.path == "/api/nav/stop":
            return self._reply(*NAV.stop())
        self._send(404, "not found", "text/plain")

    def _switch(self, key):
        if key not in MODES:
            return self._send(400, "unknown mode", "text/plain")
        # Leaving autonomous takes Nav2 down under any goal in flight. Cancel
        # it first so the rover is told to stop, rather than being cut off
        # mid-drive and coasting.
        if key != "auto":
            NAV.stop()
        # Conflicts= in the target files stops the other mode; starting the
        # one we want is the whole operation.
        code, out = systemctl("start", "--no-block", MODES[key]["target"],
                              root=True)
        self._reply(code == 0, out, code=200 if code == 0 else 500)

    def _goto(self, room):
        if room not in ROOMS:
            return self._send(400, "unknown room", "text/plain")
        # Nav2 only exists in autonomous mode, and a goal sent in remote
        # control would sit there waiting for a server that is not coming.
        if current_mode() != "auto":
            return self._reply(False, "not in autonomous mode")
        self._reply(*NAV.go(room))

    def log_message(self, *a):
        pass          # one line per poll, every 1.2s, is not worth journalling


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Rover mode switcher on http://<pi>:{PORT}/  "
          f"(teleop expected on {TELEOP_PORT})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
