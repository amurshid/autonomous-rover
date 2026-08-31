#!/usr/bin/env python3
"""Mode switcher for the rover, on a page anyone can use.

The rover runs one of two modes and never both, because Nav2 and the teleop
page would otherwise fight over /cmd_vel:

  autonomous  lidar, cartographer, relocalisation, nav2, voice
  remote      the teleop page, driven by hand

Switching is two systemd targets with Conflicts= between them, so starting
either stops the other. That is a one-line operation over SSH and no use at all
to someone holding a phone, which is what this serves.

Runs in both modes -- it is the one thing that must never go down, or there is
no way back. Needs no ROS: it only reads systemctl and starts targets, through
a sudoers rule limited to exactly those commands.

    python3 rover_mode_web.py            # port 80, or $ROVER_MODE_PORT
"""

import json
import os
import socket
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("ROVER_MODE_PORT", "80"))
TELEOP_PORT = int(os.environ.get("ROVER_TELEOP_PORT", "8080"))

# Order matters: the page shows these as a checklist, and reading it top to
# bottom should match what actually comes up.
MODES = {
    "auto": {
        "target": "rover.target",
        "label": "Autonomous",
        "blurb": "Finds itself, drives itself, listens for commands.",
        "units": ["rover-bridge", "rover-camera", "rover-lidar",
                  "rover-cartographer", "rover-initialpose",
                  "rover-relocalise", "rover-nav2", "rover-ai"],
        "steps": ["Motors", "Camera", "Lidar", "Map", "Pose bridge",
                  "Finding itself", "Navigation", "Voice"],
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


def status():
    mode = current_mode()
    out = {"mode": mode, "teleop_port": TELEOP_PORT, "units": {}}
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
</style></head><body>

<div class="wrap">
  <h1>Rover</h1>
  <p class="sub">Pick a mode.</p>
  <div id="cards"></div>
</div>

<div class="veil" id="veil"><div class="panel">
  <h2 id="vtitle">Starting</h2>
  <div class="bar"><i id="vbar"></i></div>
  <ol id="vlist"></ol>
  <p class="note" id="vnote"></p>
</div></div>

<script>
const MODES = __MODES__;
let busy = null;

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

async function poll() {
  try {
    const st = await (await fetch('/api/status')).json();
    if (busy) progress(st); else paint(st);
  } catch (e) { /* the server restarts during a switch; just retry */ }
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
    note.textContent = 'Opening the controls...';
    setTimeout(() => {
      location.href = `http://${location.hostname}:${st.teleop_port}/`;
    }, 650);
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
  if (st.mode === key) {
    // Already in this mode -- go straight through rather than restarting it.
    if (key === 'teleop' && st.ready) {
      location.href = `http://${location.hostname}:${st.teleop_port}/`;
    }
    return;
  }
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
  fetch('/api/switch/' + key, {method: 'POST'}).catch(() => {});
}

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
            self._send(200, page, "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send(200, json.dumps(status()), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if not self.path.startswith("/api/switch/"):
            return self._send(404, "not found", "text/plain")
        key = self.path.rsplit("/", 1)[-1]
        if key not in MODES:
            return self._send(400, "unknown mode", "text/plain")
        # Conflicts= in the target files stops the other mode; starting the
        # one we want is the whole operation.
        code, out = systemctl("start", "--no-block", MODES[key]["target"],
                              root=True)
        self._send(200 if code == 0 else 500,
                   json.dumps({"ok": code == 0, "detail": out}),
                   "application/json")

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
