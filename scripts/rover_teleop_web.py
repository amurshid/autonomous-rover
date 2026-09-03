#!/usr/bin/env python3
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from PIL import Image as PILImage

PAGE = """<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no,viewport-fit=cover">
<title>Rover Control</title><style>
*{box-sizing:border-box;-webkit-user-select:none;user-select:none;-webkit-tap-highlight-color:transparent}
html,body{height:100%}
body{margin:0;background:#12141a;color:#e6e8ee;font-family:system-ui,sans-serif;
display:flex;flex-direction:column;align-items:center;padding:16px;overscroll-behavior:none}
h2{margin:4px 0 12px;font-weight:600;font-size:18px}
#hint{display:none;font-size:12px;color:#9aa3b8;margin-bottom:10px}
@media (orientation:portrait) and (hover:none){#hint{display:block}}
#health{display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap}
#health span:empty{display:none}
#health span{font-size:11px;padding:2px 8px;border-radius:999px;
  color:#9aa3b8;background:#141821;border:1px solid #232838}
#health .warn{color:#f0b429;border-color:#6b5416}
#health .bad{color:#ff6b6b;border-color:#6b2020}

/* Desktop and phone-portrait: camera on top, D-pad below. */
#app{display:grid;gap:10px;width:100%;max-width:420px;
grid-template-columns:1fr 88px 1fr;justify-items:center;align-items:center;
grid-template-areas:"cam cam cam" "tog tog tog" ".  up  ." "lt stop rt"
                    ".  dn  ." "s1 s1 s1" "s2 s2 s2" "st st st"}
#lpad,#rpad,#mid,#bar{display:contents}
.cam{grid-area:cam}#camtog{grid-area:tog}#up{grid-area:up}#dn{grid-area:dn}
#lt{grid-area:lt;justify-self:end}#rt{grid-area:rt;justify-self:start}
#stop{grid-area:stop}#s1{grid-area:s1}#s2{grid-area:s2}#st{grid-area:st}

.cam{position:relative;width:100%;aspect-ratio:4/3;background:#000;
border:1px solid #333a4d;border-radius:12px;overflow:hidden;
display:flex;align-items:center;justify-content:center}
.cam img{width:100%;height:100%;object-fit:contain;display:block}
#camsg{position:absolute;font-size:13px;color:#9aa3b8}
#camtog{background:none;border:none;color:#9aa3b8;font-size:12px;padding:0}
button{background:#232735;color:#e6e8ee;border:1px solid #333a4d;border-radius:14px;
font-size:26px;touch-action:none;transition:background .08s}
button:active,button.on{background:#2f6fd0;border-color:#4a86e8}
#up,#dn,#lt,#rt{width:88px;height:88px}
#stop{width:88px;height:88px;background:#7a2230;border-color:#a03446;
font-size:16px;font-weight:600}
#stop:active{background:#a83a4c}
.sl{width:280px}
#s1{margin-top:10px}
input[type=range]{width:100%}
.lab{display:flex;justify-content:space-between;font-size:13px;color:#9aa3b8;margin-bottom:4px}
#st{font-size:13px;color:#9aa3b8;font-variant-numeric:tabular-nums}

/* Phone held sideways: a gamepad. Turning under the left thumb, throttle under
   the right, camera filling everything between them. The class is set from a
   media query in the head, so a laptop window is never demoted to this. */
.game body{height:100dvh;overflow:hidden;
padding:8px calc(8px + env(safe-area-inset-right))
        calc(8px + env(safe-area-inset-bottom)) calc(8px + env(safe-area-inset-left))}
.game h2,.game #hint{display:none}
.game #app{display:flex;flex-direction:row;align-items:center;gap:10px;
max-width:none;height:100%;touch-action:none}
.game #lpad{display:flex;flex-direction:row;gap:10px}
.game #rpad{display:flex;flex-direction:column;gap:10px}
/* column-reverse puts the slider strip above the video: the band along the
   bottom of a phone belongs to the home indicator, where a slider is either
   unusable or a swipe out of the app. */
.game #mid{display:flex;flex-direction:column-reverse;flex:1;min-width:0;height:100%;gap:6px}
.game .cam{flex:1;min-height:0;aspect-ratio:auto}
.game #bar{display:flex;align-items:center;gap:10px;width:100%}
.game #up,.game #dn,.game #lt,.game #rt{width:84px;height:84px;font-size:28px}
.game #stop{flex:none;width:84px;height:46px;font-size:14px}
.game .sl{flex:1;width:auto;min-width:90px;margin:0}
.game .lab{font-size:11px;margin-bottom:2px}
/* The bar is only as wide as the phone minus two thumbs, so the two readouts
   float in the bottom corners instead of crowding the sliders down to a width
   nobody can aim at. The corners are outside the home indicator, which sits
   centred. */
.game #camtog,.game #st{position:fixed;z-index:2;font-size:11px;white-space:nowrap;
background:rgba(18,20,26,.66);border-radius:6px;padding:3px 6px;
bottom:calc(6px + env(safe-area-inset-bottom))}
.game #camtog{left:calc(8px + env(safe-area-inset-left))}
.game #st{right:calc(8px + env(safe-area-inset-right));min-width:0;text-align:right}
</style>
<script>
// Set before first paint so the layout never visibly rearranges. ?layout=game
// or ?layout=desk forces one, which is also how this gets checked on a desktop.
(function(){
  var q=new URLSearchParams(location.search).get('layout'),
      mq=matchMedia('(orientation:landscape) and (max-height:600px) and (hover:none)');
  function set(){document.documentElement.classList.toggle('game',q?q==='game':mq.matches)}
  if(mq.addEventListener)mq.addEventListener('change',set);else mq.addListener(set);
  addEventListener('orientationchange',set); set();
})();
</script></head><body>
<h2>Rover Control</h2>
<div id="hint">turn your phone sideways for gamepad controls</div>
<div id="health"><span id="batt"></span><span id="temp"></span></div>
<div id="app">
  <div id="lpad">
    <button id="lt" data-l="0" data-a="1">&#9664;</button>
    <button id="rt" data-l="0" data-a="-1">&#9654;</button>
  </div>
  <div id="mid">
    <div class="cam"><img id="cam" alt=""><div id="camsg">connecting camera</div></div>
    <div id="bar">
      <div class="sl" id="s1"><div class="lab"><span>Speed</span><span id="sv">0.50</span></div>
      <input id="spd" type="range" min="0.1" max="1.2" step="0.05" value="0.5"></div>
      <div class="sl" id="s2"><div class="lab"><span>Turn rate</span><span id="tv">6.00</span></div>
      <input id="trn" type="range" min="1" max="16" step="0.5" value="6"></div>
      <button id="stop">STOP</button>
      <button id="camtog">turn camera off</button>
      <div id="st">idle</div>
    </div>
  </div>
  <div id="rpad">
    <button id="up" data-l="1" data-a="0">&#9650;</button>
    <button id="dn" data-l="-1" data-a="0">&#9660;</button>
  </div>
</div>
<script>
const spd=document.getElementById('spd'),trn=document.getElementById('trn'),
      st=document.getElementById('st'),sv=document.getElementById('sv'),tv=document.getElementById('tv');
spd.oninput=()=>sv.textContent=(+spd.value).toFixed(2);
trn.oninput=()=>tv.textContent=(+trn.value).toFixed(2);

// Directions are summed over everything held rather than replaced by the last
// press: on the gamepad layout each thumb owns a pad, and forward-plus-turn is
// how the rover drives an arc. Opposing presses cancel, which is also correct.
const held=new Map();
let timer=null;
function send(l,a){
  fetch(`/cmd?lin=${l}&ang=${a}`).catch(()=>st.textContent='connection lost');
  st.textContent=(l||a)?`lin ${l.toFixed(2)}  ang ${a.toFixed(2)}`:'stopped';
}
function push(){
  let l=0,a=0;
  held.forEach(v=>{l+=v[0];a+=v[1]});
  l=Math.max(-1,Math.min(1,l)); a=Math.max(-1,Math.min(1,a));
  send(l*(+spd.value),a*(+trn.value));
}
function refresh(){
  push();
  if(held.size&&!timer) timer=setInterval(push,120);
  else if(!held.size&&timer){clearInterval(timer);timer=null}
}
function hold(k,v){held.set(k,v);refresh()}
function release(k){if(held.delete(k))refresh()}
function allStop(){
  held.clear();
  if(timer){clearInterval(timer);timer=null}
  document.querySelectorAll('button[data-l]').forEach(b=>b.classList.remove('on'));
  send(0,0);
}
document.querySelectorAll('button[data-l]').forEach(b=>{
  const v=[+b.dataset.l,+b.dataset.a];
  b.addEventListener('pointerdown',e=>{e.preventDefault();b.classList.add('on');hold(b,v)});
  ['pointerup','pointerleave','pointercancel'].forEach(ev=>
    b.addEventListener(ev,()=>{b.classList.remove('on');release(b)}));
});
document.getElementById('stop').addEventListener('pointerdown',e=>{e.preventDefault();allStop()});
const KEYS={ArrowUp:[1,0],w:[1,0],ArrowDown:[-1,0],s:[-1,0],
             ArrowLeft:[0,1],a:[0,1],ArrowRight:[0,-1],d:[0,-1]};
addEventListener('keydown',e=>{const k=KEYS[e.key];if(k&&!held.has(e.key))hold(e.key,k)});
addEventListener('keyup',e=>release(e.key));
addEventListener('blur',allStop);

// Battery and Pi temperature. Slow poll: neither moves fast, and this page
// shares a warm Pi with everything else.
function health(){
  fetch('/health').then(r=>r.json()).then(h=>{
    const b=document.getElementById('batt'),t=document.getElementById('temp');
    b.textContent = h.battery_pct==null ? '' : h.battery_pct+'%  '+h.volts+'V';
    b.className = h.battery_pct==null ? ''
                : h.battery_pct<=20 ? 'bad' : h.battery_pct<=40 ? 'warn' : '';
    t.textContent = h.temp_c==null ? '' : h.temp_c+'\u00B0C';
    t.className = h.temp_state==='hot' ? 'bad'
                : h.temp_state==='warm' ? 'warn' : '';
  }).catch(()=>{});
}
health(); setInterval(health, 10000);

// Camera. /stream is one long multipart response, which is cheap and low
// latency where it works; some mobile browsers never render it, so a frame
// that has not arrived within a few seconds demotes this viewer to polling
// /snapshot.jpg, which every browser can display.
const cam=document.getElementById('cam'),camsg=document.getElementById('camsg'),
      camtog=document.getElementById('camtog');
let camOn=true,poll=null,watchdog=null,gotFrame=false,gap=150;
function msg(t){camsg.textContent=t;camsg.style.display=t?'':'none'}
function camStop(){
  if(poll){clearTimeout(poll);poll=null}
  if(watchdog){clearTimeout(watchdog);watchdog=null}
  gotFrame=false;
  cam.removeAttribute('src');
}
function camStart(){
  camStop(); msg('connecting camera');
  cam.src='/stream?'+Date.now();
  watchdog=setTimeout(()=>{if(!gotFrame&&camOn)camPoll()},4000);
}
function camPoll(){
  if(poll)return;
  cam.removeAttribute('src');
  // setTimeout, not setInterval: a camera that is gone answers instantly with
  // a 503, and retrying that every 150 ms is a request storm aimed at a Pi
  // that has nothing to send. Widen the gap on failure, snap back on a frame.
  const tick=()=>{
    poll=setTimeout(tick,gap);
    if(!document.hidden)cam.src='/snapshot.jpg?'+Date.now();
  };
  tick();
}
cam.addEventListener('load',()=>{gotFrame=true;gap=150;msg('')});
cam.addEventListener('error',()=>{
  if(!camOn)return;
  if(poll){gap=Math.min(gap*2,2000);msg('camera offline')}  // already retrying
  else{clearTimeout(watchdog);camPoll()}
});
camtog.onclick=()=>{
  camOn=!camOn;
  camtog.textContent=camOn?'turn camera off':'turn camera on';
  if(camOn)camStart(); else{camStop();msg('camera off')}
};
// A backgrounded tab holding the stream open keeps the Pi encoding for nobody.
addEventListener('visibilitychange',()=>{
  if(document.hidden)camStop();
  else if(camOn)camStart();
});
camStart();
</script></body></html>"""


class Camera:
    """Latest camera frame, JPEG-encoded for the browser.

    Encoding runs on its own thread rather than in the ROS callback: that
    callback shares an executor with the cmd_vel timer, and a timer held up
    behind a JPEG encode is a rover that keeps driving on a stale command.

    Frames are dropped rather than queued -- a viewer that falls behind wants
    the present, not a backlog -- and nothing is encoded at all unless somebody
    asked for a frame in the last few seconds, so an unwatched stream costs the
    Pi (which throttles at 80 C) one memcpy per frame.

    A camera that dies has to stay cheap. It measured 273% CPU once -- nearly
    three cores, on a Pi that throttles at 80 C -- for a stream showing
    nothing. Two causes, both handled here: readers used to wake each other
    (see the conditions below), and a reader with no frame to return used to
    block for its full timeout, so the page's 150 ms retries piled up twenty
    deep. Now a stale camera is answered immediately and the page backs off.
    """

    DEMAND_TTL = 3.0
    STALE_S = 3.0     # no frame for this long and the camera counts as gone

    def __init__(self, quality, max_fps, log):
        # Two conditions over one lock. Readers must never wake other
        # readers: a woken reader re-notifies, and with no frames arriving
        # nothing ever breaks the cycle, so the pair spins flat out. Readers
        # signal the encoder on `demand`, the encoder signals readers on
        # `frame`, and neither can wake its own kind.
        self.lock = threading.Lock()
        self.frame = threading.Condition(self.lock)
        self.demand = threading.Condition(self.lock)
        self.log = log
        self.quality = quality
        self.min_period = 1.0 / max_fps if max_fps > 0 else 0.0
        self.raw = None           # (data, height, width, encoding)
        self.raw_seq = 0
        self.jpeg = None
        self.jpeg_seq = 0
        self.last_demand = 0.0
        self.last_frame = 0.0
        self.seen = 0
        self.warned = False
        self.stale_logged = False
        threading.Thread(target=self._encode_loop, daemon=True).start()

    def _wanted(self):
        return time.monotonic() - self.last_demand < self.DEMAND_TTL

    def submit(self, msg):
        """ROS callback. Keeps a copy of the newest frame and nothing else."""
        with self.lock:
            self.seen += 1
            # Set before the _wanted() check: liveness must be observable even
            # while nothing is being encoded, or an unwatched camera looks dead.
            self.last_frame = time.monotonic()
            if self.stale_logged:
                self.stale_logged = False
                self.log('camera frames resumed')
            if not self._wanted():
                return
            self.raw = (bytes(msg.data), msg.height, msg.width, msg.encoding)
            self.raw_seq += 1
            self.demand.notify()

    def next_jpeg(self, seq, timeout):
        """Block for a frame newer than `seq`. Returns (jpeg, seq) or (None, seq)."""
        deadline = time.monotonic() + timeout
        with self.lock:
            while self.jpeg_seq == seq:
                now = time.monotonic()
                self.last_demand = now
                # Nothing is coming. Say so now rather than holding the
                # request open for its full timeout: the page retries every
                # 150 ms, and 3 s answers stack twenty threads deep per viewer.
                if now - self.last_frame >= self.STALE_S:
                    if self.seen and not self.stale_logged:
                        self.stale_logged = True
                        self.log(f'no camera frames for '
                                 f'{now - self.last_frame:.0f}s; '
                                 f'is rover-camera running?')
                    return None, seq
                self.demand.notify()      # wake the encoder if it went idle
                left = deadline - now
                if left <= 0:
                    return None, seq
                self.frame.wait(min(left, 0.5))
            self.last_demand = time.monotonic()
            return self.jpeg, self.jpeg_seq

    def _encode_loop(self):
        done = 0
        next_ok = 0.0
        while True:
            with self.lock:
                while self.raw_seq == done or not self._wanted():
                    self.demand.wait(0.5)
                raw, done = self.raw, self.raw_seq
            now = time.monotonic()
            if now < next_ok:
                time.sleep(next_ok - now)
            next_ok = time.monotonic() + self.min_period
            jpeg = self._to_jpeg(raw)
            if jpeg is None:
                continue
            with self.lock:
                self.jpeg = jpeg
                self.jpeg_seq += 1
                self.frame.notify_all()

    def _to_jpeg(self, raw):
        data, h, w, encoding = raw
        # Same encodings, and the same flip, as vpr_logger.py: libcamera's
        # RGB888 arrives byte-order BGR and the topic says so.
        if encoding not in ('rgb8', 'bgr8'):
            if not self.warned:
                self.warned = True
                self.log(f'unexpected encoding {encoding}, camera not shown')
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        try:
            arr = arr.reshape(h, w, 3)
        except ValueError:
            return None
        if encoding == 'bgr8':
            arr = arr[:, :, ::-1]
        buf = io.BytesIO()
        PILImage.fromarray(np.ascontiguousarray(arr)).save(
            buf, 'JPEG', quality=self.quality)
        return buf.getvalue()


class Teleop(Node):
    def __init__(self):
        super().__init__('rover_teleop_web')
        self.declare_parameter('timeout', 0.7)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('jpeg_quality', 70)
        self.declare_parameter('stream_fps', 12.0)
        self.timeout = self.get_parameter('timeout').value
        self.lock = threading.Lock()
        self.lin = 0.0
        self.ang = 0.0
        self.last = 0.0
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_timer(0.1, self.tick)

        self.cam = Camera(self.get_parameter('jpeg_quality').value,
                          self.get_parameter('stream_fps').value,
                          self.get_logger().warn)
        topic = self.get_parameter('image_topic').value
        self.create_subscription(
            Image, topic, self.cam.submit, qos_profile_sensor_data)

    def set_cmd(self, lin, ang):
        with self.lock:
            self.lin, self.ang = lin, ang
            self.last = time.monotonic()

    def tick(self):
        with self.lock:
            if time.monotonic() - self.last > self.timeout:
                self.lin = self.ang = 0.0
            msg = Twist()
            msg.linear.x = self.lin
            msg.angular.z = self.ang
        self.pub.publish(msg)


# A readout is not worth taking the controls down for, so a missing or
# broken rover_health leaves the pills blank rather than failing the page.
try:
    import rover_health
except Exception:
    rover_health = None


class Handler(BaseHTTPRequestHandler):
    node = None
    protocol_version = 'HTTP/1.1'

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/cmd':
            q = parse_qs(u.query)
            try:
                lin = float(q.get('lin', ['0'])[0])
                ang = float(q.get('ang', ['0'])[0])
            except ValueError:
                lin = ang = 0.0
            Handler.node.set_cmd(
                max(-1.5, min(1.5, lin)), max(-20.0, min(20.0, ang)))
            self.send_response(204)
            self.end_headers()
        elif u.path in ('/', '/index.html'):
            self.send_bytes(PAGE.encode(), 'text/html; charset=utf-8')
        elif u.path == '/snapshot.jpg':
            jpeg, _ = Handler.node.cam.next_jpeg(0, 3.0)
            if jpeg is None:
                self.send_error_empty(503)
            else:
                self.send_bytes(jpeg, 'image/jpeg')
        elif u.path == '/health':
            body = json.dumps(
                rover_health.snapshot() if rover_health else {}).encode()
            self.send_bytes(body, 'application/json')
        elif u.path == '/stream':
            self.stream()
        else:
            self.send_error_empty(404)

    def stream(self):
        cam = Handler.node.cam
        jpeg, seq = cam.next_jpeg(0, 5.0)
        if jpeg is None:                  # camera node down, or no frames yet
            self.send_error_empty(503)
            return
        self.close_connection = True      # no length to give, so no keep-alive
        self.send_response(200)
        self.send_header('Content-Type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache, private')
        self.send_header('Connection', 'close')
        self.end_headers()
        try:
            while jpeg is not None:
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\n'
                                 b'Content-Length: %d\r\n\r\n' % len(jpeg))
                self.wfile.write(jpeg)
                self.wfile.write(b'\r\n')
                jpeg, seq = cam.next_jpeg(seq, 5.0)
        except (BrokenPipeError, ConnectionResetError):
            pass                          # viewer left; the page reconnects

    def send_bytes(self, body, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        if content_type == 'image/jpeg':
            self.send_header('Cache-Control', 'no-cache, private')
        self.end_headers()
        self.wfile.write(body)

    def send_error_empty(self, code):
        self.send_response(code)
        self.send_header('Content-Length', '0')
        self.end_headers()

    def log_message(self, *a):
        pass


def main():
    rclpy.init()
    node = Teleop()
    Handler.node = node
    # Threaded: the video stream is one response that never ends, and on a
    # single-threaded server it would block every /cmd behind it.
    srv = ThreadingHTTPServer(('0.0.0.0', 8080), Handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    node.get_logger().info('Teleop web UI on http://<pi-ip>:8080')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
