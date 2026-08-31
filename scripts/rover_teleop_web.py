#!/usr/bin/env python3
import io
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
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>Rover Control</title><style>
*{box-sizing:border-box;-webkit-user-select:none;user-select:none;-webkit-tap-highlight-color:transparent}
body{margin:0;background:#12141a;color:#e6e8ee;font-family:system-ui,sans-serif;
display:flex;flex-direction:column;align-items:center;padding:16px}
h2{margin:4px 0 12px;font-weight:600;font-size:18px}
.cam{position:relative;width:100%;max-width:420px;aspect-ratio:4/3;background:#000;
border:1px solid #333a4d;border-radius:12px;overflow:hidden;margin-bottom:14px;
display:flex;align-items:center;justify-content:center}
.cam img{width:100%;height:100%;object-fit:contain;display:block}
#camsg{position:absolute;font-size:13px;color:#9aa3b8}
#camtog{background:none;border:none;color:#9aa3b8;font-size:12px;padding:0 0 12px}
.pad{display:grid;grid-template-columns:repeat(3,88px);grid-template-rows:repeat(3,88px);gap:10px}
button{background:#232735;color:#e6e8ee;border:1px solid #333a4d;border-radius:14px;
font-size:26px;touch-action:none;transition:background .08s}
button:active,button.on{background:#2f6fd0;border-color:#4a86e8}
#stop{background:#7a2230;border-color:#a03446;font-size:16px;font-weight:600}
#stop:active{background:#a83a4c}
.sl{width:280px;margin-top:20px}
input[type=range]{width:100%}
.lab{display:flex;justify-content:space-between;font-size:13px;color:#9aa3b8;margin-bottom:4px}
#st{margin-top:16px;font-size:13px;color:#9aa3b8;font-variant-numeric:tabular-nums}
</style></head><body>
<h2>Rover Control</h2>
<div class="cam"><img id="cam" alt=""><div id="camsg">connecting camera</div></div>
<button id="camtog">turn camera off</button>
<div class="pad">
  <div></div><button data-l="1" data-a="0">&#9650;</button><div></div>
  <button data-l="0" data-a="1">&#9664;</button>
  <button id="stop">STOP</button>
  <button data-l="0" data-a="-1">&#9654;</button>
  <div></div><button data-l="-1" data-a="0">&#9660;</button><div></div>
</div>
<div class="sl"><div class="lab"><span>Speed</span><span id="sv">0.50</span></div>
<input id="spd" type="range" min="0.1" max="1.2" step="0.05" value="0.5"></div>
<div class="sl"><div class="lab"><span>Turn rate</span><span id="tv">1.00</span></div>
<input id="trn" type="range" min="1" max="16" step="0.5" value="6"></div>
<div id="st">idle</div>
<script>
const spd=document.getElementById('spd'),trn=document.getElementById('trn'),
      st=document.getElementById('st'),sv=document.getElementById('sv'),tv=document.getElementById('tv');
spd.oninput=()=>sv.textContent=(+spd.value).toFixed(2);
trn.oninput=()=>tv.textContent=(+trn.value).toFixed(2);
let cur=null,timer=null;
function send(l,a){
  fetch(`/cmd?lin=${l}&ang=${a}`).catch(()=>st.textContent='connection lost');
  st.textContent=(l||a)?`lin ${l.toFixed(2)}  ang ${a.toFixed(2)}`:'stopped';
}
function start(dl,da){
  if(timer)clearInterval(timer);
  cur={dl,da};
  const tick=()=>send(dl*(+spd.value),da*(+trn.value));
  tick(); timer=setInterval(tick,120);
}
function stop(){ if(timer)clearInterval(timer); timer=null; cur=null; send(0,0); }
document.querySelectorAll('button[data-l]').forEach(b=>{
  const dl=+b.dataset.l, da=+b.dataset.a;
  b.addEventListener('pointerdown',e=>{e.preventDefault();b.classList.add('on');start(dl,da)});
  ['pointerup','pointerleave','pointercancel'].forEach(ev=>
    b.addEventListener(ev,()=>{b.classList.remove('on');stop()}));
});
document.getElementById('stop').addEventListener('pointerdown',e=>{e.preventDefault();stop()});
const KEYS={ArrowUp:[1,0],w:[1,0],ArrowDown:[-1,0],s:[-1,0],
             ArrowLeft:[0,1],a:[0,1],ArrowRight:[0,-1],d:[0,-1]};
let held=null;
addEventListener('keydown',e=>{const k=KEYS[e.key];if(k&&held!==e.key){held=e.key;start(k[0],k[1])}});
addEventListener('keyup',e=>{if(held===e.key){held=null;stop()}});
addEventListener('blur',stop);

// Camera. /stream is one long multipart response, which is cheap and low
// latency where it works; some mobile browsers never render it, so a frame
// that has not arrived within a few seconds demotes this viewer to polling
// /snapshot.jpg, which every browser can display.
const cam=document.getElementById('cam'),camsg=document.getElementById('camsg'),
      camtog=document.getElementById('camtog');
let camOn=true,poll=null,watchdog=null,gotFrame=false;
function msg(t){ camsg.textContent=t; camsg.style.display=t?'':'none'; }
function camStop(){
  if(poll){clearInterval(poll);poll=null}
  if(watchdog){clearTimeout(watchdog);watchdog=null}
  gotFrame=false;
  cam.removeAttribute('src');
}
function camStart(){
  camStop(); msg('connecting camera');
  cam.src='/stream?'+Date.now();
  watchdog=setTimeout(()=>{ if(!gotFrame&&camOn) camPoll(); },4000);
}
function camPoll(){
  if(poll)return;
  cam.removeAttribute('src');
  poll=setInterval(()=>{ if(!document.hidden) cam.src='/snapshot.jpg?'+Date.now(); },150);
}
cam.addEventListener('load',()=>{gotFrame=true;msg('')});
cam.addEventListener('error',()=>{
  if(!camOn)return;
  if(poll){ msg('camera unavailable'); }        // polling already retrying
  else { clearTimeout(watchdog); camPoll(); }
});
camtog.onclick=()=>{
  camOn=!camOn;
  camtog.textContent=camOn?'turn camera off':'turn camera on';
  if(camOn) camStart(); else { camStop(); msg('camera off'); }
};
// A backgrounded tab holding the stream open keeps the Pi encoding for nobody.
addEventListener('visibilitychange',()=>{
  if(document.hidden) camStop();
  else if(camOn) camStart();
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
    """

    DEMAND_TTL = 3.0

    def __init__(self, quality, max_fps, log):
        self.cv = threading.Condition()
        self.log = log
        self.quality = quality
        self.min_period = 1.0 / max_fps if max_fps > 0 else 0.0
        self.raw = None           # (data, height, width, encoding)
        self.raw_seq = 0
        self.jpeg = None
        self.jpeg_seq = 0
        self.last_demand = 0.0
        self.seen = 0
        self.warned = False
        threading.Thread(target=self._encode_loop, daemon=True).start()

    def _wanted(self):
        return time.monotonic() - self.last_demand < self.DEMAND_TTL

    def submit(self, msg):
        """ROS callback. Keeps a copy of the newest frame and nothing else."""
        with self.cv:
            self.seen += 1
            if not self._wanted():
                return
            self.raw = (bytes(msg.data), msg.height, msg.width, msg.encoding)
            self.raw_seq += 1
            self.cv.notify_all()

    def next_jpeg(self, seq, timeout):
        """Block for a frame newer than `seq`. Returns (jpeg, seq) or (None, seq)."""
        deadline = time.monotonic() + timeout
        with self.cv:
            while self.jpeg_seq == seq:
                self.last_demand = time.monotonic()
                self.cv.notify_all()      # wake the encoder if it went idle
                left = deadline - self.last_demand
                if left <= 0:
                    return None, seq
                self.cv.wait(min(left, 0.5))
            self.last_demand = time.monotonic()
            return self.jpeg, self.jpeg_seq

    def _encode_loop(self):
        done = 0
        next_ok = 0.0
        while True:
            with self.cv:
                while self.raw_seq == done or not self._wanted():
                    self.cv.wait(0.5)
                raw, done = self.raw, self.raw_seq
            now = time.monotonic()
            if now < next_ok:
                time.sleep(next_ok - now)
            next_ok = time.monotonic() + self.min_period
            jpeg = self._to_jpeg(raw)
            if jpeg is None:
                continue
            with self.cv:
                self.jpeg = jpeg
                self.jpeg_seq += 1
                self.cv.notify_all()

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
