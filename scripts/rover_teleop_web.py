#!/usr/bin/env python3
import threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

PAGE = """<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>Rover Control</title><style>
*{box-sizing:border-box;-webkit-user-select:none;user-select:none;-webkit-tap-highlight-color:transparent}
body{margin:0;background:#12141a;color:#e6e8ee;font-family:system-ui,sans-serif;
display:flex;flex-direction:column;align-items:center;padding:16px}
h2{margin:4px 0 12px;font-weight:600;font-size:18px}
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
</script></body></html>"""


class Teleop(Node):
    def __init__(self):
        super().__init__('rover_teleop_web')
        self.declare_parameter('timeout', 0.7)
        self.timeout = self.get_parameter('timeout').value
        self.lock = threading.Lock()
        self.lin = 0.0
        self.ang = 0.0
        self.last = 0.0
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_timer(0.1, self.tick)

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
            body = PAGE.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


def main():
    rclpy.init()
    node = Teleop()
    Handler.node = node
    srv = HTTPServer(('0.0.0.0', 8080), Handler)
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
