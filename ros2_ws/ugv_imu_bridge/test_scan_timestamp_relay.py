#!/usr/bin/env python3
"""Offline test of the relay's timing against a model of the LD19 driver.

    python3 test_scan_timestamp_relay.py [path to scan_timestamp_relay.py]

Needs the ROS environment sourced only because the relay imports rclpy; no
node is started. Simulates ldlidar_stl_ros2's 10 Hz publish loop at idle and
under load, runs the committed-before relays and this one over it, and replays
Cartographer's range_data_collator rule on the output: rays earlier than the
previous scan's last ray are dropped, and a scan ending no later than the
previous one is ignored. Exits non-zero on any failure.
"""
import importlib.util
import os
import random
import sys

RELAY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'scan_timestamp_relay.py')
spec = importlib.util.spec_from_file_location('relay', RELAY)
relay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(relay)

MS = 1_000_000
N = 501
failures = 0


def check(name, ok, detail=''):
    global failures
    failures += 0 if ok else 1
    print(f'  {"PASS" if ok else "FAIL"}  {name}' + (f'  ({detail})' if detail else ''))


# ---------------------------------------------------------------- the driver
def driver_stream(ticks, p_lidar_ms, loop_delay, relay_delay, seed):
    """Model demo.cpp: a 10 Hz WallRate loop publishing the latest completed
    revolution, stamped now(), scan_time = now - previous publish."""
    rng = random.Random(seed)
    out = []
    now, last_interval, last_rev, last_pub, prev_receipt = 0, 0, -1, None, 0
    rev0 = 50 * MS
    for _ in range(ticks):
        nxt = last_interval + 100 * MS            # rclcpp::WallRate::sleep
        if now < nxt:
            now = last_interval = nxt
        elif now - nxt > 100 * MS:
            last_interval = now
        else:
            last_interval = nxt
        now += int(loop_delay(rng))
        j = int((now - rev0) // (p_lidar_ms * MS))
        if j <= last_rev:
            now += MS
            continue                               # DATA_WAIT: nothing published
        last_rev = j
        true_end = rev0 + j * p_lidar_ms * MS
        if last_pub is None:
            last_pub = now                         # first_scan is swallowed
            continue
        scan_time = (now - last_pub) / 1e9
        last_pub = now
        receipt = max(prev_receipt + 1, now + int(relay_delay(rng)))
        prev_receipt = receipt
        out.append((now, receipt, scan_time, true_end))
        now += MS
    return out


# ---------------------------------------------------------------- the relays
def old_relay(stream):
    pub, last = [], 0
    for stamp, receipt, st, true_end in stream:
        first = receipt - int(st * 1e9)
        if first <= last:
            first = last + 1000
        last = first
        pub.append((first, int(st * 1e9), receipt, true_end))
    return pub


def relay_9b88a5e(stream):
    pub, last = [], None
    for stamp, receipt, st, true_end in stream:
        span = int(st * 1e9) or 100 * MS
        arrival = receipt - span
        if last is None or arrival >= last + span:
            s = arrival
        elif last + span <= receipt:
            s = last + span
        else:
            continue
        last = s
        pub.append((s, span, receipt, true_end))
    return pub


def new_relay(stream):
    r, pub = relay.Restamper(), []
    for stamp, receipt, st, true_end in stream:
        placed = r.place(stamp, receipt, st)
        if placed:
            pub.append((placed[0], placed[1], receipt, true_end))
    return pub


def cartographer(pub):
    """Replay Cartographer's collator: rays earlier than the previous scan's
    last ray are dropped; a scan ending no later than the previous is ignored."""
    drops = events = ignored = future = 0
    worst_future = 0.0
    lag = []
    for i, (first, span, receipt, true_end) in enumerate(pub):
        end = first + span
        if end > receipt:
            future += 1
            worst_future = max(worst_future, (end - receipt) / MS)
        lag.append((end - true_end) / MS)
        if i == 0:
            continue
        prev_end = pub[i - 1][0] + pub[i - 1][1]
        if end <= prev_end:
            ignored += 1
            continue
        if first < prev_end:
            inc = span / (N - 1)
            d = min(N, int((prev_end - first) / inc) + 1)
            drops += d
            events += 1
    return drops, events, ignored, future, worst_future, lag


# ---------------------------------------------------------------- scenarios
SCENARIOS = {
    'idle': dict(loop_delay=lambda g: g.uniform(0, 2 * MS),
                 relay_delay=lambda g: g.uniform(1 * MS, 5 * MS)),
    'load': dict(loop_delay=lambda g: g.uniform(0, 2 * MS) if g.random() < 0.85 else g.uniform(0, 150 * MS),
                 relay_delay=lambda g: g.uniform(1 * MS, 5 * MS) if g.random() < 0.8 else g.uniform(5 * MS, 80 * MS)),
}
print('== simulated driver, 3000 loop ticks, lidar at 100.3 ms/rev')
for name, sc in SCENARIOS.items():
    stream = driver_stream(3000, 100.3, sc['loop_delay'], sc['relay_delay'], seed=7)
    long_scans = sum(1 for s in stream if s[2] > 0.12)
    print(f'-- {name}: {len(stream)} driver scans, {long_scans} claiming > 120 ms')
    for label, fn in (('receipt relay', old_relay), ('9b88a5e relay', relay_9b88a5e), ('new relay', new_relay)):
        pub = fn(stream)
        drops, events, ignored, future, worst, lag = cartographer(pub)
        lag.sort()
        print(f'   {label:16s} published {len(pub):4d}  drop events {events:4d} ({drops:6d} pts)  '
              f'ignored {ignored:3d}  rays in future: {future:4d} scans (worst {worst:5.1f} ms)  '
              f'last ray vs true revolution end: median {lag[len(lag)//2]:+6.1f} ms, p95 {lag[int(len(lag)*.95)]:+6.1f}')
        if label == 'new relay':
            check(f'{name}: no Cartographer drops', drops == 0, f'{drops} pts')
            check(f'{name}: no scan ignored', ignored == 0)
            check(f'{name}: no ray stamped in the future', future == 0)
            check(f'{name}: skips under 10%', len(pub) >= 0.9 * len(stream), f'{len(stream) - len(pub)} skipped')

# ---------------------------------------------------------------- edge cases
print('== edge cases')


def run(seq):
    r = relay.Restamper()
    return r, [r.place(s * MS, rc * MS, st) for s, rc, st in seq]


r, out = run([(1000, 1003, .1), (1100, 1103, .1), (1300, 1303, .2), (1400, 1403, .1)])
check('skipped publish: all four published', all(out), str(out))
SPAN = int(100 * MS * relay.SPAN_RATIO)
check('skipped publish: every span is the same labelled span', all(o[1] == SPAN for o in out), f'{SPAN / MS:.1f} ms')
check('skipped publish: counted as mislabelled', r.mislabelled == 1, str(r.mislabelled))
ends = [o[0] + o[1] for o in out]
check('skipped publish: no overlap', all(out[i][0] > ends[i - 1] for i in range(1, 4)))

r, out = run([(1000, 1003, .1), (1100, 1103, .1), (1226, 1229, .126), (1300, 1303, .074), (1400, 1403, .1)])
pub = [o for o in out if o]
check('bunched: no overlap', all(pub[i][0] > pub[i - 1][0] + pub[i - 1][1] for i in range(1, len(pub))))
check('bunched: no ray in the future',
      all(o[0] + o[1] <= rc * MS for o, (_, rc, _) in zip(out, [(1000, 1003, 0), (1100, 1103, 0), (1226, 1229, 0), (1300, 1303, 0), (1400, 1403, 0)]) if o))
check('bunched: at most one skipped', r.bunched <= 1, f'{r.bunched} skipped')

r, out = run([(1000, 1003, .1), (1100, 1103, .1), (1095, 1203, -.005), (1300, 1303, .1)])
pub = [o for o in out if o]
check('driver stamp backwards: output stays ordered without overlap',
      all(pub[i][0] > pub[i - 1][0] + pub[i - 1][1] for i in range(1, len(pub))))

r, out = run([(11000, 1003, .1)])
check('driver clock 10 s off: falls back to receipt', out[0][0] + out[0][1] == 1003 * MS, str(out[0]))

r, out = run([(1000, 1003, .1)])
check('first scan: last ray at the driver stamp', out[0] == (1000 * MS - SPAN, SPAN), str(out[0]))

print(f'\n{"ALL PASS" if failures == 0 else f"{failures} FAILURE(S)"}')
sys.exit(1 if failures else 0)
