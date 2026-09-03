#!/usr/bin/env python3
"""Print what the Wave Rover firmware sends back, so we can find the battery.

wave_rover_bridge.py only ever writes to the serial port. The firmware also
talks, but what it emits and how often is a property of the board's firmware
version, not something to guess at from a datasheet and ship into the motor
path.

The bridge holds /dev/serial0, so stop it first or the two will split the
stream between them:

    sudo systemctl stop rover-bridge
    python3 rover_serial_peek.py
    sudo systemctl start rover-bridge

If nothing appears, the firmware only answers when asked -- try --poll, which
sends {"T":130} once a second, the usual request for base feedback.
"""

import argparse
import json
import sys
import time

import serial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='/dev/serial0')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--seconds', type=float, default=10.0)
    ap.add_argument('--poll', action='store_true',
                    help='send {"T":130} once a second to ask for feedback')
    args = ap.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.5)
    except serial.SerialException as e:
        print(f'cannot open {args.port}: {e}')
        print('is rover-bridge still running? sudo systemctl stop rover-bridge')
        return 1

    print(f'listening on {args.port} for {args.seconds:.0f}s'
          + (', polling' if args.poll else '') + '...')
    end, last_poll, seen, keys = time.time() + args.seconds, 0.0, 0, set()
    while time.time() < end:
        if args.poll and time.time() - last_poll >= 1.0:
            last_poll = time.time()
            ser.write((json.dumps({"T": 130}) + '\n').encode())
        try:
            raw = ser.readline()
        except serial.SerialException as e:
            print(f'read failed: {e}')
            break
        if not raw:
            continue
        seen += 1
        line = raw.decode('utf-8', 'replace').strip()
        print(f'  {line}')
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                keys |= set(obj)
        except ValueError:
            pass

    ser.close()
    print(f'\n{seen} lines. keys seen: {", ".join(sorted(keys)) or "(none parsed)"}')
    if not seen:
        print('nothing at all -- try --poll, or the firmware may be silent.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
