#!/usr/bin/env python3
"""Pi temperature and rover battery, for the two web pages.

Deliberately free of ROS. The mode page must keep running when everything
else is down -- that is its whole purpose -- so it cannot import rclpy, and
this is shared by both pages rather than written twice.

Temperature comes from the thermal zone directly rather than `vcgencmd`,
which would be a subprocess per poll on a board that is already warm.

Battery is whatever wave_rover_bridge.py last wrote to TELEMETRY. The bridge
owns the serial port, so nothing else can read the voltage without fighting it
for the device; it writes a small file instead, and a stale or missing file
reads as unknown rather than as an error.
"""

import json
import os
import time

THERMAL = "/sys/class/thermal/thermal_zone0/temp"
TELEMETRY = os.environ.get("ROVER_TELEMETRY", "/run/rover/telemetry.json")
STALE_S = 15.0

# 3S lithium: 12.6 V charged, 9.9 V effectively flat. Override per pack.
V_FULL = float(os.environ.get("ROVER_V_FULL", 12.6))
V_EMPTY = float(os.environ.get("ROVER_V_EMPTY", 9.9))

# The Pi throttles at 80 C and this project has measured 83.8 C, so "warm" is
# not academic -- it starved the teleop stream once.
TEMP_WARN, TEMP_HOT = 70.0, 80.0


def temperature():
    """Celsius, or None if the thermal zone is unreadable."""
    try:
        with open(THERMAL) as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def battery():
    """(volts, percent) from the bridge's telemetry file, or (None, None).

    Percent is a straight line between V_EMPTY and V_FULL. That is not a real
    discharge curve, and under load it reads low; it is meant to answer "do I
    need to charge it soon", not to be accurate.
    """
    try:
        with open(TELEMETRY) as f:
            data = json.load(f)
        if time.time() - float(data.get("t", 0)) > STALE_S:
            return None, None          # bridge stopped writing
        volts = float(data["volts"])
    except (OSError, ValueError, KeyError, TypeError):
        return None, None
    pct = (volts - V_EMPTY) / (V_FULL - V_EMPTY) * 100.0
    return round(volts, 2), max(0, min(100, round(pct)))


def snapshot():
    """Everything the pages show, in one dict."""
    c = temperature()
    volts, pct = battery()
    return {
        "temp_c": None if c is None else round(c, 1),
        "temp_state": None if c is None else (
            "hot" if c >= TEMP_HOT else "warm" if c >= TEMP_WARN else "ok"),
        "volts": volts,
        "battery_pct": pct,
    }


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=2))
