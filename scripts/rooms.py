#!/usr/bin/env python3
"""Room goal poses for the Wave Rover.

Single source of truth. Both patrol.py and rover_ai.py should import from
here so the coordinates never drift apart.

Format: (x, y, qz, qw) in the `map` frame. Orientation is yaw-only, so qx
and qy are always zero.
"""

from __future__ import annotations

ROOMS = {
    "work_room":        (  2.276,  8.183, -0.8196,  0.5730),
    "entrance":         ( -2.94,   7.20,  -0.582,  -0.813),
    "office_room":      ( -0.63,   2.85,  -0.861,   0.509),
    "dining_room":      ( -5.20,   0.03,  -0.184,   0.983),
    "kitchen":          ( -5.84,  -4.14,   0.469,   0.883),
    "breakfast_table":  ( -6.84,  -8.24,  -0.698,   0.716),
    "formal_living":    (-10.21,   1.33,   0.964,  -0.267),
    "living_room":      (-10.44,  -6.37,  -0.711,   0.703),
    "bedroom_1":     ( -9.42,   5.99,   0.993,  -0.117),
    "bedroom_2":     (-14.37,  -4.29,   0.995,   0.096),
}

# Stable ordering for the LLM tool enum.
ROOM_NAMES = sorted(ROOMS)


# What each room is called out loud, article included. The keys stay as they
# are: they match the labels in the VPR database and the recorded sessions, and
# renaming them would split the map from the data it was built from.
#
# The article travels with the name because not every room takes "the" --
# "arrived at the kitchen" is right, "arrived at bedroom 1" takes none.
#
# work_room is deliberately absent: "my room" is accepted as input (see
# ALIASES) but the rover says "the work room" back, so what it reports always
# matches the name on the map.
SPOKEN = {
    "bedroom_1": "bedroom 1",
    "bedroom_2": "bedroom 2",
}


# What a person might actually say. The model is told the canonical keys, but
# it paraphrases, and so do people -- "my room", "bedroom 1's", "the front door".
# Resolving here means a near-miss reaches the right room instead of failing.
ALIASES = {
    "my room":            "work_room",
    "work room":          "work_room",
    "person_1":            "bedroom_1",
    "bedroom 1":      "bedroom_1",
    "bedroom 1":     "bedroom_1",
    "bedroom 2":       "bedroom_2",
    "bedroom 2":      "bedroom_2",
    "bedroom 2":    "bedroom_2",
    "bedroom 2":   "bedroom_2",
    "office":             "office_room",
    "dining":             "dining_room",
    "living":             "living_room",
    "formal living room": "formal_living",
    "front door":         "entrance",
    "door":               "entrance",
    "breakfast":          "breakfast_table",
    "breakfast room":     "breakfast_table",
}


def resolve_room(name: str) -> str | None:
    """Map whatever was said to a key in ROOMS, or None if it is not a room."""
    if not name:
        return None
    n = " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())
    if n.startswith("the "):
        n = n[4:]
    for candidate in (n, n.replace(" ", "_")):
        if candidate in ROOMS:
            return candidate
    if n in ALIASES:
        return ALIASES[n]
    # "my room." from speech, or a trailing possessive
    n = n.rstrip(".!?,")
    return ALIASES.get(n)


def spoken_name(room: str) -> str:
    """'breakfast_table' -> 'the breakfast table', ready to follow a preposition."""
    if room in SPOKEN:
        return SPOKEN[room]
    return "the " + room.replace("_", " ")


if __name__ == "__main__":
    import math
    print(f"{len(ROOMS)} rooms")
    for name, (x, y, qz, qw) in sorted(ROOMS.items()):
        norm = math.hypot(qz, qw)
        yaw = math.degrees(2.0 * math.atan2(qz, qw))
        flag = "" if abs(norm - 1.0) < 1e-3 else "  <-- quaternion not normalised!"
        print(f"  {name:18s} x={x:7.2f} y={y:7.2f} yaw={yaw:7.1f} deg{flag}")
