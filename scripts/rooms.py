#!/usr/bin/env python3
"""Room goal poses for the Wave Rover.

Single source of truth for the code that resolves what somebody said into a
room the rover can drive to. Both patrol.py and rover_ai.py import from here
so nothing drifts apart.

Format: (x, y, qz, qw) in the `map` frame. Orientation is yaw-only, so qx and
qy are always zero.

The table itself is not in this repository. A list of rooms with coordinates
is a floor plan of a real home, and who sleeps in which one is nobody else's
business, so ROOMS, PEOPLE, SPOKEN and the personal aliases live in
rooms_local.py beside this file on the rover. .gitignore keeps it untracked.
See rooms_local.example.py for the shape.

Without that file ROOMS is empty and nothing will navigate. That is
deliberate. A sample table would be another house's coordinates, and a rover
driving to them finds a wall rather than a kitchen -- better that the room
buttons disappear and seed_pose says why.
"""

from __future__ import annotations

import re

try:
    import rooms_local as _local
except ImportError:          # running from a clone, or not deployed yet
    _local = None

ROOMS = getattr(_local, "ROOMS", {})
PEOPLE = getattr(_local, "PEOPLE", {})
SPOKEN = getattr(_local, "SPOKEN", {})
_LOCAL_ALIASES = getattr(_local, "ALIASES", {})

#: False when rooms_local.py is missing, so callers can say so plainly.
HAVE_ROOMS = bool(ROOMS)

# Stable ordering for the LLM tool enum.
ROOM_NAMES = sorted(ROOMS)


# What a person might actually say. The model is told the canonical keys, but
# it paraphrases, and so do people -- "my room", "the front door". Resolving
# here means a near-miss reaches the right room instead of failing. Only the
# generic ones live here; anything naming a person belongs in rooms_local.
_GENERIC_ALIASES = {
    "my room":            "work_room",
    "work room":          "work_room",
    "office":             "office_room",
    "dining":             "dining_room",
    "living":             "living_room",
    "formal living room": "formal_living",
    "front door":         "entrance",
    "door":               "entrance",
    "breakfast":          "breakfast_table",
    "breakfast room":     "breakfast_table",
}

# Filtered against ROOMS so an alias can never point at a room this rover does
# not have -- a different house may not own a formal living room.
ALIASES = {k: v for k, v in _GENERIC_ALIASES.items() if v in ROOMS}
ALIASES.update({k: v for k, v in _LOCAL_ALIASES.items() if v in ROOMS})


# A person's name is as good as their room's: going to someone and going to
# their room are the same journey. Added with setdefault so an explicit entry
# in rooms_local always wins.
for _who, _where in PEOPLE.items():
    if _where not in ROOMS:
        continue
    ALIASES.setdefault(_who, _where)
    ALIASES.setdefault(f"{_who}'s", _where)
    ALIASES.setdefault(f"{_who}s room", _where)
    ALIASES.setdefault(f"{_who}'s room", _where)


def resolve_room(name: str) -> str | None:
    """Map whatever was said to a key in ROOMS, or None if it is not a room."""
    if not name:
        return None
    # Whisper writes a curly apostrophe; the aliases above use a straight one.
    n = name.strip().lower().replace("\u2019", "'")
    n = " ".join(n.replace("_", " ").replace("-", " ").split())
    if n.startswith("the "):
        n = n[4:]
    for candidate in (n, n.replace(" ", "_")):
        if candidate in ROOMS:
            return candidate
    if n in ALIASES:
        return ALIASES[n]
    # "my room." from speech, or a trailing possessive
    n = n.rstrip(".!?,")
    if n in ALIASES:
        return ALIASES[n]
    # "bedroom 1's", "bedroom 2's" -- a possessive with the room left off.
    n = re.sub(r"'s$|'$", "", n)
    return ALIASES.get(n)


def spoken_name(room: str) -> str:
    """'breakfast_table' -> 'the breakfast table', ready to follow a preposition."""
    if room in SPOKEN:
        return SPOKEN[room]
    return "the " + room.replace("_", " ")


if __name__ == "__main__":
    import math
    if not HAVE_ROOMS:
        print("no rooms_local.py beside rooms.py -- ROOMS is empty")
        raise SystemExit(1)
    print(f"{len(ROOMS)} rooms")
    for name, (x, y, qz, qw) in sorted(ROOMS.items()):
        norm = math.hypot(qz, qw)
        yaw = math.degrees(2.0 * math.atan2(qz, qw))
        flag = "" if abs(norm - 1.0) < 1e-3 else "  <-- quaternion not normalised!"
        print(f"  {name:18s} x={x:7.2f} y={y:7.2f} yaw={yaw:7.1f} deg{flag}")
