#!/usr/bin/env python3
"""Template for rooms_local.py, which is not tracked here.

Copy to rooms_local.py beside rooms.py on the rover and fill in real values.
Poses are (x, y, qz, qw) in the `map` frame, read off /tracked_pose with the
rover parked where you want it to stop.
"""

ROOMS = {
    "work_room": (0.00, 0.00, 0.000, 1.000),
    "kitchen":   (1.00, 1.00, 0.000, 1.000),
    "bedroom_1": (2.00, 2.00, 0.000, 1.000),
}

# Who to find in which room. Both names of a shared room point at the same key.
PEOPLE = {
    "someone": "bedroom_1",
}

# What each room is called out loud, article included: "arrived at the
# kitchen" is right, "arrived at Sam's room" takes none.
SPOKEN = {
    "bedroom_1": "someone's room",
}

# Anything naming a person. The generic aliases live in rooms.py.
ALIASES = {
    "someone's room": "bedroom_1",
}
