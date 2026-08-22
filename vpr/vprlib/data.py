"""Loading and labelling for the VPR dataset.

One dataframe for all eight sessions. The logger never wrote a session column,
so it is added here from the folder name, along with the day/night condition.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "vpr_data"

# Frame counts recorded when the sessions were collected, checked at load time.
EXPECTED = {
    "day1_night_session1": 841,
    "day1_night_session2": 1556,
    "day1_night_session3": 1839,
    "day2_afternoon_300": 1786,
    "day2_afternoon_400": 1696,
    "day2_evening_500": 1976,
    "day2_evening_600": 2228,
    "day2_evening_700": 1952,
}

SESSIONS = list(EXPECTED)
NIGHT_SESSIONS = [s for s in SESSIONS if "night" in s]
DAY_SESSIONS = [s for s in SESSIONS if "night" not in s]


def _condition(session: str) -> str:
    return "night" if "night" in session else "day"


def load_session(session: str) -> pd.DataFrame:
    """Read one session's poses_clean.csv. Handles CRLF and missing images."""
    path = DATA / session / "poses_clean.csv"
    # engine='python' + explicit newline handling: the files were written on the
    # Pi and some carry CRLF, so let pandas normalise rather than trusting it.
    df = pd.read_csv(path, dtype={"filename": str})
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    df["jump"] = df["jump"].astype(int)
    for col in ("stamp", "x", "y", "yaw"):
        df[col] = df[col].astype(float)

    df.insert(0, "session", session)
    df.insert(1, "condition", _condition(session))
    df["path"] = [str(DATA / session / "images" / f) for f in df["filename"]]
    return df


def load_all(verify: bool = True) -> pd.DataFrame:
    """All eight sessions concatenated, index reset."""
    frames = [load_session(s) for s in SESSIONS]
    df = pd.concat(frames, ignore_index=True)
    if verify:
        report_counts(df)
    return df


def report_counts(df: pd.DataFrame) -> None:
    got = df.groupby("session").size().to_dict()
    for session, expected in EXPECTED.items():
        n = got.get(session, 0)
        flag = "" if n == expected else f"  <-- expected {expected}"
        print(f"  {session:22s} {n:5d}{flag}")
    print(f"  {'TOTAL':22s} {len(df):5d}   "
          f"(day {(df.condition == 'day').sum()}, night {(df.condition == 'night').sum()})")


# --- room labelling -------------------------------------------------------
# Imported from the rover's own rooms.py so the coordinates cannot drift.

def room_centres() -> dict[str, tuple[float, float]]:
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from rooms import ROOMS  # noqa: E402

    return {name: (xy[0], xy[1]) for name, xy in ROOMS.items()}


def label_rooms(df: pd.DataFrame, max_dist: float | None = None) -> pd.DataFrame:
    """Nearest-room-centre label, plus the distance to that centre.

    Naive by design -- the corridor problem: a hallway frame
    gets whichever centre happens to be closest. `room_dist` is kept so the
    labels can be thresholded later without relabelling.
    """
    centres = room_centres()
    names = list(centres)
    cx = [centres[n][0] for n in names]
    cy = [centres[n][1] for n in names]

    best_name, best_dist = [], []
    for x, y in zip(df["x"], df["y"]):
        d = [math.hypot(x - a, y - b) for a, b in zip(cx, cy)]
        i = min(range(len(d)), key=d.__getitem__)
        best_name.append(names[i])
        best_dist.append(d[i])

    df = df.copy()
    df["room"] = best_name
    df["room_dist"] = best_dist
    if max_dist is not None:
        df.loc[df["room_dist"] > max_dist, "room"] = "corridor"
    return df


def split_day_night(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The headline split: train on all day sessions, test on all night."""
    return (df[df.condition == "day"].reset_index(drop=True),
            df[df.condition == "night"].reset_index(drop=True))


def split_holdout_session(df: pd.DataFrame, holdout: str = "day2_evening_600"):
    """Within-condition sanity check: one daytime session held out."""
    day = df[df.condition == "day"]
    return (day[day.session != holdout].reset_index(drop=True),
            day[day.session == holdout].reset_index(drop=True))
