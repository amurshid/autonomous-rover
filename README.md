# Autonomous Rover

A Waveshare Wave Rover with a Raspberry Pi, an LD lidar and an Arducam, running
ROS 2 Humble. It maps a house with Cartographer, navigates between rooms with
Nav2, takes instructions in natural language, and — the part this repository is
mostly about — **works out where it is from a single camera frame.**

Everything here was built end to end: the robot, the mapping, the navigation,
the data collection, the model, and the deployment path back onto the robot.

---

## Why a camera when the rover already has lidar

Cartographer localises well once it has locked on, but two failures are logged
repeatedly in this project:

**It does not know where it is at boot.** Every recorded session needed 12–20
frames of convergence before the pose was trustworthy, and one discarded session
reported x ≈ −32 and −17 — far outside the map — for 22 frames. The usual
workaround is clicking an initial pose in RViz, which is what
`launch/start_localization.sh` does today.

**Lidar cannot tell similar rooms apart.** In one logged failure the rover sat in
the living room while Cartographer insisted it was in the parents' room, four
metres away, and stayed wrong for four minutes until a 3.30 m correction landed.

A camera sees texture and content rather than shape, so it fails differently —
which makes it a useful independent check rather than a redundant sensor. The
goal was never to beat lidar on accuracy. It was to fix the cold start and catch
the confusions.

## Result

Given one camera frame, return a map-frame pose. Measured on the two recorded
sessions the model never trained on:

| | |
|---|---|
| position error | **0.25 m median**, 90% within 1 m |
| yaw error | **6.0° median** |
| cold start: first accepted fix usable | **84.5%** (94.3% with one confirming frame) |
| gross errors among accepted fixes | **0.0–0.4%** |

Good enough to replace the hand-set RViz pose. Full method, every failed
attempt, and the caveats: **[vpr/README.md](vpr/README.md)**.

## Four failures and a diagnosis

Four approaches to beating the frozen baseline failed before one worked. They
are documented as carefully as the success, because they are what located the
cause:

1. Metric learning on the descriptors — validation improved, the night test got
   worse
2. Training against simulated night — best validation of any model, worst test
3. **The diagnosis** — interpolating between frozen and trained weights showed
   night performance peaks at 40% and then collapses while validation climbs all
   the way, so validation was selecting almost exactly the wrong model
4. Resolution and pooling sweep — no effect, ruling out descriptor fidelity

The common cause: training only ever saw one illuminant, so it could not learn
to ignore the one variable that mattered. That diagnosis made a prediction —
put one night session into training and the same code should work. It did:
0.86 m → 0.34 m, from about 11% more data and no change to the method.

Under a random split every one of those runs would have reported a win. Splitting
by session is what made the failures visible.

## Layout

```
vpr/              the place-recognition study — start with vpr/README.md
  01..11_*.py     the experiments, run once each in order
  vprlib/         data loading, features, retrieval, training, augmentation
  deploy/         build the Pi bundle, and replay a session through it
scripts/          ROS 2 nodes: teleop, Nav2 client, patrol, LLM/voice control,
                  the dataset logger, and the relocalisation node
launch/           Cartographer mapping and localization, Nav2, start scripts
config/           Cartographer .lua and Nav2 params
ros2_ws_src/      IMU bridge and laser odometry
ldlidar_src/      lidar driver
SYSTEM_NOTES.md   machine-level changes that live outside these files
```

The dataset (13,875 frames across 8 sessions) is not in the repository.

## Debugging notes

- **Cartographer drift from a pose-graph backlog**, diagnosed from a frozen
  logger counter and fixed with `optimize_every_n_nodes`.
- **Nav2 rotating forever in place.** RPP clamps angular commands to
  `current_velocity + max_angular_accel × 0.1`. With a large motor deadband the
  rover does not move, odometry correctly reports zero velocity, and the clamp
  never releases. Diagnosis and fix in `SYSTEM_NOTES.md`.
- **A silent CSV corruption** where CRLF line endings made every last field
  `1\r`, and frame numbering with gaps that made the image directory disagree
  with the pose file.

## Status

The mapping, navigation and modelling are done. The relocalisation bundle is
built and verified offline; it has not yet run on the Pi. Next step is shadow
mode on the robot — logging what it would have published alongside
Cartographer's own pose, publishing nothing, until the numbers justify turning
it on.
