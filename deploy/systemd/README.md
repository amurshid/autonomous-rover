# Boot the rover without a keyboard

Units to replace five terminals and a runbook. Power on, and the rover comes up
ready to take a command -- give Cartographer about a minute to work out where
it is before you send it anywhere.

## Two modes

The rover runs one of two, never both -- Nav2 and a human must not both be
publishing `/cmd_vel`:

    rover-common.target      motors, shared by both
        |
        +-- rover.target          autonomous: lidar, cartographer,
        |                         nav2, voice
        |
        +-- rover-teleop.target   remote control: teleop page + camera

`Conflicts=` between the two mode targets is what makes them exclusive:
starting either stops the other, with no sequencing to remember. The motors sit
outside both, so a switch never cycles them and the rover cannot glitch into
motion while changing modes.

The camera used to sit there too, for the same reason. It does not any more:
the only thing reading `/camera/image_raw` under autonomous was the camera
relocaliser, and that is out of the deployment, so it was encoding frames for
nobody at roughly 28% of a core -- on a board that throttles at 80 C and had
already starved teleop once. The price is the auto-exposure settling again on
each switch into remote control, paid once per switch rather than continuously.
`rover-relocalise` and `vpr_logger.py` still work by hand: that unit has
`Requires=rover-camera.service` and pulls the camera up wherever it lives.

`rover-mode.service` serves the switcher page on **port 80**, so the whole
address is `http://ahnaf-pi.local`. It belongs to neither mode and runs always:
if it goes down there is no way back without SSH, which is what it exists to
avoid. It needs no ROS. Picking remote control waits for the teleop port to
actually answer, then redirects to it -- systemd calls a unit active the moment
the process starts, which is well before the page is servable, and redirecting
then gives a connection-refused that reads as "it is broken".

In autonomous mode the same page also lists the rooms from `rooms.py`, so a
goal can be sent without speaking to the rover. Sending one *does* need ROS,
which that process does not have and must not acquire -- importing rclpy into
the one service that can never fail is the wrong trade. So it runs
`rover_nav.py --json <room>` as a child, under `bash -lc` with the setup
scripts sourced, and reads a line of JSON per event back:

    {"event": "sent",     "room": "kitchen", "detail": "..."}
    {"event": "feedback", "room": "kitchen", "remaining": 4.02}
    {"event": "done",     "room": "kitchen", "outcome": "arrived", "detail": ""}

One child at a time, and the page locks every other room button while it
lives -- two goals in flight would just have Nav2 preempt one with the other,
which is not what tapping a second room looks like it should do. Stop sends
SIGTERM, which `rover_nav.py` turns into a Nav2 cancel before exiting; the
`exec` in the command line matters, because a wrapping shell would swallow
that signal and the rover would keep driving. Leaving autonomous mode cancels
in the same way, rather than pulling Nav2 out from under a moving rover.

None of this needs sudo: the child runs as the same unprivileged user, and the
sudoers rule is still just the two `systemctl start` commands.

That child is invisible to `rover_ai.py`, which matters more than it sounds.
`is_navigating()` reports whether *this process* has a goal, so a drive
started from a room button read as False inside rover_ai: the gate that stops
it transcribing its own motors stayed open, it invented a command from the
noise, and its manual `drive` published /cmd_vel alongside Nav2's controller.
Two publishers at different rates is a rover that shakes and a goal that
fails. A spoken "stop" could not fix it either -- `cancel()` had no handle for
someone else's goal. `anyone_navigating()` and `cancel_any()` read Nav2's own
`navigate_to_pose/_action/status` and cancel-all service instead, which do not
care which process asked.

Starting a target needs root, so `rover-mode.sudoers` grants that user exactly
two `systemctl start` commands and `is-active`. Not blanket sudo: a page
reachable from the home network should not be able to do more than change the
mode.

## The dependency chain

    rover-bridge ─────────────────────────────────────────────┐
    rover-lidar ──> rover-cartographer ──> rover-initialpose ─┴──> rover-nav2
                                                                          │
                                                                          ▼
                                                                       rover-ai

**The camera relocaliser is not in this chain.** `rover-relocalise.service` is
still here and still works -- `systemctl start rover-relocalise` with
Cartographer up, or `vpr_relocalise.py` by hand -- but it has no `WantedBy`, so
nothing pulls it into a boot. Putting it back means restoring that line and
enabling the unit.

So finding itself is Cartographer's job now, through its own global
localization -- and with nothing publishing to `/initialpose` at boot it
starts from the map origin and searches the whole map. Measured at **167s**,
a 7.89 m jump to x=1.55 y=7.73, which is 7.88 m from the origin: the entire
correction. `wait_for_localisation.py` reports when it lands.

That number is the price of taking the camera relocaliser out, and the way to
get it back is a seed, not more patience -- from a pose near the truth
Cartographer refines in seconds. Publishing one at boot means committing to
parking the rover in a known spot, since a confident wrong seed is worse than
no seed: `start_localization.sh` asserts the mark in the work room and
measured 5.66 m wrong on an ordinary boot. Unseeded and slow is the safe
default, which is why it is the one in place.

Nav2 comes up before that finishes.
That is harmless in itself -- Nav2 does nothing until given a goal -- but a
goal sent inside that first minute is planned from a pose Cartographer has not
settled on yet. Give it a minute after the mode page goes green, or check
`/tracked_pose` has stopped moving before sending the rover anywhere.

## What systemd can and cannot guarantee

`After=` orders **starts**, not readiness. It cannot know when the lidar is
producing scans, or when Cartographer is ready to accept a pose.

`set_initial_pose.py` handles its own case -- it calls `wait_for_service` on
`/finish_trajectory`, so a late Cartographer is fine.

The one that still matters is **Nav2 against Cartographer's convergence**,
which no ordering can express: Nav2 is up long before the rover knows where it
is. Nothing breaks, because Nav2 acts only on a goal, but the first minute is
not a good time to send one.

`/initialpose` is also volatile QoS, so anything publishing a pose must wait
for `set_initial_pose.py` to have subscribed -- publish first and the
middleware drops it silently. Only relevant if you relocalise by hand;
`vpr_relocalise.py --publish` already waits for a subscriber and exits 3 if
none appears.

## Install

    sudo cp *.service *.target /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable rover.target
    sudo systemctl enable rover-bridge rover-lidar rover-cartographer \
         rover-initialpose rover-nav2 rover-ai
    sudo systemctl enable rover-camera rover-teleop rover-mode

**Use `reenable`, not `enable`, when a unit's `[Install]` section has moved.**
`enable` leaves an existing symlink alone, so a unit enabled under an older
`WantedBy=` keeps pointing at the old target -- and `is-enabled` still says
`enabled`, because it only checks that a symlink exists somewhere, not where.
That is how `rover-bridge` and `rover-camera` ended up in
`rover.target.wants/` after they moved to `rover-common.target`: present under
autonomous, gone the moment you switched to remote control, with
`rover-common.target` reporting active and holding nothing.

    sudo systemctl reenable rover-bridge rover-camera
    ls /etc/systemd/system/rover-common.target.wants/

`rover-teleop` and `rover-mode` are easy to miss. `WantedBy=` only takes
effect on `enable` -- without it `rover-teleop.target` comes up active with no
service under it, nothing listens on 8080, and the page has a mode selected
that never finishes arriving.

## When something fails

Long-running units restart themselves: the drivers (`bridge`, `lidar`,
`camera`) with `Restart=always`, the rest with `Restart=on-failure`.

The case worth understanding is **Cartographer restarting**. It comes back on a
default pose, having lost everything it had worked out, so `initialpose` and
`nav2` are `PartOf=rover-cartographer.service` and go down and back with it.
Nav2 planning from a pose that no longer means anything is worse than Nav2
being briefly absent, and it will need another minute to find itself either
way.

**A camera that dies is now cheap.** It was not: `rover_teleop_web.py` once
measured 273% CPU -- nearly three cores, on a Pi that throttles at 80 C --
streaming nothing. `rover-camera` has `Restart=always`, so this only shows up
if the camera fails in a way that restarting cannot fix. `journalctl -u
rover-teleop` says so directly now:

    no camera frames for 3s; is rover-camera running?
    camera frames resumed

To re-run the whole sequence by hand:

    sudo systemctl restart rover.target

`PartOf=rover.target` on every unit is what makes that work -- `WantedBy=`
starts units but never stops them, so without it the command silently does
nothing.

## Operating it

    systemctl status 'rover-*'          # what is up
    journalctl -u rover-cartographer -b # this boot's localisation
    journalctl -f -u 'rover-*'          # everything, live
    journalctl -f -u rover-nav2 -u rover-ai   # who moved the rover, in order

Every unit running Python sets `PYTHONUNBUFFERED=1`. Without it `print()` to a
pipe is block-buffered, and these scripts report themselves through `print()`
-- `[ignored while driving: ...]`, `[searched: ...]`, `[mic calibrated: ...]`
would arrive in 4KB chunks or not until the process exited, which makes
`journalctl -f` look like nothing is happening.

Nav2 is one launch file, so its nodes all log to `rover-nav2` with the node
name in the prefix: `[controller_server-2]`, `[bt_navigator-4]`. For a goal
that failed, `Failed to make progress` or `patience exceeded` from
`controller_server` means the rover did not go where it was told -- which is
what a second publisher on /cmd_vel looks like from Nav2's side. To settle
that directly, while it drives:

    ros2 topic info /cmd_vel --verbose

`controller_server` should be the only publisher. `rover_motions` beside it is
the fight.

    sudo systemctl stop rover.target
    sudo systemctl start rover.target

To debug one component by hand, stop its unit and run the command from its
`ExecStart` in a terminal. Nothing here changes how the scripts themselves work.

## Before enabling it at boot

Test the chain with the Pi already running, so a mistake is a failed unit
rather than a robot that will not boot:

    sudo systemctl start rover.target
    journalctl -f -u 'rover-*'

Enable at boot only once that has worked twice.

## Voice, not the text REPL

`rover_ai.py` defaults to its text REPL — `use_text = args.text or not
args.voice` — which reads stdin. A systemd service has no stdin, so the unit
passes `--voice` explicitly. It also needs `SupplementaryGroups=audio`: a login
shell grants access to `arecord` and `aplay`, a service does not inherit it.

Two things to check on the rover, since they only bite at boot:

- **Card numbering.** `rover_voice.py` defaults to `plughw:1,0` for the mic and
  `plughw:0,0` for the speaker. The speaker is the built-in analog jack and
  cannot renumber; the USB mic can, if anything else audio-capable is plugged
  in first. `arecord -l` after a cold boot confirms it.
- **The mic exists at all.** Voice mode is the only interface once there is no
  keyboard attached, so if `arecord` fails the rover boots deaf and there is
  nothing to notice it — `journalctl -u rover-ai -b` is where that shows up.

## The API key

`rover_ai.py` exits immediately without `GROQ_API_KEY`, and systemd gives a
service almost none of your shell's environment — a key that works when you
type the command by hand is simply absent under systemd. It must not go in the
unit file either, since units in `/etc/systemd/system` are world-readable:

    sudo install -d -m 750 /etc/rover
    printf 'GROQ_API_KEY=gsk_...\n' | sudo tee /etc/rover/env >/dev/null
    sudo chmod 640 /etc/rover/env
    sudo chown root:amurshid /etc/rover/env

## Correcting the pose by hand

Systemd does not set a pose at all — Cartographer finds itself. Two ways to
override it if it lands somewhere wrong:

    ~/start_localization.sh    # asserts the marked spot in the work room

That is only right if the rover is physically on the mark; it measured 5.66 m
wrong on an ordinary boot. `rover-initialpose` is already running, so the
bridge half of that script is redundant — but harmless.

    sudo systemctl start rover-relocalise    # the camera, if you want it back

That unit is still installed and still works. It is out of the boot by
request, not because it stopped working: it published a pose 5.66 m closer
than the marked spot and Cartographer's scan matcher then held it to 4 cm.

## Still missing

Nothing. All eight units are here.
