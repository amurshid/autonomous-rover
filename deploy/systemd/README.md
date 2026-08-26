# Boot the rover without a keyboard

Units to replace five terminals and a runbook. Power on, and the rover comes up
localised and ready to take a command.

## The dependency chain

    rover-bridge ──────────────────────────────────────────────┐
    rover-lidar ──> rover-cartographer ──> rover-initialpose ──┤
    rover-camera ──────────────────────────────────────────────┴──> rover-relocalise
                                                                          │
                                                                          ▼
                                                                      rover-nav2
                                                                          │
                                                                          ▼
                                                                       rover-ai

`rover-relocalise` is `Type=oneshot` with `RemainAfterExit=yes`, so Nav2's
`After=` means *after it has finished*, not after it started. A planner brought
up before the robot knows where it is plans from the wrong place.

## What systemd can and cannot guarantee

`After=` orders **starts**, not readiness. It cannot know when the lidar is
producing scans, or when Cartographer is ready to accept a pose.

Two places where that mattered:

- **Cartographer before a pose.** `set_initial_pose.py` already calls
  `wait_for_service` on `/finish_trajectory`, so a late Cartographer is fine.
- **The bridge before a pose is published.** This one silently lost poses.
  `/initialpose` is volatile QoS: publish before `set_initial_pose.py` has
  subscribed and the middleware drops the message with no error, leaving the
  rover on the hardcoded pose. `vpr_relocalise.py --publish` now blocks until
  the topic has a subscriber, and exits 3 if none appears within 30 s. The
  `sleep 5` it replaces was a guess that happened to work on a warm system.

## Failure is not fatal

`rover-relocalise` sets `SuccessExitStatus=2 3`, so neither "no confirmed fix"
(2) nor "no bridge" (3) stops the boot. Cartographer still runs, and its own
global localization may find the pose unaided, in about a minute. A rover that
boots unlocalised is recoverable; a boot that halts partway is not.

## Install

    sudo cp *.service *.target /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable rover.target
    sudo systemctl enable rover-bridge rover-lidar rover-camera \
         rover-cartographer rover-initialpose rover-relocalise rover-nav2 rover-ai

## When something fails

Long-running units restart themselves: the drivers (`bridge`, `lidar`,
`camera`) with `Restart=always`, the rest with `Restart=on-failure`.

The case worth understanding is **Cartographer restarting**. It comes back on a
default pose, having lost the one it was relocalised to, so `initialpose`,
`relocalise` and `nav2` are all `PartOf=rover-cartographer.service` and go down
and back with it. That re-runs the camera relocalisation rather than leaving
Nav2 to plan from a pose that no longer means anything.

`relocalise` itself is never restarted: it is a oneshot that succeeded, and
re-publishing a pose mid-navigation would restart Cartographer's trajectory
underneath the planner.

To re-run the whole sequence by hand:

    sudo systemctl restart rover.target

`PartOf=rover.target` on every unit is what makes that work -- `WantedBy=`
starts units but never stops them, so without it the command silently does
nothing.

## Operating it

    systemctl status 'rover-*'          # what is up
    journalctl -u rover-relocalise -b   # this boot's relocalisation
    journalctl -f -u 'rover-*'          # everything, live

    sudo systemctl stop rover.target
    sudo systemctl start rover.target

To debug one component by hand, stop its unit and run the command from its
`ExecStart` in a terminal. Nothing here changes how the scripts themselves work.

## Before enabling it at boot

Test the chain with the Pi already running, so a mistake is a failed unit
rather than a robot that will not boot:

    sudo systemctl start rover.target
    journalctl -u rover-relocalise -f

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

## What happened to start_localization.sh

Nothing — it still works, and is still the way to relocalise by hand. Systemd
does not call it. It does the same two things as separate units
(`rover-initialpose` and `rover-relocalise`) because systemd needs to order and
restart them independently.

One deliberate difference: **the systemd path has no hardcoded-pose fallback.**
`start_localization.sh` publishes the marked spot when VPR finds no fix. That
made sense when it was the only option. It no longer is, and it is probably
worse than nothing now — the marked spot is only correct if someone physically
put the rover there, it measured 5.66 m wrong on an ordinary boot, and
Cartographer's own global localization recovers unaided in about a minute.
Handing a confidently wrong pose to a scan matcher that was going to work it
out anyway is a bad trade.

If you want the fallback back, add to `rover-relocalise.service`:

    ExecStartPost=/bin/bash -c 'test $$EXIT_STATUS -eq 0 || \
        ros2 topic pub --once /initialpose ...'

## Still missing

Nothing. All eight units are here.
