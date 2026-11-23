# DogBot

Keeps the dogs off the couch when no person is around. A Raspberry Pi 5 with a
Hailo-8L and a Camera Module 3 watches a couch zone. When a dog stays in the
zone and nobody has been seen for the grace period, it plays escalating sound
blasts. A LAN web app shows the live view, the arming state and the event log,
and lets you draw the zone, tune settings and test the sound.

- Host: `ai-robot`, `admin@192.168.7.208`
- Web app: <http://192.168.7.208:8080> (LAN only, no login)
- Model: `/home/admin/models/yolo26s.hef` (YOLO26s, Hailo Model Zoo v2.20, hailo8l)

## How it behaves

- **Arming.** It starts **disarmed**. Any frame with a person (confidence at
  least `person_conf`) disarms it and restarts the grace timer. It arms after
  `rearm_grace_s` (default 60 s) with no person seen.
- **Escalation.** A dog in the zone for `dwell_s` starts round 1. Round *n*
  plays *n* blasts at level *n* (n = 1, 2, 3), with `round_gap_s` between
  rounds. The 1→2→3 sequence runs `sequences` times (default 2). After that it
  logs `unresolved` and stays silent until the couch has been clear for
  `clear_s`.
- **Stops at once** when a person appears, when you switch it off in the app,
  or when the camera or Hailo fails (`fault`, shown in the app).

## Layout on the Pi

| What | Where |
|---|---|
| Code | `/home/admin/dogbot` |
| Python venv | `/home/admin/dogbot/.venv` (`--system-site-packages`) |
| Settings | `/home/admin/dogbot/config/settings.json` (missing = defaults) |
| Event DB and snapshots | `/home/admin/dogbot/data/events.db`, `data/snapshots/` |
| Service | `/etc/systemd/system/dogbot.service` (runs as `admin`) |

## Develop on the Mac

```sh
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest -q -W error     # `pi`-marked tests are skipped off the Pi
.venv/bin/python tools/dev_server.py       # web app with a simulated pipeline
```

## Deploy

```sh
./deploy/deploy.sh                        # DOGBOT_HOST=admin@ai-robot to override
```

The script rsyncs the repo to the Pi and leaves `.venv/`, `config/` and `data/`
alone. It creates the venv if it is missing, runs `pip install -e .` and
installs and enables the systemd unit. Then it restarts the service.

The venv uses the system `numpy` 1.24.x, because `picamera2` and
`hailo_platform` are built against it. `pip` runs with a constraints file that
pins that exact version, and the script stops if `numpy` changes anyway. Never
`pip install -U numpy` in this venv.

## Run and operate

```sh
sudo systemctl status dogbot
journalctl -u dogbot -f                   # logs
sudo systemctl restart dogbot
sudo systemctl stop dogbot                # stops in about 1 s, even with the live view open
```

### Choosing the camera

DogBot uses the Pi camera (IMX708) by default. To use a USB (UVC) camera
instead, such as the wide-angle Huddly GO, set two environment variables in a
systemd drop-in. A drop-in survives `deploy.sh`, which reinstalls the main
unit. Use the stable `/dev/v4l/by-id/...` path (`ls /dev/v4l/by-id/`), not
`/dev/videoN`, which can change between boots:

```sh
sudo mkdir -p /etc/systemd/system/dogbot.service.d
sudo tee /etc/systemd/system/dogbot.service.d/camera.conf <<'EOF'
[Service]
Environment=DOGBOT_CAMERA=uvc
Environment=DOGBOT_CAMERA_DEVICE=/dev/v4l/by-id/usb-Huddly_Huddly_GO_<serial>-video-index0
EOF
sudo systemctl daemon-reload && sudo systemctl restart dogbot
```

The USB camera is read as MJPG 1280x720 at 30 fps, and black warm-up frames
are skipped. Remove the drop-in to go back to the Pi camera. Redraw the couch
zone after switching, because the field of view is different.

To run it by hand for debugging, stop the service first, because the camera
and the Hailo can only be opened by one process:

```sh
cd /home/admin/dogbot
.venv/bin/python -m dogbot --port 8080 [--hef PATH] [--lens-position 2.0]
.venv/bin/python -m dogbot.pipeline --once 50   # camera + Hailo smoke test, no sound, no DB
```

The service restarts on failure, every 5 s. If the camera or the Hailo cannot
be opened at startup, the web app is not available and the reason is in
`journalctl -u dogbot`. A watchdog also exits the service (so systemd restarts
it) when the camera loop stops updating for 3 s; the app shows FAULT "No data
from the camera loop" while the status is more than 2 s old.

The web app only answers requests whose `Host` is one of the Pi's own names or
addresses (localhost, 127.0.0.1, the hostname, `<hostname>.local` and every IPv4
address at startup), which blocks DNS-rebinding attacks from a LAN browser.
Anything else gets `400 Invalid host header`. To use another name, pass
`--allowed-hosts name1,name2` or set `DOGBOT_ALLOWED_HOSTS` (this replaces the
default list). If the Pi's IP address changes, restart the service.

### Hailo driver after a kernel update

After `apt full-upgrade` installs a new kernel, `/dev/hailo0` can be missing
after the reboot, because DKMS only built the `hailo_pci` module for the old
kernel. The service then keeps restarting with a Hailo error. To fix it:

```sh
ls /dev/hailo0 || { sudo dkms autoinstall && sudo modprobe hailo_pci; }
sudo systemctl restart dogbot
```

PCIe Gen3 is enabled in `/boot/firmware/config.txt` (`dtparam=pciex1_gen=3`)
for the Hailo-8L. Keep it when you edit that file.

## Owner checklist

These are the steps of the end-to-end check that need a person, a speaker and a
dog. Everything else (deploy, every API route, the live stream at about 30 fps,
`systemctl restart`, and a full reboot) was checked on 2026-11-20.

1. **Plug in the USB speaker.** Run `aplay -l` on the Pi and find its card name,
   for example `card 2: Device [USB Audio Device]` → `Device`.
2. **Set `audio_device`** in the app's settings to `plughw:CARD=<name>,DEV=0`,
   for example `plughw:CARD=Device,DEV=0`. Do not keep `"default"` or
   `"sysdefault"`. On this Pi `default` is PulseAudio, which the service cannot
   reach (it runs without a desktop session), so nothing plays. Only plain ALSA
   names are accepted: `default`, `sysdefault[:CARD=x]` and
   `[plug]hw:[CARD=]x[,[DEV=]n]`. If the speaker has a volume control,
   set it to maximum with `alsamixer -c <name>` and keep the loudness in the
   app's levels.
3. **Check that you actually hear a test blast.** Press Test on the L1 row. If
   `aplay` fails at once (no such device, device busy), the test blast reports
   the error, and every `aplay` failure is logged as a `sound` error event. A
   device that opens but has no speaker attached, or a muted mixer, still
   reports OK while nothing plays.
4. **Calibrate L1–L3.** Hold a phone SPL meter app where the dogs lie on the
   couch. Press Test on the L1, L2 and L3 rows, adjust the levels (software gain 0–1) until
   they sound right, and enter the measured dB for each level.
5. **Draw the couch zone** on the live view (at least 3 points) and save it.
6. **Stand in view** and check that the app shows DISARMED.
7. **Shorten the grace period for the test.** Set `rearm_grace_s` to 15 (the
   minimum is 5). Leave the room and check that it shows ARMED after the grace
   period.
8. **Trigger it with a dog** (or a dog photo or plush toy on the couch). Check
   rounds 1 → 2 → 3, then walk in mid-sequence and check that the sound stops
   at once.
9. **Check the event log** for the intrusion, round and stop entries, and their
   snapshots.
10. **Restore `rearm_grace_s`** to 60 (or the value you prefer).
