# DogBot MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python service on the Pi 5 + Hailo-8L that detects dogs and people with YOLO26s, arms itself only when no person is seen, and plays escalating sound blasts when a dog stays on the couch, with a LAN web app for live view, on/off, zone editing, settings and an event log.

**Architecture:** One process, `python -m dogbot`. A pipeline thread does camera → Hailo inference → YOLO26 decode → ByteTrack → zone presence → arming → rules engine → sound and events, and publishes an annotated JPEG. A FastAPI app in the same process serves the UI and API. Pure logic (decode, zones, arming, rules, events, config) lives in hardware-free modules that are unit-tested on the Mac. Hardware glue (camera, Hailo, ALSA) is tested on the Pi.

**Tech Stack:** Python 3.11 (Pi system Python), venv with `--system-site-packages` (for `picamera2` and `hailo_platform` 4.20), numpy, opencv-python-headless, supervision (ByteTrack), FastAPI + uvicorn, sqlite3 (stdlib), `aplay` (alsa-utils), pytest.

**Spec:** `docs/superpowers/specs/2026-11-20-dogbot-mvp-design.md`

## Global Constraints

- Host `ai-robot`, `admin@192.168.7.208`; code deployed to `/home/admin/dogbot`, venv at `/home/admin/dogbot/.venv`.
- Model: `/home/admin/models/yolo26s.hef` (Hailo Model Zoo v2.20, hailo8l). Fallback: `/usr/share/hailo-models/yolov8s_h8l.hef`.
- HEF input `UINT8 NHWC 640x640x3`. Outputs per stride (8/16/32 → grids 80/40/20): a box head with 4 channels and a class head with 80 channels (COCO). Read outputs as FLOAT32 (dequantized).
- COCO class IDs: `person = 0`, `dog = 16`.
- No cloud dependencies. Web app is LAN-only, no auth (MVP).
- Never blast when the system is disabled, disarmed, or in a fault state. Fail silent.
- Lights, recording, pan-tilt and the launcher are out of scope. Do not add code for them; the `Action` interface leaves room.
- Default settings (all editable in the UI): `dog_conf 0.50`, `person_conf 0.40`, `dwell_s 1.0`, `rearm_grace_s 60`, `clear_s 10`, `round_gap_s 5`, `blast_s 0.5`, `blast_spacing_s 0.4`, `tone_hz 4000`, `levels {1: 0.4, 2: 0.7, 3: 1.0}` (software gain 0–1), `level_db {1: null, 2: null, 3: null}` (calibrated dB shown in UI), `sequences 2`, `enabled true`, `audio_device "default"`.
- Escalation: round n plays n blasts at level n (n = 1, 2, 3); the full 1→2→3 sequence runs `sequences` (2) times; then an `unresolved` event and silence until the couch has been clear for `clear_s`.
- Escalation is per couch, not per dog. A dog back in the zone before `clear_s` resumes at the next round; after a full clear the next intrusion starts at round 1.
- Startup state is **disarmed**; it arms only after `rearm_grace_s` with no person seen.
- Zones are stored normalized (0–1) in the settings file `config/settings.json` (path overridable by `DOGBOT_CONFIG`).

## Review Focus

1. **Person flickers in and out of detection** (partly visible, turned away): any single frame with a person ≥ `person_conf` must disarm and restart the grace timer. Pinned in Task 4.
2. **Camera or Hailo failure mid-run:** the pipeline must stop any sound, enter `fault`, log an `error` event, show the fault in the UI, and never blast until it recovers. Pinned in Task 8.
3. **Disable or disarm during a blast:** sound stops within one pipeline tick (≤100 ms). Pinned in Tasks 5 and 7.
4. **Two dogs or tracker ID churn on the couch:** must not restart the sequence. Presence is "any dog in zone". Pinned in Task 3.
5. **Bad zone input from the UI** (fewer than 3 points, coordinates outside 0–1): rejected with HTTP 422; the old zone stays. Pinned in Task 9.

---

### Task 1: Project scaffold and settings

**Files:**
- Create: `pyproject.toml`, `dogbot/__init__.py`, `dogbot/config.py`, `tests/test_config.py`, `.gitignore`

**Interfaces:**
- Produces: `@dataclass Settings` with every field in Global Constraints, plus `zone: list[tuple[float, float]]` (default `[]`). `load_settings(path: Path) -> Settings` (missing file → defaults). `save_settings(s: Settings, path: Path) -> None` (atomic write via temp file + rename). `Settings.validate() -> None`, which raises `ValueError` on bad values. `settings_path() -> Path` (reads `DOGBOT_CONFIG`).

- [ ] **Step 1: Write failing tests** in `tests/test_config.py`:
  - `test_defaults_when_missing`: `load_settings(tmp/"x.json").dog_conf == 0.5` and `.sequences == 2`.
  - `test_roundtrip`: change `rearm_grace_s=30` and `zone=[(0.1,0.1),(0.9,0.1),(0.5,0.9)]`; save; load; the values are equal.
  - `test_validate_rejects_short_zone`: a zone with 2 points raises `ValueError`. An empty zone is allowed (no couch configured).
  - `test_validate_rejects_out_of_range`: point `(1.2, 0.5)` raises; `dog_conf=1.5` raises; level gain `-0.1` raises.
- [ ] **Step 2:** `pytest tests/test_config.py -v` → FAIL (import error).
- [ ] **Step 3:** Implement `dogbot/config.py`. `pyproject.toml` declares the package `dogbot`, `requires-python>=3.11`, deps `numpy`, `fastapi`, `uvicorn`, `supervision`, `opencv-python-headless`, and dev extra `pytest`, `httpx`.
- [ ] **Step 4:** `pytest tests/test_config.py -v` → PASS.
- [ ] **Step 5:** Commit: `feat: settings model with validation`.

### Task 2: YOLO26 output decoder

**Files:**
- Create: `dogbot/detection.py`, `dogbot/yolo26.py`, `tests/test_yolo26.py`, `tests/fixtures/yolo26_outputs.npz`, `tools/capture_fixture.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) Detection(cls_id: int, conf: float, box: tuple[float,float,float,float])`, with the box as normalized `x1,y1,x2,y2`. `decode_yolo26(outputs: dict[str, np.ndarray], conf_thresh: float, keep: set[int] = {0, 16}, input_size: int = 640) -> list[Detection]`.

- [ ] **Step 1: Probe the real output on the Pi.** Write `tools/capture_fixture.py`. It loads the HEF with `hailo_platform`, sets output format FLOAT32, runs one frame of a known image, and saves the raw outputs (keyed by layer name) plus shapes to `yolo26_outputs.npz`. Use any COCO-style photo with a dog and a person, e.g. an image fetched to the Pi. Record which layers are box (4 ch) vs class (80 ch) and their grid sizes. Check whether the class values are already in [0,1]: if the max is ≤1 and the min is ≥0 they're probabilities; otherwise they're logits and need a sigmoid. Write the answer as a comment at the top of `yolo26.py`.
- [ ] **Step 2: Write failing tests:**
  - `test_synthetic_single_box`: build outputs for one stride-32 cell (gx=10, gy=5) with box distances `l=t=r=b=1.0`, and dog class score high in that cell only. Expect exactly one detection, `cls_id=16`, box ≈ `((10.5-1)*32/640, (5.5-1)*32/640, (10.5+1)*32/640, (5.5+1)*32/640)` (abs tol 1e-3).
  - `test_threshold_filters`: the same input with `conf_thresh` above the score gives `[]`.
  - `test_ignores_other_classes`: a high score on class 2 (car) gives `[]`.
  - `test_fixture_finds_dog_and_person`: decoding the captured `.npz` at 0.4 gives ≥1 `cls_id=16` and ≥1 `cls_id=0`.
- [ ] **Step 3: Implement `decode_yolo26`.** For each stride s ∈ {8,16,32}, match the box and class layers by grid size. Anchor = `(gx+0.5, gy+0.5)`, box = `(ax-l, ay-t, ax+r, ay+b) * s / input_size`, clipped to [0,1]. Score = the class value (sigmoid only if Step 1 found logits). Keep the max class per cell if it's in `keep` and ≥ threshold. YOLO26 is NMS-free, but run `cv2.dnn.NMSBoxes` at IoU 0.7 per class as a guard against duplicate boxes.
- [ ] **Step 4:** `pytest tests/test_yolo26.py -v` → PASS.
- [ ] **Step 5:** Commit: `feat: YOLO26 Hailo output decoder`.

### Task 3: Zone geometry and couch presence

**Files:**
- Create: `dogbot/zones.py`, `tests/test_zones.py`

**Interfaces:**
- Consumes: `Detection` (Task 2).
- Produces: `anchor(box) -> tuple[float,float]` (bottom-center). `point_in_polygon(pt, poly: list[tuple[float,float]]) -> bool` (ray casting; an empty or <3-point polygon → False). `class CouchPresence(dwell_s: float)` with `update(now: float, any_dog_in_zone: bool) -> bool`. It returns True once dogs have been continuously in the zone for ≥ `dwell_s`, and False as soon as no dog is in the zone. Clearing and hysteresis are the rules engine's job.

- [ ] **Step 1: Write failing tests:**
  - `test_anchor_bottom_center`: `anchor((0.2,0.2,0.4,0.6)) == (0.3, 0.6)`.
  - `test_point_in_square` / `test_point_outside` / `test_empty_polygon_false`.
  - `test_dwell`: dog in zone at t=0 → False; at t=0.9 → False; at t=1.0 → True.
  - `test_dog_leaves_resets_dwell`: in at 0, out at 0.5, in at 0.6 → still False at 1.2, True at 1.6.
  - `test_two_dogs_alternating_counts_as_continuous` (Review Focus 4): the caller passes `any_dog_in_zone=True` every tick while dog A and then dog B (different track IDs) sit there. Presence stays True with no gap.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `dogbot/zones.py`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: couch zone geometry and dwell presence`.

### Task 4: Arming

**Files:**
- Create: `dogbot/arming.py`, `tests/test_arming.py`

**Interfaces:**
- Produces: `class Arming(rearm_grace_s: float, start: float)` with `update(now: float, person_present: bool) -> bool` (armed?) and `seconds_until_armed(now) -> float`. Starts disarmed. Any `person_present=True` → disarmed immediately and the grace timer restarts. Armed once `now - last_person_or_start >= rearm_grace_s`. The `enabled` switch is applied by the rules engine, not here.

- [ ] **Step 1: Write failing tests:**
  - `test_starts_disarmed`: at t=0 and t=59 → False; at t=60 → True (grace 60).
  - `test_person_disarms_immediately`: armed at 60; person at 61 → False.
  - `test_single_frame_person_restarts_grace` (Review Focus 1): person only at t=100 → False until 160, True at 160.
  - `test_seconds_until_armed`: right after a person at t=100, `seconds_until_armed(130) == 30`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: person-based auto arming`.

### Task 5: Rules engine (couch guard)

**Files:**
- Create: `dogbot/rules.py`, `tests/test_rules.py`

**Interfaces:**
- Consumes: `Settings` (Task 1).
- Produces:
  - Commands: `@dataclass PlayBlasts(count: int, level: int)` and `@dataclass StopSound()`.
  - Events: `@dataclass RuleEvent(kind: str, detail: dict)`, where kind ∈ `intrusion, round, cleared, unresolved, stopped`.
  - `class CouchGuard(settings: Settings)` with:
    - `update(now: float, active: bool, occupied: bool) -> tuple[list[Command], list[RuleEvent]]`, where `active = enabled and armed and not fault`.
    - `state -> str` ∈ `idle, blasting, gap, paused, unresolved`.
    - `round -> int`, `sequence -> int`.
  - Round duration = `count*blast_s + (count-1)*blast_spacing_s`.

- [ ] **Step 1: Write failing tests** (default settings, ticks of 0.1 s):
  - `test_full_escalation`: occupied from t=0 → at t=0, `PlayBlasts(1,1)` + `intrusion` + `round{seq:1,round:1}`. Round 1 lasts 0.5 s, then a 5 s gap, so at t≈5.5 → `PlayBlasts(2,2)`. Round 2 lasts 1.4 s, so at t≈11.9 → `PlayBlasts(3,3)`. Round 3 lasts 2.3 s, so at t≈19.2 → sequence 2 starts with `PlayBlasts(1,1)` … after sequence 2's round 3 and its gap → `unresolved`, with no further `PlayBlasts` while occupied.
  - `test_dog_leaves_then_clear`: occupied 0–3, then False. No new rounds start while paused. At t=13 (3+`clear_s`) → `cleared` event, `state=='idle'`.
  - `test_resume_before_clear`: after round 1, the dog leaves at t=2 and returns at t=6 (<`clear_s`). The next command is `PlayBlasts(2,2)`, not `(1,1)`.
  - `test_restart_after_full_clear`: after `cleared`, a new intrusion starts at `PlayBlasts(1,1)`.
  - `test_deactivate_stops_sound` (Review Focus 3): in `blasting`, `update(active=False)` → `[StopSound()]` + `stopped` event in the same call, then `state=='idle'`.
  - `test_unresolved_silent_until_clear`: after `unresolved`, occupied stays True for 60 s with no commands. Not occupied for `clear_s` → `cleared` → idle.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement `CouchGuard`.** States:
  - `idle --occupied&active--> blasting`
  - `blasting --round time elapsed--> gap`
  - `gap --gap elapsed & occupied--> blasting(next round)`
  - `gap/blasting --!occupied--> paused` (no new rounds; a blast in progress finishes)
  - `paused --occupied--> gap` (the gap restarts, then the next round)
  - `paused --!occupied for clear_s--> idle + cleared`
  - after the final round and its gap: `unresolved`
  - `unresolved --!occupied for clear_s--> idle + cleared`
  - `active=False` from any non-idle state → `StopSound` + `stopped` → idle.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: couch guard escalation state machine`.

### Task 6: Event store

**Files:**
- Create: `dogbot/events.py`, `tests/test_events.py`

**Interfaces:**
- Produces: `class EventStore(db_path: Path, snapshot_dir: Path)` with:
  - `log(kind: str, detail: dict, snapshot_jpeg: bytes | None = None) -> int`
  - `list(limit: int = 100, before_id: int | None = None, kinds: set[str] | None = None) -> list[dict]`, newest first. Each dict has `id, ts (ISO-8601 local), kind, detail, has_snapshot`.
  - `snapshot_path(event_id) -> Path | None`
  - Thread-safe: one connection with `check_same_thread=False` plus a lock.
  - Kinds used app-wide: `armed, disarmed, intrusion, round, cleared, unresolved, stopped, enabled, disabled, settings, zone, test_blast, error, startup`.

- [ ] **Step 1: Write failing tests:** log 3 events; `list()` returns them newest first; the `kinds` filter works; `before_id` paginates; a snapshot is written to disk and `has_snapshot` is True; `detail` round-trips as a dict.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Snapshots are saved as `snapshot_dir/{id}.jpg`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5:** Commit: `feat: sqlite event log with snapshots`.

### Task 7: Sound action

**Files:**
- Create: `dogbot/actions/__init__.py`, `dogbot/actions/sound.py`, `tests/test_sound.py`

**Interfaces:**
- Produces:
  - `class Action(Protocol)`, with `handle(cmd) -> None` and `stop() -> None`.
  - `synth_blasts(count: int, gain: float, tone_hz: float, blast_s: float, spacing_s: float, rate: int = 48000) -> np.ndarray` (int16 mono).
  - `class SoundAction(settings_getter: Callable[[], Settings], runner=subprocess.Popen)`. It plays `PlayBlasts` by piping the raw samples to `aplay -q -D {audio_device} -f S16_LE -r 48000 -c 1`. `StopSound` / `stop()` kills the running `aplay`. A new `PlayBlasts` kills any running playback first.

- [ ] **Step 1: Write failing tests:**
  - `synth_blasts(3, 1.0, 4000, 0.5, 0.4)` has length `round((3*0.5+2*0.4)*48000)`, peak ≈ 32767*1.0 (±2%), and the spacing samples are zero.
  - Gain 0.4 → peak ≈ 0.4*32767.
  - Each blast has a 10 ms fade in and out (the first sample is 0) to avoid clicks.
  - With a fake `runner`: `handle(PlayBlasts(2,2))` launches aplay with the right args and gain `levels[2]`. `stop()` calls `terminate()` on the fake process within the call (Review Focus 3).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement. Write to stdin from a daemon thread so `handle` never blocks the pipeline.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Manual check on the Pi** with the USB speaker plugged in. Run `aplay -l` to confirm the card, and set `audio_device` (e.g. `plughw:CARD=...`). Run `python -c` to play `PlayBlasts(1,1)` and confirm it's audible.
- [ ] **Step 6:** Commit: `feat: ALSA sound action with escalating blasts`.

### Task 8: Hailo runner, overlay, and pipeline

**Files:**
- Create: `dogbot/hailo_runner.py`, `dogbot/overlay.py`, `dogbot/pipeline.py`, `tests/test_pipeline.py`, `tests/test_overlay.py`

**Interfaces:**
- Consumes: Tasks 1–7.
- Produces:
  - `class HailoDetector(hef_path: str)`, with `infer(rgb640: np.ndarray) -> dict[str, np.ndarray]` and `close()`. It uses `hailo_platform` (VDevice, InferModel or InferVStreams) with FLOAT32 outputs.
  - `draw_overlay(frame_bgr, tracks: list[TrackedDet], zone_norm, status: dict) -> np.ndarray`, where `TrackedDet(det: Detection, track_id: int | None, in_zone: bool)`. Drawing rules:
    - People: thin box.
    - Dogs: box, plus a **reticle** on tracked dogs (a circle with crosshair ticks at the box center, radius 0.35×min(w,h), red if `in_zone` else amber).
    - Couch polygon: green when idle, amber when occupied, red when blasting.
    - Top banner: `ARMED` / `DISARMED (arming in Ns)` / `DISABLED` / `FAULT`, plus state and round.
  - `class Pipeline(settings_getter, events: EventStore, sound: Action, frame_source, detector, clock=time.monotonic, presence_sources: list[Callable[[], bool]] = [])`. `person_present` = camera person detected **or** any extra presence source returns True. The list is empty in the MVP; a later PIR or second camera plugs in here.
    - `step() -> None`, one tick.
    - `latest_jpeg() -> bytes | None`.
    - `status() -> dict`, with keys `armed, enabled, fault, state, round, sequence, seconds_until_armed, fps, dogs, people`.
    - `run_forever(stop_event)`.
    - `frame_source` is any object with `capture() -> (main_bgr, lores_rgb640)`. The real one is `PicameraSource` (Picamera2, main 1280×720 BGR888, lores 640×640 RGB888, autofocus continuous at start then locked by `lens_position` if set).

- [ ] **Step 1: Write failing tests** with a fake `frame_source`, a fake detector that returns scripted `Detection` lists per tick, a fake `Action`, an in-memory `EventStore` (tmp path), and a fake clock:
  - `test_dog_on_couch_when_armed_blasts`: no person for 61 s, then a dog with its anchor inside the zone for 1.1 s → the sound got `PlayBlasts(1,1)`; events include `armed`, `intrusion`, and an `intrusion` snapshot exists.
  - `test_person_present_never_blasts`: a person and a dog on the couch for 120 s → no `PlayBlasts`.
  - `test_detector_exception_enters_fault` (Review Focus 2): the detector raises mid-blast → `StopSound` is sent, `status()['fault']` is True, an `error` event is logged, and no further `PlayBlasts` happen. After 3 good ticks, the fault clears and a fresh `Arming` grace starts.
  - `test_disabled_never_blasts`: `enabled=False` → no commands.
  - `test_overlay_draws_reticle`: draw on a black frame with one tracked dog → pixels at the reticle center column are non-black, and the output shape equals the input.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement.** Each `step`:
  1. capture
  2. infer
  3. `decode_yolo26` with `min(dog_conf, person_conf)`, then filter per class
  4. `sv.ByteTrack` on dogs only
  5. `CouchPresence` from the anchors
  6. `Arming`
  7. `CouchGuard.update(active=enabled and armed and not fault, ...)`
  8. dispatch commands to `sound`
  9. log `RuleEvent`s (attach a snapshot for `intrusion` and `unresolved`) and `armed`/`disarmed` transitions
  10. render the overlay; JPEG-encode at quality 75 every 2nd frame

  Settings are re-read each tick via `settings_getter`. A change to `rearm_grace_s` or `dwell_s` rebuilds only that component.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: On the Pi**, run `python -m dogbot.pipeline --once 50` (a small `__main__` hook that runs 50 ticks and prints FPS and detections). Expect ≥15 FPS and person detections when you stand in view.
- [ ] **Step 6:** Commit: `feat: camera-to-sound pipeline with overlay and fault handling`.

### Task 9: Web app

**Files:**
- Create: `dogbot/web/app.py`, `dogbot/web/static/index.html`, `dogbot/web/static/app.js`, `dogbot/web/static/style.css`, `tests/test_web.py`

**Interfaces:**
- Consumes: `Pipeline.status/latest_jpeg`, `EventStore`, `load/save_settings`, `SoundAction`.
- Produces: `create_app(state: AppState) -> FastAPI`, where `AppState` holds the settings (with a lock), pipeline, events, sound and settings path. Routes:
  - `GET /` serves the UI.
  - `GET /stream.mjpg` serves a multipart MJPEG stream of `latest_jpeg()` at ≤15 fps.
  - `GET /frame.jpg` returns a single frame, used by the zone editor.
  - `GET /api/status` returns `status()`.
  - `GET /api/settings`; `PUT /api/settings` takes a partial update, runs validate → save → logs a `settings` event with the changed keys, and returns 422 on `ValueError`.
  - `PUT /api/zone` with `{points:[[x,y],...]}`: validate, save, log a `zone` event; 422 if <3 points or out of range (Review Focus 5).
  - `POST /api/enabled` with `{enabled: bool}` logs `enabled`/`disabled`.
  - `POST /api/test-blast` with `{level:1|2|3}` plays one blast at that level even when disarmed (calibration) and logs `test_blast`.
  - `GET /api/events?limit=&before_id=&kinds=`; `GET /api/events/{id}/snapshot.jpg`.

- [ ] **Step 1: Write failing tests** with `fastapi.testclient` and fakes:
  - `PUT /api/zone` with 2 points → 422, and the stored zone is unchanged.
  - A valid zone → 200 and it persists in the settings file.
  - `PUT /api/settings {"dog_conf": 2}` → 422.
  - `POST /api/enabled false` → status shows `enabled: false` and a `disabled` event exists.
  - `POST /api/test-blast {"level":3}` → the fake sound got `PlayBlasts(1,3)`.
  - `GET /api/events?kinds=intrusion` filters.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement the API**, then the UI: one page, vanilla JS, no build step.
  - Live view (`<img src=/stream.mjpg>`) with a status bar polled every 1 s.
  - Big on/off toggle.
  - Zone editor: freeze `/frame.jpg`, click to add points, drag to move, right-click to delete, Save/Cancel; the polygon is drawn on a `<canvas>` over the image.
  - Settings form: thresholds as sliders, timings as number inputs, L1–L3 gain sliders each with a "Test" button and a "Measured dB" field.
  - Event log: a table with time, kind, detail and a snapshot thumbnail (click to enlarge), a filter by kind, and "load more".
  - Must work on a phone browser.
- [ ] **Step 4:** Run → PASS. Then manual: open `http://192.168.7.208:8080` on a laptop and a phone.
- [ ] **Step 5:** Commit: `feat: web app for live view, zones, settings, event log`.

### Task 10: Service entry point, deploy, and end-to-end check

**Files:**
- Create: `dogbot/__main__.py`, `deploy/dogbot.service`, `deploy/deploy.sh`, `README.md`

**Interfaces:**
- Consumes: everything.
- Produces:
  - `python -m dogbot [--port 8080] [--hef PATH]` starts the pipeline thread and uvicorn on `0.0.0.0:8080`, and logs a `startup` event. On SIGTERM it stops sound, stops the pipeline and closes the camera and Hailo.
  - `deploy.sh` rsyncs to `admin@192.168.7.208:/home/admin/dogbot` (excluding `.venv`, `config/`, `data/`), creates the venv with `--system-site-packages` if missing, `pip install -e .`, installs the systemd unit, then runs `systemctl restart dogbot`.
  - The unit runs as `admin` with `Restart=on-failure` and `WorkingDirectory=/home/admin/dogbot`.

- [ ] **Step 1: Run the full suite on the Mac:** `pytest -q` → all pass (Tasks 2/8 tests that need hardware are marked `@pytest.mark.pi` and skipped off-Pi).
- [ ] **Step 2: Deploy:** `./deploy/deploy.sh`, then `ssh admin@192.168.7.208 systemctl is-active dogbot` → `active`.
- [ ] **Step 3: End-to-end on the Pi with the owner:**
  1. Draw the couch zone.
  2. Use Test buttons + a phone SPL app to calibrate L1–L3 and enter the dB values.
  3. Stand in view and confirm DISARMED.
  4. Leave and confirm it shows ARMED after the grace period (temporarily set `rearm_grace_s=15` for the test).
  5. Have a dog (or a dog photo or plush toy on the couch, if needed) trigger rounds 1→2→3.
  6. Walk in mid-sequence and confirm the sound stops at once.
  7. Check the event log entries and snapshots.
  8. Restore the grace setting.
- [ ] **Step 4: Reboot test:** `sudo reboot` → the service comes back, `/dev/hailo0` exists, and the UI loads. If the Hailo driver is missing after a kernel update, run `sudo dkms autoinstall` (documented in the README).
- [ ] **Step 5:** Commit: `feat: service entry point, deploy script, README`.
