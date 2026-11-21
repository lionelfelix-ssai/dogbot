# DogBot MVP — Design

Date: 2026-11-20 · Host: `ai-robot` (Pi 5 4 GB, 192.168.7.208, user `admin`)

## Intent

A fixed AI camera that keeps the dogs off one specific sofa **when no person is around**. It is the first "job" of a longer-lived multi-job AI camera platform (later: pan-tilt foam launcher, other detection jobs, recording). The owner has wanted this for over a year; it should feel like a polished little robot, not a script.

## Goals (MVP)

1. Detect dogs and people in the camera frame in real time on the Hailo-8.
2. **Arming is automatic:** no person visible for a grace period → *armed*; any person visible → *disarmed*. A manual master switch in the web app overrides (off = never fires).
3. Track each dog with a stable ID; draw a bounding box plus a **reticle** on tracked dogs in the live view.
4. One **forbidden zone** (the couch), drawn as a polygon in the web app.
5. When an armed dog is in the forbidden zone: escalating sound — **round 1: 1 blast at level L1; round 2: 2 blasts at L2; round 3: 3 blasts at L3** — stopping as soon as the dog leaves.
6. Web app: live annotated view, on/off switch, zone editor, sensitivity settings, event log.

## Non-goals (MVP)

- Lights/LEDs (deferred).
- Video recording / clips (v2.0).
- Pan-tilt, foam launcher (later; architecture must leave room).
- Telling individual dogs apart; custom model training.
- Warning zone (only the forbidden zone in MVP; the zone model supports more).

## Anti-goals

- Firing when a person is present, or on a person/cushion/blanket misdetected as a dog. A false blast with someone on the couch is worse than a missed dog.
- Endless or unbounded sound. Every escalation is capped.
- Anything that needs the cloud to work.

## Constraints

- Hardware: Pi 5 4 GB, Hailo-8L (13 TOPS, PCIe; use `*_h8l.hef` models), IMX708 12 MP autofocus camera (no on-sensor AI), Raspberry Pi OS Bookworm.
- **No audio output exists today** besides HDMI. A USB speaker (or USB audio adapter + small powered speaker) must be added.
- "Volume in dB": the Pi cannot measure SPL. Levels L1–L3 are configured as output gain and calibrated once with a phone SPL-meter app at the couch; the UI shows the calibrated dB next to each level.
- Hailo runs compiled `.hef` models only — not PyTorch/Ultralytics weights directly.

## Approaches considered

| | Approach | Trade-off |
|---|---|---|
| A | Frigate NVR + automations | Least code, recording free; logic (person suppression, escalation) split across systems, poor base for future jobs/aiming. **Rejected** — owner wants custom. |
| B | Hailo's official GStreamer example pipeline + callbacks | Fastest; GStreamer is rigid for overlays, multiple jobs, aiming. |
| C | **Custom Python service: Picamera2 + HailoRT + tracker + rules engine + web app** | One process, all logic in our code, easy to extend; more code to own. **Chosen.** |

## Where Ultralytics fits

- Ultralytics publishes the YOLO models (YOLOv8 / YOLO11) and the training toolkit. **Hailo does not run Ultralytics models as-is**: a model is exported to ONNX and compiled to `.hef` with Hailo's Dataflow Compiler (x86 Linux).
- **MVP uses no Ultralytics code at runtime.** It uses Ultralytics' **YOLO26s**, precompiled for Hailo-8L by the Hailo Model Zoo v2.20 (COCO already includes `dog` and `person`). Verified on this Pi with HailoRT 4.20: yolo26s ≈23 FPS, yolo26m ≈11 FPS, yolov8s ≈30 FPS (raw `hailortcli run`, PCIe Gen2). With `dtparam=pciex1_gen=3` (now enabled): yolo26s ≈41 FPS, yolov8s ≈58 FPS. YOLO26 is NMS-free, but the Hailo build outputs raw heads, so the host decodes the boxes; the builder verifies the output format with `hailortcli parse-hef`. Fallback: `yolov8s_h8l.hef` (ships with `hailo-all`; Picamera2/rpicam post-processing already supports it).
- **v2 role:** fine-tune YOLO26 with Ultralytics on our own footage (e.g. individual dog IDs, "dog on couch", fewer false positives in this room), then `model.export(format="hailo")` (Ultralytics ≥8.4.97). Compiling still needs the Hailo Dataflow Compiler on x86 Linux. The `ultralytics/skills` Claude plugin helps with this phase.
- Licensing: Ultralytics is AGPL-3.0 (enterprise license for closed commercial use). Fine for a personal project; revisit if this becomes a product.

## Design

**Runtime (one Python service, systemd, on the Pi):**

1. **Capture:** Picamera2, two streams — a low-res stream for inference (model input size, e.g. 640×640) and a main stream (~1280×720) for the web view. Autofocus fixed at the couch distance once set.
2. **Detect:** HailoRT via Picamera2's Hailo helper, YOLOv8 (s or m, chosen by measured FPS; target ≥15 fps). Classes kept: `dog`, `person`.
3. **Track:** ByteTrack (via the `supervision` library) for stable dog IDs.
4. **Zones:** couch polygon stored in normalized coordinates. A dog is "in" when its anchor point (bottom-center of the box) is inside the polygon for ≥ *dwell* seconds.
5. **Arming:** person seen (confidence ≥ person threshold) → disarmed immediately; armed after *rearm grace* seconds with no person.
6. **Rules engine (state machine per job):** `idle → intrusion → round1 → round2 → round3 → cooldown`. A round plays N blasts at level Ln, then waits *round gap* seconds; it advances only if the dog is still in the zone. Dog out of the zone for *clear time* → stop and reset. After round 3, either stop or repeat round 3 up to a cap (see open question). Disarming at any point stops the sound at once.
7. **Actions:** a pluggable action interface (`sound` now; `lights`, `launcher` later). Sound is played via ALSA with per-level gain.
8. **Events:** every state change (armed/disarmed, intrusion, each round, cleared, manual toggles, settings changes) is written to SQLite with a timestamp and a JPEG snapshot for intrusions.

**Web app (served by the same service, LAN only):**

- Live view (MJPEG) with boxes, labels, track IDs, reticle on tracked dogs, couch polygon colored by state, armed/disarmed banner.
- Master on/off switch.
- Zone editor: click polygon points on a frozen frame.
- Settings: dog and person confidence thresholds, dwell time, rearm grace, clear time, round gap, L1–L3 levels with a "test blast" button for calibration.
- Event log: filterable list with snapshots.

**Jobs:** a job is a config file (model, classes, zones, rules, actions). MVP ships one job, `couch-guard`.

## Decided by the owner

Fixed camera · custom build (not Frigate) · Hailo for inference · arming automatic from person presence · reticle on tracked dogs · 1/2/3 escalating blasts at rising volume · web app with on/off, zones, sensitivity, and event log in MVP · lights deferred · recording in v2.0.

## Left to the builder

Exact model variant (by measured FPS), web framework (proposed: FastAPI + vanilla JS), deterrent sound file/frequency, default timing values, project layout, tests.

## Answers from the owner (builder check)

- **Coverage:** camera only for MVP. Blind spots will be covered later by a PIR sensor and/or the owner's second camera. Design: presence comes from a list of "presence sources" (MVP: camera person detection only), so adding a PIR (GPIO) or a second camera later is just another source. Any source reporting presence disarms.
- **Dark:** not required. The MVP runs in a lit room only.
- **After round 3:** run the full 1→2→3 sequence **twice** in total. If the dog is still on the couch, stop, log an `unresolved` event with a snapshot, and stay silent until the couch has been clear for *clear time*.
- **Audio:** the Pi 5 has no analog/aux output. Use the owner's USB speaker, or a USB audio dongle (3.5 mm out) into a powered speaker. Volume is software ALSA gain per level.

## Builder's calls (owner may override)

- Escalation is **per couch, not per dog**: any dog in the zone drives one shared sequence.
- If a dog gets back on during the cooldown (before *clear time* has passed), the sequence resumes where it left off instead of restarting at round 1. After a full clear, it starts again from round 1.
