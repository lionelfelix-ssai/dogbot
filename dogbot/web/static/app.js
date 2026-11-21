/* DogBot control panel: status, on/off, couch zone editor, tuning, event log.
   Vanilla JS, no build step, no network dependencies beyond the DogBot API. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const STATUS_MS = 1000;
const EVENTS_MS = 5000;
const PAGE = 50;

// ---------------------------------------------------------------- API helper

async function api(method, url, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch (_) { /* empty body */ }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    if (data && data.detail) {
      msg = Array.isArray(data.detail)
        ? data.detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
        : String(data.detail);
    }
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return data;
}

let toastTimer = null;
function toast(msg, bad = false) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.toggle("bad", bad);
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, bad ? 5000 : 2600);
}

function busy(btn, on) {
  btn.classList.toggle("busy", on);
  btn.disabled = on;
}

// ------------------------------------------------------------------- status

let lastStatus = null;
let powerBusy = false;

function describe(st) {
  if (st.fault) {
    return ["fault", "FAULT", "Camera or detector problem. DogBot stays silent until it recovers."];
  }
  if (!st.enabled) {
    return ["disabled", "DISABLED", "Guard is off. DogBot will not make any sound."];
  }
  if (st.armed) {
    switch (st.state) {
      case "blasting": return ["armed", "ARMED", `Sounding: round ${st.round} of sequence ${st.sequence}.`];
      case "gap": return ["armed", "ARMED", `Dog on the couch. Next round in a moment (round ${st.round}, sequence ${st.sequence} done).`];
      case "paused": return ["armed", "ARMED", "Dog left the couch. Waiting to see if it stays off."];
      case "unresolved": return ["armed", "ARMED", "Gave up for now. Silent until the couch has been clear for a while."];
      case "held": return ["armed", "ARMED", "Resuming where it left off if the dog is still there."];
      default: return ["armed", "ARMED", "No one around. Watching the couch."];
    }
  }
  const secs = Math.ceil(st.seconds_until_armed || 0);
  if (st.people > 0) {
    return ["disarmed", "DISARMED", `Person in view. Arms ${fmtDuration(secs)} after everyone leaves.`];
  }
  return ["disarmed", "DISARMED", secs > 0 ? `Arming in ${fmtDuration(secs)}…` : "Arming…"];
}

function fmtDuration(s) {
  if (s < 90) return `${s} s`;
  const m = Math.floor(s / 60), r = s % 60;
  return r ? `${m} min ${r} s` : `${m} min`;
}

function renderStatus(st) {
  lastStatus = st;
  const [state, title, sub] = describe(st);
  const banner = $("#banner");
  banner.dataset.state = state;
  banner.classList.toggle("blasting", !!st.armed && st.state === "blasting" && st.enabled && !st.fault);
  document.body.dataset.state = state;
  $("#banner-title").textContent = title;
  $("#banner-sub").textContent = sub;

  const power = $("#power");
  if (!powerBusy) {
    power.disabled = false;
    power.setAttribute("aria-pressed", String(!!st.enabled));
    $("#power-label").textContent = st.enabled ? "Guard on" : "Guard off";
    power.title = st.enabled ? "Turn the guard off" : "Turn the guard on";
  }

  setReadout("#r-dogs", st.dogs, st.dogs > 0 ? "warm" : "");
  setReadout("#r-people", st.people, "");
  setReadout("#r-couch", st.occupied ? "DOG" : "clear", st.occupied ? "hot" : "");
  setReadout("#r-state", st.fault ? "fault" : st.state, st.state === "blasting" ? "hot" : "");
  setReadout("#r-round", st.round ? `R${st.round} · S${st.sequence}` : "–", "");
  setReadout("#r-fps", Number(st.fps || 0).toFixed(1), "");
  $("#zone-warn").hidden = st.zone_set !== false;
  if (live.naturalWidth > 0 && !zone.editing) $("#nosignal").hidden = true;  // MJPEG load events vary by browser

  setLink(true);
}

function setReadout(sel, value, cls) {
  const el = $(sel);
  el.textContent = String(value);
  el.className = "readout-v" + (cls ? " " + cls : "");
}

function setLink(up) {
  const link = $("#link");
  link.classList.toggle("up", up);
  link.classList.toggle("down", !up);
  $("#link-text").textContent = up ? "online" : "offline";
  if (!up) {
    const banner = $("#banner");
    banner.dataset.state = "offline";
    banner.classList.remove("blasting");
    document.body.dataset.state = "offline";
    $("#banner-title").textContent = "OFFLINE";
    $("#banner-sub").textContent = "Can't reach DogBot. Retrying…";
    $("#power").disabled = true;
  }
}

async function pollStatus() {
  try {
    renderStatus(await api("GET", "/api/status"));
  } catch (_) {
    setLink(false);
  } finally {
    setTimeout(pollStatus, STATUS_MS);
  }
}

$("#power").addEventListener("click", async () => {
  if (!lastStatus) return;
  const want = !lastStatus.enabled;
  const power = $("#power");
  powerBusy = true;
  power.disabled = true;
  power.setAttribute("aria-pressed", String(want));   // optimistic
  $("#power-label").textContent = want ? "Guard on" : "Guard off";
  try {
    const st = await api("POST", "/api/enabled", { enabled: want });
    powerBusy = false;
    renderStatus(st);
    toast(want ? "Guard on" : "Guard off. No sounds.");
    refreshEvents();
  } catch (e) {
    powerBusy = false;
    power.disabled = false;
    power.setAttribute("aria-pressed", String(!want));
    toast(`Couldn't switch: ${e.message}`, true);
  }
});

// -------------------------------------------------------------- live stream

const live = $("#live");
let streamOn = true;

function startStream() {
  streamOn = true;
  $("#nosignal").hidden = false;   // until the first frame arrives
  live.src = "/stream.mjpg?t=" + Date.now();
}
function stopStream() {
  streamOn = false;
  $("#nosignal").hidden = false;   // never show the bare <img> alt text
  live.removeAttribute("src");   // closes the connection
}
live.addEventListener("load", () => { $("#nosignal").hidden = true; });
live.addEventListener("error", () => {
  if (!streamOn) return;
  $("#nosignal").hidden = false;
  setTimeout(() => { if (streamOn && !zone.editing && !document.hidden) startStream(); }, 2000);
});
document.addEventListener("visibilitychange", () => {
  if (zone.editing) return;
  if (document.hidden) stopStream(); else startStream();
});

// ---------------------------------------------------------------- settings

let settings = null;
const form = $("#settings-form");
const NUM_FIELDS = ["dog_conf", "person_conf", "dwell_s", "rearm_grace_s", "clear_s",
                    "round_gap_s", "sequences", "tone_hz", "blast_s", "blast_spacing_s"];
const levelRows = {};

function buildLevelRows() {
  const tpl = $("#level-row");
  for (const n of [1, 2, 3]) {
    const row = tpl.content.firstElementChild.cloneNode(true);
    row.dataset.level = n;
    row.querySelector(".level-tag").textContent = "L" + n;
    const gain = row.querySelector(".level-gain");
    const db = row.querySelector(".level-db");
    gain.setAttribute("aria-label", `Level ${n} gain`);
    db.setAttribute("aria-label", `Level ${n} measured dB`);
    const btn = row.querySelector(".test-btn");
    btn.setAttribute("aria-label", `Play one test blast at level ${n}`);
    btn.addEventListener("click", () => testBlast(n, btn));
    $("#levels").appendChild(row);
    levelRows[n] = { row, gain, db, out: row.querySelector(".level-out"), btn };
  }
}

function fillForm(s) {
  for (const name of NUM_FIELDS) form.elements[name].value = s[name];
  form.elements.audio_device.value = s.audio_device;
  for (const n of [1, 2, 3]) {
    levelRows[n].gain.value = s.levels[n];
    const db = s.level_db[n];
    levelRows[n].db.value = db === null || db === undefined ? "" : db;
  }
  for (const el of form.querySelectorAll("input")) el.classList.remove("invalid");
  syncOutputs();
  updateDirty();
}

function syncOutputs() {
  for (const input of form.querySelectorAll('input[type="range"]')) {
    const pct = ((input.value - input.min) / (input.max - input.min)) * 100;
    input.style.setProperty("--pct", pct + "%");
  }
  for (const name of ["dog_conf", "person_conf"]) {
    form.querySelector(`output[data-for="${name}"]`).textContent = Number(form.elements[name].value).toFixed(2);
  }
  for (const n of [1, 2, 3]) {
    levelRows[n].out.textContent = Math.round(levelRows[n].gain.value * 100) + "%";
  }
}

function readNumber(input) {
  if (input.value.trim() === "") return NaN;
  return Number(input.value);
}

/** Changed fields only, as a partial update. Marks unparsable inputs invalid. */
function formChanges() {
  const out = {};
  let bad = false;
  for (const name of NUM_FIELDS) {
    const el = form.elements[name];
    const v = readNumber(el);
    const invalid = !Number.isFinite(v) || (name === "sequences" && !Number.isInteger(v));
    el.classList.toggle("invalid", invalid);
    if (invalid) { bad = true; continue; }
    if (v !== settings[name]) out[name] = v;
  }
  const dev = form.elements.audio_device.value.trim();
  if (dev !== settings.audio_device) out.audio_device = dev;
  const levels = {}, dbs = {};
  for (const n of [1, 2, 3]) {
    const g = Number(levelRows[n].gain.value);
    if (g !== settings.levels[n]) levels[n] = g;
    const raw = levelRows[n].db.value.trim();
    const db = raw === "" ? null : Number(raw);
    const invalid = db !== null && !Number.isFinite(db);
    levelRows[n].db.classList.toggle("invalid", invalid);
    if (invalid) { bad = true; continue; }
    if (db !== settings.level_db[n]) dbs[n] = db;
  }
  if (Object.keys(levels).length) out.levels = levels;
  if (Object.keys(dbs).length) out.level_db = dbs;
  return { changes: out, bad };
}

function updateDirty() {
  if (!settings) return;
  const { changes, bad } = formChanges();
  const dirty = Object.keys(changes).length > 0 || bad;
  $("#dirty-pill").hidden = !dirty;
  $("#save").disabled = !dirty || bad;
  $("#revert").disabled = !dirty;
}

function showFormError(msg) {
  const el = $("#form-error");
  el.textContent = msg || "";
  el.hidden = !msg;
}

form.addEventListener("input", () => { syncOutputs(); updateDirty(); showFormError(""); });

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const { changes, bad } = formChanges();
  if (bad || !Object.keys(changes).length) return;
  const btn = $("#save");
  busy(btn, true);
  try {
    settings = await api("PUT", "/api/settings", changes);
    fillForm(settings);
    toast("Settings saved");
    refreshEvents();
  } catch (e) {
    showFormError(e.message);
  } finally {
    btn.classList.remove("busy");
    updateDirty();
  }
});

$("#revert").addEventListener("click", () => { fillForm(settings); showFormError(""); });

async function testBlast(level, btn) {
  busy(btn, true);
  try {
    // Calibrate what you see: save this level's gain first if it was moved.
    const g = Number(levelRows[level].gain.value);
    if (settings && g !== settings.levels[level]) {
      settings = await api("PUT", "/api/settings", { levels: { [level]: g } });
      const keep = formChanges().changes;     // keep other unsaved edits
      fillForm(settings);
      reapply(keep);
    }
    const r = await api("POST", "/api/test-blast", { level });
    btn.classList.add("fired");
    setTimeout(() => btn.classList.remove("fired"), 700);
    toast(`Level ${level} test blast at ${Math.round(r.gain * 100)}% gain`);
    refreshEvents();
  } catch (e) {
    toast(`Test blast failed: ${e.message}`, true);
  } finally {
    btn.classList.remove("busy");
    btn.disabled = false;
  }
}

function reapply(changes) {
  for (const [k, v] of Object.entries(changes)) {
    if (k === "levels") for (const [n, g] of Object.entries(v)) levelRows[n].gain.value = g;
    else if (k === "level_db") for (const [n, d] of Object.entries(v)) levelRows[n].db.value = d === null ? "" : d;
    else if (form.elements[k]) form.elements[k].value = v;
  }
  syncOutputs();
  updateDirty();
}

async function loadSettings() {
  try {
    settings = await api("GET", "/api/settings");
    fillForm(settings);
    zone.draw();
  } catch (_) {
    setTimeout(loadSettings, 2000);
  }
}

// ------------------------------------------------------------- zone editor

const zone = {
  editing: false,
  points: [],      // [[x, y], ...] normalized 0-1
  history: [],
  drag: null,      // {index, pointerId, startX, startY, moved}
  longPress: null,
  hover: -1,
  canvas: $("#zone-canvas"),
  still: $("#still"),

  hitRadius(pointerType) { return pointerType === "mouse" ? 12 : 24; },

  /** Image content rect inside the canvas (object-fit: contain), CSS px. */
  rect() {
    const cw = this.canvas.clientWidth, ch = this.canvas.clientHeight;
    const iw = this.still.naturalWidth || 16, ih = this.still.naturalHeight || 9;
    const scale = Math.min(cw / iw, ch / ih);
    const w = iw * scale, h = ih * scale;
    return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
  },

  toNorm(ev) {
    const b = this.canvas.getBoundingClientRect();
    const r = this.rect();
    const x = (ev.clientX - b.left - r.x) / r.w;
    const y = (ev.clientY - b.top - r.y) / r.h;
    return [clamp01(x), clamp01(y)];
  },

  toPx([x, y]) {
    const r = this.rect();
    return [r.x + x * r.w, r.y + y * r.h];
  },

  hit(ev) {
    const b = this.canvas.getBoundingClientRect();
    const px = ev.clientX - b.left, py = ev.clientY - b.top;
    const rad = this.hitRadius(ev.pointerType);
    let best = -1, bestD = Infinity;
    this.points.forEach((p, i) => {
      const [x, y] = this.toPx(p);
      const d = Math.hypot(x - px, y - py);
      if (d <= rad && d < bestD) { best = i; bestD = d; }
    });
    return best;
  },

  /** Where a new point goes: after 3 points, into the closest edge. */
  insertIndex(p) {
    const n = this.points.length;
    if (n < 3) return n;
    let best = n, bestD = Infinity;
    for (let i = 0; i < n; i++) {
      const d = segDist(p, this.points[i], this.points[(i + 1) % n]);
      if (d < bestD) { bestD = d; best = i + 1; }
    }
    return best;
  },

  snapshot() {
    this.history.push(this.points.map((p) => p.slice()));
    if (this.history.length > 100) this.history.shift();
  },

  async open() {
    if (this.editing) return;
    const btn = $("#zone-edit");
    busy(btn, true);
    try {
      const res = await fetch("/frame.jpg?t=" + Date.now(), { cache: "no-store" });
      if (!res.ok) throw new Error(res.status === 503 ? "no camera frame yet" : res.statusText);
      const url = URL.createObjectURL(await res.blob());
      await new Promise((ok, fail) => {
        this.still.onload = ok;
        this.still.onerror = () => fail(new Error("bad frame"));
        if (this.still.src.startsWith("blob:")) URL.revokeObjectURL(this.still.src);
        this.still.src = url;
      });
    } catch (e) {
      toast(`Can't freeze the view: ${e.message}`, true);
      btn.classList.remove("busy");
      btn.disabled = false;
      return;
    }
    btn.classList.remove("busy");
    this.editing = true;
    this.points = (settings ? settings.zone : []).map((p) => p.slice());
    this.history = [];
    stopStream();
    live.hidden = true;
    $("#nosignal").hidden = true;
    this.still.hidden = false;
    this.canvas.hidden = false;
    $("#viewport").classList.add("editing");
    $("#zone-bar").hidden = false;
    btn.hidden = true;
    $("#view-title").textContent = "Couch zone (frozen frame)";
    const coarse = window.matchMedia("(pointer: coarse)").matches;
    $("#zone-hint").textContent = coarse
      ? "Tap to add a corner. Drag a corner to move it. Long-press a corner to delete it."
      : "Click to add a corner. Drag a corner to move it. Right-click a corner to delete it.";
    this.draw();
  },

  close() {
    this.editing = false;
    this.drag = null;
    clearTimeout(this.longPress);
    this.still.hidden = true;
    this.canvas.hidden = true;
    live.hidden = false;
    $("#viewport").classList.remove("editing");
    $("#zone-bar").hidden = true;
    const btn = $("#zone-edit");
    btn.hidden = false;
    btn.disabled = false;
    $("#view-title").textContent = "Live view";
    startStream();
  },

  async save() {
    if (this.points.length < 3) {
      toast("A zone needs at least 3 corners", true);
      return;
    }
    const btn = $("#zone-save");
    busy(btn, true);
    try {
      settings = await api("PUT", "/api/zone", { points: this.points });
      toast("Couch zone saved");
      this.close();
      refreshEvents();
      pollOnce();
    } catch (e) {
      toast(`Zone not saved: ${e.message}`, true);
    } finally {
      btn.classList.remove("busy");
      btn.disabled = false;
    }
  },

  draw() {
    const c = this.canvas;
    if (c.hidden) return;
    const dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
      c.width = Math.round(w * dpr);
      c.height = Math.round(h * dpr);
    }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const pts = this.points.map((p) => this.toPx(p));
    if (pts.length) {
      g.beginPath();
      pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
      if (pts.length >= 3) {
        g.closePath();
        g.fillStyle = "rgba(255, 176, 32, 0.22)";
        g.fill();
      }
      g.lineWidth = 2.5;
      g.strokeStyle = "#ffb020";
      g.setLineDash(pts.length >= 3 ? [] : [8, 6]);
      g.stroke();
      g.setLineDash([]);
    }
    pts.forEach(([x, y], i) => {
      const r = i === this.hover || (this.drag && this.drag.index === i) ? 10 : 7;
      g.beginPath();
      g.arc(x, y, r, 0, Math.PI * 2);
      g.fillStyle = "#ffb020";
      g.fill();
      g.lineWidth = 2.5;
      g.strokeStyle = "#ffffff";
      g.stroke();
    });
    const undo = $("#zone-undo");
    undo.disabled = this.history.length === 0;
    $("#zone-clear").disabled = this.points.length === 0;
    $("#zone-save").disabled = this.points.length < 3;
  },
};

function clamp01(v) { return Math.min(1, Math.max(0, v)); }

function segDist(p, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len2 = dx * dx + dy * dy;
  let t = len2 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
}

zone.canvas.addEventListener("pointerdown", (ev) => {
  if (ev.button !== 0) return;
  ev.preventDefault();
  let index = zone.hit(ev);
  zone.snapshot();
  if (index < 0) {
    const p = zone.toNorm(ev);
    index = zone.insertIndex(p);
    zone.points.splice(index, 0, p);
  } else if (ev.pointerType !== "mouse") {
    // long-press on an existing corner deletes it (no right-click on phones)
    zone.longPress = setTimeout(() => {
      if (zone.drag && !zone.drag.moved) {
        zone.points.splice(zone.drag.index, 1);
        zone.drag = null;
        if (navigator.vibrate) navigator.vibrate(30);
        zone.draw();
      }
    }, 550);
  }
  zone.drag = { index, pointerId: ev.pointerId, x: ev.clientX, y: ev.clientY, moved: false };
  zone.canvas.setPointerCapture(ev.pointerId);
  zone.draw();
});

zone.canvas.addEventListener("pointermove", (ev) => {
  if (zone.drag && zone.drag.pointerId === ev.pointerId) {
    if (Math.hypot(ev.clientX - zone.drag.x, ev.clientY - zone.drag.y) > 6) {
      zone.drag.moved = true;
      clearTimeout(zone.longPress);
    }
    zone.points[zone.drag.index] = zone.toNorm(ev);
    zone.draw();
  } else if (ev.pointerType === "mouse") {
    const h = zone.hit(ev);
    if (h !== zone.hover) {
      zone.hover = h;
      zone.canvas.style.cursor = h >= 0 ? "grab" : "crosshair";
      zone.draw();
    }
  }
});

function endDrag(ev) {
  if (zone.drag && zone.drag.pointerId === ev.pointerId) {
    zone.drag = null;
    clearTimeout(zone.longPress);
    zone.draw();
  }
}
zone.canvas.addEventListener("pointerup", endDrag);
zone.canvas.addEventListener("pointercancel", endDrag);

zone.canvas.addEventListener("contextmenu", (ev) => {
  ev.preventDefault();
  if (zone.drag) return;   // touch long-press: handled by the pointer timer
  const i = zone.hit({ pointerType: "mouse", clientX: ev.clientX, clientY: ev.clientY });
  if (i >= 0) {
    zone.snapshot();
    zone.points.splice(i, 1);
    zone.hover = -1;
    zone.draw();
  }
});

$("#zone-edit").addEventListener("click", () => zone.open());
$("#zone-cancel").addEventListener("click", () => zone.close());
$("#zone-save").addEventListener("click", () => zone.save());
$("#zone-undo").addEventListener("click", () => {
  if (zone.history.length) { zone.points = zone.history.pop(); zone.draw(); }
});
$("#zone-clear").addEventListener("click", () => { zone.snapshot(); zone.points = []; zone.draw(); });

document.addEventListener("keydown", (ev) => {
  if (!zone.editing) return;
  if (ev.key === "Escape" && !$("#lightbox").open) zone.close();
  if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "z") {
    ev.preventDefault();
    $("#zone-undo").click();
  }
});

new ResizeObserver(() => zone.draw()).observe(zone.canvas);

// --------------------------------------------------------------- event log

const log = { kinds: "", newest: 0, oldest: null };

function fmtTime(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return iso;
  const today = new Date();
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  if (d.toDateString() === today.toDateString()) return time;
  return d.toLocaleDateString([], { month: "short", day: "numeric" }) + " " + time;
}

const LABELS = {
  settings: {
    dog_conf: "dog conf", person_conf: "person conf", dwell_s: "dwell", rearm_grace_s: "grace",
    clear_s: "clear", round_gap_s: "gap", blast_s: "blast", blast_spacing_s: "spacing",
    tone_hz: "tone", levels: "gains", level_db: "dB", sequences: "sequences",
    enabled: "enabled", audio_device: "device",
  },
};

function fmtValue(v) {
  if (v === null) return "—";
  if (typeof v === "object") {
    return Object.entries(v).map(([k, x]) => `L${k} ${x === null ? "—" : x}`).join(", ");
  }
  return String(v);
}

function fmtDetail(kind, d) {
  d = d || {};
  switch (kind) {
    case "round": return `Round ${d.round} · sequence ${d.sequence}`;
    case "unresolved": return `Still there after ${d.sequences} sequence(s). Going quiet.`;
    case "intrusion": return "Dog on the couch";
    case "cleared": return "Couch clear";
    case "held": return `Paused at round ${d.round}, sequence ${d.sequence}`;
    case "stopped": return d.reason === "inactive" ? "Sound cut (guard inactive)" : (d.reason || "");
    case "disarmed": return d.reason ? `Reason: ${d.reason}` : "Person seen";
    case "armed": return "No person for the grace period";
    case "zone": return `${d.points} corners`;
    case "test_blast": return `Level ${d.level} at ${Math.round((d.gain || 0) * 100)}% gain`;
    case "enabled": return d.via ? "Guard on (settings)" : "Guard on";
    case "disabled": return d.via ? "Guard off (settings)" : "Guard off";
    case "settings": {
      const ch = d.changed || {};
      return Object.entries(ch).map(([k, v]) => `${LABELS.settings[k] || k} → ${fmtValue(v)}`).join("; ");
    }
    case "error":
      if (d.recovered) return `${d.source || "system"} recovered`;
      return `${d.source || "system"}: ${d.error || ""}`;
    default: {
      const keys = Object.keys(d);
      return keys.length ? JSON.stringify(d) : "";
    }
  }
}

function eventRow(e, fresh) {
  const tr = document.createElement("tr");
  if (fresh) tr.className = "fresh";
  const t = document.createElement("td");
  t.className = "t";
  t.textContent = fmtTime(e.ts);
  t.title = e.ts;
  const k = document.createElement("td");
  k.className = "k";
  const badge = document.createElement("span");
  badge.className = "kind";
  badge.dataset.k = e.kind;
  badge.textContent = e.kind.replace("_", " ");
  k.appendChild(badge);
  const d = document.createElement("td");
  d.className = "d";
  d.textContent = fmtDetail(e.kind, e.detail);
  const s = document.createElement("td");
  s.className = "s";
  if (e.has_snapshot) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "thumb";
    b.setAttribute("aria-label", `Enlarge snapshot of ${e.kind} at ${fmtTime(e.ts)}`);
    const img = document.createElement("img");
    img.loading = "lazy";
    img.alt = "";
    img.src = `/api/events/${e.id}/snapshot.jpg`;
    b.appendChild(img);
    b.addEventListener("click", () => openLightbox(e));
    s.appendChild(b);
  }
  tr.append(t, k, d, s);
  return tr;
}

function eventsUrl(extra) {
  const q = new URLSearchParams({ limit: String(PAGE), ...extra });
  if (log.kinds) q.set("kinds", log.kinds);
  return "/api/events?" + q.toString();
}

async function loadEvents() {
  const body = $("#log-body");
  try {
    const rows = await api("GET", eventsUrl({}));
    body.replaceChildren(...rows.map((e) => eventRow(e, false)));
    log.newest = rows.length ? rows[0].id : 0;
    log.oldest = rows.length ? rows[rows.length - 1].id : null;
    $("#log-more").hidden = rows.length < PAGE;
    $("#log-empty").hidden = rows.length > 0;
  } catch (e) {
    $("#log-empty").hidden = false;
    $("#log-empty").textContent = `Couldn't load events: ${e.message}`;
  }
}

async function refreshEvents() {
  try {
    const rows = await api("GET", eventsUrl({}));
    const fresh = rows.filter((e) => e.id > log.newest);
    if (!fresh.length) return;
    const body = $("#log-body");
    body.prepend(...fresh.map((e) => eventRow(e, true)));
    log.newest = fresh[0].id;
    if (log.oldest === null) log.oldest = fresh[fresh.length - 1].id;
    $("#log-empty").hidden = true;
  } catch (_) { /* status poll shows offline */ }
}

$("#log-more").addEventListener("click", async (ev) => {
  const btn = ev.currentTarget;
  if (log.oldest === null) return;
  busy(btn, true);
  try {
    const rows = await api("GET", eventsUrl({ before_id: String(log.oldest) }));
    $("#log-body").append(...rows.map((e) => eventRow(e, false)));
    if (rows.length) log.oldest = rows[rows.length - 1].id;
    btn.hidden = rows.length < PAGE;
  } catch (e) {
    toast(`Couldn't load more: ${e.message}`, true);
  } finally {
    btn.classList.remove("busy");
    btn.disabled = false;
  }
});

$("#log-filter").addEventListener("change", (ev) => {
  log.kinds = ev.target.value;
  try { localStorage.setItem("dogbot.logFilter", log.kinds); } catch (_) { /* private mode */ }
  loadEvents();
});

function openLightbox(e) {
  $("#lightbox-img").src = `/api/events/${e.id}/snapshot.jpg`;
  $("#lightbox-cap").textContent = `${e.kind.replace("_", " ")} · ${fmtTime(e.ts)}`;
  $("#lightbox").showModal();
}
$("#lightbox").addEventListener("click", (ev) => {
  if (ev.target === ev.currentTarget) ev.currentTarget.close();  // backdrop click
});

// -------------------------------------------------------------------- boot

async function pollOnce() {
  try { renderStatus(await api("GET", "/api/status")); } catch (_) { setLink(false); }
}

function boot() {
  buildLevelRows();
  try {
    const saved = localStorage.getItem("dogbot.logFilter");
    if (saved !== null && [...$("#log-filter").options].some((o) => o.value === saved)) {
      $("#log-filter").value = saved;
      log.kinds = saved;
    }
  } catch (_) { /* storage blocked */ }
  pollStatus();
  loadSettings();
  loadEvents();
  setInterval(() => { if (!document.hidden) refreshEvents(); }, EVENTS_MS);
}

boot();
