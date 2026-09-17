/* Speech Plan 2D-lane annotator */

const LANE_ORDER = [
  "content",
  "pitch",
  "energy",
  "rate",
  "emphasis",
  "pause",
  "nonverbal",
  "boundary",
];

const LANE_TYPE = {
  content: "WORD",
  pitch: "PITCH",
  energy: "ENERGY",
  rate: "RATE",
  emphasis: "EMPHASIS",
  pause: "PAUSE",
  boundary: "BOUNDARY",
  nonverbal: "NONVERBAL",
};

const DEFAULT_VALUE = {
  pitch: "rising",
  energy: "stronger",
  rate: "slower",
  emphasis: "strong",
  pause: 200,
  boundary: "completing",
  nonverbal: "hmm",
};

let meta = null;
let clips = [];
let currentId = null;
let plan = null;
let wavesurfer = null;
let selected = null; // { lane, index }
let dragState = null;

const el = (id) => document.getElementById(id);

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`${res.status}: ${t}`);
  }
  return res.json();
}

function setStatus(msg) {
  el("status").textContent = msg || "";
}

function formatTime(t) {
  return (t || 0).toFixed(2);
}

function syncTimeLabel() {
  if (!wavesurfer || !plan) return;
  const cur = wavesurfer.getCurrentTime();
  el("time-label").textContent = `${formatTime(cur)} / ${formatTime(plan.duration)}`;
  document.querySelectorAll(".playhead").forEach((ph) => {
    const pct = plan.duration > 0 ? (cur / plan.duration) * 100 : 0;
    ph.style.left = `${pct}%`;
  });
}

function fillGlobalStateSelect() {
  const sel = el("global-state");
  sel.innerHTML = '<option value="">— unset —</option>';
  (meta?.vocab?.global_state || []).forEach((v) => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v;
    sel.appendChild(o);
  });
}

function defaultNonverbalValue() {
  const lang = (plan?.language || "").toLowerCase();
  const forms = meta?.vocab?.nonverbal_catalog?.indic_forms || [];
  const match = forms.find((f) => f.language === lang);
  if (match) return match.annotation_value || match.native;
  return DEFAULT_VALUE.nonverbal;
}

/** Returns [{value, label}] for select options, or null for free text. */
function valueOptionsForLane(lane) {
  if (lane === "content") return null;
  if (lane === "pause") return null;
  if (lane === "nonverbal") {
    const catalog = meta?.vocab?.nonverbal_catalog;
    if (catalog?.options?.length) {
      return catalog.options.map((o) => ({
        value: o.id,
        label: o.label || o.id,
        group: o.group || "",
      }));
    }
    // Fallback: flat ids (includes native Indic scripts)
    return (meta?.vocab?.nonverbal || []).map((v) => ({
      value: v,
      label: v,
      group: "",
    }));
  }
  return (meta?.vocab?.[lane] || []).map((v) => ({
    value: v,
    label: v,
    group: "",
  }));
}

function renderClipList() {
  const ul = el("clip-list");
  ul.innerHTML = "";
  clips.forEach((c) => {
    const li = document.createElement("li");
    li.textContent = c.id;
    if (c.id === currentId) li.classList.add("active");
    li.onclick = () => loadClip(c.id);
    ul.appendChild(li);
  });
}

function buildLanesDom() {
  const root = el("lanes");
  root.innerHTML = "";
  LANE_ORDER.forEach((lane) => {
    const row = document.createElement("div");
    row.className = `lane lane-${lane}`;
    row.dataset.lane = lane;
    row.innerHTML = `
      <div class="lane-label">${lane}</div>
      <div class="lane-track" data-lane="${lane}">
        <div class="playhead"></div>
      </div>`;
    root.appendChild(row);
    const track = row.querySelector(".lane-track");
    // Content is ASR-sourced; still allow nonverbal + other lanes to drag-add.
    if (lane !== "content") {
      track.addEventListener("mousedown", (e) => onTrackMouseDown(e, lane));
    }
  });
}

function renderSpans() {
  if (!plan) return;
  LANE_ORDER.forEach((lane) => {
    const track = document.querySelector(`.lane-track[data-lane="${lane}"]`);
    if (!track) return;
    track.querySelectorAll(".span").forEach((n) => n.remove());
    const events = plan.lanes[lane] || [];
    const dur = plan.duration || 1;
    events.forEach((ev, index) => {
      const span = document.createElement("div");
      const cls = lane === "content" ? "word" : lane;
      span.className = `span ${cls}`;
      const left = (ev.start / dur) * 100;
      let width = ((ev.end - ev.start) / dur) * 100;
      if (lane === "boundary") {
        span.style.left = `${left}%`;
        span.style.width = "14px";
        span.style.marginLeft = "-7px";
      } else {
        span.style.left = `${left}%`;
        span.style.width = `${Math.max(width, 0.4)}%`;
      }
      const label =
        lane === "content"
          ? String(ev.value ?? "")
          : lane === "pause"
            ? `${ev.value ?? ""}ms`
            : String(ev.value ?? "");
      span.textContent = label;
      span.title = `${ev.type} ${ev.start.toFixed(2)}–${ev.end.toFixed(2)} (${ev.source})`;
      if (selected && selected.lane === lane && selected.index === index) {
        span.classList.add("selected");
      }
      span.onclick = (e) => {
        e.stopPropagation();
        openEditor(lane, index);
      };
      track.appendChild(span);
    });
  });
  syncTimeLabel();
}

function onTrackMouseDown(e, lane) {
  if (e.button !== 0) return;
  if (e.target.classList.contains("span")) return;
  const track = e.currentTarget;
  const rect = track.getBoundingClientRect();
  const startX = e.clientX;
  const startFrac = (e.clientX - rect.left) / rect.width;
  dragState = { lane, track, rect, startX, startFrac, startTime: startFrac * plan.duration };
  const ghost = document.createElement("div");
  ghost.className = `span ${lane}`;
  ghost.style.opacity = "0.5";
  ghost.id = "drag-ghost";
  track.appendChild(ghost);

  const onMove = (ev) => {
    const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
    const t0 = Math.min(startFrac, frac) * plan.duration;
    const t1 = Math.max(startFrac, frac) * plan.duration;
    ghost.style.left = `${(t0 / plan.duration) * 100}%`;
    ghost.style.width = `${Math.max(((t1 - t0) / plan.duration) * 100, 0.3)}%`;
  };
  const onUp = (ev) => {
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
    ghost.remove();
    const frac = Math.min(1, Math.max(0, (ev.clientX - rect.left) / rect.width));
    let t0 = Math.min(startFrac, frac) * plan.duration;
    let t1 = Math.max(startFrac, frac) * plan.duration;
    if (t1 - t0 < 0.04) {
      // click without drag → short marker
      t0 = Math.max(0, startFrac * plan.duration - 0.02);
      t1 = Math.min(plan.duration, startFrac * plan.duration + 0.02);
    }
    addEvent(lane, t0, t1);
    dragState = null;
  };
  document.addEventListener("mousemove", onMove);
  document.addEventListener("mouseup", onUp);
}

function addEvent(lane, start, end) {
  if (!plan.lanes[lane]) plan.lanes[lane] = [];
  const type = LANE_TYPE[lane] || lane.toUpperCase();
  let value = DEFAULT_VALUE[lane] ?? "";
  if (lane === "pause") value = Math.round((end - start) * 1000);
  if (lane === "nonverbal") value = defaultNonverbalValue();
  const ev = {
    type,
    start: Math.round(start * 1000) / 1000,
    end: Math.round(end * 1000) / 1000,
    value,
    intensity: null,
    source: "human",
  };
  plan.lanes[lane].push(ev);
  selected = { lane, index: plan.lanes[lane].length - 1 };
  renderSpans();
  openEditor(lane, selected.index);
  setStatus("Added event (unsaved)");
}

function openEditor(lane, index) {
  selected = { lane, index };
  const ev = plan.lanes[lane][index];
  if (!ev) return;
  el("editor").classList.remove("hidden");
  el("ed-lane").value = lane;
  el("ed-type").value = ev.type;
  el("ed-start").value = ev.start;
  el("ed-end").value = ev.end;

  const opts = valueOptionsForLane(lane);
  const sel = el("ed-value");
  const text = el("ed-value-text");
  if (opts && opts.length) {
    sel.classList.remove("hidden");
    text.classList.add("hidden");
    sel.innerHTML = "";
    const byGroup = {};
    opts.forEach((o) => {
      const g = o.group || "_";
      if (!byGroup[g]) byGroup[g] = [];
      byGroup[g].push(o);
    });
    const groupOrder = [
      "indic",
      "conversational",
      "emotional",
      "breathing",
      "thinking",
      "questioning",
      "physical",
      "interaction",
      "_",
    ];
    const seen = new Set();
    groupOrder.forEach((g) => {
      if (!byGroup[g]) return;
      seen.add(g);
      const parent = g === "_" ? sel : document.createElement("optgroup");
      if (g !== "_") {
        parent.label = g === "indic" ? "Indic (native transcript)" : g;
        sel.appendChild(parent);
      }
      byGroup[g].forEach((item) => {
        const o = document.createElement("option");
        o.value = item.value;
        o.textContent = item.label;
        if (String(ev.value) === String(item.value)) o.selected = true;
        parent.appendChild(o);
      });
    });
    Object.keys(byGroup).forEach((g) => {
      if (seen.has(g)) return;
      const parent = document.createElement("optgroup");
      parent.label = g;
      sel.appendChild(parent);
      byGroup[g].forEach((item) => {
        const o = document.createElement("option");
        o.value = item.value;
        o.textContent = item.label;
        if (String(ev.value) === String(item.value)) o.selected = true;
        parent.appendChild(o);
      });
    });
  } else {
    sel.classList.add("hidden");
    text.classList.remove("hidden");
    text.value = ev.value ?? "";
  }
  renderSpans();
}

function applyEditor() {
  if (!selected) return;
  const { lane, index } = selected;
  const ev = plan.lanes[lane][index];
  ev.type = el("ed-type").value || ev.type;
  ev.start = parseFloat(el("ed-start").value);
  ev.end = parseFloat(el("ed-end").value);
  const opts = valueOptionsForLane(lane);
  if (opts && opts.length) {
    ev.value = el("ed-value").value;
  } else {
    const raw = el("ed-value-text").value;
    ev.value = lane === "pause" ? Number(raw) : raw;
  }
  if (lane !== "content") ev.source = "human";
  renderSpans();
  setStatus("Edited (unsaved)");
}

function deleteSelected() {
  if (!selected) return;
  const { lane, index } = selected;
  if (lane === "content") {
    setStatus("Content words come from ASR — edit carefully");
  }
  plan.lanes[lane].splice(index, 1);
  selected = null;
  el("editor").classList.add("hidden");
  renderSpans();
  setStatus("Deleted (unsaved)");
}

async function loadClip(id) {
  setStatus("Loading…");
  selected = null;
  el("editor").classList.add("hidden");
  const data = await api(`/api/clips/${encodeURIComponent(id)}`);
  currentId = id;
  plan = data.plan;
  renderClipList();

  el("global-state").value = plan.global_state?.value || "";

  if (wavesurfer) {
    wavesurfer.destroy();
    wavesurfer = null;
  }
  wavesurfer = WaveSurfer.create({
    container: "#waveform",
    waveColor: "#4a6280",
    progressColor: "#4ea1ff",
    cursorColor: "#ffffff",
    height: 96,
    normalize: true,
  });
  wavesurfer.on("audioprocess", syncTimeLabel);
  wavesurfer.on("seeking", syncTimeLabel);
  wavesurfer.on("timeupdate", syncTimeLabel);
  wavesurfer.on("play", () => {
    el("btn-play").textContent = "Pause";
  });
  wavesurfer.on("pause", () => {
    el("btn-play").textContent = "Play";
  });

  if (data.audio_url) {
    try {
      await wavesurfer.load(data.audio_url);
    } catch (err) {
      console.error(err);
      setStatus(`Audio load failed: ${err}`);
    }
  }
  if (wavesurfer.getDuration()) {
    plan.duration = plan.duration || wavesurfer.getDuration();
  }
  buildLanesDom();
  renderSpans();
  setStatus(`Loaded ${id}`);
}

async function saveClip() {
  if (!currentId || !plan) return;
  plan.global_state = {
    value: el("global-state").value || null,
    source: "human",
  };
  // Mirror nonverbal lane to top-level convenience field.
  plan.nonverbal = plan.lanes.nonverbal || [];
  setStatus("Saving…");
  await api(`/api/clips/${encodeURIComponent(currentId)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan }),
  });
  setStatus("Saved");
}

function nextClip() {
  if (!clips.length) return;
  const idx = clips.findIndex((c) => c.id === currentId);
  const next = clips[(idx + 1) % clips.length];
  loadClip(next.id);
}

async function init() {
  meta = await api("/api/meta");
  el("root-label").textContent = meta.root;
  fillGlobalStateSelect();
  const data = await api("/api/clips");
  clips = data.clips || [];
  renderClipList();
  buildLanesDom();

  el("btn-play").onclick = () => {
    if (!wavesurfer) return;
    wavesurfer.playPause();
  };
  el("btn-save").onclick = () => saveClip().catch((e) => setStatus(String(e)));
  el("btn-apply").onclick = applyEditor;
  el("btn-delete").onclick = deleteSelected;
  el("btn-cancel-edit").onclick = () => {
    selected = null;
    el("editor").classList.add("hidden");
    renderSpans();
  };
  el("global-state").onchange = () => setStatus("Global state changed (unsaved)");

  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.code === "Space") {
      e.preventDefault();
      if (wavesurfer) wavesurfer.playPause();
    }
    if (e.key === "n" || e.key === "N") nextClip();
  });

  if (clips.length) loadClip(clips[0].id);
  else setStatus("No speech_plan.json under root — run extract first");
}

init().catch((e) => {
  console.error(e);
  setStatus(String(e));
});
