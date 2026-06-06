"use strict";

/* fluister — local transcription UI
 * Vanilla JS, no build step. Talks to the FastAPI backend at same origin.
 */

(function () {
  // ── Constants ────────────────────────────────────────────────────────
  const API = "/api/jobs";
  const PERSONS_API = "/api/persons";
  const POLL_MS = 3000;
  const ACTIVE = new Set(["queued", "converting", "transcribing"]);
  const STATUS_LABEL = {
    queued: "Queued",
    converting: "Converting",
    transcribing: "Transcribing",
    done: "Done",
    error: "Error",
    interrupted: "Interrupted",
  };

  // ── State ────────────────────────────────────────────────────────────
  // jobs: id -> job object (latest known server state)
  const jobs = new Map();
  // ui: id -> { expanded, streaming, es, segs:[], liveDetected, liveProgress, liveStatus }
  const ui = new Map();
  // jobJson: id -> { sig, data } — cached structured transcript (download/json).
  //   `sig` is the transcript length the cache was built for, so a changed
  //   transcript (or a persons edit, which clears the map) forces a refetch.
  const jobJson = new Map();
  // ids with an in-flight json fetch, to avoid duplicate requests.
  const jobJsonPending = new Set();
  // persons: latest list from the server (for the Speakers page).
  let persons = [];
  // Active speaker-name filter for the transcriptions list ("" = all).
  let speakerFilterValue = "";

  // ── DOM refs ─────────────────────────────────────────────────────────
  const $ = (sel) => document.querySelector(sel);
  const dropzone = $("#dropzone");
  const fileInput = $("#fileInput");
  const languageSel = $("#language");
  const jobList = $("#jobList");
  const emptyState = $("#emptyState");
  const jobsCount = $("#jobsCount");
  const banner = $("#banner");
  const navButtons = Array.from(document.querySelectorAll(".nav__btn"));
  const views = {
    transcriptions: $("#view-transcriptions"),
    speakers: $("#view-speakers"),
  };
  const personList = $("#personList");
  const speakersEmpty = $("#speakersEmpty");
  const speakersCount = $("#speakersCount");
  const speakerFilter = $("#speakerFilter");
  const clearAllBtn = $("#clearAll");

  // ── Small helpers ────────────────────────────────────────────────────
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        const v = attrs[k];
        if (v == null || v === false) continue;
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else if (k === "html") node.innerHTML = v;
        else if (k.startsWith("on") && typeof v === "function") {
          node.addEventListener(k.slice(2).toLowerCase(), v);
        } else if (k === "dataset") {
          for (const d in v) node.dataset[d] = v[d];
        } else {
          node.setAttribute(k, v === true ? "" : v);
        }
      }
    }
    if (children != null) {
      const arr = Array.isArray(children) ? children : [children];
      for (const c of arr) {
        if (c == null) continue;
        node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
      }
    }
    return node;
  }

  function showBanner(msg) {
    banner.textContent = msg;
    banner.hidden = false;
  }
  function clearBanner() {
    banner.hidden = true;
    banner.textContent = "";
  }

  function safeNumber(v) {
    const n = typeof v === "number" ? v : parseFloat(v);
    return Number.isFinite(n) ? n : null;
  }

  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  // Parse an ISO timestamp defensively. Backend may send either a naive
  // local string ("2026-06-06T09:44:49") or a UTC string with offset/Z.
  function parseTs(iso) {
    if (!iso || typeof iso !== "string") return null;
    const hasZone = /[zZ]|[+-]\d{2}:?\d{2}$/.test(iso);
    let d;
    if (hasZone) {
      d = new Date(iso);
    } else {
      // Treat as local time. Let the Date constructor handle it; if it
      // can't, fall back to manual parsing of YYYY-MM-DDTHH:MM:SS.
      d = new Date(iso);
      if (isNaN(d.getTime())) {
        const m = iso.match(/^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?/);
        if (m) {
          d = new Date(+m[1], +m[2] - 1, +m[3], +(m[4] || 0), +(m[5] || 0), +(m[6] || 0));
        }
      }
    }
    return d && !isNaN(d.getTime()) ? d : null;
  }

  function formatMsgTime(job) {
    const d = parseTs(job.msg_timestamp);
    if (!d) return null;
    const datePart = `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}`;
    // msg_has_time can be 1/0/null. Only show clock when explicitly true-ish.
    const hasTime = job.msg_has_time === 1 || job.msg_has_time === true;
    if (!hasTime) return datePart;
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    return `${datePart}, ${hh}:${mm}`;
  }

  function formatDuration(seconds) {
    const s = safeNumber(seconds);
    if (s == null || s < 0) return null;
    const total = Math.floor(s);
    const sec = total % 60;
    const min = Math.floor(total / 60) % 60;
    const hr = Math.floor(total / 3600);
    const pad = (n) => String(n).padStart(2, "0");
    return hr > 0 ? `${hr}:${pad(min)}:${pad(sec)}` : `${min}:${pad(sec)}`;
  }

  function langLabel(code) {
    if (!code) return null;
    const map = { nl: "Nederlands", en: "English", auto: "Auto-detect" };
    return map[code] || code.toUpperCase();
  }

  // Stable hue (0–359) from a string, so the same speaker keeps the same color
  // everywhere (transcript chips + Speakers page swatch). FNV-1a-ish hash.
  function hueFor(key) {
    const s = String(key == null ? "" : key);
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    return (h >>> 0) % 360;
  }

  // ── UI state accessor ────────────────────────────────────────────────
  function uiFor(id) {
    let u = ui.get(id);
    if (!u) {
      u = {
        expanded: false,
        streaming: false,
        es: null,
        segs: [],
        liveDetected: null,
        liveProgress: null,
        liveStatus: null,
        // Audio playback (lazily created per card).
        audio: null,
        wordSpans: null, // ordered array of clickable word spans
        wordStarts: null, // parallel array of numeric start times
        activeWordIdx: -1, // index of the currently highlighted word span
        audioBroken: false, // true once the audio element errors (404/decode)
      };
      ui.set(id, u);
    }
    return u;
  }

  // ── Rendering ────────────────────────────────────────────────────────
  function effectiveStatus(job, u) {
    // Prefer live SSE status while streaming.
    if (u.streaming && u.liveStatus) return u.liveStatus;
    return job.status || "queued";
  }

  function effectiveProgress(job, u) {
    if (u.streaming && u.liveProgress != null) return u.liveProgress;
    return safeNumber(job.progress) || 0;
  }

  function effectiveDetected(job, u) {
    if (u.liveDetected) return u.liveDetected;
    return job.detected_language || null;
  }

  // Distinct identified speakers on a job, parsed from its `speakers` map.
  function jobSpeakers(job) {
    if (!job || !job.speakers) return [];
    let m = job.speakers;
    if (typeof m === "string") {
      try { m = JSON.parse(m); } catch (e) { return []; }
    }
    if (!m || typeof m !== "object") return [];
    const seen = new Set();
    const out = [];
    for (const k in m) {
      const v = m[k] || {};
      const name = v.name || null;
      const key = v.person_id || k;
      if (!name || seen.has(key)) continue;
      seen.add(key);
      out.push({ name, key });
    }
    return out;
  }

  // Rebuild the speaker filter <select> from the speakers present across jobs.
  function populateSpeakerFilter(all) {
    const names = new Set();
    for (const j of all) for (const s of jobSpeakers(j)) names.add(s.name);
    if (speakerFilterValue && !names.has(speakerFilterValue)) speakerFilterValue = "";
    const sorted = Array.from(names).sort((a, b) => a.localeCompare(b));
    speakerFilter.innerHTML = "";
    speakerFilter.appendChild(el("option", { value: "", text: "All speakers" }));
    for (const n of sorted) speakerFilter.appendChild(el("option", { value: n, text: n }));
    speakerFilter.value = speakerFilterValue;
    speakerFilter.hidden = sorted.length === 0;
  }

  function render() {
    const all = Array.from(jobs.values());
    // Newest first by created_at (fallback to insertion as-is).
    all.sort((a, b) => {
      const ta = parseTs(a.created_at);
      const tb = parseTs(b.created_at);
      const va = ta ? ta.getTime() : 0;
      const vb = tb ? tb.getTime() : 0;
      return vb - va;
    });

    populateSpeakerFilter(all);
    if (clearAllBtn) clearAllBtn.hidden = all.length === 0;
    const shown = speakerFilterValue
      ? all.filter((j) => jobSpeakers(j).some((s) => s.name === speakerFilterValue))
      : all;

    jobsCount.textContent = shown.length ? `${shown.length}` : "";
    if (speakerFilterValue && shown.length === 0) {
      emptyState.textContent = `No transcriptions with “${speakerFilterValue}”.`;
      emptyState.hidden = false;
    } else {
      emptyState.textContent = "No transcriptions yet. Drop a file above to get started.";
      emptyState.hidden = shown.length > 0;
    }

    // Track which cards should exist.
    const wanted = new Set(shown.map((j) => j.id));

    // Remove stale cards.
    Array.from(jobList.children).forEach((node) => {
      if (!wanted.has(node.dataset.id)) {
        node.remove();
      }
    });

    // Upsert in order.
    let prev = null;
    for (const job of shown) {
      let card = jobList.querySelector(`[data-id="${cssEscape(job.id)}"]`);
      if (!card) {
        card = buildCard(job);
        jobList.insertBefore(card, prev ? prev.nextSibling : jobList.firstChild);
      } else {
        updateCard(card, job);
      }
      // Ensure DOM order matches sorted order.
      if (prev) {
        if (prev.nextSibling !== card) jobList.insertBefore(card, prev.nextSibling);
      } else {
        if (jobList.firstChild !== card) jobList.insertBefore(card, jobList.firstChild);
      }
      prev = card;
    }
  }

  function cssEscape(s) {
    if (window.CSS && CSS.escape) return CSS.escape(s);
    return String(s).replace(/["\\\]]/g, "\\$&");
  }

  function buildCard(job) {
    const card = el("div", { class: "card", dataset: { id: job.id } });
    card.appendChild(buildHead(job));
    card.appendChild(buildProgress(job));
    const body = el("div", { class: "card__body" });
    body.hidden = true;
    card.appendChild(body);
    updateCard(card, job); // fill content
    return card;
  }

  function buildHead(job) {
    const head = el("div", { class: "card__head" });
    head.addEventListener("click", (e) => {
      if (e.target.closest(".icon-btn")) return;
      toggleExpand(job.id);
    });

    const main = el("div", { class: "card__main" });
    const title = el("div", { class: "card__title" });
    title.appendChild(chevronIcon());
    title.appendChild(el("span", { class: "card__name", title: job.original_filename, text: job.original_filename || "(unnamed)" }));
    main.appendChild(title);
    main.appendChild(el("div", { class: "card__meta" }));

    const right = el("div", { class: "card__right" });
    right.appendChild(el("span", { class: "badge" }));
    const del = el("button", {
      class: "icon-btn",
      title: "Delete",
      "aria-label": "Delete transcription",
    });
    del.innerHTML =
      '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>';
    del.addEventListener("click", () => deleteJob(job.id));
    right.appendChild(del);

    head.appendChild(main);
    head.appendChild(right);
    return head;
  }

  function chevronIcon() {
    const span = el("span", { class: "chev" });
    span.innerHTML =
      '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg>';
    return span;
  }

  function buildProgress(job) {
    const wrap = el("div", { class: "progress" });
    wrap.appendChild(el("div", { class: "progress__bar" }));
    wrap.hidden = true;
    return wrap;
  }

  function updateCard(card, job) {
    const u = uiFor(job.id);
    const status = effectiveStatus(job, u);
    const progress = effectiveProgress(job, u);

    card.classList.toggle("is-open", u.expanded);

    // Badge
    const badge = card.querySelector(".badge");
    badge.className = `badge badge--${status}`;
    badge.innerHTML = "";
    badge.appendChild(el("span", { class: "pip" }));
    badge.appendChild(document.createTextNode(STATUS_LABEL[status] || status));

    // Meta line
    const meta = card.querySelector(".card__meta");
    meta.innerHTML = "";
    const when = formatMsgTime(job);
    if (when) {
      meta.appendChild(el("span", { text: when }));
      if (job.msg_timestamp_source === "mtime") {
        meta.appendChild(el("span", { class: "tag", title: "Inferred from file date", text: "file date" }));
      }
    } else {
      meta.appendChild(el("span", { class: "tag", text: "no date" }));
    }
    const dur = formatDuration(job.duration);
    if (dur) {
      meta.appendChild(el("span", { class: "dot" }));
      meta.appendChild(el("span", { text: dur }));
    }
    for (const sp of jobSpeakers(job)) {
      const chip = el("span", { class: "chip", text: sp.name });
      chip.style.setProperty("--chip-h", String(hueFor(sp.key)));
      meta.appendChild(chip);
    }

    // Progress bar
    const prog = card.querySelector(".progress");
    const bar = prog.querySelector(".progress__bar");
    const isActive = ACTIVE.has(status);
    const showBar = status === "converting" || status === "transcribing";
    prog.hidden = !showBar;
    if (showBar) {
      const pct = Math.max(0, Math.min(1, progress)) * 100;
      if (pct <= 0.0001) {
        prog.classList.add("indeterminate");
      } else {
        prog.classList.remove("indeterminate");
        bar.style.width = pct + "%";
      }
    }

    // Body (only render details when expanded for cheap updates)
    if (u.expanded) {
      // A full rebuild wipes the transcript node and resets the user's scroll
      // position (and clears any text selection). Avoid it unless the rendered
      // content actually changed.
      //  - Streaming: keep the live transcript, refresh only the header.
      //  - Static (done/error): rebuild only when the body signature changes;
      //    otherwise just refresh the header so late metadata stays fresh.
      const liveArea = card.querySelector('.transcript[data-live="1"]');
      const body = card.querySelector(".card__body");
      if (u.streaming && liveArea) {
        refreshBodyHeader(card, job, u);
      } else {
        // For a finished job, make sure its structured transcript is loaded
        // (or refreshed if stale). The fetch resolves asynchronously and then
        // re-enters updateCard, bumping the signature so the body rebuilds once.
        if (status === "done") ensureJobJson(job.id);
        const sig = bodySignature(job, u);
        if (body.hidden || card.dataset.bodySig !== sig) {
          renderBody(card, job, u);
          card.dataset.bodySig = sig;
        } else {
          refreshBodyHeader(card, job, u);
        }
      }
    }

    // Manage SSE lifecycle.
    if (u.expanded && isActive && !u.streaming) {
      startStream(job.id);
    } else if (!isActive && u.streaming) {
      stopStream(job.id);
    }
  }

  function buildStatLine(job, u) {
    const detected = effectiveDetected(job, u);
    const stat = el("div", { class: "card__statline" });
    const reqLang = langLabel(job.language);
    if (detected) {
      stat.appendChild(el("span", { html: `Language: <strong>${escapeHtml(langLabel(detected))}</strong>` }));
    } else if (reqLang && job.language !== "auto") {
      stat.appendChild(el("span", { html: `Language: <strong>${escapeHtml(reqLang)}</strong>` }));
    }
    const dur = formatDuration(job.duration);
    if (dur) stat.appendChild(el("span", { html: `Duration: <strong>${escapeHtml(dur)}</strong>` }));
    if (job.model_name) stat.appendChild(el("span", { html: `Model: <strong>${escapeHtml(job.model_name)}</strong>` }));
    return stat.children.length ? stat : null;
  }

  // Signature of everything that affects the static (non-streaming) body. When
  // it's unchanged we skip rebuilding the transcript node, preserving the user's
  // scroll position and text selection across background poll refreshes.
  //
  // The diarized-transcript marker is included so that the body is rebuilt
  // exactly once when the structured JSON arrives (or after a persons edit that
  // invalidated the cache), and NOT on every poll while it stays the same.
  function bodySignature(job, u) {
    const status = effectiveStatus(job, u);
    const len = (job.transcript_text || "").length;
    let diarSig = "0";
    if (status === "done") {
      const cached = jobJson.get(job.id);
      diarSig = cached ? `1:${cached.sig}` : "0";
    }
    return `${status}|${job.error || ""}|${len}|${diarSig}`;
  }

  // Length of a job's current plain transcript — the key the JSON cache is
  // validated against. When it changes, the cached structured transcript is
  // stale and we refetch.
  function transcriptLen(job) {
    return (job.transcript_text || "").length;
  }

  // Ensure the structured (diarized) transcript JSON for a done job is loaded.
  // Cached per id; refetched only when the transcript changes or the cache was
  // invalidated (persons edit). On arrival we re-run updateCard so the body
  // signature changes and the transcript is rebuilt once into its colored form.
  function ensureJobJson(id) {
    const job = jobs.get(id);
    if (!job || job.status !== "done") return;
    const cached = jobJson.get(id);
    if (cached && cached.sig === transcriptLen(job)) return; // fresh
    if (jobJsonPending.has(id)) return; // already fetching
    jobJsonPending.add(id);
    const sigAtFetch = transcriptLen(job);
    fetch(`${API}/${encodeURIComponent(id)}/transcript`, {
      headers: { Accept: "application/json" },
    })
      .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`HTTP ${res.status}`))))
      .then((data) => {
        jobJson.set(id, { sig: sigAtFetch, data: data || null });
      })
      .catch(() => {
        // Leave the cache empty; the transcript falls back to plain text. A
        // later poll/expand will retry.
      })
      .finally(() => {
        jobJsonPending.delete(id);
        const u = ui.get(id);
        const card = jobList.querySelector(`[data-id="${cssEscape(id)}"]`);
        if (card && u && u.expanded) updateCard(card, jobs.get(id) || {});
      });
  }

  // True when the cached JSON actually carries per-segment speakers.
  function hasSpeakers(data) {
    if (!data || !Array.isArray(data.segments)) return false;
    return data.segments.some((s) => s && s.speaker != null);
  }

  // Update only the stat line in place (used during live streaming so the
  // transcript area and its scroll position are left untouched).
  function refreshBodyHeader(card, job, u) {
    const body = card.querySelector(".card__body");
    const existing = body.querySelector(".card__statline");
    const fresh = buildStatLine(job, u);
    if (existing && fresh) {
      existing.replaceWith(fresh);
    } else if (existing && !fresh) {
      existing.remove();
    } else if (!existing && fresh) {
      body.insertBefore(fresh, body.firstChild);
    }
  }

  function renderBody(card, job, u) {
    const body = card.querySelector(".card__body");
    body.hidden = false;
    // Stop any prior audio and drop stale span caches before wiping the body so
    // the about-to-be-removed audio element / word spans don't leak or play on.
    teardownAudio(job.id);
    body.innerHTML = "";

    const status = effectiveStatus(job, u);

    // Stat line
    const stat = buildStatLine(job, u);
    if (stat) body.appendChild(stat);

    // Error
    if ((status === "error" || status === "interrupted")) {
      const msg = job.error || (status === "interrupted" ? "Interrupted before completion." : "An unknown error occurred.");
      body.appendChild(el("div", { class: "error-box", text: msg }));
    }

    // Transcript area
    if (status === "done") {
      const cached = jobJson.get(job.id);
      const data = cached ? cached.data : null;
      const hasSegments = data && Array.isArray(data.segments) && data.segments.length > 0;
      let t;
      if (data && hasSpeakers(data)) {
        t = buildDiarizedTranscript(data);
      } else if (hasSegments) {
        t = buildWordTranscript(data);
      } else {
        const text = job.transcript_text || "";
        t = el("div", { class: "transcript" });
        if (text.trim()) t.textContent = text;
        else t.appendChild(el("span", { class: "placeholder", text: "(empty transcript)" }));
      }
      body.appendChild(t);
      // Cache this card's word spans for click-to-seek + live highlight, then
      // wire clicks and append the player bar below the transcript.
      cacheWordSpans(u, t);
      wireWordClicks(card, job.id, t);
      body.appendChild(buildPlayerBar(card, job.id, u));
      body.appendChild(buildActions(job));
    } else if (u.streaming || ACTIVE.has(status)) {
      const t = el("div", { class: "transcript", dataset: { live: "1" } });
      u.segs.forEach((s, i) => {
        t.appendChild(el("span", { class: "seg", text: (i > 0 ? " " : "") + s }));
      });
      t.appendChild(el("span", { class: "caret", "aria-hidden": "true" }));
      body.appendChild(t);
    } else if (status === "error" || status === "interrupted") {
      // no transcript area beyond error-box; still offer delete via head
    }
  }

  // Append one segment's words into `container` as clickable spans. Each word
  // span carries its numeric start time in dataset.start; a trailing text node
  // keeps natural spacing so copy/selection reads cleanly. Segments without a
  // `words` array (older jobs) render as a plain text node — no click target.
  function appendWordSpans(container, seg) {
    if (Array.isArray(seg.words) && seg.words.length) {
      for (const w of seg.words) {
        if (!w || typeof w.word !== "string") continue;
        const start = safeNumber(w.start);
        const span = el("span", {
          class: "word",
          text: w.word,
          dataset: { start: start == null ? "" : String(start) },
        });
        container.appendChild(span);
        container.appendChild(document.createTextNode(" "));
      }
    } else {
      container.appendChild(document.createTextNode(typeof seg.text === "string" ? seg.text : ""));
    }
  }

  // Build a flat (non-diarized) transcript of word spans from structured
  // segments. Used for done jobs that have structured data but no speakers.
  function buildWordTranscript(data) {
    const t = el("div", { class: "transcript" });
    const segs = (data && data.segments) || [];
    let any = false;
    let first = true;
    for (const seg of segs) {
      if (!seg) continue;
      const text = typeof seg.text === "string" ? seg.text.trim() : "";
      if (!text && !(Array.isArray(seg.words) && seg.words.length)) continue;
      if (!first) t.appendChild(document.createTextNode(" "));
      appendWordSpans(t, seg);
      first = false;
      any = true;
    }
    if (!any) {
      t.appendChild(el("span", { class: "placeholder", text: "(empty transcript)" }));
    }
    return t;
  }

  // Render structured segments grouped into speaker turns. Consecutive segments
  // by the same speaker are merged into one turn led by a colored name chip.
  // The chip hue is derived from a stable hash of person_id||speaker so it
  // matches the swatch on the Speakers page.
  function buildDiarizedTranscript(data) {
    const t = el("div", { class: "transcript transcript--diarized" });
    const speakers = (data && data.speakers) || {};
    const segs = (data && data.segments) || [];

    let turn = null;
    let turnKey = null; // identity used to decide when a new turn starts

    const flush = () => {
      if (turn) t.appendChild(turn.node);
      turn = null;
      turnKey = null;
    };

    for (const seg of segs) {
      if (!seg) continue;
      const text = typeof seg.text === "string" ? seg.text.trim() : "";
      if (!text) continue;

      // Resolve a stable color key and a display name. Prefer the live speaker
      // map (so renames/merges reflected in the JSON win), else the segment.
      const mapEntry = seg.speaker ? speakers[seg.speaker] : null;
      const personId = (mapEntry && mapEntry.person_id) || seg.person_id || null;
      const name =
        (mapEntry && mapEntry.name) || seg.speaker || (personId ? "Speaker" : null);
      const key = personId || seg.speaker || "__none";

      if (turnKey !== key) {
        flush();
        const node = el("div", { class: "turn" });
        if (name) {
          const chip = el("span", { class: "chip", text: name });
          chip.style.setProperty("--chip-h", String(hueFor(key)));
          node.appendChild(chip);
        }
        const textNode = el("div", { class: "turn__text" });
        node.appendChild(textNode);
        turn = { node, textNode };
        turnKey = key;
        appendWordSpans(textNode, seg);
      } else {
        turn.textNode.appendChild(document.createTextNode(" "));
        appendWordSpans(turn.textNode, seg);
      }
    }
    flush();

    if (!t.children.length) {
      t.appendChild(el("span", { class: "placeholder", text: "(empty transcript)" }));
    }
    return t;
  }

  // ── Audio playback + player bar ───────────────────────────────────────
  // At most one card's audio plays at a time. Track the currently-playing id.
  let playingId = null;

  // Format seconds as "m:ss" (no hours; transcripts are typically short, and
  // the existing duration field handles long-form display elsewhere).
  function formatClock(seconds) {
    const s = safeNumber(seconds);
    if (s == null || s < 0) return "0:00";
    const total = Math.floor(s);
    const sec = total % 60;
    const min = Math.floor(total / 60);
    return `${min}:${String(sec).padStart(2, "0")}`;
  }

  // Cache the ordered word spans (and their numeric starts) for a transcript
  // node so the timeupdate highlighter can do a cheap linear/binary search.
  function cacheWordSpans(u, transcriptNode) {
    const spans = Array.from(transcriptNode.querySelectorAll(".word"));
    const starts = [];
    const kept = [];
    for (const sp of spans) {
      const v = safeNumber(sp.dataset.start);
      if (v == null) continue; // skip words without a usable start time
      kept.push(sp);
      starts.push(v);
    }
    u.wordSpans = kept;
    u.wordStarts = starts;
    u.activeWordIdx = -1;
  }

  // Lazily create (once) the card's <audio> element and wire its events.
  function ensureAudio(card, jobId, u) {
    if (u.audio) return u.audio;
    const audio = el("audio", { preload: "none" });
    audio.style.display = "none";
    u.audio = audio;

    audio.addEventListener("loadedmetadata", () => {
      updatePlayerTime(card, u);
    });
    audio.addEventListener("timeupdate", () => {
      updatePlayerTime(card, u);
      highlightAtTime(u, audio.currentTime);
    });
    audio.addEventListener("play", () => {
      playingId = jobId;
      setPlayIcon(card, true);
    });
    audio.addEventListener("pause", () => {
      if (playingId === jobId) playingId = null;
      setPlayIcon(card, false);
      clearHighlight(u);
    });
    audio.addEventListener("ended", () => {
      if (playingId === jobId) playingId = null;
      setPlayIcon(card, false);
      clearHighlight(u);
    });
    audio.addEventListener("error", () => {
      u.audioBroken = true;
      markPlayerUnavailable(card);
      clearHighlight(u);
    });

    card.appendChild(audio);
    return audio;
  }

  function audioSrcFor(jobId) {
    return `${API}/${encodeURIComponent(jobId)}/audio`;
  }

  // Pause whatever card is currently playing (other than `keepId`), so only one
  // plays at a time.
  function pauseOthers(keepId) {
    if (playingId == null || playingId === keepId) return;
    const other = ui.get(playingId);
    if (other && other.audio && !other.audio.paused) {
      try { other.audio.pause(); } catch (e) { /* noop */ }
    }
  }

  // Seek the card's audio to startSec (minus a small lead) and play it.
  function seekAndPlay(card, jobId, startSec) {
    const u = uiFor(jobId);
    if (u.audioBroken) return;
    const audio = ensureAudio(card, jobId, u);
    pauseOthers(jobId);
    const target = Math.max(0, (safeNumber(startSec) || 0) - 0.25);
    const src = audioSrcFor(jobId);
    if (!audio.src || audio.src.indexOf(src) === -1) {
      audio.src = src;
    }
    if (audio.readyState >= 1) {
      try { audio.currentTime = target; } catch (e) { /* noop */ }
      const p = audio.play();
      if (p && p.catch) p.catch(() => {});
    } else {
      const onReady = () => {
        audio.removeEventListener("loadedmetadata", onReady);
        audio.removeEventListener("canplay", onReady);
        try { audio.currentTime = target; } catch (e) { /* noop */ }
        const p = audio.play();
        if (p && p.catch) p.catch(() => {});
      };
      audio.addEventListener("loadedmetadata", onReady);
      audio.addEventListener("canplay", onReady);
      // Kick off loading if it hasn't started.
      try { audio.load(); } catch (e) { /* noop */ }
    }
  }

  // Attach a single delegated click handler on the transcript so any .word
  // seeks + plays. No-op when the audio is known to be broken.
  function wireWordClicks(card, jobId, transcriptNode) {
    transcriptNode.addEventListener("click", (e) => {
      const span = e.target.closest && e.target.closest(".word");
      if (!span || !transcriptNode.contains(span)) return;
      const u = uiFor(jobId);
      if (u.audioBroken) return;
      const start = safeNumber(span.dataset.start);
      if (start == null) return;
      seekAndPlay(card, jobId, start);
    });
  }

  // Build the small player bar (play/pause + "m:ss / m:ss"). Lazily creates the
  // audio element so metadata (and a possible error) can populate the bar.
  function buildPlayerBar(card, jobId, u) {
    const bar = el("div", { class: "player" });
    const btn = el("button", {
      class: "player__btn",
      type: "button",
      "aria-label": "Play",
      title: "Play / pause",
    });
    btn.innerHTML = playIconSvg(false);
    btn.addEventListener("click", () => togglePlay(card, jobId));
    bar.appendChild(btn);

    const time = el("span", { class: "player__time", text: "0:00 / 0:00" });
    bar.appendChild(time);

    // Create the audio now so word clicks and the bar share one element, and so
    // a src error (404/decode) flips the bar into its disabled state via the
    // audio "error" handler wired in ensureAudio().
    const audio = ensureAudio(card, jobId, u);
    if (!audio.src) audio.src = audioSrcFor(jobId);
    return bar;
  }

  function togglePlay(card, jobId) {
    const u = uiFor(jobId);
    if (u.audioBroken) return;
    const audio = ensureAudio(card, jobId, u);
    if (audio.paused) {
      pauseOthers(jobId);
      const src = audioSrcFor(jobId);
      if (!audio.src || audio.src.indexOf(src) === -1) audio.src = src;
      const p = audio.play();
      if (p && p.catch) p.catch(() => {});
    } else {
      audio.pause();
    }
  }

  function playIconSvg(isPlaying) {
    return isPlaying
      ? '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>'
      : '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
  }

  function setPlayIcon(card, isPlaying) {
    const btn = card.querySelector(".player__btn");
    if (!btn) return;
    btn.innerHTML = playIconSvg(isPlaying);
    btn.setAttribute("aria-label", isPlaying ? "Pause" : "Play");
  }

  function updatePlayerTime(card, u) {
    const time = card.querySelector(".player__time");
    if (!time || !u.audio) return;
    const cur = u.audio.currentTime || 0;
    const dur = Number.isFinite(u.audio.duration) ? u.audio.duration : 0;
    time.textContent = `${formatClock(cur)} / ${formatClock(dur)}`;
  }

  // Disable the player bar and show an "audio unavailable" state.
  function markPlayerUnavailable(card) {
    const bar = card.querySelector(".player");
    if (!bar) return;
    bar.classList.add("player--unavailable");
    const btn = bar.querySelector(".player__btn");
    if (btn) btn.disabled = true;
    const time = bar.querySelector(".player__time");
    if (time) time.textContent = "audio unavailable";
  }

  // ── Live highlight ────────────────────────────────────────────────────
  function setActiveWord(u, idx) {
    if (idx === u.activeWordIdx) return;
    if (u.activeWordIdx >= 0 && u.wordSpans && u.wordSpans[u.activeWordIdx]) {
      u.wordSpans[u.activeWordIdx].classList.remove("word--active");
    }
    u.activeWordIdx = idx;
    if (idx >= 0 && u.wordSpans && u.wordSpans[idx]) {
      u.wordSpans[idx].classList.add("word--active");
    }
  }

  function clearHighlight(u) {
    setActiveWord(u, -1);
  }

  // Highlight the word whose [start, nextStart) interval contains `t` (the last
  // word's upper bound is Infinity). Binary search over the cached starts.
  function highlightAtTime(u, t) {
    const starts = u.wordStarts;
    if (!starts || !starts.length) return;
    const time = safeNumber(t);
    if (time == null) return;
    if (time < starts[0]) {
      setActiveWord(u, -1);
      return;
    }
    let lo = 0;
    let hi = starts.length - 1;
    let idx = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (starts[mid] <= time) {
        idx = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    setActiveWord(u, idx);
  }

  // Stop and reset a card's audio, and drop its cached spans/highlight. Called
  // when a body is rebuilt or a card is collapsed/closed.
  function teardownAudio(id) {
    const u = ui.get(id);
    if (!u) return;
    if (u.audio) {
      try { u.audio.pause(); } catch (e) { /* noop */ }
      try { u.audio.removeAttribute("src"); u.audio.load(); } catch (e) { /* noop */ }
      if (u.audio.parentNode) u.audio.parentNode.removeChild(u.audio);
    }
    if (playingId === id) playingId = null;
    u.audio = null;
    u.wordSpans = null;
    u.wordStarts = null;
    u.activeWordIdx = -1;
    u.audioBroken = false;
  }

  function buildActions(job) {
    const actions = el("div", { class: "actions" });
    const copy = el("button", { class: "btn btn--ghost" });
    copy.appendChild(copyIcon());
    copy.appendChild(document.createTextNode("Copy"));
    copy.addEventListener("click", () => copyTranscript(job, copy));
    actions.appendChild(copy);
    return actions;
  }

  function copyIcon() {
    const span = document.createElement("span");
    span.style.display = "inline-flex";
    span.innerHTML =
      '<svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
    return span;
  }

  function escapeHtml(s) {
    if (s == null) return "";
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  // ── Confirm modal (native <dialog>, replaces window.confirm) ──────────
  let _dialog = null;
  function ensureDialog() {
    if (_dialog) return _dialog;
    const dlg = el("dialog", { class: "modal" });
    const card = el("div", { class: "modal__card" });
    card.appendChild(el("h3", { class: "modal__title" }));
    card.appendChild(el("p", { class: "modal__msg" }));
    const actions = el("div", { class: "modal__actions" });
    actions.appendChild(el("button", { type: "button", class: "btn btn--ghost", dataset: { act: "cancel" }, text: "Cancel" }));
    actions.appendChild(el("button", { type: "button", class: "btn btn--danger", dataset: { act: "confirm" }, text: "Delete" }));
    card.appendChild(actions);
    dlg.appendChild(card);
    document.body.appendChild(dlg);
    _dialog = dlg;
    return dlg;
  }

  // Returns a Promise<boolean>. Esc / backdrop / Cancel resolve false.
  function confirmModal(opts) {
    const o = opts || {};
    const dlg = ensureDialog();
    dlg.querySelector(".modal__title").textContent = o.title || "Are you sure?";
    dlg.querySelector(".modal__msg").textContent = o.message || "";
    const confirmBtn = dlg.querySelector('[data-act="confirm"]');
    const cancelBtn = dlg.querySelector('[data-act="cancel"]');
    confirmBtn.textContent = o.confirmLabel || "Delete";
    confirmBtn.className = "btn " + (o.danger === false ? "" : "btn--danger");
    cancelBtn.hidden = o.info === true; // info mode: single OK button
    return new Promise((resolve) => {
      const close = (val) => {
        confirmBtn.onclick = null;
        cancelBtn.onclick = null;
        dlg.removeEventListener("cancel", onCancel);
        dlg.removeEventListener("click", onBackdrop);
        if (dlg.open) dlg.close();
        resolve(val);
      };
      const onCancel = (e) => { e.preventDefault(); close(false); }; // Esc
      const onBackdrop = (e) => { if (e.target === dlg) close(false); };
      confirmBtn.onclick = () => close(true);
      cancelBtn.onclick = () => close(false);
      dlg.addEventListener("cancel", onCancel);
      dlg.addEventListener("click", onBackdrop);
      dlg.showModal();
      confirmBtn.focus();
    });
  }

  // One-button info dialog (reuses the confirm modal).
  function notify(title, message) {
    return confirmModal({ title, message, confirmLabel: "OK", danger: false, info: true });
  }

  // ── Interactions ─────────────────────────────────────────────────────
  function toggleExpand(id) {
    const u = uiFor(id);
    u.expanded = !u.expanded;
    const card = jobList.querySelector(`[data-id="${cssEscape(id)}"]`);
    const job = jobs.get(id);
    if (!card || !job) return;

    if (u.expanded) {
      // Seed live segments from stored transcript if the job is active but
      // we don't yet have any streamed segments (rare; keeps area non-empty).
      updateCard(card, job);
    } else {
      const body = card.querySelector(".card__body");
      body.hidden = true;
      teardownAudio(id); // stop playback + drop span cache on collapse
      body.innerHTML = "";
      delete card.dataset.bodySig;
      card.classList.remove("is-open");
      // Keep streaming in background? No — stop to save resources; polling
      // will keep the badge/progress fresh.
      if (u.streaming) stopStream(id);
      updateCard(card, job);
    }
  }

  async function copyTranscript(job, btn) {
    const text = (jobs.get(job.id) || job).transcript_text || "";
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        legacyCopy(text);
      }
      flashCopied(btn);
    } catch (err) {
      try {
        legacyCopy(text);
        flashCopied(btn);
      } catch (e) {
        showBanner("Couldn't copy to clipboard.");
      }
    }
  }

  function legacyCopy(text) {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    document.body.removeChild(ta);
  }

  function flashCopied(btn) {
    btn.classList.add("is-copied");
    if (btn.lastChild && btn.lastChild.nodeType === Node.TEXT_NODE) {
      btn.lastChild.textContent = "Copied";
    }
    setTimeout(() => {
      btn.classList.remove("is-copied");
      if (btn.lastChild && btn.lastChild.nodeType === Node.TEXT_NODE) {
        btn.lastChild.textContent = "Copy";
      }
    }, 1600);
  }

  async function deleteJob(id) {
    const job = jobs.get(id);
    const name = job ? job.original_filename : "this transcription";
    const ok = await confirmModal({
      title: "Delete transcription",
      message: `Delete “${name}”? This removes the job and its files.`,
      confirmLabel: "Delete",
    });
    if (!ok) return;
    // Optimistic UI: stop stream + audio, remove card.
    const u = uiFor(id);
    if (u.streaming) stopStream(id);
    teardownAudio(id);
    try {
      const res = await fetch(`${API}/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (!res.ok && res.status !== 404) {
        throw new Error(`HTTP ${res.status}`);
      }
      jobs.delete(id);
      ui.delete(id);
      jobJson.delete(id);
      jobJsonPending.delete(id);
      render();
      clearBanner();
    } catch (err) {
      showBanner("Couldn't delete the job. Is the server running?");
    }
  }

  async function clearAllJobs() {
    if (jobs.size === 0) return;
    const ok = await confirmModal({
      title: "Clear all transcriptions",
      message: "Delete all transcriptions and their files? Your speakers (voice profiles) are kept.",
      confirmLabel: "Clear all",
    });
    if (!ok) return;
    try {
      const res = await fetch(`${API}/clear`, { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      for (const id of Array.from(ui.keys())) {
        const u = ui.get(id);
        if (u && u.streaming) stopStream(id);
        teardownAudio(id);
      }
      jobs.clear();
      ui.clear();
      jobJson.clear();
      jobJsonPending.clear();
      speakerFilterValue = "";
      clearBanner();
      render();
    } catch (err) {
      showBanner("Couldn't clear transcriptions.");
    }
  }

  // ── SSE streaming ────────────────────────────────────────────────────
  function startStream(id) {
    const u = uiFor(id);
    if (u.streaming || u.es) return;
    let es;
    try {
      es = new EventSource(`${API}/${encodeURIComponent(id)}/events`);
    } catch (err) {
      return;
    }
    u.es = es;
    u.streaming = true;
    u.segs = []; // fresh stream; collect from scratch

    es.addEventListener("status", (e) => {
      const data = parseEvent(e);
      if (!data) return;
      if (data.status) u.liveStatus = data.status;
      const p = safeNumber(data.progress);
      if (p != null) u.liveProgress = p;
      if (data.detected_language) u.liveDetected = data.detected_language;
      refreshLive(id);
    });

    es.addEventListener("segment", (e) => {
      const data = parseEvent(e);
      if (!data || typeof data.text !== "string") return;
      const txt = data.text.trim();
      if (txt) {
        const isFirst = u.segs.length === 0;
        u.segs.push(txt);
        appendSegment(id, txt, isFirst);
      }
    });

    es.addEventListener("done", (e) => {
      const data = parseEvent(e);
      if (data && typeof data === "object" && data.id) {
        jobs.set(data.id, data);
      } else {
        const cur = jobs.get(id);
        if (cur) cur.status = "done";
      }
      stopStream(id);
      const card = jobList.querySelector(`[data-id="${cssEscape(id)}"]`);
      if (card) updateCard(card, jobs.get(id) || {});
    });

    es.addEventListener("error", (e) => {
      // Could be a protocol "error" event OR a connection drop.
      const data = parseEvent(e);
      if (data && data.message) {
        const cur = jobs.get(id) || {};
        cur.status = "error";
        cur.error = data.message;
        jobs.set(id, cur);
        u.liveStatus = "error";
        stopStream(id);
        const card = jobList.querySelector(`[data-id="${cssEscape(id)}"]`);
        if (card) updateCard(card, jobs.get(id));
      }
      // On a plain connection error (no parsable data), let polling reconcile;
      // EventSource will attempt reconnection on its own. If the job already
      // finished, the next poll will flip it and stopStream() runs.
    });

    // Reflect the freshly-opened streaming state.
    const card = jobList.querySelector(`[data-id="${cssEscape(id)}"]`);
    if (card) updateCard(card, jobs.get(id) || {});
  }

  function parseEvent(e) {
    if (!e || typeof e.data !== "string" || e.data === "") return null;
    try {
      return JSON.parse(e.data);
    } catch (err) {
      return null;
    }
  }

  function stopStream(id) {
    const u = ui.get(id);
    if (!u) return;
    if (u.es) {
      try { u.es.close(); } catch (e) { /* noop */ }
    }
    u.es = null;
    u.streaming = false;
    u.liveStatus = null;
    u.liveProgress = null;
  }

  // Light-touch updates while streaming (avoid full body re-render churn).
  function refreshLive(id) {
    const card = jobList.querySelector(`[data-id="${cssEscape(id)}"]`);
    const job = jobs.get(id);
    if (!card || !job) return;
    updateCard(card, job);
  }

  function appendSegment(id, txt, isFirst) {
    const u = uiFor(id);
    if (!u.expanded) return;
    const card = jobList.querySelector(`[data-id="${cssEscape(id)}"]`);
    if (!card) return;
    const t = card.querySelector('.transcript[data-live="1"]');
    if (!t) {
      // Body not in live mode yet; do a full render.
      updateCard(card, jobs.get(id) || {});
      return;
    }
    const caret = t.querySelector(".caret");
    const seg = el("span", { class: "seg", text: (isFirst ? "" : " ") + txt });
    if (caret) t.insertBefore(seg, caret);
    else t.appendChild(seg);
    // Autoscroll if user is near the bottom.
    const nearBottom = t.scrollHeight - t.scrollTop - t.clientHeight < 60;
    if (nearBottom) t.scrollTop = t.scrollHeight;
  }

  // ── Polling ──────────────────────────────────────────────────────────
  async function poll() {
    try {
      const res = await fetch(API, { headers: { Accept: "application/json" } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const list = await res.json();
      if (!Array.isArray(list)) throw new Error("bad payload");
      clearBanner();
      reconcile(list);
    } catch (err) {
      // Don't nuke the UI on a transient failure; just hint once.
      showBanner("Lost contact with the server. Retrying…");
    }
  }

  function reconcile(list) {
    const seen = new Set();
    for (const incoming of list) {
      if (!incoming || !incoming.id) continue;
      seen.add(incoming.id);
      const u = ui.get(incoming.id);
      const existing = jobs.get(incoming.id);
      if (u && u.streaming) {
        // Don't clobber live content. Merge non-live fields so finished/static
        // metadata stays fresh, but keep server status only if it's already
        // terminal (the stream owns the active lifecycle).
        const merged = Object.assign({}, incoming);
        // If polling reports a terminal state the stream missed, honor it.
        if (!ACTIVE.has(incoming.status)) {
          jobs.set(incoming.id, merged);
          stopStream(incoming.id);
          const card = jobList.querySelector(`[data-id="${cssEscape(incoming.id)}"]`);
          if (card) updateCard(card, merged);
        } else {
          // keep existing live-leaning view; still update static fields
          jobs.set(incoming.id, Object.assign({}, existing, merged));
        }
      } else {
        jobs.set(incoming.id, incoming);
      }
    }
    // Drop jobs that disappeared server-side (deleted elsewhere).
    for (const id of Array.from(jobs.keys())) {
      if (!seen.has(id)) {
        const u = ui.get(id);
        if (u && u.streaming) stopStream(id);
        teardownAudio(id);
        jobs.delete(id);
        ui.delete(id);
        jobJson.delete(id);
        jobJsonPending.delete(id);
      }
    }
    render();
  }

  // ── Upload ───────────────────────────────────────────────────────────
  async function uploadFiles(fileList) {
    const files = Array.from(fileList || []).filter(Boolean);
    if (!files.length) return;

    const fd = new FormData();
    for (const f of files) fd.append("files", f, f.name);
    fd.append("language", languageSel.value || "auto");

    // Optimistic placeholder cards so they appear immediately.
    const placeholders = files.map((f) => makePlaceholder(f));
    for (const p of placeholders) jobs.set(p.id, p);
    render();

    try {
      const res = await fetch(API, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const createdJobs = Array.isArray(data) ? data : (data && data.created) || [];
      const dups = (data && data.duplicates) || [];
      // Remove placeholders, insert real jobs.
      for (const p of placeholders) {
        jobs.delete(p.id);
        ui.delete(p.id);
      }
      for (const job of createdJobs) {
        if (job && job.id) jobs.set(job.id, job);
      }
      clearBanner();
      render();
      if (dups.length) {
        const names = dups.map((d) => d.filename).join(", ");
        notify(
          "Duplicates skipped",
          `${dups.length} file${dups.length === 1 ? "" : "s"} already transcribed and skipped: ${names}`
        );
      }
      // Refresh shortly to catch the worker picking jobs up.
      setTimeout(poll, 500);
    } catch (err) {
      for (const p of placeholders) {
        jobs.delete(p.id);
        ui.delete(p.id);
      }
      render();
      showBanner("Upload failed. Check that the server is running and the file type is supported.");
    }
  }

  let phCounter = 0;
  function makePlaceholder(file) {
    phCounter += 1;
    return {
      id: `__pending_${Date.now()}_${phCounter}`,
      original_filename: file.name,
      msg_timestamp: null,
      msg_timestamp_source: null,
      msg_has_time: null,
      language: languageSel.value || "auto",
      detected_language: null,
      duration: null,
      status: "queued",
      error: null,
      progress: 0,
      transcript_text: null,
      model_name: null,
      created_at: new Date().toISOString(),
      started_at: null,
      finished_at: null,
      _pending: true,
    };
  }

  // ── Drag & drop + picker wiring ──────────────────────────────────────
  function wireUploader() {
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });

    fileInput.addEventListener("change", () => {
      uploadFiles(fileInput.files);
      fileInput.value = ""; // allow re-selecting same file
    });

    let dragDepth = 0;
    const onOver = (e) => {
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
    };
    dropzone.addEventListener("dragenter", (e) => {
      e.preventDefault();
      dragDepth += 1;
      dropzone.classList.add("is-dragover");
    });
    dropzone.addEventListener("dragover", onOver);
    dropzone.addEventListener("dragleave", () => {
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) dropzone.classList.remove("is-dragover");
    });
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dragDepth = 0;
      dropzone.classList.remove("is-dragover");
      const dt = e.dataTransfer;
      if (dt && dt.files && dt.files.length) uploadFiles(dt.files);
    });

    // Prevent the browser from navigating away if a file is dropped
    // outside the zone.
    ["dragover", "drop"].forEach((evt) => {
      window.addEventListener(evt, (e) => {
        if (!dropzone.contains(e.target)) e.preventDefault();
      });
    });
  }

  // ── Navigation / views ───────────────────────────────────────────────
  function showView(name) {
    if (!views[name]) name = "transcriptions";
    for (const key in views) {
      if (views[key]) views[key].hidden = key !== name;
    }
    navButtons.forEach((btn) => {
      const active = btn.dataset.view === name;
      btn.classList.toggle("is-active", active);
      if (active) btn.setAttribute("aria-current", "page");
      else btn.removeAttribute("aria-current");
    });
    if (name === "speakers") loadPersons();
  }

  function wireNav() {
    navButtons.forEach((btn) => {
      btn.addEventListener("click", () => showView(btn.dataset.view));
    });
  }

  // ── Speakers page ─────────────────────────────────────────────────────
  async function loadPersons() {
    try {
      const res = await fetch(PERSONS_API, { headers: { Accept: "application/json" } });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const list = await res.json();
      persons = Array.isArray(list) ? list : [];
      clearBanner();
      renderPersons();
    } catch (err) {
      showBanner("Couldn't load speakers. Is the server running?");
    }
  }

  function renderPersons() {
    speakersCount.textContent = persons.length ? `${persons.length}` : "";
    speakersEmpty.hidden = persons.length > 0;
    personList.innerHTML = "";
    for (const p of persons) {
      personList.appendChild(buildPersonRow(p));
    }
  }

  function buildPersonRow(p) {
    const row = el("div", { class: "person", dataset: { id: p.id } });

    const main = el("div", { class: "person__main" });
    const swatch = el("span", { class: "swatch", "aria-hidden": "true" });
    swatch.style.setProperty("--chip-h", String(hueFor(p.id)));
    main.appendChild(swatch);

    const nameInput = el("input", {
      class: "person__name",
      type: "text",
      value: p.name || "",
      "aria-label": "Speaker name",
      maxlength: "120",
    });
    const commit = () => {
      const next = nameInput.value.trim();
      if (!next || next === p.name) {
        nameInput.value = p.name || "";
        return;
      }
      renamePerson(p, next);
    };
    nameInput.addEventListener("blur", commit);
    nameInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); nameInput.blur(); }
      else if (e.key === "Escape") { nameInput.value = p.name || ""; nameInput.blur(); }
    });
    main.appendChild(nameInput);

    const n = safeNumber(p.n_samples) || 0;
    main.appendChild(el("span", {
      class: "person__meta",
      text: `${n} ${n === 1 ? "sample" : "samples"}`,
    }));
    row.appendChild(main);

    // Actions: merge-into select + delete
    const actions = el("div", { class: "person__actions" });

    const others = persons.filter((o) => o.id !== p.id);
    if (others.length) {
      const sel = el("select", {
        class: "merge-select",
        "aria-label": `Merge ${p.name || "this speaker"} into another speaker`,
      });
      sel.appendChild(el("option", { value: "", text: "Merge into…" }));
      for (const o of others) {
        sel.appendChild(el("option", { value: o.id, text: o.name || "(unnamed)" }));
      }
      sel.addEventListener("change", () => {
        const dst = sel.value;
        sel.value = "";
        if (dst) mergePersons(p, dst);
      });
      actions.appendChild(sel);
    }

    const del = el("button", {
      class: "icon-btn",
      title: "Delete speaker",
      "aria-label": `Delete ${p.name || "speaker"}`,
    });
    del.innerHTML =
      '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>';
    del.addEventListener("click", () => deletePerson(p));
    actions.appendChild(del);

    row.appendChild(actions);
    return row;
  }

  // After any persons mutation, drop the cached transcript JSON so speaker
  // chips pick up the new/merged names on the next render, then reload.
  function invalidatePersonsCache() {
    jobJson.clear();
    refreshExpandedTranscripts();
    poll(); // refresh job cards so on-card speaker chips reflect the change
  }

  // Re-run updateCard for currently expanded done cards so they refetch JSON
  // and re-render their (possibly recolored/renamed) speaker chips.
  function refreshExpandedTranscripts() {
    for (const [id, u] of ui.entries()) {
      if (!u.expanded) continue;
      const job = jobs.get(id);
      const card = jobList.querySelector(`[data-id="${cssEscape(id)}"]`);
      if (job && card) updateCard(card, job);
    }
  }

  async function renamePerson(p, name) {
    try {
      const res = await fetch(`${PERSONS_API}/${encodeURIComponent(p.id)}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      clearBanner();
      invalidatePersonsCache();
      await loadPersons();
    } catch (err) {
      showBanner("Couldn't rename the speaker.");
      renderPersons(); // revert input to last-known value
    }
  }

  async function deletePerson(p) {
    const ok = await confirmModal({
      title: "Delete speaker",
      message: `Delete speaker “${p.name || "(unnamed)"}”? This can't be undone.`,
      confirmLabel: "Delete",
    });
    if (!ok) return;
    try {
      const res = await fetch(`${PERSONS_API}/${encodeURIComponent(p.id)}`, { method: "DELETE" });
      if (!res.ok && res.status !== 404) throw new Error(`HTTP ${res.status}`);
      clearBanner();
      invalidatePersonsCache();
      await loadPersons();
    } catch (err) {
      showBanner("Couldn't delete the speaker.");
    }
  }

  async function mergePersons(src, dstId) {
    const dst = persons.find((o) => o.id === dstId);
    const dstName = dst ? (dst.name || "(unnamed)") : "another speaker";
    const ok = await confirmModal({
      title: "Merge speakers",
      message: `Merge “${src.name || "(unnamed)"}” into “${dstName}”? Samples move to ${dstName}.`,
      confirmLabel: "Merge",
      danger: false,
    });
    if (!ok) return;
    try {
      const res = await fetch(`${PERSONS_API}/merge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ src: src.id, dst: dstId }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      clearBanner();
      invalidatePersonsCache();
      await loadPersons();
    } catch (err) {
      showBanner("Couldn't merge the speakers.");
    }
  }

  // ── Boot ─────────────────────────────────────────────────────────────
  function init() {
    wireUploader();
    wireNav();
    if (speakerFilter) {
      speakerFilter.addEventListener("change", () => {
        speakerFilterValue = speakerFilter.value;
        render();
      });
    }
    if (clearAllBtn) clearAllBtn.addEventListener("click", clearAllJobs);
    poll();
    setInterval(poll, POLL_MS);
    // Tidy up streams + audio on unload.
    window.addEventListener("beforeunload", () => {
      for (const id of ui.keys()) { stopStream(id); teardownAudio(id); }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
