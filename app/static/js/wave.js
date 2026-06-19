"use strict";

/* The halftone dot-matrix soundwave that crowns the app.
 *
 * Fully procedural (no image): a continuous 1-D signal — layered value noise,
 * reseeded every page load so the pattern is fresh each visit — is *scrolled*
 * left→right through a center-weighted envelope and dithered into dots whose
 * size/opacity fall off from the centre line. So it reads like a real waveform
 * flowing through the matrix, not a fixed shape wobbling in place.
 *
 * Hue is swept across the width from the active theme; dot lightness adapts to
 * light/dark. The flow runs continuously, a touch livelier while a job works.
 * Honors prefers-reduced-motion (one static — but still randomly-seeded — frame).
 */
import { ACCENTS, DEFAULT_ACCENT, resolveMode } from "./palette.js";

const canvas = document.getElementById("waveCanvas");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const REST = 0.6;     // idle signal strength
const ACTIVE = 1.0;   // signal strength while a job is transcribing
const SPAN_U = 6.5;   // signal units visible across the width (≈ how many features)
const SCROLL = 1.0;   // signal units travelled per second

const seed = Math.random() * 1000;  // fresh waveform shape on every page load

let raf = null;
let t = 0;            // accumulated seconds of travel
let last = 0;
let intensity = REST;
let target = REST;
let cw = 0, ch = 0;
let resizeTimer = null;

// ── Continuous procedural signal (smooth value noise) ────────────────────
function fract(x) { return x - Math.floor(x); }
function hash1(n) { return fract(Math.sin((n + seed * 131.7) * 12.9898) * 43758.5453); }
function vnoise(u) {
  const i = Math.floor(u), f = u - i;
  const a = hash1(i), b = hash1(i + 1);
  const w = f * f * (3 - 2 * f);            // smoothstep
  return a + (b - a) * w;
}
function signal(u) {
  let v = 0.55 * vnoise(u * 0.6) + 0.30 * vnoise(u * 1.7 + 3.1) + 0.15 * vnoise(u * 3.9 + 7.7);
  return Math.pow(Math.max(0, Math.min(1, v)), 1.35);  // sharpen into speech-like bursts
}
// Center-weighted hero envelope: high in the middle, tapers to the edges.
function win(x) { return Math.pow(Math.sin(Math.PI * x), 0.7); }

function lerp(a, b, t) { return a + (b - a) * t; }
function hueAt(stops, x) {
  const seg = x * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(seg));
  const h = lerp(stops[i], stops[i + 1], seg - i);
  return ((h % 360) + 360) % 360;
}

// Column amplitude at horizontal position x for a given scroll offset: a faint
// continuous spine + a centred mass + the scrolling signal riding on top.
function amplitudeAt(x, scroll) {
  const s = signal(x * SPAN_U - scroll);
  const a = 0.055 + win(x) * (0.22 + (0.30 + 0.70 * intensity) * s);
  return Math.max(0.05, Math.min(1, a));
}

function draw(scroll) {
  if (!canvas) return;
  const accent = ACCENTS[document.documentElement.dataset.accent] || ACCENTS[DEFAULT_ACCENT];
  const mode = resolveMode();
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (!W || !H) return;
  if (W !== cw || H !== ch) { canvas.width = W * dpr; canvas.height = H * dpr; cw = W; ch = H; }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const spacing = Math.max(6, H / 26);
  const maxR = spacing * 0.52;
  const cy = H / 2;
  const vSpan = (H / 2) * 0.74;
  const rows = Math.ceil(H / spacing) + 2;
  const cols = Math.ceil(W / spacing) + 2;
  const baseL = mode === "dark" ? 54 : 44;
  const presL = mode === "dark" ? 12 : 8;

  for (let c = 0; c < cols; c++) {
    const cx = c * spacing + spacing / 2;
    const x = cx / W;
    const a = amplitudeAt(x, scroll);
    const hue = hueAt(accent.stops, x);
    for (let r = 0; r < rows; r++) {
      const y = (r - rows / 2) * spacing + cy + spacing / 2;
      const ry = Math.abs((y - cy) / vSpan);
      if (ry > 1.05) continue;
      let pres = (a - ry + 0.12) / 0.30;
      pres = Math.max(0, Math.min(1, pres));
      if (pres <= 0.02) continue;
      const rad = maxR * pres;
      if (rad < 0.4) continue;
      const light = baseL + presL * pres + 12 * pres * intensity;
      const alpha = 0.26 + 0.74 * pres;
      ctx.fillStyle = `hsla(${hue.toFixed(0)}, 82%, ${light.toFixed(1)}%, ${alpha.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(cx, y, rad, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function frame(ts) {
  if (!last) last = ts;
  const dt = (ts - last) / 1000;
  if (dt >= 0.03) {                                   // ~30fps is plenty for a calm flow
    t += dt * (0.85 + 0.5 * intensity);               // a touch faster while working
    intensity += (target - intensity) * Math.min(1, dt * 3);
    last = ts;
    draw(t * SCROLL);
  }
  raf = window.requestAnimationFrame(frame);
}

function start() { if (reduceMotion || raf) return; last = 0; raf = window.requestAnimationFrame(frame); }

// A job is "working" whenever a live status badge is on screen. Polling the DOM
// keeps wave.js fully decoupled from the jobs/SSE code; it only nudges the flow.
function activeJobsPresent() {
  return !!document.querySelector(".badge--converting, .badge--transcribing, .badge--tidying");
}
window.setInterval(() => { target = activeJobsPresent() ? ACTIVE : REST; }, 700);

if (reduceMotion) {
  // Static: one frame (still randomly seeded, so each load differs), recoloured
  // on theme/resize. No motion.
  intensity = REST;
  window.addEventListener("load", () => draw(0));
  document.addEventListener("fluister:themechange", () => draw(0));
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => draw(0), 150); });
  setTimeout(() => draw(0), 30);
} else {
  window.addEventListener("load", start);
  document.addEventListener("fluister:themechange", () => { if (!raf) draw(t * SCROLL); });
  setTimeout(start, 30);
}
