"use strict";

/* The halftone dot-matrix soundwave that crowns the app.
 *
 * Procedurally generated on a <canvas>: a central swell, smaller side swells
 * and a continuous thin spine, dithered into dots whose size and opacity fall
 * off from the horizontal centre line. Hue is swept across the width using the
 * active theme's `stops`; dot lightness adapts to light/dark.
 *
 * Motion is a gentle left→right *flow*: a brightness/size crest travels across
 * the wave, so it reads like energy moving through it rather than pulsing up
 * and down. The flow runs continuously at a calm rest level and lifts while a
 * job is working. Honors prefers-reduced-motion (paints once, static).
 */
import { ACCENTS, DEFAULT_ACCENT, resolveMode } from "./palette.js";

const canvas = document.getElementById("waveCanvas");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const REST = 0.65;   // idle flow depth (visible at rest)
const ACTIVE = 1.0;  // flow depth while a job is transcribing

let raf = null;
let t = 0;                 // travel phase (advances over time → crest moves right)
let last = 0;
let intensity = REST;      // eased current flow depth
let target = REST;         // flow depth we're easing toward
let cw = 0, ch = 0;        // cached backing-store size
let resizeTimer = null;

// ── Waveform shape (deterministic) ───────────────────────────────────────
function gauss(x, mu, s) { return Math.exp(-((x - mu) * (x - mu)) / (2 * s * s)); }
function hash(i) { const s = Math.sin(i * 127.1 + 0.7) * 43758.5453; return s - Math.floor(s); }
function amp(x) {
  let v = 0.09;
  v += 1.00 * gauss(x, 0.50, 0.085);
  v += 0.52 * gauss(x, 0.34, 0.05);
  v += 0.46 * gauss(x, 0.66, 0.055);
  v += 0.26 * gauss(x, 0.18, 0.05);
  v += 0.24 * gauss(x, 0.83, 0.05);
  v *= 0.78 + 0.42 * hash(Math.floor(x * 230));
  return Math.max(0.05, Math.min(1, v));
}
function lerp(a, b, t) { return a + (b - a) * t; }
function hueAt(stops, x) {
  const seg = x * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(seg));
  const h = lerp(stops[i], stops[i + 1], seg - i);
  return ((h % 360) + 360) % 360;
}

function draw() {
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
    const a = amp(x);
    const hue = hueAt(accent.stops, x);
    // Traveling crest: 0..1, peaks sweep from left to right as t grows.
    const flow = 0.5 + 0.5 * Math.sin(x * Math.PI * 2.4 - t);
    // Travelling height ripple: each column's reach grows/shrinks as the crest
    // passes, so the silhouette visibly flows left→right (not just a shimmer).
    const aMod = a * (1 - intensity * 0.45 * (1 - flow));
    for (let r = 0; r < rows; r++) {
      const y = (r - rows / 2) * spacing + cy + spacing / 2;
      const ry = Math.abs((y - cy) / vSpan);
      if (ry > 1.05) continue;
      let pres = (aMod - ry + 0.12) / 0.30;
      pres = Math.max(0, Math.min(1, pres));
      if (pres <= 0.02) continue;
      const rad = maxR * pres * (0.86 + 0.16 * flow);
      if (rad < 0.4) continue;
      const light = baseL + presL * pres + 18 * intensity * flow;
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
  if (ts - last >= 32) {              // ~30fps is plenty for a calm flow
    t += 0.05 * (0.7 + 0.6 * intensity);
    intensity += (target - intensity) * 0.06;
    last = ts;
    draw();
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
  // Static: a single pleasant frame, recoloured on theme/resize.
  t = Math.PI / 2; intensity = REST;
  window.addEventListener("load", draw);
  document.addEventListener("fluister:themechange", draw);
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(draw, 150); });
  setTimeout(draw, 30);
} else {
  window.addEventListener("load", start);
  document.addEventListener("fluister:themechange", () => { if (!raf) draw(); });
  setTimeout(start, 30);
}
