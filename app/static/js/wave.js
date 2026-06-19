"use strict";

/* The halftone dot-matrix soundwave that crowns the app.
 *
 * Procedurally generated on a <canvas>: a central swell, smaller side swells
 * and a continuous thin spine, dithered into dots whose size and opacity fall
 * off from the horizontal centre line. Hue is swept across the width using the
 * active theme's `stops`; dot lightness adapts to light/dark so it reads on
 * both backgrounds.
 *
 * While any job is actively working the wave "breathes" (a slow vertical
 * pulse); idle, it paints once and rests. Honors prefers-reduced-motion.
 */
import { ACCENTS, DEFAULT_ACCENT, resolveMode } from "./palette.js";

const canvas = document.getElementById("waveCanvas");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let raf = null;
let phase = 0;
let animating = false;

// ── Waveform shape (deterministic, so every paint is identical) ──────────
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

// `breathe` (~1.0) scales the wave's vertical reach for the idle/active pulse.
function draw(breathe) {
  if (!canvas) return;
  const accent = ACCENTS[document.documentElement.dataset.accent] || ACCENTS[DEFAULT_ACCENT];
  const mode = resolveMode();
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  if (!W || !H) return;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const spacing = Math.max(6, H / 26);
  const maxR = spacing * 0.52;
  const cy = H / 2;
  const vSpan = (H / 2) * 0.74 * (breathe || 1);
  const rows = Math.ceil(H / spacing) + 2;
  const cols = Math.ceil(W / spacing) + 2;

  for (let c = 0; c < cols; c++) {
    const cx = c * spacing + spacing / 2;
    const x = cx / W;
    const a = amp(x);
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
      const light = mode === "dark" ? 56 + 12 * pres : 44 + 8 * pres;
      const alpha = 0.30 + 0.70 * pres;
      ctx.fillStyle = `hsla(${hue.toFixed(0)}, 82%, ${light}%, ${alpha.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(cx, y, rad, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function render() { draw(1); }

function tick() {
  phase += 0.018;
  draw(1 + 0.05 * Math.sin(phase * 1.6));
  raf = window.requestAnimationFrame(tick);
}

function setAnimating(on) {
  if (reduceMotion) on = false;
  if (on === animating) return;
  animating = on;
  if (on) {
    if (!raf) raf = window.requestAnimationFrame(tick);
  } else {
    if (raf) { window.cancelAnimationFrame(raf); raf = null; }
    render();
  }
}

// A job is "working" whenever a live status badge is on screen. Polling the DOM
// keeps wave.js fully decoupled from the jobs/SSE code.
function activeJobsPresent() {
  return !!document.querySelector(".badge--converting, .badge--transcribing, .badge--tidying");
}

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(render, 150);
});
document.addEventListener("fluister:themechange", () => { if (!animating) render(); });

window.setInterval(() => setAnimating(activeJobsPresent()), 700);
window.addEventListener("load", render);
// Fonts/layout may already be ready; paint once immediately too.
setTimeout(render, 30);
