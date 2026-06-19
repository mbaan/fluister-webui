"use strict";

/* The halftone dot-matrix soundwave that crowns the app.
 *
 * Fully procedural (no image): a continuous 1-D signal — gated, spiky value
 * noise, reseeded every page load so the pattern is fresh each visit — dithered
 * into dots whose size/opacity fall off from the centre line, under a
 * center-weighted envelope so it reads as a hero "portrait".
 *
 * It sits STILL by default (a pretty, fully-formed waveform). It only scrolls —
 * right→left, like a player — while a job is actually working (converting /
 * transcribing / polishing), then freezes again when idle. Hue follows the
 * active theme; lightness adapts to light/dark. Honors prefers-reduced-motion
 * (never scrolls; one static, still randomly-seeded frame).
 */
import { ACCENTS, DEFAULT_ACCENT, resolveMode } from "./palette.js";

const canvas = document.getElementById("waveCanvas");
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const SPAN_U = 13;    // signal units visible across the width — higher = denser
const SCROLL = 0.75;  // signal units travelled per second while working (calm pace)

const seed = Math.random() * 1000;  // fresh waveform shape on every page load

let raf = null;
let scroll = 0;       // current travel offset (frozen unless a job is working)
let last = 0;
let active = false;   // is a job working right now?
let glow = 0;         // eased 0→1 brightness lift while working
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
  // Slow "activity" gate → carves quiet gaps between bursts.
  let env = vnoise(u * 0.5 + 1.7);
  env = env < 0.40 ? 0 : (env - 0.40) / 0.60;
  env = env * env * (3 - 2 * env);                     // smooth the gate edges
  // Spiky multi-octave detail → tall peaks with small ripples between.
  let d = 0.5 * vnoise(u * 2.6 + 3.1)
        + 0.3 * vnoise(u * 5.7 + 7.7)
        + 0.2 * vnoise(u * 11.3 + 9.1);
  d = Math.pow(d, 2.4);
  return Math.min(1, env * (0.12 + 0.88 * d) * 2.2);   // gain so peaks reach full height
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

// Column amplitude: a faint continuous spine + a centred mass + the signal.
// Full height regardless of state, so the idle (frozen) wave still looks rich.
function amplitudeAt(x, sc) {
  const s = signal(x * SPAN_U + sc);        // +sc → features travel right→left
  const a = 0.05 + win(x) * (0.12 + 0.78 * s);
  return Math.max(0.05, Math.min(1, a));
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
      const light = baseL + presL * pres + 12 * pres * glow;
      const alpha = 0.26 + 0.74 * pres;
      ctx.fillStyle = `hsla(${hue.toFixed(0)}, 82%, ${light.toFixed(1)}%, ${alpha.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(cx, y, rad, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// ── Motion: only runs while a job is working ─────────────────────────────
function frame(ts) {
  if (!last) last = ts;
  const dt = (ts - last) / 1000;
  if (dt >= 0.03) {                          // ~30fps is plenty for a calm flow
    scroll += dt * SCROLL;
    glow += (1 - glow) * Math.min(1, dt * 2);
    last = ts;
    draw();
  }
  raf = window.requestAnimationFrame(frame);
}
function startMotion() { if (reduceMotion || raf) return; last = 0; raf = window.requestAnimationFrame(frame); }
function stopMotion() {
  if (raf) { window.cancelAnimationFrame(raf); raf = null; }
  glow = 0;
  draw();                                    // freeze on the current frame, static
}

function activeJobsPresent() {
  return !!document.querySelector(".badge--converting, .badge--transcribing, .badge--tidying");
}
window.setInterval(() => {
  const now = activeJobsPresent();
  if (now && !active) { active = true; startMotion(); }
  else if (!now && active) { active = false; stopMotion(); }
}, 700);

// Static paints (the default state, plus theme/resize while idle). The motion
// loop owns redraws while it's running.
function paintIfStill() { if (!raf) draw(); }
window.addEventListener("load", paintIfStill);
document.addEventListener("fluister:themechange", paintIfStill);
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(paintIfStill, 150); });
setTimeout(paintIfStill, 30);
