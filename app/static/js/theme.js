"use strict";

/* Theme controls: light/dark mode (auto by default, with manual override) and
 * the four switchable color themes. Both choices persist in localStorage and
 * survive reloads; an inline <script> in index.html applies them before first
 * paint to avoid a flash, so this module's job is to keep state, wire the
 * controls, and broadcast changes (the wave listens for `fluister:themechange`).
 */
import { ACCENTS, DEFAULT_ACCENT } from "./palette.js";

const root = document.documentElement;
const THEME_KEY = "fluister.theme";   // auto | light | dark
const ACCENT_KEY = "fluister.accent"; // spectrum | aurora | sunset | meadow

let pref = localStorage.getItem(THEME_KEY) || "auto";
let accent = localStorage.getItem(ACCENT_KEY) || DEFAULT_ACCENT;
if (!["auto", "light", "dark"].includes(pref)) pref = "auto";
if (!ACCENTS[accent]) accent = DEFAULT_ACCENT;

function dispatch() {
  document.dispatchEvent(new CustomEvent("fluister:themechange"));
}

function applyMode() {
  root.dataset.theme = pref; // CSS reads [data-theme]; "auto" falls through to the media query
  document.querySelectorAll("[data-mode]").forEach((b) =>
    b.classList.toggle("is-active", b.dataset.mode === pref));
}

function applyAccent() {
  const a = ACCENTS[accent];
  root.style.setProperty("--accent", a.accent);
  root.style.setProperty("--accent-2", a.accent2);
  root.dataset.accent = accent;
  document.querySelectorAll("[data-accent-pick]").forEach((b) =>
    b.classList.toggle("is-active", b.dataset.accentPick === accent));
}

function setPref(p) {
  pref = p;
  localStorage.setItem(THEME_KEY, p);
  applyMode();
  dispatch();
}

function setAccent(a) {
  if (!ACCENTS[a]) return;
  accent = a;
  localStorage.setItem(ACCENT_KEY, a);
  applyAccent();
  dispatch();
}

function wire() {
  applyMode();
  applyAccent();

  document.querySelectorAll("[data-mode]").forEach((b) =>
    b.addEventListener("click", () => setPref(b.dataset.mode)));

  // Palette popover
  const palBtn = document.getElementById("paletteBtn");
  const palPop = document.getElementById("palettePop");
  if (palBtn && palPop) {
    const close = () => { palPop.hidden = true; palBtn.setAttribute("aria-expanded", "false"); };
    const open = () => { palPop.hidden = false; palBtn.setAttribute("aria-expanded", "true"); };
    palBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      palPop.hidden ? open() : close();
    });
    palPop.querySelectorAll("[data-accent-pick]").forEach((b) =>
      b.addEventListener("click", () => { setAccent(b.dataset.accentPick); close(); }));
    document.addEventListener("click", (e) => {
      if (!palPop.hidden && !palPop.contains(e.target) && e.target !== palBtn) close();
    });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
  }

  // When following the system, repaint on OS light/dark changes.
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (pref === "auto") dispatch();
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", wire);
} else {
  wire();
}
