"use strict";

/* Shared color source of truth for the four switchable themes.
 *
 *  - `accent` / `accent2`  drive UI chrome: the wordmark gradient, progress
 *    bars, soft fills and the readable accent-ink derived in CSS.
 *  - `stops`               are HSL hues swept left→right across the halftone
 *    hero wave (see wave.js). Kept vivid on purpose — the wave is decorative.
 *
 * Imported by both theme.js (chrome) and wave.js (the canvas), so the two can
 * never drift apart. The anti-FOUC <script> in index.html inlines the same
 * accent pairs to paint the right color before this module loads.
 */
export const ACCENTS = {
  spectrum: { label: "Spectrum", accent: "#a855f7", accent2: "#ec4899", stops: [270, 330, 200, 185] },
  aurora:   { label: "Aurora",   accent: "#14b8a6", accent2: "#6366f1", stops: [168, 190, 220, 250] },
  sunset:   { label: "Sunset",   accent: "#f97316", accent2: "#ec4899", stops: [45, 22, 0, -28] },
  meadow:   { label: "Meadow",   accent: "#84cc16", accent2: "#06b6d4", stops: [95, 135, 170, 196] },
};

export const DEFAULT_ACCENT = "spectrum";

/* Resolve the effective light/dark mode: an explicit data-theme on <html>
 * wins; otherwise follow the system preference. */
export function resolveMode() {
  const attr = document.documentElement.dataset.theme;
  if (attr === "light" || attr === "dark") return attr;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
