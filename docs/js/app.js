"use strict";

/* ── State ─────────────────────────────────────────────── */

let manifest = null;
let pyodide = null;
let collapseCode = null;
let fineBoundaries = null; // Float64Array from manifest
let nFineGroups = 0;
let nFineAngleBins = 0;

/* ── DOM refs ──────────────────────────────────────────── */

const $ = (id) => document.getElementById(id);
const banner = $("loading-banner");
const gridSummary = $("grid-summary");
const tempSelect = $("temperature-select");
const nGroupsInput = $("n-groups");
const nGroupsHint = $("n-groups-hint");
const boundariesDisplay = $("boundaries-display");
const boundariesAuto = $("boundaries-auto");
const boundariesCustom = $("boundaries-custom");
const customCheck = $("use-custom-boundaries");
const customInput = $("custom-boundaries-input");
const customError = $("custom-boundaries-error");
const nAnglesInput = $("n-angles");
const nAnglesHint = $("n-angles-hint");
const angleInfo = $("angle-info");
const summaryShape = $("summary-shape");
const summarySize = $("summary-size");
const btnDownload = $("btn-download");
const progressContainer = $("progress-container");
const progressBar = $("progress-bar");
const progressLabel = $("progress-label");

/* ── Collapsible cards ─────────────────────────────────── */

window.toggleCard = function (cardId) {
  const card = document.getElementById(cardId);
  const title = card.querySelector(".card-title");
  const body = card.querySelector(".card-body");
  const isCollapsed = body.classList.contains("collapsed");
  if (isCollapsed) {
    body.style.maxHeight = body.scrollHeight + "px";
    body.classList.remove("collapsed");
    title.classList.remove("collapsed");
  } else {
    body.style.maxHeight = body.scrollHeight + "px";
    requestAnimationFrame(() => {
      body.classList.add("collapsed");
      title.classList.add("collapsed");
    });
  }
};

/* ── Helpers ───────────────────────────────────────────── */

function formatTemp(t) {
  const keV = t.temperature_K / 1.16045e7;
  return `T${String(t._index).padStart(3, "0")}:  ${t.temperature_K.toExponential(3)} K  (${keV.toPrecision(4)} keV)`;
}

function getValidAngleDivisors(M) {
  const divs = [];
  for (let d = 1; d <= M; d++) {
    if (M % d === 0) divs.push(d);
  }
  return divs;
}

function nearestDivisor(n, M) {
  const divs = getValidAngleDivisors(M);
  let best = divs[0];
  for (const d of divs) {
    if (Math.abs(d - n) < Math.abs(best - n)) best = d;
  }
  return best;
}

function pickLogSpacedSubset(boundaries, nCoarse) {
  const G = boundaries.length - 1;
  if (nCoarse >= G) {
    return Array.from({ length: G + 1 }, (_, i) => i);
  }
  const indices = [0];
  for (let i = 1; i < nCoarse; i++) {
    const frac = i / nCoarse;
    const target = Math.round(frac * G);
    if (target > indices[indices.length - 1] && target < G) {
      indices.push(target);
    }
  }
  indices.push(G);
  // Deduplicate and ensure we have exactly nCoarse groups
  while (indices.length - 1 < nCoarse) {
    // Fill gaps by splitting the largest interval
    let maxGap = 0, maxIdx = 0;
    for (let i = 0; i < indices.length - 1; i++) {
      const gap = indices[i + 1] - indices[i];
      if (gap > maxGap) { maxGap = gap; maxIdx = i; }
    }
    if (maxGap <= 1) break;
    indices.splice(maxIdx + 1, 0, Math.floor((indices[maxIdx] + indices[maxIdx + 1]) / 2));
  }
  while (indices.length - 1 > nCoarse) {
    // Remove the index causing the smallest interval
    let minGap = Infinity, minIdx = 1;
    for (let i = 1; i < indices.length - 1; i++) {
      const merged = indices[i + 1] - indices[i - 1];
      if (merged < minGap) { minGap = merged; minIdx = i; }
    }
    indices.splice(minIdx, 1);
  }
  return indices;
}

function findNearestBoundaryIndex(value, boundaries) {
  let best = 0;
  let bestDist = Math.abs(Math.log(value) - Math.log(boundaries[0]));
  for (let i = 1; i < boundaries.length; i++) {
    const dist = Math.abs(Math.log(value) - Math.log(boundaries[i]));
    if (dist < bestDist) { bestDist = dist; best = i; }
  }
  return best;
}

/* ── UI update functions ───────────────────────────────── */

function getCurrentGroupIndices() {
  if (customCheck.checked) {
    return parseCustomBoundaries();
  }
  const n = Math.max(1, Math.min(nFineGroups, parseInt(nGroupsInput.value) || nFineGroups));
  return pickLogSpacedSubset(fineBoundaries, n);
}

function getCurrentAngleFactor() {
  const desired = parseInt(nAnglesInput.value) || nFineAngleBins;
  const actual = nearestDivisor(desired, nFineAngleBins);
  return nFineAngleBins / actual;
}

function getCurrentAngleBins() {
  const desired = parseInt(nAnglesInput.value) || nFineAngleBins;
  return nearestDivisor(desired, nFineAngleBins);
}

function updateBoundariesDisplay() {
  if (!fineBoundaries) return;
  if (customCheck.checked) return;

  const indices = getCurrentGroupIndices();
  if (!indices) return;
  const vals = indices.map((i) => fineBoundaries[i].toExponential(3));
  boundariesDisplay.textContent = vals.join("  ");
}

function updateSummary() {
  const indices = getCurrentGroupIndices();
  if (!indices) return;

  const nG = indices.length - 1;
  const nA = getCurrentAngleBins();

  summaryShape.innerHTML = `${nG} &times; ${nG} &times; ${nA}`;
  const bytes = nG * nG * nA * 8;
  if (bytes < 1024) {
    summarySize.textContent = bytes + " B";
  } else if (bytes < 1024 * 1024) {
    summarySize.textContent = (bytes / 1024).toFixed(1) + " KB";
  } else {
    summarySize.textContent = (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }
}

function updateAngleHints() {
  const divs = getValidAngleDivisors(nFineAngleBins);
  nAnglesHint.textContent = "Valid choices: " + divs.join(", ");
  const actual = getCurrentAngleBins();
  const factor = nFineAngleBins / actual;
  angleInfo.textContent =
    `Each output bin spans ${factor} fine bin${factor > 1 ? "s" : ""}; ` +
    `bin width: \u0394\u03BE = ${(2 / actual).toFixed(4)}`;
}

function parseCustomBoundaries() {
  const text = customInput.value.trim();
  if (!text) { customError.textContent = ""; return null; }

  const tokens = text.split(/[\s,]+/).filter(Boolean);
  const values = tokens.map(Number);
  if (values.some(isNaN)) {
    customError.textContent = "Non-numeric value found.";
    customError.classList.add("error");
    return null;
  }
  if (values.length < 2) {
    customError.textContent = "Need at least 2 boundary values.";
    customError.classList.add("error");
    return null;
  }
  for (let i = 1; i < values.length; i++) {
    if (values[i] <= values[i - 1]) {
      customError.textContent = "Boundaries must be strictly increasing.";
      customError.classList.add("error");
      return null;
    }
  }

  const tol = 1e-6;
  if (Math.abs(Math.log(values[0]) - Math.log(fineBoundaries[0])) > tol) {
    customError.textContent = `First boundary must equal ${fineBoundaries[0].toExponential(3)}.`;
    customError.classList.add("error");
    return null;
  }
  if (Math.abs(Math.log(values[values.length - 1]) - Math.log(fineBoundaries[fineBoundaries.length - 1])) > tol) {
    customError.textContent = `Last boundary must equal ${fineBoundaries[fineBoundaries.length - 1].toExponential(3)}.`;
    customError.classList.add("error");
    return null;
  }

  const indices = [];
  for (const v of values) {
    const idx = findNearestBoundaryIndex(v, fineBoundaries);
    const ratio = Math.abs(Math.log(v) - Math.log(fineBoundaries[idx]));
    if (ratio > tol) {
      customError.textContent = `Value ${v.toExponential(3)} does not match any fine grid boundary.`;
      customError.classList.add("error");
      return null;
    }
    indices.push(idx);
  }

  for (let i = 1; i < indices.length; i++) {
    if (indices[i] <= indices[i - 1]) {
      customError.textContent = "Resolved indices are not strictly increasing (duplicates?).";
      customError.classList.add("error");
      return null;
    }
  }

  customError.textContent = `\u2713 ${indices.length - 1} groups resolved.`;
  customError.classList.remove("error");
  return indices;
}

/* ── Event wiring ──────────────────────────────────────── */

function wireEvents() {
  nGroupsInput.addEventListener("input", () => {
    const v = parseInt(nGroupsInput.value);
    if (v < 1 || v > nFineGroups) {
      nGroupsHint.textContent = `Must be between 1 and ${nFineGroups}.`;
      nGroupsHint.classList.add("error");
    } else {
      nGroupsHint.textContent = "";
      nGroupsHint.classList.remove("error");
    }
    updateBoundariesDisplay();
    updateSummary();
  });

  customCheck.addEventListener("change", () => {
    if (customCheck.checked) {
      boundariesAuto.style.display = "none";
      boundariesCustom.style.display = "block";
      nGroupsInput.disabled = true;
    } else {
      boundariesAuto.style.display = "";
      boundariesCustom.style.display = "none";
      nGroupsInput.disabled = false;
      customError.textContent = "";
      updateBoundariesDisplay();
    }
    updateSummary();
  });

  customInput.addEventListener("input", () => {
    parseCustomBoundaries();
    updateSummary();
  });

  nAnglesInput.addEventListener("input", () => {
    updateAngleHints();
    updateSummary();
  });

  nAnglesInput.addEventListener("change", () => {
    const desired = parseInt(nAnglesInput.value) || nFineAngleBins;
    const actual = nearestDivisor(desired, nFineAngleBins);
    nAnglesInput.value = actual;
    updateAngleHints();
    updateSummary();
  });

  tempSelect.addEventListener("change", () => {
    btnDownload.disabled = !tempSelect.value || !pyodide;
  });

  btnDownload.addEventListener("click", handleDownload);
}

/* ── Download handler ──────────────────────────────────── */

async function handleDownload() {
  const tempIdx = parseInt(tempSelect.value);
  const tempEntry = manifest.temperatures.find((t) => t.index === tempIdx);
  if (!tempEntry || !pyodide) return;

  const groupIndices = getCurrentGroupIndices();
  if (!groupIndices) return;
  const angleFactor = getCurrentAngleFactor();
  const nCoarseGroups = groupIndices.length - 1;
  const nCoarseAngles = getCurrentAngleBins();

  btnDownload.disabled = true;
  progressContainer.classList.add("visible");
  progressBar.style.width = "10%";
  progressLabel.textContent = `Fetching ${tempEntry.file}\u2026`;

  try {
    const npzUrl = `data/${tempEntry.file}`;
    const resp = await fetch(npzUrl);
    if (!resp.ok) throw new Error(`Failed to fetch ${npzUrl}: ${resp.status}`);
    const npzBuf = await resp.arrayBuffer();

    progressBar.style.width = "50%";
    progressLabel.textContent = "Collapsing matrix\u2026";

    pyodide.FS.writeFile("/tmp/input.npz", new Uint8Array(npzBuf));

    const npyBytes = await pyodide.runPythonAsync(`
collapse(
    open("/tmp/input.npz", "rb").read(),
    [${groupIndices.join(",")}],
    ${angleFactor},
)
    `);

    progressBar.style.width = "90%";
    progressLabel.textContent = "Preparing download\u2026";

    const blob = new Blob([npyBytes.toJs()], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `compton_sigma_T${String(tempIdx).padStart(3, "0")}_${nCoarseGroups}g_${nCoarseAngles}a.npy`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    progressBar.style.width = "100%";
    progressLabel.textContent = "Download complete.";
    setTimeout(() => {
      progressContainer.classList.remove("visible");
      progressBar.style.width = "0%";
    }, 2000);
  } catch (err) {
    progressLabel.textContent = "Error: " + err.message;
    progressBar.style.width = "0%";
    console.error(err);
  } finally {
    btnDownload.disabled = false;
  }
}

/* ── Initialization ────────────────────────────────────── */

async function loadManifest() {
  const resp = await fetch("data/manifest.json");
  manifest = await resp.json();

  fineBoundaries = new Float64Array(manifest.boundaries_keV);
  nFineGroups = manifest.n_groups;
  nFineAngleBins = manifest.n_angle_bins;

  nGroupsInput.max = nFineGroups;
  nGroupsInput.value = nFineGroups;
  nAnglesInput.max = nFineAngleBins;
  nAnglesInput.value = nFineAngleBins;

  const tMin = manifest.temperatures[0];
  const tMax = manifest.temperatures[manifest.temperatures.length - 1];

  gridSummary.innerHTML =
    `${nFineGroups} energy groups (${fineBoundaries[0].toExponential(1)} \u2013 ${fineBoundaries[fineBoundaries.length - 1].toExponential(1)} keV, log-spaced)<br>` +
    `${nFineAngleBins} angle bins (\u03BE = cos\u03B8 \u2208 [\u22121, 1])<br>` +
    `${manifest.temperatures.length} temperatures (${tMin.temperature_K.toExponential(1)} \u2013 ${tMax.temperature_K.toExponential(1)} K, log-spaced)`;

  tempSelect.innerHTML = '<option value="">-- select temperature --</option>';
  for (const t of manifest.temperatures) {
    t._index = t.index;
    const opt = document.createElement("option");
    opt.value = t.index;
    opt.textContent = formatTemp(t);
    tempSelect.appendChild(opt);
  }
  tempSelect.disabled = false;

  updateBoundariesDisplay();
  updateAngleHints();
  updateSummary();
}

async function loadPyodide_() {
  banner.textContent = "Loading Python runtime\u2026";
  const pyodideModule = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.6/full/",
  });
  await pyodideModule.loadPackage("numpy");

  const collapseResp = await fetch("py/collapse.py");
  collapseCode = await collapseResp.text();
  await pyodideModule.runPythonAsync(collapseCode);

  pyodide = pyodideModule;

  banner.classList.add("hidden");
  if (tempSelect.value) btnDownload.disabled = false;
}

async function init() {
  wireEvents();

  const manifestPromise = loadManifest();

  const script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/pyodide/v0.27.6/full/pyodide.js";
  script.addEventListener("load", () => {
    loadPyodide_().catch((err) => {
      banner.textContent = "Failed to load Python runtime: " + err.message;
      console.error(err);
    });
  });
  document.head.appendChild(script);

  await manifestPromise;
}

init();
