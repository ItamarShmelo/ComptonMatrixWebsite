import { renderLinePlot, renderHeatmap, plotDataToCSV } from './plot.js';

/* ── State ─────────────────────────────────────────────── */

let manifest = null;
let pyodide = null;
let collapseCode = null;
let fineBoundaries = null;
let nFineGroups = 0;
let nFineAngleBins = 0;
let jszipLoaded = false;
let lastPlotCSV = null;

/* ── DOM refs ──────────────────────────────────────────── */

const $ = (id) => document.getElementById(id);
const loadingMsg = $("loading-message");
const downloadForm = $("download-form");
const gridSummary = $("grid-summary");
const tempSelect = $("temperature-select");
const customTempInput = $("custom-temperature-input");
const customTempError = $("custom-temperature-error");
const tempHint = $("temperature-hint");
const nGroupsInput = $("n-groups");
const nGroupsHint = $("n-groups-hint");
const boundariesDisplay = $("boundaries-display");
const boundariesAuto = $("boundaries-auto");
const customBoundariesInput = $("custom-boundaries-input");
const customBoundariesError = $("custom-boundaries-error");
const nAnglesInput = $("n-angles");
const nAnglesHint = $("n-angles-hint");
const angleInfo = $("angle-info");
const customAngleInput = $("custom-angle-input");
const customAngleError = $("custom-angle-error");
const summaryShape = $("summary-shape");
const summaryAxes = $("summary-axes");
const summarySize = $("summary-size");
const btnDownload = $("btn-download");
const btnDownloadAll = $("btn-download-all");
const btnDownloadMetadata = $("btn-download-metadata");
const progressContainer = $("progress-container");
const progressBar = $("progress-bar");
const progressLabel = $("progress-label");

const plotForm = $("plot-form");
const plotPlaceholder = $("plot-placeholder");
const plotModeSelect = $("plot-mode-select");
const plotTempSelect = $("plot-temperature-select");
const plotButton = $("plot-button");
const plotDownloadButton = $("plot-download-button");
const plotSection = $("plot-section");
const plotChart = $("plot-chart");
const plotDescription = $("plot-description");
const plotSelectionInfo = $("plot-selection-info");
const plotFormError = $("plot-form-error");

/* ── Mode helpers ──────────────────────────────────────── */

function getRadioValue(name) {
  const el = document.querySelector(`input[name="${name}"]:checked`);
  return el ? el.value : null;
}

function isTempCustom() { return getRadioValue("temp-mode") === "custom"; }
function isEnergyCustom() { return getRadioValue("energy-mode") === "custom"; }
function isAngleCustom() { return getRadioValue("angle-mode") === "custom"; }

/* ── Helpers ───────────────────────────────────────────── */

function formatTemp(t) {
  const keV = t.temperature_K / 1.16045e7;
  return `T${String(t._index).padStart(3, "0")}:  ${t.temperature_K.toExponential(3)} K  (${keV.toPrecision(4)} keV)`;
}

function pickLogSpacedBoundaries(fineBounds, nCoarse) {
  const G = fineBounds.length - 1;
  if (nCoarse >= G) return Array.from(fineBounds);
  return Array.from({ length: nCoarse + 1 }, (_, i) => {
    const frac = i / nCoarse;
    const logMin = Math.log(fineBounds[0]);
    const logMax = Math.log(fineBounds[G]);
    return Math.exp(logMin + frac * (logMax - logMin));
  });
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

/* ── Temperature helpers ──────────────────────────────── */

function getTemperatureRange() {
  if (!manifest) return { min: 1000, max: 1e9 };
  const temps = manifest.temperatures;
  return { min: temps[0].temperature_K, max: temps[temps.length - 1].temperature_K };
}

function findBracketingTemps(T_target) {
  const temps = manifest.temperatures;
  if (T_target <= temps[0].temperature_K) return { lo: 0, hi: 0, exact: true };
  if (T_target >= temps[temps.length - 1].temperature_K)
    return { lo: temps.length - 1, hi: temps.length - 1, exact: true };
  for (let i = 0; i < temps.length - 1; i++) {
    const tLo = temps[i].temperature_K;
    const tHi = temps[i + 1].temperature_K;
    if (Math.abs(T_target - tLo) / tLo < 1e-9) return { lo: i, hi: i, exact: true };
    if (Math.abs(T_target - tHi) / tHi < 1e-9) return { lo: i + 1, hi: i + 1, exact: true };
    if (T_target > tLo && T_target < tHi) return { lo: i, hi: i + 1, exact: false };
  }
  return { lo: temps.length - 1, hi: temps.length - 1, exact: true };
}

function getSelectedTemperatures() {
  if (isTempCustom()) return parseCustomTemperatures();
  const selected = Array.from(tempSelect.selectedOptions)
    .map((opt) => parseInt(opt.value))
    .filter((v) => !isNaN(v));
  if (selected.length === 0) return null;
  const temps = selected.map((idx) => {
    const entry = manifest.temperatures.find((t) => t.index === idx);
    return entry ? entry.temperature_K : null;
  }).filter((v) => v !== null);
  return deduplicateAndSort(temps);
}

function deduplicateAndSort(temps) {
  const sorted = [...temps].sort((a, b) => a - b);
  const deduped = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    if (Math.abs(sorted[i] - deduped[deduped.length - 1]) / sorted[i] > 1e-9)
      deduped.push(sorted[i]);
  }
  return deduped;
}

function parseCustomTemperatures() {
  const text = customTempInput.value.trim();
  if (!text) { customTempError.textContent = ""; return null; }
  const tokens = text.split(/[\s,;]+/).filter(Boolean);
  const values = tokens.map(Number);
  if (values.some(isNaN)) {
    customTempError.textContent = "Non-numeric value found.";
    customTempError.classList.add("error");
    return null;
  }
  if (values.length < 1) {
    customTempError.textContent = "Enter at least one temperature.";
    customTempError.classList.add("error");
    return null;
  }
  if (values.some((v) => v <= 0)) {
    customTempError.textContent = "Temperatures must be positive.";
    customTempError.classList.add("error");
    return null;
  }
  const range = getTemperatureRange();
  if (values.some((v) => v < range.min * (1 - 1e-6) || v > range.max * (1 + 1e-6))) {
    customTempError.textContent =
      `Temperatures must lie within [${range.min.toExponential(3)}, ${range.max.toExponential(3)}] K.`;
    customTempError.classList.add("error");
    return null;
  }
  const sorted = deduplicateAndSort(values);
  customTempError.textContent = `\u2713 ${sorted.length} temperature(s) specified.`;
  customTempError.classList.remove("error");
  return sorted;
}

/* ── UI update functions ───────────────────────────────── */

function getCurrentEnergyBoundaries() {
  if (isEnergyCustom()) return parseCustomBoundaries();
  const n = Math.max(1, Math.min(nFineGroups, parseInt(nGroupsInput.value) || nFineGroups));
  return pickLogSpacedBoundaries(fineBoundaries, n);
}

function getCurrentAngleBins() {
  if (isAngleCustom()) {
    const bounds = parseCustomAngleBoundaries();
    if (bounds) return { boundaries: bounds };
    return null;
  }
  const v = parseInt(nAnglesInput.value);
  if (isNaN(v) || v < 1) return { count: 1 };
  return { count: v };
}

function getAngleBinCount() {
  const a = getCurrentAngleBins();
  if (!a) return 0;
  return a.count || (a.boundaries.length - 1);
}

function getAngleBoundariesArray() {
  const a = getCurrentAngleBins();
  if (!a) return null;
  if (a.boundaries) return a.boundaries;
  const nA = a.count;
  const bounds = [];
  for (let i = 0; i <= nA; i++) bounds.push(-1 + (2 * i) / nA);
  return bounds;
}

function updateBoundariesDisplay() {
  if (!fineBoundaries || isEnergyCustom()) return;
  const bounds = getCurrentEnergyBoundaries();
  if (!bounds) return;
  boundariesDisplay.textContent = bounds.map((v) => v.toExponential(3)).join("  ");
}

function updateSummary() {
  const bounds = getCurrentEnergyBoundaries();
  if (!bounds) return;
  const angleCfg = getCurrentAngleBins();
  if (!angleCfg) return;
  const nG = bounds.length - 1;
  const nA = getAngleBinCount();
  const temps = getSelectedTemperatures();
  const nT = temps ? temps.length : 0;

  if (nT > 1) {
    summaryShape.innerHTML = `${nT} &times; ${nG} &times; ${nG} &times; ${nA}`;
    summaryAxes.textContent = "[temperature] \u00D7 [incoming group] \u00D7 [outgoing group] \u00D7 [angle bin]";
    summarySize.textContent = formatSize(nT * nG * nG * nA * 8);
  } else if (nG === 1 && nA === 1) {
    summaryShape.textContent = "scalar";
    summaryAxes.textContent = "\u2014";
    summarySize.textContent = "8 B";
  } else if (nG === 1) {
    summaryShape.textContent = `${nA}`;
    summaryAxes.textContent = "[angle bin]";
    summarySize.textContent = formatSize(nA * 8);
  } else if (nA === 1) {
    summaryShape.innerHTML = `${nG} &times; ${nG}`;
    summaryAxes.textContent = "[incoming group] \u00D7 [outgoing group]";
    summarySize.textContent = formatSize(nG * nG * 8);
  } else {
    summaryShape.innerHTML = `${nG} &times; ${nG} &times; ${nA}`;
    summaryAxes.textContent = "[incoming group] \u00D7 [outgoing group] \u00D7 [angle bin]";
    summarySize.textContent = formatSize(nG * nG * nA * 8);
  }
}

function updateAngleHints() {
  if (isAngleCustom()) return;
  const aInfo = getCurrentAngleBins();
  if (!aInfo) return;
  const nA = aInfo.count || 1;
  nAnglesHint.textContent = `Any integer from 1 to ${nFineAngleBins}. Values above ${nFineAngleBins} exceed stored resolution.`;
  angleInfo.textContent = `\u0394\u03BE = ${(2 / nA).toFixed(6)}`;
}

function parseCustomBoundaries() {
  const text = customBoundariesInput.value.trim();
  if (!text) { customBoundariesError.textContent = ""; return null; }
  const tokens = text.split(/[\s,;]+/).filter(Boolean);
  const values = tokens.map(Number);
  if (values.some(isNaN)) {
    customBoundariesError.textContent = "Non-numeric value found.";
    customBoundariesError.classList.add("error");
    return null;
  }
  if (values.length < 2) {
    customBoundariesError.textContent = "Need at least 2 boundary values.";
    customBoundariesError.classList.add("error");
    return null;
  }
  for (let i = 1; i < values.length; i++) {
    if (values[i] <= values[i - 1]) {
      customBoundariesError.textContent = "Boundaries must be strictly increasing.";
      customBoundariesError.classList.add("error");
      return null;
    }
  }
  if (values[0] <= 0) {
    customBoundariesError.textContent = "Boundaries must be positive.";
    customBoundariesError.classList.add("error");
    return null;
  }
  const eMin = fineBoundaries[0];
  const eMax = fineBoundaries[fineBoundaries.length - 1];
  if (values[0] < eMin * (1 - 1e-6) || values[values.length - 1] > eMax * (1 + 1e-6)) {
    customBoundariesError.textContent = `Boundaries must lie within [${eMin.toExponential(3)}, ${eMax.toExponential(3)}] keV.`;
    customBoundariesError.classList.add("error");
    return null;
  }
  customBoundariesError.textContent = `\u2713 ${values.length - 1} groups defined.`;
  customBoundariesError.classList.remove("error");
  return values;
}

function parseCustomAngleBoundaries() {
  const text = customAngleInput.value.trim();
  if (!text) { customAngleError.textContent = ""; return null; }
  const tokens = text.split(/[\s,;]+/).filter(Boolean);
  const values = tokens.map(Number);
  if (values.some(isNaN)) {
    customAngleError.textContent = "Non-numeric value found.";
    customAngleError.classList.add("error");
    return null;
  }
  if (values.length < 2) {
    customAngleError.textContent = "Need at least 2 boundary values.";
    customAngleError.classList.add("error");
    return null;
  }
  for (let i = 1; i < values.length; i++) {
    if (values[i] <= values[i - 1]) {
      customAngleError.textContent = "Boundaries must be strictly increasing.";
      customAngleError.classList.add("error");
      return null;
    }
  }
  if (values[0] < -1 - 1e-9 || values[values.length - 1] > 1 + 1e-9) {
    customAngleError.textContent = "Boundaries must lie within [\u22121, 1].";
    customAngleError.classList.add("error");
    return null;
  }
  customAngleError.textContent = `\u2713 ${values.length - 1} angle bins defined.`;
  customAngleError.classList.remove("error");
  return values;
}

/* ── Mode panel switching ──────────────────────────────── */

function setupModeRadios(name, panels) {
  document.querySelectorAll(`input[name="${name}"]`).forEach((radio) => {
    radio.addEventListener("change", () => {
      const val = radio.value;
      for (const [key, panelId] of Object.entries(panels)) {
        const panel = $(panelId);
        if (panel) panel.hidden = key !== val;
      }
      updateBoundariesDisplay();
      updateSummary();
      updateButtonStates();
    });
  });
}

/* ── Event wiring ──────────────────────────────────────── */

function wireDownloadEvents() {
  setupModeRadios("temp-mode", { stored: "temp-stored-panel", custom: "temp-custom-panel" });
  setupModeRadios("energy-mode", { standard: "energy-standard-panel", custom: "energy-custom-panel" });
  setupModeRadios("angle-mode", { standard: "angle-standard-panel", custom: "angle-custom-panel" });

  nGroupsInput.addEventListener("input", () => {
    const v = parseInt(nGroupsInput.value);
    if (v < 1 || v > nFineGroups) {
      nGroupsHint.textContent = `Must be between 1 and ${nFineGroups}.`;
      nGroupsHint.classList.add("error");
    } else {
      nGroupsHint.textContent = "\u00a0";
      nGroupsHint.classList.remove("error");
    }
    updateBoundariesDisplay();
    updateSummary();
  });

  customBoundariesInput.addEventListener("input", () => {
    parseCustomBoundaries();
    updateSummary();
  });

  nAnglesInput.addEventListener("input", () => {
    updateAngleHints();
    updateSummary();
  });

  customAngleInput.addEventListener("input", () => {
    parseCustomAngleBoundaries();
    updateSummary();
  });

  customTempInput.addEventListener("input", () => {
    parseCustomTemperatures();
    updateSummary();
    updateButtonStates();
  });

  tempSelect.addEventListener("change", () => {
    updateSummary();
    updateButtonStates();
  });

  btnDownload.addEventListener("click", handleDownload);
  btnDownloadAll.addEventListener("click", handleDownloadAll);
  btnDownloadMetadata.addEventListener("click", handleDownloadMetadata);
}

function updateButtonStates() {
  const hasTemp = isTempCustom()
    ? parseCustomTemperatures() !== null
    : tempSelect.selectedOptions.length > 0;
  btnDownload.disabled = !hasTemp || !pyodide;
  btnDownloadAll.disabled = !jszipLoaded || !pyodide || !manifest;
  btnDownloadMetadata.disabled = !hasTemp || !pyodide;
  plotButton.disabled = !pyodide || !manifest;
}

/* ── Progress helpers ──────────────────────────────────── */

function showProgress(pct, label) {
  progressContainer.hidden = false;
  progressBar.style.width = pct + "%";
  progressLabel.textContent = label;
}

function hideProgress() {
  setTimeout(() => {
    progressContainer.hidden = true;
    progressBar.style.width = "0%";
  }, 2000);
}

/* ── Collapse call builders ────────────────────────────── */

function buildCollapseToArrayCall(energyBounds, angleInfo) {
  const anglePyArg = angleInfo.boundaries
    ? `angle_boundaries=[${angleInfo.boundaries.join(",")}]`
    : `n_angle_bins=${angleInfo.count}`;
  return `_collapse_to_array(\n    open("/tmp/input.npz", "rb").read(),\n    [${energyBounds.join(",")}],\n    ${anglePyArg},\n)`;
}

function buildCollapseInterpCall(energyBounds, angleInfo, T_lo, T_hi, T_target) {
  const anglePyArg = angleInfo.boundaries
    ? `angle_boundaries=[${angleInfo.boundaries.join(",")}]`
    : `n_angle_bins=${angleInfo.count}`;
  return `collapse_interp(\n    open("/tmp/input_lo.npz", "rb").read(),\n    open("/tmp/input_hi.npz", "rb").read(),\n    ${T_lo}, ${T_hi}, ${T_target},\n    [${energyBounds.join(",")}],\n    ${anglePyArg},\n)`;
}

function isIdentityGrid(energyBounds, angleInfo) {
  if (angleInfo.boundaries) return false;
  if (angleInfo.count !== nFineAngleBins) return false;
  if (energyBounds.length !== fineBoundaries.length) return false;
  for (let i = 0; i < energyBounds.length; i++) {
    if (Math.abs(energyBounds[i] - fineBoundaries[i]) > fineBoundaries[i] * 1e-9) return false;
  }
  return true;
}

/* ── Single/multi-temperature download ─────────────────── */

async function handleDownload() {
  const temps = getSelectedTemperatures();
  if (!temps || temps.length === 0 || !pyodide) return;

  const energyBounds = getCurrentEnergyBoundaries();
  if (!energyBounds) return;
  const angleInfo = getCurrentAngleBins();
  if (!angleInfo) return;
  const nCoarseGroups = energyBounds.length - 1;
  const nA = getAngleBinCount();
  const nT = temps.length;

  btnDownload.disabled = true;
  showProgress(5, "Processing\u2026");

  try {
    const collapseCache = new Map();

    async function getCollapsedForStoredIndex(idx) {
      if (collapseCache.has(idx)) return collapseCache.get(idx);
      const entry = manifest.temperatures[idx];
      const resp = await fetch(`data/uniform/${entry.file}`);
      if (!resp.ok) throw new Error(`Failed to fetch ${entry.file}: ${resp.status}`);
      const npzBuf = await resp.arrayBuffer();
      pyodide.FS.writeFile("/tmp/input.npz", new Uint8Array(npzBuf));
      const arr = await pyodide.runPythonAsync(buildCollapseToArrayCall(energyBounds, angleInfo));
      const result = arr.toJs({ create_proxies: false });
      collapseCache.set(idx, result);
      return result;
    }

    const arrays = [];
    for (let i = 0; i < nT; i++) {
      showProgress(5 + Math.round((i / nT) * 80), `Processing temperature ${i + 1}/${nT}\u2026`);
      const T = temps[i];
      const bracket = findBracketingTemps(T);

      if (bracket.exact) {
        arrays.push(await getCollapsedForStoredIndex(bracket.lo));
      } else {
        const entryLo = manifest.temperatures[bracket.lo];
        const entryHi = manifest.temperatures[bracket.hi];
        let loData = collapseCache.get(bracket.lo);
        let hiData = collapseCache.get(bracket.hi);

        if (loData && hiData) {
          const alpha = (Math.log(T) - Math.log(entryLo.temperature_K)) /
            (Math.log(entryHi.temperature_K) - Math.log(entryLo.temperature_K));
          const interpResult = await pyodide.runPythonAsync(`
import numpy as np
_lo = np.array(${JSON.stringify(Array.from(loData))}).reshape(${nCoarseGroups}, ${nCoarseGroups}, ${nA})
_hi = np.array(${JSON.stringify(Array.from(hiData))}).reshape(${nCoarseGroups}, ${nCoarseGroups}, ${nA})
(1.0 - ${alpha}) * _lo + ${alpha} * _hi
`);
          arrays.push(interpResult.toJs({ create_proxies: false }));
        } else {
          const [respLo, respHi] = await Promise.all([
            fetch(`data/uniform/${entryLo.file}`),
            fetch(`data/uniform/${entryHi.file}`),
          ]);
          if (!respLo.ok) throw new Error(`Failed to fetch ${entryLo.file}`);
          if (!respHi.ok) throw new Error(`Failed to fetch ${entryHi.file}`);
          const [bufLo, bufHi] = await Promise.all([respLo.arrayBuffer(), respHi.arrayBuffer()]);
          pyodide.FS.writeFile("/tmp/input_lo.npz", new Uint8Array(bufLo));
          pyodide.FS.writeFile("/tmp/input_hi.npz", new Uint8Array(bufHi));
          const interpArr = await pyodide.runPythonAsync(
            buildCollapseInterpCall(energyBounds, angleInfo, entryLo.temperature_K, entryHi.temperature_K, T)
          );
          arrays.push(interpArr.toJs({ create_proxies: false }));

          if (!collapseCache.has(bracket.lo)) {
            pyodide.FS.writeFile("/tmp/input.npz", new Uint8Array(bufLo));
            const loArr = await pyodide.runPythonAsync(buildCollapseToArrayCall(energyBounds, angleInfo));
            collapseCache.set(bracket.lo, loArr.toJs({ create_proxies: false }));
          }
          if (!collapseCache.has(bracket.hi)) {
            pyodide.FS.writeFile("/tmp/input.npz", new Uint8Array(bufHi));
            const hiArr = await pyodide.runPythonAsync(buildCollapseToArrayCall(energyBounds, angleInfo));
            collapseCache.set(bracket.hi, hiArr.toJs({ create_proxies: false }));
          }
        }
      }
    }

    showProgress(88, "Building output array\u2026");

    let npyBytes;
    if (nT === 1) {
      const flatData = JSON.stringify(Array.from(arrays[0]));
      npyBytes = await pyodide.runPythonAsync(`
import numpy as np, io
_arr = np.array(${flatData}).reshape(${nCoarseGroups}, ${nCoarseGroups}, ${nA})
N, _, K = _arr.shape
if N == 1 and K == 1:
    _arr = _arr.ravel()[0]
elif N == 1:
    _arr = _arr[0, 0, :]
elif K == 1:
    _arr = _arr[:, :, 0]
_buf = io.BytesIO()
np.save(_buf, _arr)
_buf.getvalue()
`);
    } else {
      const allFlat = arrays.map((a) => Array.from(a));
      npyBytes = await pyodide.runPythonAsync(`
import numpy as np, io
_data = ${JSON.stringify(allFlat)}
_stacked = np.array(_data, dtype=np.float64).reshape(${nT}, ${nCoarseGroups}, ${nCoarseGroups}, ${nA})
_buf = io.BytesIO()
np.save(_buf, _stacked)
_buf.getvalue()
`);
    }

    showProgress(95, "Preparing download\u2026");
    const blob = new Blob([npyBytes.toJs()], { type: "application/octet-stream" });
    triggerDownload(blob, buildFilename(temps, nCoarseGroups, nA));
    showProgress(100, "Download complete.");
    hideProgress();
  } catch (err) {
    progressLabel.textContent = "Error: " + err.message;
    progressBar.style.width = "0%";
    console.error(err);
  } finally {
    updateButtonStates();
  }
}

function buildFilename(temps, nG, nA) {
  const nT = temps.length;
  if (nT === 1) {
    const bracket = findBracketingTemps(temps[0]);
    if (bracket.exact) return `compton_sigma_T${String(bracket.lo).padStart(3, "0")}_${nG}g_${nA}a.npy`;
    return `compton_sigma_T${temps[0].toExponential(3)}K_${nG}g_${nA}a.npy`;
  }
  return `compton_sigma_${nT}T_${nG}g_${nA}a.npy`;
}

/* ── Bulk download ─────────────────────────────────────── */

async function handleDownloadAll() {
  if (!manifest || !jszipLoaded || !pyodide) return;

  const energyBounds = getCurrentEnergyBoundaries();
  if (!energyBounds) return;
  const angleInfo = getCurrentAngleBins();
  if (!angleInfo) return;
  const nCoarseGroups = energyBounds.length - 1;
  const nA = getAngleBinCount();
  const identity = isIdentityGrid(energyBounds, angleInfo);
  const total = manifest.temperatures.length;
  const totalEstimate = nCoarseGroups * nCoarseGroups * nA * 8 * total;

  if (totalEstimate > 200 * 1024 * 1024) {
    const sizeMB = (totalEstimate / (1024 * 1024)).toFixed(0);
    if (!confirm(`Estimated total size: ~${sizeMB} MB.\nThis may use significant browser memory. Proceed?`)) return;
  }

  btnDownloadAll.disabled = true;

  if (identity) {
    showProgress(0, `Fetching 0/${total}\u2026`);
    try {
      const zip = new JSZip();
      for (let i = 0; i < total; i++) {
        const t = manifest.temperatures[i];
        const resp = await fetch(`data/uniform/${t.file}`);
        if (!resp.ok) throw new Error(`Failed to fetch ${t.file}: ${resp.status}`);
        zip.file(t.file, await resp.arrayBuffer());
        showProgress(Math.round(((i + 1) / total) * 90), `Fetching ${i + 1}/${total}\u2026`);
      }
      showProgress(92, "Compressing ZIP\u2026");
      const blob = await zip.generateAsync({ type: "blob" });
      triggerDownload(blob, "compton_all_temperatures_raw.zip");
      showProgress(100, "Download complete.");
      hideProgress();
    } catch (err) {
      progressLabel.textContent = "Error: " + err.message;
      console.error(err);
    } finally {
      updateButtonStates();
    }
    return;
  }

  showProgress(0, `Processing 0/${total}\u2026`);
  try {
    const allArrays = [];
    for (let i = 0; i < total; i++) {
      const t = manifest.temperatures[i];
      const resp = await fetch(`data/uniform/${t.file}`);
      if (!resp.ok) throw new Error(`Failed to fetch ${t.file}: ${resp.status}`);
      pyodide.FS.writeFile("/tmp/input.npz", new Uint8Array(await resp.arrayBuffer()));
      const arr = await pyodide.runPythonAsync(buildCollapseToArrayCall(energyBounds, angleInfo));
      allArrays.push(Array.from(arr.toJs({ create_proxies: false })));
      showProgress(Math.round(((i + 1) / total) * 85), `Processing ${i + 1}/${total}\u2026`);
    }
    showProgress(87, "Building 4D array\u2026");
    const npyBytes = await pyodide.runPythonAsync(`
import numpy as np, io
_data = ${JSON.stringify(allArrays)}
_stacked = np.array(_data, dtype=np.float64).reshape(${total}, ${nCoarseGroups}, ${nCoarseGroups}, ${nA})
_buf = io.BytesIO()
np.save(_buf, _stacked)
_buf.getvalue()
`);
    const blob = new Blob([npyBytes.toJs()], { type: "application/octet-stream" });
    triggerDownload(blob, `compton_sigma_all_${total}T_${nCoarseGroups}g_${nA}a.npy`);
    showProgress(100, "Download complete.");
    hideProgress();
  } catch (err) {
    progressLabel.textContent = "Error: " + err.message;
    console.error(err);
  } finally {
    updateButtonStates();
  }
}

/* ── Metadata download ─────────────────────────────────── */

async function handleDownloadMetadata() {
  const temps = getSelectedTemperatures();
  if (!temps || !pyodide) return;
  const energyBounds = getCurrentEnergyBoundaries();
  if (!energyBounds) return;
  const angleBounds = getAngleBoundariesArray();
  if (!angleBounds) return;

  btnDownloadMetadata.disabled = true;
  try {
    const tempBytes = await pyodide.runPythonAsync(`
import numpy as np, io
_buf = io.BytesIO()
np.save(_buf, np.array([${temps.join(",")}], dtype=np.float64))
_buf.getvalue()
`);
    triggerDownload(new Blob([tempBytes.toJs()], { type: "application/octet-stream" }), "temperatures.npy");

    const energyBytes = await pyodide.runPythonAsync(`
import numpy as np, io
_buf = io.BytesIO()
np.save(_buf, np.array([${energyBounds.join(",")}], dtype=np.float64))
_buf.getvalue()
`);
    triggerDownload(new Blob([energyBytes.toJs()], { type: "application/octet-stream" }), "energy_boundaries_keV.npy");

    const angleBytes = await pyodide.runPythonAsync(`
import numpy as np, io
_buf = io.BytesIO()
np.save(_buf, np.array([${angleBounds.join(",")}], dtype=np.float64))
_buf.getvalue()
`);
    triggerDownload(new Blob([angleBytes.toJs()], { type: "application/octet-stream" }), "angle_boundaries_xi.npy");
  } catch (err) {
    console.error("Metadata download error:", err);
  } finally {
    updateButtonStates();
  }
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/* ── Plot Instrument ───────────────────────────────────── */

function wirePlotEvents() {
  plotModeSelect.addEventListener("change", updatePlotPanels);
  plotButton.addEventListener("click", handlePlot);
  plotDownloadButton.addEventListener("click", handlePlotDownload);

  [$("plot-e-group"), $("plot-xi-bin"), $("plot-e-group-xi"),
   $("plot-ep-group-xi"), $("plot-xi-bin-matrix"), plotTempSelect,
   plotModeSelect].forEach((el) => {
    if (el) el.addEventListener("change", updatePlotSelectionInfo);
  });
  [$("plot-e-group"), $("plot-xi-bin"), $("plot-e-group-xi"),
   $("plot-ep-group-xi"), $("plot-xi-bin-matrix")].forEach((el) => {
    if (el) el.addEventListener("input", updatePlotSelectionInfo);
  });
}

function updatePlotPanels() {
  const mode = plotModeSelect.value;
  $("plot-params-ep").hidden = mode !== "ep-profile";
  $("plot-params-xi").hidden = mode !== "xi-profile";
  $("plot-params-matrix").hidden = mode !== "matrix";
  $("plot-scale-switches").hidden = mode === "matrix";
  updatePlotSelectionInfo();
}

function groupMidpoint(idx) {
  if (!fineBoundaries || idx < 0 || idx >= nFineGroups) return NaN;
  return Math.sqrt(fineBoundaries[idx] * fineBoundaries[idx + 1]);
}

function angleBinMidpoint(idx) {
  if (idx < 0 || idx >= nFineAngleBins) return NaN;
  const w = 2 / nFineAngleBins;
  return -1 + (idx + 0.5) * w;
}

function updatePlotSelectionInfo() {
  if (!manifest) return;
  const mode = plotModeSelect.value;
  let info = "";
  if (mode === "ep-profile") {
    const eIdx = parseInt($("plot-e-group").value);
    const xiIdx = parseInt($("plot-xi-bin").value);
    const eMid = groupMidpoint(eIdx);
    const xiMid = angleBinMidpoint(xiIdx);
    info = `E = ${isNaN(eMid) ? "?" : eMid.toExponential(3)} keV (group ${eIdx}), ` +
           `\u03BE = ${isNaN(xiMid) ? "?" : xiMid.toFixed(4)} (bin ${xiIdx})`;
  } else if (mode === "xi-profile") {
    const eIdx = parseInt($("plot-e-group-xi").value);
    const epIdx = parseInt($("plot-ep-group-xi").value);
    const eMid = groupMidpoint(eIdx);
    const epMid = groupMidpoint(epIdx);
    info = `E = ${isNaN(eMid) ? "?" : eMid.toExponential(3)} keV (group ${eIdx}), ` +
           `E\u2032 = ${isNaN(epMid) ? "?" : epMid.toExponential(3)} keV (group ${epIdx})`;
  } else {
    const xiIdx = parseInt($("plot-xi-bin-matrix").value);
    const xiMid = angleBinMidpoint(xiIdx);
    info = `\u03BE = ${isNaN(xiMid) ? "?" : xiMid.toFixed(4)} (bin ${xiIdx})`;
  }
  plotSelectionInfo.textContent = info;
}

const npzCache = new Map();

async function fetchNpzMatrix(tempIndex) {
  if (npzCache.has(tempIndex)) return npzCache.get(tempIndex);
  const entry = manifest.temperatures[tempIndex];
  const resp = await fetch(`data/uniform/${entry.file}`);
  if (!resp.ok) throw new Error(`Failed to fetch ${entry.file}: ${resp.status}`);
  const npzBuf = await resp.arrayBuffer();
  pyodide.FS.writeFile("/tmp/plot_input.npz", new Uint8Array(npzBuf));
  const matProxy = await pyodide.runPythonAsync(`
import numpy as np
_f = np.load("/tmp/plot_input.npz")
_m = _f["sigma_matrix"].astype(np.float64).ravel()
_m
`);
  const flat = matProxy.toJs();
  matProxy.destroy();
  const result = { data: flat, nG: nFineGroups, nA: nFineAngleBins };
  npzCache.set(tempIndex, result);
  return result;
}

function getMatrixValue(flat, nG, nA, gIn, gOut, aBin) {
  return flat[gIn * nG * nA + gOut * nA + aBin];
}

async function handlePlot() {
  if (!pyodide || !manifest) return;
  const mode = plotModeSelect.value;
  const tempIdx = parseInt(plotTempSelect.value);
  if (isNaN(tempIdx)) return;

  plotButton.disabled = true;
  plotFormError.hidden = true;

  try {
    const { data: flat, nG, nA } = await fetchNpzMatrix(tempIdx);
    const tempEntry = manifest.temperatures[tempIdx];
    const tempK = tempEntry.temperature_K;

    if (mode === "ep-profile") {
      const eIdx = parseInt($("plot-e-group").value);
      const xiIdx = parseInt($("plot-xi-bin").value);
      if (eIdx < 0 || eIdx >= nG || xiIdx < 0 || xiIdx >= nA) {
        plotFormError.textContent = `Group/bin index out of range (0\u2013${nG - 1} for groups, 0\u2013${nA - 1} for bins).`;
        plotFormError.hidden = false;
        return;
      }
      const xVals = [], yVals = [];
      for (let g = 0; g < nG; g++) {
        xVals.push(groupMidpoint(g));
        yVals.push(getMatrixValue(flat, nG, nA, eIdx, g, xiIdx));
      }
      const logX = $("plot-log-x").checked;
      const logY = $("plot-log-y").checked;
      const xLabel = "Outgoing energy E\u2032 [keV]";
      const yLabel = "\u03C3(E \u2192 E\u2032, \u03BE) [cm\u00B2]";
      const title = `E\u2032 profile: E = ${groupMidpoint(eIdx).toExponential(3)} keV, \u03BE\u2248${angleBinMidpoint(xiIdx).toFixed(3)}, T = ${tempK.toExponential(3)} K`;
      renderLinePlot(plotChart, xVals, yVals, { logX, logY, xLabel, yLabel, title });
      plotDescription.textContent = title;
      lastPlotCSV = plotDataToCSV(xVals, yVals, "Ep_keV", "sigma_cm2");
    } else if (mode === "xi-profile") {
      const eIdx = parseInt($("plot-e-group-xi").value);
      const epIdx = parseInt($("plot-ep-group-xi").value);
      if (eIdx < 0 || eIdx >= nG || epIdx < 0 || epIdx >= nG) {
        plotFormError.textContent = `Group index out of range (0\u2013${nG - 1}).`;
        plotFormError.hidden = false;
        return;
      }
      const xVals = [], yVals = [];
      for (let a = 0; a < nA; a++) {
        xVals.push(angleBinMidpoint(a));
        yVals.push(getMatrixValue(flat, nG, nA, eIdx, epIdx, a));
      }
      const logX = false;
      const logY = $("plot-log-y").checked;
      const xLabel = "\u03BE = cos \u03B8";
      const yLabel = "\u03C3(E \u2192 E\u2032, \u03BE) [cm\u00B2]";
      const title = `\u03BE profile: E = ${groupMidpoint(eIdx).toExponential(3)} keV, E\u2032 = ${groupMidpoint(epIdx).toExponential(3)} keV, T = ${tempK.toExponential(3)} K`;
      renderLinePlot(plotChart, xVals, yVals, { logX, logY, xLabel, yLabel, title });
      plotDescription.textContent = title;
      lastPlotCSV = plotDataToCSV(xVals, yVals, "xi_cos_theta", "sigma_cm2");
    } else {
      const xiIdx = parseInt($("plot-xi-bin-matrix").value);
      if (xiIdx < 0 || xiIdx >= nA) {
        plotFormError.textContent = `Bin index out of range (0\u2013${nA - 1}).`;
        plotFormError.hidden = false;
        return;
      }
      const matrixData = [];
      for (let gIn = 0; gIn < nG; gIn++) {
        const row = [];
        for (let gOut = 0; gOut < nG; gOut++) {
          row.push(getMatrixValue(flat, nG, nA, gIn, gOut, xiIdx));
        }
        matrixData.push(row);
      }
      const xEdges = Array.from(fineBoundaries);
      const yEdges = Array.from(fineBoundaries);
      const title = `E/E\u2032 matrix: \u03BE\u2248${angleBinMidpoint(xiIdx).toFixed(3)} (bin ${xiIdx}), T = ${tempK.toExponential(3)} K`;
      renderHeatmap(plotChart, matrixData, xEdges, yEdges, {
        xLabel: "Outgoing energy E\u2032 [keV]",
        yLabel: "Incoming energy E [keV]",
        zLabel: "\u03C3 [cm\u00B2]",
        title
      });
      plotDescription.textContent = title;
      lastPlotCSV = null;
    }

    plotSection.hidden = false;
    plotPlaceholder.hidden = true;
    plotSection.scrollIntoView({ behavior: "smooth", block: "start" });
    plotDownloadButton.disabled = (lastPlotCSV === null);
  } catch (err) {
    plotFormError.textContent = "Plot error: " + err.message;
    plotFormError.hidden = false;
    console.error(err);
  } finally {
    plotButton.disabled = false;
  }
}

function handlePlotDownload() {
  if (!lastPlotCSV) return;
  const blob = new Blob([lastPlotCSV], { type: "text/csv" });
  triggerDownload(blob, "mace_plot_data.csv");
}

/* ── Initialization ────────────────────────────────────── */

async function loadManifest() {
  const resp = await fetch("data/uniform/manifest.json");
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

  function texExp(v) {
    const s = v.toExponential(1);
    const [m, e] = s.split("e");
    return `${m}\\times10^{${parseInt(e, 10)}}`;
  }
  gridSummary.innerHTML =
    `${nFineGroups} energy groups (\\(${texExp(fineBoundaries[0])}\\) \u2013 \\(${texExp(fineBoundaries[fineBoundaries.length - 1])}\\) keV, log-spaced)<br>` +
    `${nFineAngleBins} angle bins (\\(\\xi = \\cos\\theta \\in [-1,\\,1]\\))<br>` +
    `${manifest.temperatures.length} temperatures (\\(${texExp(tMin.temperature_K)}\\) \u2013 \\(${texExp(tMax.temperature_K)}\\) K, log-spaced)`;
  if (window.renderMathInElement) renderMathInElement(gridSummary, { delimiters: [{ left: "$$", right: "$$", display: true }, { left: "\\(", right: "\\)", display: false }] });

  tempSelect.innerHTML = "";
  for (const t of manifest.temperatures) {
    t._index = t.index;
    const opt = document.createElement("option");
    opt.value = t.index;
    opt.textContent = formatTemp(t);
    tempSelect.appendChild(opt);
  }

  plotTempSelect.innerHTML = "";
  for (const t of manifest.temperatures) {
    const opt = document.createElement("option");
    opt.value = t.index;
    opt.textContent = formatTemp(t);
    plotTempSelect.appendChild(opt);
  }

  [$("plot-e-group"), $("plot-e-group-xi"), $("plot-ep-group-xi")].forEach((el) => {
    if (el) { el.max = nFineGroups - 1; }
  });
  [$("plot-xi-bin"), $("plot-xi-bin-matrix")].forEach((el) => {
    if (el) { el.max = nFineAngleBins - 1; }
  });

  loadingMsg.hidden = true;
  downloadForm.hidden = false;
  plotForm.hidden = false;

  updateBoundariesDisplay();
  updateAngleHints();
  updateSummary();
  updateButtonStates();
  updatePlotPanels();
  updatePlotSelectionInfo();
}

async function loadPyodide_() {
  loadingMsg.textContent = "Loading Python runtime\u2026";
  const pyodideModule = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.6/full/",
  });
  await pyodideModule.loadPackage("numpy");

  const collapseResp = await fetch("py/collapse.py");
  collapseCode = await collapseResp.text();
  await pyodideModule.runPythonAsync(collapseCode);

  pyodide = pyodideModule;
  loadingMsg.hidden = true;
  downloadForm.hidden = false;
  updateButtonStates();
}

async function init() {
  wireDownloadEvents();
  wirePlotEvents();

  const manifestPromise = loadManifest();

  const pyodideScript = document.createElement("script");
  pyodideScript.src = "https://cdn.jsdelivr.net/pyodide/v0.27.6/full/pyodide.js";
  pyodideScript.addEventListener("load", () => {
    loadPyodide_().catch((err) => {
      loadingMsg.textContent = "Failed to load Python runtime: " + err.message;
      console.error(err);
    });
  });
  document.head.appendChild(pyodideScript);

  const jszipScript = document.createElement("script");
  jszipScript.src = "https://cdn.jsdelivr.net/npm/jszip@3.10.1/dist/jszip.min.js";
  jszipScript.addEventListener("load", () => {
    jszipLoaded = true;
    updateButtonStates();
  });
  document.head.appendChild(jszipScript);

  await manifestPromise;
}

init();
