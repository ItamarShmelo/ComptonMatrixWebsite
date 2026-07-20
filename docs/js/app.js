"use strict";

/* ── State ─────────────────────────────────────────────── */

let manifest = null;
let pyodide = null;
let collapseCode = null;
let fineBoundaries = null; // Float64Array from manifest
let nFineGroups = 0;
let nFineAngleBins = 0;
let jszipLoaded = false;

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
const customAngleCheck = $("use-custom-angle-boundaries");
const customAngleDiv = $("angle-boundaries-custom");
const customAngleInput = $("custom-angle-input");
const customAngleError = $("custom-angle-error");
const summaryShape = $("summary-shape");
const summaryAxes = $("summary-axes");
const summarySize = $("summary-size");
const btnDownload = $("btn-download");
const btnDownloadAll = $("btn-download-all");
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

function pickLogSpacedBoundaries(fineBounds, nCoarse) {
  const G = fineBounds.length - 1;
  if (nCoarse >= G) {
    return Array.from(fineBounds);
  }
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

/* ── UI update functions ───────────────────────────────── */

function getCurrentEnergyBoundaries() {
  if (customCheck.checked) {
    return parseCustomBoundaries();
  }
  const n = Math.max(1, Math.min(nFineGroups, parseInt(nGroupsInput.value) || nFineGroups));
  return pickLogSpacedBoundaries(fineBoundaries, n);
}

function getCurrentAngleBins() {
  if (customAngleCheck.checked) {
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

function updateBoundariesDisplay() {
  if (!fineBoundaries) return;
  if (customCheck.checked) return;

  const bounds = getCurrentEnergyBoundaries();
  if (!bounds) return;
  const vals = bounds.map((v) => v.toExponential(3));
  boundariesDisplay.textContent = vals.join("  ");
}

function updateSummary() {
  const bounds = getCurrentEnergyBoundaries();
  if (!bounds) return;
  const angleCfg = getCurrentAngleBins();
  if (!angleCfg) return;

  const nG = bounds.length - 1;
  const nA = getAngleBinCount();

  if (nG === 1 && nA === 1) {
    summaryShape.textContent = "scalar";
    summaryAxes.textContent = "\u2014";
  } else if (nG === 1) {
    summaryShape.textContent = `${nA}`;
    summaryAxes.textContent = "[angle bin]";
  } else if (nA === 1) {
    summaryShape.innerHTML = `${nG} &times; ${nG}`;
    summaryAxes.textContent = "[incoming group] \u00D7 [outgoing group]";
  } else {
    summaryShape.innerHTML = `${nG} &times; ${nG} &times; ${nA}`;
    summaryAxes.textContent = "[incoming group] \u00D7 [outgoing group] \u00D7 [angle bin]";
  }
  const bytes = Math.max(1, nG * nG * nA) * 8;
  summarySize.textContent = formatSize(bytes);
}

function updateAngleHints() {
  if (customAngleCheck.checked) return;
  const aInfo = getCurrentAngleBins();
  if (!aInfo) return;
  const nA = aInfo.count || 1;
  nAnglesHint.textContent = `Any integer from 1 to ${nFineAngleBins}. Values above ${nFineAngleBins} exceed stored resolution.`;
  angleInfo.textContent = `Bin width: \u0394\u03BE = ${(2 / nA).toFixed(6)}`;
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
  if (values[0] <= 0) {
    customError.textContent = "Boundaries must be positive.";
    customError.classList.add("error");
    return null;
  }

  const eMin = fineBoundaries[0];
  const eMax = fineBoundaries[fineBoundaries.length - 1];
  if (values[0] < eMin * (1 - 1e-6) || values[values.length - 1] > eMax * (1 + 1e-6)) {
    customError.textContent = `Boundaries must lie within [${eMin.toExponential(3)}, ${eMax.toExponential(3)}] keV.`;
    customError.classList.add("error");
    return null;
  }

  customError.textContent = `\u2713 ${values.length - 1} groups defined.`;
  customError.classList.remove("error");
  return values;
}

function parseCustomAngleBoundaries() {
  const text = customAngleInput.value.trim();
  if (!text) { customAngleError.textContent = ""; return null; }

  const tokens = text.split(/[\s,]+/).filter(Boolean);
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

  customAngleCheck.addEventListener("change", () => {
    if (customAngleCheck.checked) {
      nAnglesInput.disabled = true;
      customAngleDiv.style.display = "block";
      nAnglesHint.textContent = "";
      angleInfo.textContent = "";
    } else {
      nAnglesInput.disabled = false;
      customAngleDiv.style.display = "none";
      customAngleError.textContent = "";
      updateAngleHints();
    }
    updateSummary();
  });

  customAngleInput.addEventListener("input", () => {
    parseCustomAngleBoundaries();
    updateSummary();
  });

  tempSelect.addEventListener("change", updateButtonStates);

  btnDownload.addEventListener("click", handleDownload);
  btnDownloadAll.addEventListener("click", handleDownloadAll);
}

function updateButtonStates() {
  btnDownload.disabled = !tempSelect.value || !pyodide;
  btnDownloadAll.disabled = !jszipLoaded || !pyodide || !manifest;
}

/* ── Progress helpers ──────────────────────────────────── */

function showProgress(pct, label) {
  progressContainer.classList.add("visible");
  progressBar.style.width = pct + "%";
  progressLabel.textContent = label;
}

function hideProgress() {
  setTimeout(() => {
    progressContainer.classList.remove("visible");
    progressBar.style.width = "0%";
  }, 2000);
}

/* ── Collapse call builder ─────────────────────────────── */

function buildCollapseCall(energyBounds, angleInfo) {
  const anglePyArg = angleInfo.boundaries
    ? `angle_boundaries=[${angleInfo.boundaries.join(",")}]`
    : `n_angle_bins=${angleInfo.count}`;
  return `collapse(\n    open("/tmp/input.npz", "rb").read(),\n    [${energyBounds.join(",")}],\n    ${anglePyArg},\n)`;
}

function isIdentityGrid(energyBounds, angleInfo) {
  if (angleInfo.boundaries) return false;
  if (angleInfo.count !== nFineAngleBins) return false;
  if (energyBounds.length !== fineBoundaries.length) return false;
  for (let i = 0; i < energyBounds.length; i++) {
    if (Math.abs(energyBounds[i] - fineBoundaries[i]) > fineBoundaries[i] * 1e-9)
      return false;
  }
  return true;
}

/* ── Single-temperature download ───────────────────────── */

async function handleDownload() {
  const tempIdx = parseInt(tempSelect.value);
  const tempEntry = manifest.temperatures.find((t) => t.index === tempIdx);
  if (!tempEntry || !pyodide) return;

  const energyBounds = getCurrentEnergyBoundaries();
  if (!energyBounds) return;
  const angleInfo = getCurrentAngleBins();
  if (!angleInfo) return;
  const nCoarseGroups = energyBounds.length - 1;
  const nA = getAngleBinCount();

  btnDownload.disabled = true;
  showProgress(10, `Fetching ${tempEntry.file}\u2026`);

  try {
    const npzUrl = `data/${tempEntry.file}`;
    const resp = await fetch(npzUrl);
    if (!resp.ok) throw new Error(`Failed to fetch ${npzUrl}: ${resp.status}`);
    const npzBuf = await resp.arrayBuffer();

    showProgress(50, "Collapsing matrix\u2026");

    pyodide.FS.writeFile("/tmp/input.npz", new Uint8Array(npzBuf));

    const npyBytes = await pyodide.runPythonAsync(buildCollapseCall(energyBounds, angleInfo));

    showProgress(90, "Preparing download\u2026");

    const blob = new Blob([npyBytes.toJs()], { type: "application/octet-stream" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `compton_sigma_T${String(tempIdx).padStart(3, "0")}_${nCoarseGroups}g_${nA}a.npy`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    showProgress(100, "Download complete.");
    hideProgress();
  } catch (err) {
    progressLabel.textContent = "Error: " + err.message;
    progressBar.style.width = "0%";
    console.error(err);
  } finally {
    btnDownload.disabled = false;
  }
}

/* ── Bulk download: all temperatures ───────────────────── */

async function handleDownloadAll() {
  if (!manifest || !jszipLoaded || !pyodide) return;

  const energyBounds = getCurrentEnergyBoundaries();
  if (!energyBounds) return;
  const angleInfo = getCurrentAngleBins();
  if (!angleInfo) return;
  const nCoarseGroups = energyBounds.length - 1;
  const nA = getAngleBinCount();
  const identity = isIdentityGrid(energyBounds, angleInfo);

  const perFileBytes = nCoarseGroups * nCoarseGroups * nA * 8;
  const totalEstimate = perFileBytes * manifest.temperatures.length;
  if (totalEstimate > 200 * 1024 * 1024) {
    const sizeMB = (totalEstimate / (1024 * 1024)).toFixed(0);
    if (!confirm(
      `Estimated total size: ~${sizeMB} MB.\n\n` +
      `This may use significant browser memory. Consider using a coarser grid.\n\nProceed?`
    )) return;
  }

  const total = manifest.temperatures.length;
  btnDownloadAll.disabled = true;
  showProgress(0, `${identity ? "Fetching" : "Processing"} 0/${total}\u2026`);

  try {
    const collapseExpr = identity ? null : buildCollapseCall(energyBounds, angleInfo);
    const zip = new JSZip();
    for (let i = 0; i < total; i++) {
      const t = manifest.temperatures[i];
      const resp = await fetch(`data/${t.file}`);
      if (!resp.ok) throw new Error(`Failed to fetch ${t.file}: ${resp.status}`);
      const npzBuf = await resp.arrayBuffer();

      if (identity) {
        zip.file(t.file, npzBuf);
      } else {
        pyodide.FS.writeFile("/tmp/input.npz", new Uint8Array(npzBuf));
        const npyBytes = await pyodide.runPythonAsync(collapseExpr);
        const fname = `compton_sigma_T${String(t.index).padStart(3, "0")}_${nCoarseGroups}g_${nA}a.npy`;
        zip.file(fname, npyBytes.toJs());
      }
      showProgress(
        Math.round(((i + 1) / total) * 90),
        `${identity ? "Fetching" : "Processing"} ${i + 1}/${total}\u2026`
      );
    }

    showProgress(92, "Compressing ZIP\u2026");
    const blob = await zip.generateAsync({ type: "blob" });

    showProgress(98, "Preparing download\u2026");
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = identity
      ? "compton_all_temperatures_raw.zip"
      : `compton_all_temperatures_${nCoarseGroups}g_${nA}a.zip`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

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

  function texExp(v) {
    const s = v.toExponential(1);
    const [m, e] = s.split("e");
    const exp = parseInt(e, 10);
    return `${m}\\times10^{${exp}}`;
  }
  const eLoTex = texExp(fineBoundaries[0]);
  const eHiTex = texExp(fineBoundaries[fineBoundaries.length - 1]);
  const tLoTex = texExp(tMin.temperature_K);
  const tHiTex = texExp(tMax.temperature_K);
  gridSummary.innerHTML =
    `${nFineGroups} energy groups (\\(${eLoTex}\\) \u2013 \\(${eHiTex}\\) keV, log-spaced)<br>` +
    `${nFineAngleBins} angle bins (\\(\\xi = \\cos\\theta \\in [-1,\\,1]\\))<br>` +
    `${manifest.temperatures.length} temperatures (\\(${tLoTex}\\) \u2013 \\(${tHiTex}\\) K, log-spaced)`;
  if (window.renderMathInElement) renderMathInElement(gridSummary, {delimiters:[{left:'$$',right:'$$',display:true},{left:'\\(',right:'\\)',display:false}]});

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
  updateButtonStates();
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
  updateButtonStates();
}

async function init() {
  wireEvents();

  const manifestPromise = loadManifest();

  const pyodideScript = document.createElement("script");
  pyodideScript.src = "https://cdn.jsdelivr.net/pyodide/v0.27.6/full/pyodide.js";
  pyodideScript.addEventListener("load", () => {
    loadPyodide_().catch((err) => {
      banner.textContent = "Failed to load Python runtime: " + err.message;
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
