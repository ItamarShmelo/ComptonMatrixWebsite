const SVG_NS = "http://www.w3.org/2000/svg";

function el(tag, attrs, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const c of children) {
    if (typeof c === "string") node.appendChild(document.createTextNode(c));
    else if (c) node.appendChild(c);
  }
  return node;
}

/* ── Axis tick generation ──────────────────────────────── */

function logTicks(lo, hi, maxTicks = 12) {
  const logLo = Math.floor(Math.log10(lo));
  const logHi = Math.ceil(Math.log10(hi));
  const ticks = [];
  for (let e = logLo; e <= logHi; e++) {
    const v = Math.pow(10, e);
    if (v >= lo * 0.999 && v <= hi * 1.001) ticks.push(v);
  }
  if (ticks.length > maxTicks) {
    const step = Math.ceil(ticks.length / maxTicks);
    return ticks.filter((_, i) => i % step === 0);
  }
  if (ticks.length < 2) {
    return linearTicks(lo, hi, 6);
  }
  return ticks;
}

function linearTicks(lo, hi, target = 8) {
  const range = hi - lo;
  if (range <= 0) return [lo];
  const rough = range / target;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  let step;
  const norm = rough / mag;
  if (norm < 1.5) step = mag;
  else if (norm < 3.5) step = 2 * mag;
  else if (norm < 7.5) step = 5 * mag;
  else step = 10 * mag;
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 0.01; v += step) {
    ticks.push(parseFloat(v.toPrecision(12)));
  }
  return ticks;
}

function formatTickLabel(v, isLog) {
  if (isLog) {
    const e = Math.round(Math.log10(v));
    if (Math.abs(v - Math.pow(10, e)) / v < 0.01) return `10${superscript(e)}`;
    return v.toExponential(1);
  }
  if (Math.abs(v) < 1e-10) return "0";
  if (Math.abs(v) >= 1e4 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(1);
  const s = v.toPrecision(4);
  return parseFloat(s).toString();
}

function superscript(n) {
  const map = { "0": "\u2070", "1": "\u00B9", "2": "\u00B2", "3": "\u00B3",
    "4": "\u2074", "5": "\u2075", "6": "\u2076", "7": "\u2077",
    "8": "\u2078", "9": "\u2079", "-": "\u207B" };
  return String(n).split("").map((c) => map[c] || c).join("");
}

/* ── Line plot rendering ──────────────────────────────── */

export function renderLinePlot(container, xVals, yVals, options = {}) {
  const { logX = true, logY = true, xLabel = "x", yLabel = "y", title = "" } = options;

  const width = 920, height = 540;
  const margin = { top: 50, right: 40, bottom: 65, left: 85 };
  const pw = width - margin.left - margin.right;
  const ph = height - margin.top - margin.bottom;

  const drawable = [];
  for (let i = 0; i < xVals.length; i++) {
    const x = xVals[i], y = yVals[i];
    if (!isFinite(x) || !isFinite(y)) continue;
    if (logX && x <= 0) continue;
    if (logY && y <= 0) continue;
    drawable.push({ x, y });
  }

  if (drawable.length === 0) {
    container.innerHTML = '<p style="padding:2rem;color:var(--muted);">No drawable data points (all zero or negative on log scale).</p>';
    return;
  }

  let xMin = Math.min(...drawable.map((d) => d.x));
  let xMax = Math.max(...drawable.map((d) => d.x));
  let yMin = Math.min(...drawable.map((d) => d.y));
  let yMax = Math.max(...drawable.map((d) => d.y));

  if (logX) {
    if (xMin === xMax) { xMin *= 0.5; xMax *= 2; }
  } else {
    const pad = (xMax - xMin) * 0.05 || 0.5;
    xMin -= pad; xMax += pad;
  }
  if (logY) {
    if (yMin === yMax) { yMin *= 0.5; yMax *= 2; }
    const logRange = Math.log10(yMax) - Math.log10(yMin);
    yMin = Math.pow(10, Math.log10(yMin) - logRange * 0.05);
    yMax = Math.pow(10, Math.log10(yMax) + logRange * 0.05);
  } else {
    const pad = (yMax - yMin) * 0.08 || 0.5;
    yMin -= pad; yMax += pad;
  }

  function px(v) {
    if (logX) return margin.left + (Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin)) * pw;
    return margin.left + (v - xMin) / (xMax - xMin) * pw;
  }
  function py(v) {
    if (logY) return margin.top + (1 - (Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * ph;
    return margin.top + (1 - (v - yMin) / (yMax - yMin)) * ph;
  }

  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "plot-svg",
    "aria-label": title,
  });

  svg.appendChild(el("rect", { x: 0, y: 0, width, height, class: "plot-background" }));

  const xTicks = logX ? logTicks(xMin, xMax) : linearTicks(xMin, xMax);
  const yTicks = logY ? logTicks(yMin, yMax) : linearTicks(yMin, yMax);

  for (const t of xTicks) {
    const x = px(t);
    if (x < margin.left || x > width - margin.right) continue;
    svg.appendChild(el("line", { x1: x, y1: margin.top, x2: x, y2: height - margin.bottom, class: "plot-grid-line" }));
    svg.appendChild(el("text", { x, y: height - margin.bottom + 18, "text-anchor": "middle", class: "plot-tick-label" }, formatTickLabel(t, logX)));
  }
  for (const t of yTicks) {
    const y = py(t);
    if (y < margin.top || y > height - margin.bottom) continue;
    svg.appendChild(el("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, class: "plot-grid-line" }));
    svg.appendChild(el("text", { x: margin.left - 10, y: y + 4, "text-anchor": "end", class: "plot-tick-label" }, formatTickLabel(t, logY)));
  }

  svg.appendChild(el("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: "plot-axis-line" }));
  svg.appendChild(el("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, class: "plot-axis-line" }));

  svg.appendChild(el("text", {
    x: margin.left + pw / 2, y: height - 8,
    "text-anchor": "middle", class: "plot-axis-label"
  }, xLabel));

  const yLabelEl = el("text", {
    x: 18, y: margin.top + ph / 2,
    "text-anchor": "middle", class: "plot-axis-label",
    transform: `rotate(-90, 18, ${margin.top + ph / 2})`
  }, yLabel);
  svg.appendChild(yLabelEl);

  if (title) {
    svg.appendChild(el("text", {
      x: margin.left + pw / 2, y: 24,
      "text-anchor": "middle", class: "plot-axis-label",
      "font-size": "14"
    }, title));
  }

  const pathParts = [];
  let prevValid = false;
  for (const d of drawable) {
    const x = px(d.x), y = py(d.y);
    if (x < margin.left || x > width - margin.right || y < margin.top || y > height - margin.bottom) {
      prevValid = false;
      continue;
    }
    pathParts.push(prevValid ? `L${x.toFixed(2)} ${y.toFixed(2)}` : `M${x.toFixed(2)} ${y.toFixed(2)}`);
    prevValid = true;
  }

  if (pathParts.length > 0) {
    svg.appendChild(el("path", { d: pathParts.join(" "), class: "plot-data-line" }));
  }

  for (const d of drawable) {
    const x = px(d.x), y = py(d.y);
    if (x < margin.left || x > width - margin.right || y < margin.top || y > height - margin.bottom) continue;
    const circle = el("circle", { cx: x, cy: y, r: 2.8, class: "plot-data-point" });
    circle.appendChild(el("title", {}, `(${d.x.toExponential(4)}, ${d.y.toExponential(4)})`));
    svg.appendChild(circle);
  }

  container.replaceChildren(svg);
}

/* ── Heatmap rendering ─────────────────────────────────── */

function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const c = [
    [0.267, 0.004, 0.329],
    [0.282, 0.140, 0.458],
    [0.253, 0.265, 0.530],
    [0.207, 0.372, 0.553],
    [0.164, 0.471, 0.558],
    [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518],
    [0.267, 0.749, 0.441],
    [0.478, 0.821, 0.318],
    [0.741, 0.873, 0.150],
    [0.993, 0.906, 0.144],
  ];
  const idx = t * (c.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, c.length - 1);
  const f = idx - lo;
  const r = Math.round((c[lo][0] * (1 - f) + c[hi][0] * f) * 255);
  const g = Math.round((c[lo][1] * (1 - f) + c[hi][1] * f) * 255);
  const b = Math.round((c[lo][2] * (1 - f) + c[hi][2] * f) * 255);
  return `rgb(${r},${g},${b})`;
}

export function renderHeatmap(container, matrix, xEdges, yEdges, options = {}) {
  const { xLabel = "x", yLabel = "y", zLabel = "z", title = "" } = options;

  const nRows = matrix.length;
  const nCols = matrix[0].length;

  let zMin = Infinity, zMax = -Infinity;
  for (let i = 0; i < nRows; i++) {
    for (let j = 0; j < nCols; j++) {
      const v = matrix[i][j];
      if (v > 0 && isFinite(v)) {
        if (v < zMin) zMin = v;
        if (v > zMax) zMax = v;
      }
    }
  }
  if (!isFinite(zMin) || !isFinite(zMax) || zMin >= zMax) {
    zMin = 1e-30; zMax = 1;
  }

  const logZMin = Math.log10(zMin);
  const logZMax = Math.log10(zMax);

  const width = 960, height = 740;
  const margin = { top: 50, right: 120, bottom: 70, left: 90 };
  const pw = width - margin.left - margin.right;
  const ph = height - margin.top - margin.bottom;

  const logXMin = Math.log10(xEdges[0]);
  const logXMax = Math.log10(xEdges[xEdges.length - 1]);
  const logYMin = Math.log10(yEdges[0]);
  const logYMax = Math.log10(yEdges[yEdges.length - 1]);

  function px(v) { return margin.left + (Math.log10(v) - logXMin) / (logXMax - logXMin) * pw; }
  function py(v) { return margin.top + (1 - (Math.log10(v) - logYMin) / (logYMax - logYMin)) * ph; }

  const svg = el("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "plot-svg",
    "aria-label": title,
  });

  svg.appendChild(el("rect", { x: 0, y: 0, width, height, class: "plot-background" }));

  for (let i = 0; i < nRows; i++) {
    const y1 = py(yEdges[i + 1]);
    const y2 = py(yEdges[i]);
    const cellH = y2 - y1;
    if (cellH < 0.2) continue;
    for (let j = 0; j < nCols; j++) {
      const x1 = px(xEdges[j]);
      const x2 = px(xEdges[j + 1]);
      const cellW = x2 - x1;
      if (cellW < 0.2) continue;
      const v = matrix[i][j];
      let color;
      if (v <= 0 || !isFinite(v)) {
        color = "rgb(10,10,20)";
      } else {
        const t = (Math.log10(v) - logZMin) / (logZMax - logZMin);
        color = viridis(t);
      }
      const rect = el("rect", {
        x: x1.toFixed(2), y: y1.toFixed(2),
        width: Math.max(cellW, 0.5).toFixed(2),
        height: Math.max(cellH, 0.5).toFixed(2),
        fill: color,
        class: "heatmap-cell"
      });
      rect.appendChild(el("title", {}, `E[${i}]→E'[${j}]: ${v.toExponential(4)}`));
      svg.appendChild(rect);
    }
  }

  const xTicks = logTicks(xEdges[0], xEdges[xEdges.length - 1]);
  const yTicks = logTicks(yEdges[0], yEdges[yEdges.length - 1]);

  for (const t of xTicks) {
    const x = px(t);
    if (x < margin.left || x > width - margin.right) continue;
    svg.appendChild(el("line", { x1: x, y1: margin.top, x2: x, y2: height - margin.bottom, stroke: "rgba(255,255,255,0.12)", "stroke-width": 0.5 }));
    svg.appendChild(el("text", { x, y: height - margin.bottom + 18, "text-anchor": "middle", class: "plot-tick-label" }, formatTickLabel(t, true)));
  }
  for (const t of yTicks) {
    const y = py(t);
    if (y < margin.top || y > height - margin.bottom) continue;
    svg.appendChild(el("line", { x1: margin.left, y1: y, x2: width - margin.right, y2: y, stroke: "rgba(255,255,255,0.12)", "stroke-width": 0.5 }));
    svg.appendChild(el("text", { x: margin.left - 10, y: y + 4, "text-anchor": "end", class: "plot-tick-label" }, formatTickLabel(t, true)));
  }

  svg.appendChild(el("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: "plot-axis-line" }));
  svg.appendChild(el("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, class: "plot-axis-line" }));

  svg.appendChild(el("text", {
    x: margin.left + pw / 2, y: height - 10,
    "text-anchor": "middle", class: "plot-axis-label"
  }, xLabel));

  svg.appendChild(el("text", {
    x: 18, y: margin.top + ph / 2,
    "text-anchor": "middle", class: "plot-axis-label",
    transform: `rotate(-90, 18, ${margin.top + ph / 2})`
  }, yLabel));

  if (title) {
    svg.appendChild(el("text", {
      x: margin.left + pw / 2, y: 24,
      "text-anchor": "middle", class: "plot-axis-label",
      "font-size": "14"
    }, title));
  }

  const cbX = width - margin.right + 25;
  const cbW = 16;
  const cbH = ph;
  const nStops = 128;
  for (let k = 0; k < nStops; k++) {
    const t = k / (nStops - 1);
    const y = margin.top + (1 - t) * cbH;
    const h = cbH / nStops + 0.5;
    svg.appendChild(el("rect", {
      x: cbX, y: y.toFixed(1), width: cbW, height: h.toFixed(1),
      fill: viridis(t), "shape-rendering": "crispEdges"
    }));
  }

  svg.appendChild(el("rect", {
    x: cbX, y: margin.top, width: cbW, height: cbH,
    fill: "none", stroke: "rgba(255,255,255,0.2)", "stroke-width": 1
  }));

  const cbTicks = logTicks(zMin, zMax, 6);
  for (const t of cbTicks) {
    const frac = (Math.log10(t) - logZMin) / (logZMax - logZMin);
    const y = margin.top + (1 - frac) * cbH;
    if (y < margin.top || y > margin.top + cbH) continue;
    svg.appendChild(el("line", {
      x1: cbX + cbW, y1: y, x2: cbX + cbW + 4, y2: y,
      stroke: "rgba(255,255,255,0.5)", "stroke-width": 1
    }));
    svg.appendChild(el("text", {
      x: cbX + cbW + 7, y: y + 3.5,
      "text-anchor": "start", class: "colorbar-label"
    }, formatTickLabel(t, true)));
  }

  svg.appendChild(el("text", {
    x: cbX + cbW / 2, y: margin.top - 10,
    "text-anchor": "middle", class: "colorbar-title"
  }, zLabel));

  container.replaceChildren(svg);
}

/* ── CSV export ────────────────────────────────────────── */

export function plotDataToCSV(xVals, yVals, xHeader, yHeader) {
  const lines = [`${xHeader},${yHeader}`];
  for (let i = 0; i < xVals.length; i++) {
    lines.push(`${xVals[i]},${yVals[i]}`);
  }
  return lines.join("\n");
}
