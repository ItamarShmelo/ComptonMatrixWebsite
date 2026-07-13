#!/usr/bin/env python3
"""
Generate docs/data/manifest.json from the existing .npz files in docs/data/.

Scans for T{idx}_{temp}K.npz files, validates every file for consistency
(shape, required keys, matching boundaries), and writes a manifest that the
website JavaScript reads on page load.

Usage:
  python scripts/prepare_website.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "docs" / "data"

FILENAME_RE = re.compile(r"^T(\d{3})_(.+)K\.npz$")
REQUIRED_KEYS = {"sigma_matrix", "boundaries_keV", "temperature_K"}


def main() -> None:
    npz_files = sorted(DATA_DIR.glob("T*_*K.npz"))
    if not npz_files:
        print(f"ERROR: no .npz files found in {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    ref_shape = None
    ref_boundaries = None
    temperatures = []
    errors = []

    for path in npz_files:
        m = FILENAME_RE.match(path.name)
        if not m:
            errors.append(f"{path.name}: filename does not match T{{idx}}_{{temp}}K.npz")
            continue

        tidx = int(m.group(1))

        with np.load(path) as data:
            keys = set(data.files)
            missing = REQUIRED_KEYS - keys
            if missing:
                errors.append(f"{path.name}: missing keys {missing}")
                continue

            sigma = data["sigma_matrix"]
            boundaries = data["boundaries_keV"]
            temp_K = float(data["temperature_K"])

            if sigma.ndim != 3:
                errors.append(
                    f"{path.name}: sigma_matrix is {sigma.ndim}D (expected 3D), "
                    f"shape={sigma.shape}"
                )
                continue

            if ref_shape is None:
                ref_shape = sigma.shape
                ref_boundaries = boundaries
            else:
                if sigma.shape != ref_shape:
                    errors.append(
                        f"{path.name}: shape {sigma.shape} != reference {ref_shape}"
                    )
                    continue
                if not np.array_equal(boundaries, ref_boundaries):
                    errors.append(
                        f"{path.name}: boundaries differ from reference"
                    )
                    continue

        temperatures.append({
            "index": tidx,
            "temperature_K": temp_K,
            "file": path.name,
        })

    if errors:
        print(f"WARNING: {len(errors)} file(s) skipped:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)

    if not temperatures:
        print("ERROR: no valid .npz files found.", file=sys.stderr)
        sys.exit(1)

    G, _, M = ref_shape

    manifest = {
        "description": "Multigroup Compton scattering matrices",
        "boundaries_keV": ref_boundaries.tolist(),
        "n_groups": G,
        "n_angle_bins": M,
        "xi_min": -1.0,
        "xi_max": 1.0,
        "weight_function": "uniform",
        "temperatures": sorted(temperatures, key=lambda t: t["index"]),
    }

    out_path = DATA_DIR / "manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_temps = len(temperatures)
    print(f"Manifest written: {out_path}")
    print(f"  {n_temps} temperatures, {G} groups, {M} angle bins")
    print(f"  Energy range: {ref_boundaries[0]:.2e} -- {ref_boundaries[-1]:.2e} keV")


if __name__ == "__main__":
    main()
