#!/usr/bin/env python3
"""
Convert existing uniform float64 .npz files to float32, dropping dsigma_dT_matrix.

Reads from docs/data/T*K.npz and writes to docs/data/uniform/T*K.npz.
This preserves the high-quadrature uniform tables (indices 0-20) without
recomputing them.

Usage:
  python scripts/convert_uniform_to_float32.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "docs" / "data"
DST_DIR = SRC_DIR / "uniform"


def main() -> None:
    src_files = sorted(SRC_DIR.glob("T*_*K.npz"))
    if not src_files:
        print(f"No .npz files found in {SRC_DIR}")
        return

    DST_DIR.mkdir(parents=True, exist_ok=True)

    total_src = 0
    total_dst = 0

    for src in src_files:
        data = np.load(src)
        sigma = data["sigma_matrix"]
        boundaries = data["boundaries_keV"]
        temp = float(data["temperature_K"])
        data.close()

        dst = DST_DIR / src.name
        np.savez_compressed(
            dst,
            sigma_matrix=sigma.astype(np.float32),
            boundaries_keV=boundaries,
            temperature_K=temp,
        )

        src_size = src.stat().st_size
        dst_size = dst.stat().st_size
        total_src += src_size
        total_dst += dst_size
        print(f"  {src.name}: {src_size/1024:.0f} KB -> {dst_size/1024:.0f} KB "
              f"({dst_size/src_size:.1%})")

    print(f"\nConverted {len(src_files)} files: "
          f"{total_src/1e6:.1f} MB -> {total_dst/1e6:.1f} MB "
          f"({total_dst/total_src:.1%})")


if __name__ == "__main__":
    main()
