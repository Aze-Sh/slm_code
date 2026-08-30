"""Acquire and analyze a physical delta-z scan with an SLM and AVT camera.

The optical WGS scan remains in :mod:`delta_z_scan`.  This module consumes its
ordered BMP outputs.  Hardware libraries are imported lazily so scan manifests
and CCD data can be analyzed on a computer without wxPython or Vimba X.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ScanPoint:
    """One independently optimized phase pattern in hardware display order."""

    delta_z_mm: float
    bmp_path: Path
    scan_label: str


def load_scan_points(scan_dir: str | Path) -> list[ScanPoint]:
    """Load the ordered phase manifest produced by ``delta_z_scan.py``."""
    scan_path = Path(scan_dir)
    csv_path = scan_path / "delta_z_metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"scan manifest not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"scan manifest contains no scan points: {csv_path}")

    required_columns = {"delta_z_mm", "bmp_file"}
    missing_columns = required_columns.difference(rows[0])
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"scan manifest is missing columns: {names}")

    points: list[ScanPoint] = []
    delta_z_values: set[float] = set()
    bitmap_paths: set[Path] = set()
    for row in rows:
        delta_z_mm = float(row["delta_z_mm"])
        if delta_z_mm in delta_z_values:
            raise ValueError(f"duplicate delta_z in scan manifest: {delta_z_mm}")
        delta_z_values.add(delta_z_mm)

        bitmap_name = Path(row["bmp_file"])
        if bitmap_name.is_absolute() or len(bitmap_name.parts) != 1:
            raise ValueError(
                f"bmp_file must be a filename inside the scan directory: "
                f"{bitmap_name}"
            )
        bitmap_path = scan_path / bitmap_name
        if bitmap_path in bitmap_paths:
            raise ValueError(f"duplicate BMP in scan manifest: {bitmap_name}")
        bitmap_paths.add(bitmap_path)
        if not bitmap_path.is_file():
            raise FileNotFoundError(f"scan BMP not found: {bitmap_path}")

        prefix = "slm_phase_"
        if not bitmap_path.stem.startswith(prefix):
            raise ValueError(
                f"scan BMP must start with {prefix!r}: {bitmap_path.name}"
            )
        points.append(
            ScanPoint(
                delta_z_mm=delta_z_mm,
                bmp_path=bitmap_path,
                scan_label=bitmap_path.stem[len(prefix) :],
            )
        )
    return points


def apply_slm_calibration(
    phase: np.ndarray,
    *,
    correction: np.ndarray | None,
    lut: int,
) -> np.ndarray:
    """Apply factory phase correction and the notebook's phase-level LUT."""
    phase_array = np.asarray(phase)
    if phase_array.ndim != 2:
        raise ValueError("SLM phase must be a two-dimensional grayscale image")
    if np.any(phase_array < 0) or np.any(phase_array > 255):
        raise ValueError("SLM phase values must be inside the 8-bit range")
    if not 1 <= lut <= 256:
        raise ValueError("lut must be between 1 and 256")

    modular_phase = phase_array.astype(np.uint16)
    if correction is not None:
        correction_array = np.asarray(correction)
        if correction_array.shape != phase_array.shape:
            raise ValueError(
                "SLM correction shape must match the physical phase image"
            )
        if np.any(correction_array < 0) or np.any(correction_array > 255):
            raise ValueError("SLM correction values must be inside the 8-bit range")
        modular_phase = modular_phase + correction_array.astype(np.uint16)
    modular_phase = np.remainder(modular_phase, 256)
    return (modular_phase.astype(np.float64) / 256 * lut).astype(np.uint8)


def center_phase_on_display(
    phase: np.ndarray, display_size_xy: tuple[int, int]
) -> np.ndarray:
    """Center-pad a physical LCOS phase onto a display without interpolation."""
    phase_array = np.asarray(phase)
    if phase_array.ndim != 2:
        raise ValueError("SLM phase must be a two-dimensional grayscale image")
    display_width, display_height = display_size_xy
    phase_height, phase_width = phase_array.shape
    if phase_width > display_width or phase_height > display_height:
        raise ValueError(
            f"SLM phase {phase_width}x{phase_height} does not fit display "
            f"transport {display_width}x{display_height}"
        )

    frame = np.zeros((display_height, display_width), dtype=np.uint8)
    start_x = (display_width - phase_width) // 2
    start_y = (display_height - phase_height) // 2
    frame[
        start_y : start_y + phase_height,
        start_x : start_x + phase_width,
    ] = phase_array.astype(np.uint8)
    return frame
