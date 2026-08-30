"""Acquire and analyze a physical delta-z scan with an SLM and AVT camera.

The optical WGS scan remains in :mod:`delta_z_scan`.  This module consumes its
ordered BMP outputs.  Hardware libraries are imported lazily so scan manifests
and CCD data can be analyzed on a computer without wxPython or Vimba X.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ScanPoint:
    """One independently optimized phase pattern in hardware display order."""

    delta_z_mm: float
    bmp_path: Path
    scan_label: str


@dataclass(frozen=True)
class AcquiredPoint:
    """A camera average and its provenance for one displayed phase pattern."""

    point: ScanPoint
    average: np.ndarray
    saturation_fraction: float
    peak_raw: float
    average_npy: Path
    average_tiff: Path
    raw_frames_npy: Path | None


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


def _prepare_experiment_directory(output_dir: str | Path) -> Path:
    output_path = Path(output_dir)
    if output_path.exists() and not output_path.is_dir():
        raise NotADirectoryError(
            f"experiment output path is not a directory: {output_path}"
        )
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            "experimental scan outputs already exist; choose a fresh output "
            f"directory: {output_path}"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _saturation_statistics(
    frame_stack: np.ndarray, saturation_level: float | None
) -> tuple[float, float]:
    peak_raw = float(np.max(frame_stack))
    if saturation_level is None:
        if np.issubdtype(frame_stack.dtype, np.integer):
            saturation_level = float(np.iinfo(frame_stack.dtype).max)
        else:
            return peak_raw, 0.0
    if saturation_level <= 0:
        raise ValueError("saturation_level must be positive")
    saturation_fraction = float(np.mean(frame_stack >= saturation_level))
    return peak_raw, saturation_fraction


def acquire_scan_points(
    points: list[ScanPoint],
    display,
    camera,
    output_dir: str | Path,
    *,
    exposure_us: float,
    frames_per_point: int,
    settle_seconds: float,
    correction: np.ndarray | None,
    lut: int,
    save_raw_frames: bool,
    saturation_level: float | None = None,
    sleep_fn=time.sleep,
) -> list[AcquiredPoint]:
    """Display every phase and acquire a fixed-exposure AVT frame average."""
    if not points:
        raise ValueError("at least one scan point is required")
    if frames_per_point <= 0:
        raise ValueError("frames_per_point must be positive")
    if exposure_us <= 0:
        raise ValueError("exposure_us must be positive")
    if settle_seconds < 0:
        raise ValueError("settle_seconds must be non-negative")

    output_path = _prepare_experiment_directory(output_dir)
    display_size_xy = tuple(int(value) for value in display.getSize())
    if len(display_size_xy) != 2:
        raise ValueError("display getSize() must return (width, height)")

    acquired: list[AcquiredPoint] = []
    camera_shape: tuple[int, int] | None = None
    for point in points:
        with Image.open(point.bmp_path) as bitmap:
            phase = np.asarray(bitmap.convert("L"), dtype=np.uint8)
        calibrated = apply_slm_calibration(
            phase, correction=correction, lut=lut
        )
        display_frame = center_phase_on_display(calibrated, display_size_xy)
        display.updateArray(display_frame)
        sleep_fn(settle_seconds)

        frames = []
        point_shape: tuple[int, int] | None = None
        for _ in range(frames_per_point):
            frame = np.asarray(camera.capture(exposure_us))
            if frame.ndim == 3 and frame.shape[-1] == 1:
                frame = frame[..., 0]
            if frame.ndim != 2:
                raise ValueError("camera frame must be a two-dimensional image")
            if point_shape is None:
                point_shape = tuple(frame.shape)
            elif tuple(frame.shape) != point_shape:
                raise ValueError(
                    "camera frame shape changed during one scan point: "
                    f"{point_shape} -> {tuple(frame.shape)}"
                )
            frames.append(frame.copy())

        frame_stack = np.stack(frames)
        if camera_shape is None:
            camera_shape = tuple(frame_stack.shape[1:])
        elif tuple(frame_stack.shape[1:]) != camera_shape:
            raise ValueError(
                "camera frame shape changed between scan points: "
                f"{camera_shape} -> {tuple(frame_stack.shape[1:])}"
            )

        average = frame_stack.astype(np.float64).mean(axis=0)
        peak_raw, saturation_fraction = _saturation_statistics(
            frame_stack, saturation_level
        )
        average_npy = output_path / f"camera_average_{point.scan_label}.npy"
        average_tiff = output_path / f"camera_average_{point.scan_label}.tiff"
        np.save(average_npy, average)
        average_uint16 = np.clip(np.rint(average), 0, 65535).astype(np.uint16)
        Image.fromarray(average_uint16).save(average_tiff)

        raw_frames_npy = None
        if save_raw_frames:
            raw_frames_npy = (
                output_path / f"camera_raw_{point.scan_label}.npy"
            )
            np.save(raw_frames_npy, frame_stack)

        acquired.append(
            AcquiredPoint(
                point=point,
                average=average,
                saturation_fraction=saturation_fraction,
                peak_raw=peak_raw,
                average_npy=average_npy,
                average_tiff=average_tiff,
                raw_frames_npy=raw_frames_npy,
            )
        )
    return acquired
