"""Acquire and analyze a physical delta-z scan with an SLM and AVT camera.

The optical WGS scan remains in :mod:`delta_z_scan`.  This module consumes its
ordered BMP outputs.  Hardware libraries are imported lazily so scan manifests
and CCD data can be analyzed on a computer without wxPython or Vimba X.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Callable, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


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
    saturation_fraction_source: str = "raw_frames"
    saturation_fraction_is_exact: bool = True
    peak_intensity_source: str = "raw_frames"
    average_peak: float | None = None
    acquisition_exposure_us: float | None = None
    acquisition_frames_per_point: int | None = None
    effective_saturation_level: float | None = None


@dataclass(frozen=True)
class _SpotWindow:
    """One small circular spot mask, indexed relative to the analysis ROI."""

    y_slice: slice
    x_slice: slice
    mask: np.ndarray
    global_y_start: int
    global_x_start: int


@dataclass(frozen=True)
class _CameraAverage:
    average: np.ndarray
    frame_dtype: np.dtype
    frame_shape: tuple[int, int]
    effective_saturation_level: float | None
    peak_raw: float
    saturation_fraction: float


@dataclass(frozen=True)
class _ExistingPointInfo:
    point: ScanPoint
    average_npy: Path
    average_tiff: Path
    raw_frames_npy: Path | None
    saturation_fraction: float
    peak_raw: float
    saturation_fraction_source: str
    saturation_fraction_is_exact: bool
    peak_intensity_source: str
    average_peak: float
    acquisition_exposure_us: float | None
    acquisition_frames_per_point: int | None
    effective_saturation_level: float | None


ProgressFn = Callable[[str], None] | None


def _report(progress_fn: ProgressFn, message: str) -> None:
    if progress_fn is not None:
        progress_fn(message)


def _close_memmap(array: np.ndarray) -> None:
    memory_map = getattr(array, "_mmap", None)
    if memory_map is not None and not memory_map.closed:
        memory_map.close()


def _close_acquired_memmaps(acquired: Sequence[AcquiredPoint]) -> None:
    for result in acquired:
        _close_memmap(result.average)


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


def _effective_saturation_level(
    dtype: np.dtype, saturation_level: float | None
) -> float | None:
    if saturation_level is not None:
        if saturation_level <= 0:
            raise ValueError("saturation_level must be positive")
        return float(saturation_level)
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return None


def _saturation_statistics(
    frames: np.ndarray, saturation_level: float | None
) -> tuple[float, float]:
    """Calculate raw statistics frame-by-frame without a full boolean copy."""
    frame_array = np.asarray(frames)
    if frame_array.ndim not in (2, 3):
        raise ValueError("camera frames must be a 2-D image or 3-D stack")
    effective_level = _effective_saturation_level(
        frame_array.dtype, saturation_level
    )
    iterable = frame_array[None, ...] if frame_array.ndim == 2 else frame_array
    peak_raw = -np.inf
    saturated_pixels = 0
    total_pixels = 0
    for frame in iterable:
        peak_raw = max(peak_raw, float(np.max(frame)))
        if effective_level is not None:
            saturated_pixels += int(np.count_nonzero(frame >= effective_level))
        total_pixels += int(frame.size)
    fraction = (
        saturated_pixels / total_pixels if effective_level is not None else 0.0
    )
    return float(peak_raw), float(fraction)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def _temporary_numpy_path(path: Path) -> Path:
    return path.with_name(f".{path.stem}.tmp{path.suffix}")


def _save_numpy_atomic(path: Path, array: np.ndarray) -> None:
    temporary_path = _temporary_numpy_path(path)
    np.save(temporary_path, array)
    temporary_path.replace(path)


def _save_average_tiff(
    average: np.ndarray, path: Path, source_dtype: np.dtype
) -> None:
    """Preserve an 8-bit camera as 8-bit so ordinary viewers are not black."""
    if np.issubdtype(source_dtype, np.integer):
        if np.iinfo(source_dtype).max <= 255:
            pixels = np.clip(np.rint(average), 0, 255).astype(np.uint8)
            Image.fromarray(pixels, mode="L").save(path)
        else:
            pixels = np.clip(np.rint(average), 0, 65535).astype(np.uint16)
            Image.fromarray(pixels).save(path)
    else:
        Image.fromarray(np.asarray(average, dtype=np.float32), mode="F").save(
            path
        )


def _save_preview_png(
    image: np.ndarray, path: Path, *, upper: float | None = None
) -> None:
    image_array = np.asarray(image)
    if upper is None:
        upper = float(np.max(image_array))
    preview = np.empty(image_array.shape, dtype=np.uint8)
    if not np.isfinite(upper) or upper <= 0:
        preview.fill(0)
    else:
        rows_per_chunk = max(1, 1_000_000 // max(1, image_array.shape[1]))
        scale = np.float32(255.0 / upper)
        for row_start in range(0, image_array.shape[0], rows_per_chunk):
            row_stop = min(image_array.shape[0], row_start + rows_per_chunk)
            scaled = np.array(
                image_array[row_start:row_stop], dtype=np.float32, copy=True
            )
            np.multiply(scaled, scale, out=scaled)
            np.clip(scaled, 0, 255, out=scaled)
            preview[row_start:row_stop] = scaled.astype(np.uint8)
    Image.fromarray(preview, mode="L").save(path)


def _capture_camera_average(
    camera,
    *,
    exposure_us: float,
    frames_per_point: int,
    expected_shape: tuple[int, int] | None,
    saturation_level: float | None,
    raw_frames_npy: Path | None,
) -> _CameraAverage:
    point_shape: tuple[int, int] | None = None
    frame_dtype: np.dtype | None = None
    average: np.ndarray | None = None
    raw_frames: np.memmap | None = None
    raw_temporary_path = (
        _temporary_numpy_path(raw_frames_npy)
        if raw_frames_npy is not None
        else None
    )
    effective_level: float | None = None
    peak_raw = -np.inf
    saturated_pixels = 0
    total_pixels = 0
    try:
        for frame_index in range(frames_per_point):
            frame = np.asarray(camera.capture(exposure_us))
            if frame.ndim == 3 and frame.shape[-1] == 1:
                frame = frame[..., 0]
            if frame.ndim != 2:
                raise ValueError("camera frame must be a two-dimensional image")
            if not np.issubdtype(frame.dtype, np.number):
                raise ValueError("camera frame must have a numeric dtype")
            if point_shape is None:
                point_shape = tuple(int(value) for value in frame.shape)
                frame_dtype = frame.dtype
                if expected_shape is not None and point_shape != expected_shape:
                    raise ValueError(
                        "camera frame shape changed between scan points: "
                        f"{expected_shape} -> {point_shape}"
                    )
                average = np.zeros(point_shape, dtype=np.float32)
                effective_level = _effective_saturation_level(
                    frame_dtype, saturation_level
                )
                if raw_temporary_path is not None:
                    raw_frames = np.lib.format.open_memmap(
                        raw_temporary_path,
                        mode="w+",
                        dtype=frame_dtype,
                        shape=(frames_per_point, *point_shape),
                    )
            elif tuple(frame.shape) != point_shape:
                raise ValueError(
                    "camera frame shape changed during one scan point: "
                    f"{point_shape} -> {tuple(frame.shape)}"
                )
            elif frame.dtype != frame_dtype:
                raise ValueError(
                    "camera frame dtype changed during one scan point: "
                    f"{frame_dtype} -> {frame.dtype}"
                )
            if average is None:
                raise RuntimeError("camera average accumulator was not initialized")
            frame_minimum = float(np.min(frame))
            frame_maximum = float(np.max(frame))
            if not np.isfinite(frame_minimum) or not np.isfinite(frame_maximum):
                raise ValueError("camera frame contains non-finite values")
            np.add(average, frame, out=average, casting="unsafe")
            if raw_frames is not None:
                raw_frames[frame_index] = frame
            peak_raw = max(peak_raw, frame_maximum)
            if effective_level is not None:
                saturated_pixels += int(
                    np.count_nonzero(frame >= effective_level)
                )
            total_pixels += int(frame.size)

        if point_shape is None or frame_dtype is None or average is None:
            raise RuntimeError("no camera frames were acquired")
        average *= np.float32(1.0 / frames_per_point)
        saturation_fraction = (
            saturated_pixels / total_pixels
            if effective_level is not None
            else 0.0
        )
        if raw_frames is not None and raw_temporary_path is not None:
            raw_frames.flush()
            _close_memmap(raw_frames)
            raw_frames = None
            raw_temporary_path.replace(raw_frames_npy)
        return _CameraAverage(
            average=average,
            frame_dtype=frame_dtype,
            frame_shape=point_shape,
            effective_saturation_level=effective_level,
            peak_raw=float(peak_raw),
            saturation_fraction=float(saturation_fraction),
        )
    except BaseException:
        if raw_frames is not None:
            _close_memmap(raw_frames)
        if raw_temporary_path is not None:
            raw_temporary_path.unlink(missing_ok=True)
        raise


def _acquire_one_scan_point(
    point: ScanPoint,
    display,
    camera,
    output_path: Path,
    display_size_xy: tuple[int, int],
    expected_camera_shape: tuple[int, int] | None,
    *,
    exposure_us: float,
    frames_per_point: int,
    settle_seconds: float,
    correction: np.ndarray | None,
    lut: int,
    save_raw_frames: bool,
    saturation_level: float | None,
    sleep_fn,
) -> AcquiredPoint:
    with Image.open(point.bmp_path) as bitmap:
        phase = np.asarray(bitmap.convert("L"), dtype=np.uint8)
    calibrated = apply_slm_calibration(phase, correction=correction, lut=lut)
    display_frame = center_phase_on_display(calibrated, display_size_xy)
    display.updateArray(display_frame)
    sleep_fn(settle_seconds)

    raw_frames_npy = (
        output_path / f"camera_raw_{point.scan_label}.npy"
        if save_raw_frames
        else None
    )
    captured = _capture_camera_average(
        camera,
        exposure_us=exposure_us,
        frames_per_point=frames_per_point,
        expected_shape=expected_camera_shape,
        saturation_level=saturation_level,
        raw_frames_npy=raw_frames_npy,
    )
    average_peak = float(np.max(captured.average))
    average_npy = output_path / f"camera_average_{point.scan_label}.npy"
    average_tiff = output_path / f"camera_average_{point.scan_label}.tiff"
    _save_numpy_atomic(average_npy, captured.average)

    statistics_path = output_path / f"camera_stats_{point.scan_label}.json"
    _write_json_atomic(
        statistics_path,
        {
            "scan_label": point.scan_label,
            "delta_z_mm": point.delta_z_mm,
            "frame_dtype": str(captured.frame_dtype),
            "frame_shape_yx": list(captured.frame_shape),
            "frames_per_point": frames_per_point,
            "exposure_us": exposure_us,
            "effective_saturation_level": (
                captured.effective_saturation_level
            ),
            "peak_raw": captured.peak_raw,
            "saturation_fraction": captured.saturation_fraction,
            "statistics_source": "live_frames",
        },
    )

    _save_average_tiff(captured.average, average_tiff, captured.frame_dtype)
    preview_path = output_path / f"camera_preview_{point.scan_label}.png"
    _save_preview_png(captured.average, preview_path)

    average_mmap: np.memmap | None = None
    try:
        average_mmap = np.load(
            average_npy, mmap_mode="r", allow_pickle=False
        )
        return AcquiredPoint(
            point=point,
            average=average_mmap,
            saturation_fraction=captured.saturation_fraction,
            peak_raw=captured.peak_raw,
            average_npy=average_npy,
            average_tiff=average_tiff,
            raw_frames_npy=raw_frames_npy,
            saturation_fraction_source="live_frames",
            saturation_fraction_is_exact=(
                captured.effective_saturation_level is not None
            ),
            peak_intensity_source="live_frames",
            average_peak=average_peak,
            acquisition_exposure_us=float(exposure_us),
            acquisition_frames_per_point=frames_per_point,
            effective_saturation_level=captured.effective_saturation_level,
        )
    except BaseException:
        if average_mmap is not None:
            _close_memmap(average_mmap)
        raise


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
    progress_fn: ProgressFn = print,
) -> list[AcquiredPoint]:
    """Display phases and stream fixed-exposure frames into float32 averages."""
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
    point_count = len(points)
    for point_index, point in enumerate(points, start=1):
        _report(
            progress_fn,
            f"[acquire {point_index}/{point_count}] Displaying "
            f"{point.scan_label}",
        )
        try:
            result = _acquire_one_scan_point(
                point,
                display,
                camera,
                output_path,
                display_size_xy,
                camera_shape,
                exposure_us=exposure_us,
                frames_per_point=frames_per_point,
                settle_seconds=settle_seconds,
                correction=correction,
                lut=lut,
                save_raw_frames=save_raw_frames,
                saturation_level=saturation_level,
                sleep_fn=sleep_fn,
            )
        except BaseException:
            _close_acquired_memmaps(acquired)
            raise
        if camera_shape is None:
            camera_shape = tuple(int(value) for value in result.average.shape)
        acquired.append(result)
        _report(
            progress_fn,
            f"[acquire {point_index}/{point_count}] Saved camera average "
            f"(peak={result.peak_raw:g}, "
            f"saturation={result.saturation_fraction:.3g})",
        )
    return acquired


def _validate_background_percentile(background_percentile: float) -> None:
    if not 0 <= background_percentile < 100:
        raise ValueError("background_percentile must be in [0, 100)")


def _background_correct(
    image: np.ndarray, background_percentile: float
) -> np.ndarray:
    _validate_background_percentile(background_percentile)
    image_array = np.array(image, dtype=np.float32, copy=True)
    if image_array.ndim != 2:
        raise ValueError("camera image must be two-dimensional")
    image_minimum = float(np.min(image_array))
    image_maximum = float(np.max(image_array))
    if not np.isfinite(image_minimum) or not np.isfinite(image_maximum):
        raise ValueError("camera image contains non-finite values")
    baseline = float(np.percentile(image_array, background_percentile))
    np.subtract(image_array, baseline, out=image_array)
    np.maximum(image_array, 0.0, out=image_array)
    return image_array


def _roi_slices(
    image_shape: tuple[int, int],
    roi_xywh: tuple[int, int, int, int] | None,
) -> tuple[slice, slice]:
    height, width = image_shape
    if roi_xywh is None:
        return slice(0, height), slice(0, width)
    if len(roi_xywh) != 4:
        raise ValueError("roi_xywh must contain x, y, width, height")
    x, y, roi_width, roi_height = (int(value) for value in roi_xywh)
    if x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0:
        raise ValueError("camera ROI must have a non-negative origin and size")
    if x + roi_width > width or y + roi_height > height:
        raise ValueError("camera ROI lies outside the acquired image")
    return slice(y, y + roi_height), slice(x, x + roi_width)


def detect_common_spots(
    images: Sequence[np.ndarray],
    expected_count: int,
    *,
    roi_xywh: tuple[int, int, int, int] | None = None,
    min_peak_distance_px: int = 8,
    background_percentile: float = 10.0,
    progress_fn: ProgressFn = None,
) -> np.ndarray:
    """Detect fixed centers from an equal-weight, ROI-sized reference image."""
    if not images:
        raise ValueError("at least one camera image is required")
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if min_peak_distance_px <= 0:
        raise ValueError("min_peak_distance_px must be positive")

    first_shape = tuple(np.asarray(images[0]).shape)
    if len(first_shape) != 2:
        raise ValueError("camera image must be two-dimensional")
    y_slice, x_slice = _roi_slices(first_shape, roi_xywh)
    roi_shape = (
        int(y_slice.stop - y_slice.start),
        int(x_slice.stop - x_slice.start),
    )
    reference = np.zeros(roi_shape, dtype=np.float32)
    image_count = len(images)
    for image_index, image in enumerate(images, start=1):
        if tuple(np.asarray(image).shape) != first_shape:
            raise ValueError("all camera images must have the same shape")
        corrected = _background_correct(
            np.asarray(image)[y_slice, x_slice], background_percentile
        )
        total = float(np.sum(corrected, dtype=np.float64))
        if total <= 0:
            raise ValueError("camera image has no signal above background")
        corrected *= np.float32(1.0 / total)
        reference += corrected
        _report(
            progress_fn,
            f"[detect {image_index}/{image_count}] Added normalized ROI "
            "to common reference",
        )
    reference *= np.float32(1.0 / image_count)

    filter_size = 2 * min_peak_distance_px + 1
    local_maximum = ndimage.maximum_filter(
        reference,
        size=filter_size,
        mode="constant",
        cval=-np.inf,
    )
    maxima_mask = (reference == local_maximum) & (reference > 0)
    labels, component_count = ndimage.label(maxima_mask)
    candidates = []
    component_slices = ndimage.find_objects(labels, max_label=component_count)
    for label_index, component_slice in enumerate(component_slices, start=1):
        if component_slice is None:
            continue
        component_labels = labels[component_slice]
        component_reference = reference[component_slice]
        component_values = np.where(
            component_labels == label_index, component_reference, -np.inf
        )
        local_flat_index = int(np.argmax(component_values))
        local_y, local_x = np.unravel_index(
            local_flat_index, component_values.shape
        )
        roi_y = int(component_slice[0].start + local_y)
        roi_x = int(component_slice[1].start + local_x)
        candidate_y = int(roi_y + y_slice.start)
        candidate_x = int(roi_x + x_slice.start)
        candidates.append(
            (float(reference[roi_y, roi_x]), candidate_y, candidate_x)
        )
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    selected: list[tuple[int, int]] = []
    minimum_distance_squared = float(min_peak_distance_px**2)
    for _, candidate_y, candidate_x in candidates:
        if all(
            (candidate_y - selected_y) ** 2
            + (candidate_x - selected_x) ** 2
            >= minimum_distance_squared
            for selected_y, selected_x in selected
        ):
            selected.append((candidate_y, candidate_x))
        if len(selected) == expected_count:
            break
    if len(selected) != expected_count:
        raise ValueError(
            f"detected {len(selected)} spots, expected {expected_count}; "
            "adjust the camera ROI or minimum peak distance"
        )
    return np.asarray(sorted(selected), dtype=int)


def _infer_spot_radius(centers: np.ndarray) -> float:
    if len(centers) < 2:
        raise ValueError("spot_radius_px is required for a single target spot")
    differences = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt(np.sum(np.square(differences), axis=-1))
    np.fill_diagonal(distances, np.inf)
    nearest_neighbor_distance = float(np.median(np.min(distances, axis=1)))
    if not np.isfinite(nearest_neighbor_distance) or nearest_neighbor_distance <= 0:
        raise ValueError("spot centers must be distinct")
    return max(2.0, 0.35 * nearest_neighbor_distance)


def _build_spot_windows(
    image_shape: tuple[int, int],
    centers: np.ndarray,
    radius: float,
    roi_xywh: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, list[_SpotWindow], tuple[int, int]]:
    """Build one ROI union mask and one tiny mask per target spot."""
    height, width = image_shape
    center_array = np.asarray(centers, dtype=int)
    if np.any(center_array[:, 0] < 0) or np.any(center_array[:, 0] >= height):
        raise ValueError("spot center y coordinate lies outside the image")
    if np.any(center_array[:, 1] < 0) or np.any(center_array[:, 1] >= width):
        raise ValueError("spot center x coordinate lies outside the image")

    roi_y_slice, roi_x_slice = _roi_slices(image_shape, roi_xywh)
    roi_y_start = int(roi_y_slice.start)
    roi_y_stop = int(roi_y_slice.stop)
    roi_x_start = int(roi_x_slice.start)
    roi_x_stop = int(roi_x_slice.stop)
    roi_shape = (
        roi_y_stop - roi_y_start,
        roi_x_stop - roi_x_start,
    )
    target_mask = np.zeros(roi_shape, dtype=bool)
    windows: list[_SpotWindow] = []
    half_width = int(np.ceil(radius))
    for center_y, center_x in center_array:
        if not (
            roi_y_start <= center_y < roi_y_stop
            and roi_x_start <= center_x < roi_x_stop
        ):
            raise ValueError("spot center lies outside the camera ROI")
        global_y_start = max(roi_y_start, int(center_y) - half_width)
        global_y_stop = min(roi_y_stop, int(center_y) + half_width + 1)
        global_x_start = max(roi_x_start, int(center_x) - half_width)
        global_x_stop = min(roi_x_stop, int(center_x) + half_width + 1)
        y_coordinates = np.arange(global_y_start, global_y_stop)[:, None]
        x_coordinates = np.arange(global_x_start, global_x_stop)[None, :]
        mask = (
            (y_coordinates - center_y) ** 2
            + (x_coordinates - center_x) ** 2
            <= radius**2
        )
        local_y_slice = slice(
            global_y_start - roi_y_start, global_y_stop - roi_y_start
        )
        local_x_slice = slice(
            global_x_start - roi_x_start, global_x_stop - roi_x_start
        )
        target_region = target_mask[local_y_slice, local_x_slice]
        np.logical_or(target_region, mask, out=target_region)
        windows.append(
            _SpotWindow(
                y_slice=local_y_slice,
                x_slice=local_x_slice,
                mask=mask,
                global_y_start=global_y_start,
                global_x_start=global_x_start,
            )
        )
    return target_mask, windows, (roi_y_start, roi_x_start)


def _spot_shape_metrics(
    signal_patch: np.ndarray,
    mask: np.ndarray,
    global_y_start: int,
    global_x_start: int,
) -> tuple[float, float, float]:
    values = signal_patch[mask]
    spot_sum = float(np.sum(values, dtype=np.float64))
    if values.size == 0 or spot_sum <= 0:
        return float("inf"), float("inf"), 0.0

    local_y, local_x = np.nonzero(mask)
    mask_y = local_y + global_y_start
    mask_x = local_x + global_x_start
    peak_index = int(np.argmax(values))
    peak_y = float(mask_y[peak_index])
    peak_x = float(mask_x[peak_index])
    half_maximum_area = int(np.count_nonzero(values >= 0.5 * values[peak_index]))
    equivalent_fwhm = 2.0 * np.sqrt(half_maximum_area / np.pi)

    distances = np.sqrt((mask_y - peak_y) ** 2 + (mask_x - peak_x) ** 2)
    order = np.argsort(distances)
    ordered_values = values[order]
    cumulative = np.cumsum(ordered_values, dtype=np.float64)
    radius_index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    radius_index = min(radius_index, len(order) - 1)
    encircled_energy_radius_50 = float(distances[order[radius_index]])

    sharpness = float(
        np.sum(np.square(values), dtype=np.float64) / spot_sum**2
    )
    return float(equivalent_fwhm), encircled_energy_radius_50, sharpness


def analyze_acquired_points(
    acquired: list[AcquiredPoint],
    centers: np.ndarray,
    *,
    spot_radius_px: float | None = None,
    roi_xywh: tuple[int, int, int, int] | None = None,
    background_percentile: float = 10.0,
    progress_fn: ProgressFn = None,
) -> tuple[list[dict[str, float | str]], float]:
    """Measure each scan point with one fixed set of circular target ROIs."""
    if not acquired:
        raise ValueError("at least one acquired scan point is required")
    center_array = np.asarray(centers, dtype=int)
    if center_array.ndim != 2 or center_array.shape[1] != 2 or not len(center_array):
        raise ValueError("centers must be a non-empty array of (y, x) points")
    radius = (
        _infer_spot_radius(center_array)
        if spot_radius_px is None
        else float(spot_radius_px)
    )
    if radius <= 0:
        raise ValueError("spot_radius_px must be positive")

    image_shape = tuple(np.asarray(acquired[0].average).shape)
    if len(image_shape) != 2:
        raise ValueError("camera image must be two-dimensional")
    target_mask, spot_windows, (roi_y_start, roi_x_start) = (
        _build_spot_windows(image_shape, center_array, radius, roi_xywh)
    )
    roi_y_slice, roi_x_slice = _roi_slices(image_shape, roi_xywh)
    roi_height, roi_width = target_mask.shape

    rows: list[dict[str, float | str]] = []
    result_count = len(acquired)
    for result_index, result in enumerate(acquired, start=1):
        if tuple(np.asarray(result.average).shape) != image_shape:
            raise ValueError("all acquired camera images must have the same shape")
        signal = _background_correct(
            np.asarray(result.average)[roi_y_slice, roi_x_slice],
            background_percentile,
        )
        total_signal = float(np.sum(signal, dtype=np.float64))
        if total_signal <= 0:
            raise ValueError(
                f"camera image has no signal above background: "
                f"{result.point.scan_label}"
            )

        spot_sums = np.asarray(
            [
                float(
                    np.sum(
                        signal[window.y_slice, window.x_slice][window.mask],
                        dtype=np.float64,
                    )
                )
                for window in spot_windows
            ]
        )
        maximum_spot_sum = float(np.max(spot_sums))
        target_uniformity = (
            float(np.min(spot_sums) / maximum_spot_sum)
            if maximum_spot_sum > 0
            else 0.0
        )
        mean_spot_sum = float(np.mean(spot_sums))
        spot_cv = (
            float(np.std(spot_sums) / mean_spot_sum)
            if mean_spot_sum > 0
            else float("inf")
        )
        target_signal = float(
            np.sum(signal[target_mask], dtype=np.float64)
        )
        target_efficiency = target_signal / total_signal
        background_halo = 1.0 - target_efficiency

        row_sums = np.sum(signal, axis=1, dtype=np.float64)
        column_sums = np.sum(signal, axis=0, dtype=np.float64)
        centroid_y = float(
            np.dot(
                np.arange(roi_y_start, roi_y_start + roi_height), row_sums
            )
            / total_signal
        )
        centroid_x = float(
            np.dot(
                np.arange(roi_x_start, roi_x_start + roi_width), column_sums
            )
            / total_signal
        )
        peak_flat_index = int(np.argmax(signal))
        local_peak_y, local_peak_x = divmod(peak_flat_index, roi_width)
        peak_y = local_peak_y + roi_y_start
        peak_x = local_peak_x + roi_x_start
        centroid_peak_offset = float(
            np.hypot(centroid_x - peak_x, centroid_y - peak_y)
        )

        per_spot_metrics = [
            _spot_shape_metrics(
                signal[window.y_slice, window.x_slice],
                window.mask,
                window.global_y_start,
                window.global_x_start,
            )
            for window in spot_windows
        ]
        mean_fwhm = float(np.mean([metric[0] for metric in per_spot_metrics]))
        mean_radius_50 = float(
            np.mean([metric[1] for metric in per_spot_metrics])
        )
        mean_sharpness = float(
            np.mean([metric[2] for metric in per_spot_metrics])
        )

        rows.append(
            {
                "delta_z_mm": result.point.delta_z_mm,
                "scan_label": result.point.scan_label,
                "bmp_file": result.point.bmp_path.name,
                "camera_average_file": result.average_npy.name,
                "target_plane_uniformity": target_uniformity,
                "spot_intensity_cv": spot_cv,
                "target_efficiency": target_efficiency,
                "background_halo": background_halo,
                "peak_intensity": result.peak_raw,
                "saturation_fraction": result.saturation_fraction,
                "saturation_fraction_source": (
                    result.saturation_fraction_source
                ),
                "saturation_fraction_is_exact": (
                    result.saturation_fraction_is_exact
                ),
                "peak_intensity_source": result.peak_intensity_source,
                "centroid_x_px": centroid_x,
                "centroid_y_px": centroid_y,
                "peak_x_px": float(peak_x),
                "peak_y_px": float(peak_y),
                "centroid_peak_offset_px": centroid_peak_offset,
                "mean_fwhm_px": mean_fwhm,
                "mean_encircled_energy_radius_50_px": mean_radius_50,
                "mean_spot_sharpness": mean_sharpness,
            }
        )
        _report(
            progress_fn,
            f"[analyze {result_index}/{result_count}] Measured "
            f"{result.point.scan_label}",
        )
    return rows, radius


def _normalize_metric(values: np.ndarray, *, higher_is_better: bool) -> np.ndarray:
    if not np.all(np.isfinite(values)):
        raise ValueError("quality metric contains non-finite values")
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if np.isclose(minimum, maximum):
        normalized = np.full(values.shape, 0.5, dtype=np.float64)
    else:
        normalized = (values - minimum) / (maximum - minimum)
    return normalized if higher_is_better else 1.0 - normalized


def rank_quality(
    rows: list[dict[str, float | str]],
    *,
    maximum_saturation_fraction: float = 0.001,
) -> tuple[list[dict[str, float | str | bool]], dict[str, float | str | bool]]:
    """Add a scan-relative quality score and select the best unsaturated row."""
    if not rows:
        raise ValueError("at least one metric row is required")
    if maximum_saturation_fraction < 0:
        raise ValueError("maximum_saturation_fraction must be non-negative")

    uniformity = np.asarray(
        [float(row["target_plane_uniformity"]) for row in rows]
    )
    sharpness = np.asarray([float(row["mean_spot_sharpness"]) for row in rows])
    halo = np.asarray([float(row["background_halo"]) for row in rows])
    fwhm = np.asarray([float(row["mean_fwhm_px"]) for row in rows])
    components = np.column_stack(
        [
            _normalize_metric(uniformity, higher_is_better=True),
            _normalize_metric(sharpness, higher_is_better=True),
            _normalize_metric(halo, higher_is_better=False),
            _normalize_metric(fwhm, higher_is_better=False),
        ]
    )
    quality_scores = np.mean(components, axis=1)

    ranked_rows: list[dict[str, float | str | bool]] = []
    eligible = []
    for row, score in zip(rows, quality_scores):
        row_copy: dict[str, float | str | bool] = dict(row)
        row_is_eligible = (
            float(row["saturation_fraction"])
            <= maximum_saturation_fraction
        )
        row_copy["quality_score"] = float(score)
        row_copy["eligible_for_best"] = bool(row_is_eligible)
        ranked_rows.append(row_copy)
        eligible.append(row_is_eligible)
    eligible_array = np.asarray(eligible, dtype=bool)
    if not np.any(eligible_array):
        raise ValueError("all scan points are saturated; reduce the exposure")
    eligible_scores = np.where(eligible_array, quality_scores, -np.inf)
    best_index = int(np.argmax(eligible_scores))
    return ranked_rows, ranked_rows[best_index]


def _write_experimental_metrics(
    rows: list[dict[str, float | str | bool]], output_path: Path
) -> None:
    csv_path = output_path / "experimental_metrics.csv"
    temporary_path = output_path / ".experimental_metrics.csv.tmp"
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(csv_path)


def _write_experimental_plot(
    rows: list[dict[str, float | str | bool]], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    delta_z = np.asarray([float(row["delta_z_mm"]) for row in rows])
    series = (
        ("quality_score", "Experimental quality score"),
        ("target_plane_uniformity", "Spot uniformity (min/max)"),
        ("mean_fwhm_px", "Mean equivalent FWHM (px)"),
        ("background_halo", "Background / halo fraction"),
        ("mean_spot_sharpness", "Mean spot sharpness"),
        ("saturation_fraction", "Saturated pixel fraction"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(10, 11), sharex=True)
    for axis, (key, title) in zip(axes.flat, series):
        axis.plot(delta_z, [float(row[key]) for row in rows], "o-")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
    for axis in axes[-1]:
        axis.set_xlabel("delta_z (mm)")
    figure.tight_layout()
    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary_path, dpi=180)
    plt.close(figure)
    temporary_path.replace(path)


def _write_common_previews(
    acquired: Sequence[AcquiredPoint],
    output_path: Path,
    *,
    progress_fn: ProgressFn = None,
) -> None:
    """Write comparable 8-bit previews using one scale for the whole scan."""
    common_upper = max(
        (
            float(result.average_peak)
            if result.average_peak is not None
            else float(np.max(result.average))
        )
        for result in acquired
    )
    result_count = len(acquired)
    for result_index, result in enumerate(acquired, start=1):
        preview_path = (
            output_path / f"camera_preview_{result.point.scan_label}.png"
        )
        _save_preview_png(result.average, preview_path, upper=common_upper)
        _close_memmap(result.average)
        _report(
            progress_fn,
            f"[preview {result_index}/{result_count}] Wrote "
            f"{preview_path.name}",
        )


def _write_spot_overlay(
    acquired: Sequence[AcquiredPoint],
    centers: np.ndarray,
    radius: float,
    path: Path,
    *,
    roi_xywh: tuple[int, int, int, int] | None = None,
    progress_fn: ProgressFn = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    image_shape = tuple(np.asarray(acquired[0].average).shape)
    y_slice, x_slice = _roi_slices(image_shape, roi_xywh)
    reference = np.zeros(
        (y_slice.stop - y_slice.start, x_slice.stop - x_slice.start),
        dtype=np.float32,
    )
    result_count = len(acquired)
    for result_index, result in enumerate(acquired, start=1):
        image = np.array(
            np.asarray(result.average)[y_slice, x_slice],
            dtype=np.float32,
            copy=True,
        )
        scale = float(np.max(image))
        if scale > 0:
            image *= np.float32(1.0 / scale)
        reference += image
        _report(
            progress_fn,
            f"[overlay {result_index}/{result_count}] Added "
            f"{result.point.scan_label}",
        )
    reference *= np.float32(1.0 / result_count)

    figure, axis = plt.subplots(figsize=(8, 7))
    image_handle = axis.imshow(
        reference,
        cmap="gray",
        origin="upper",
        extent=(
            x_slice.start - 0.5,
            x_slice.stop - 0.5,
            y_slice.stop - 0.5,
            y_slice.start - 0.5,
        ),
    )
    for index, (center_y, center_x) in enumerate(centers):
        axis.add_patch(
            Circle(
                (center_x, center_y),
                radius,
                fill=False,
                edgecolor="tab:red",
                linewidth=0.8,
            )
        )
        axis.text(
            center_x,
            center_y,
            str(index),
            color="yellow",
            fontsize=6,
            ha="center",
            va="center",
        )
    axis.set_title("Common CCD spot ROIs used for every delta_z")
    figure.colorbar(image_handle, ax=axis, shrink=0.8)
    figure.tight_layout()
    temporary_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    figure.savefig(temporary_path, dpi=180)
    plt.close(figure)
    temporary_path.replace(path)


def _load_source_scan_parameters(scan_directory: Path) -> dict[str, object] | None:
    parameters_path = scan_directory / "scan_parameters.json"
    if not parameters_path.is_file():
        return None
    with parameters_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_point_statistics(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"camera statistics must contain an object: {path}")
    return payload


def _validate_point_statistics(
    payload: dict[str, object],
    path: Path,
    point: ScanPoint,
    image_shape: tuple[int, int],
) -> None:
    required = {
        "scan_label",
        "delta_z_mm",
        "frame_dtype",
        "frame_shape_yx",
        "frames_per_point",
        "exposure_us",
        "effective_saturation_level",
        "peak_raw",
        "saturation_fraction",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(
            f"camera statistics are missing {sorted(missing)}: {path}"
        )
    try:
        delta_z_mm = float(payload["delta_z_mm"])
        frames_per_point = int(payload["frames_per_point"])
        exposure_us = float(payload["exposure_us"])
        peak_raw = float(payload["peak_raw"])
        saturation_fraction = float(payload["saturation_fraction"])
        frame_shape = tuple(int(value) for value in payload["frame_shape_yx"])
        saved_level = payload["effective_saturation_level"]
        effective_level = (
            None if saved_level is None else float(saved_level)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"camera statistics contain invalid values: {path}") from error
    if payload["scan_label"] != point.scan_label:
        raise ValueError(f"camera statistics scan_label does not match: {path}")
    if not np.isfinite(delta_z_mm) or not np.isclose(
        delta_z_mm, point.delta_z_mm
    ):
        raise ValueError(f"camera statistics delta_z does not match: {path}")
    if frame_shape != image_shape:
        raise ValueError(f"camera statistics frame shape does not match: {path}")
    if frames_per_point <= 0 or frames_per_point != payload["frames_per_point"]:
        raise ValueError(f"camera statistics frame count is invalid: {path}")
    if not np.isfinite(exposure_us) or exposure_us <= 0:
        raise ValueError(f"camera statistics exposure is invalid: {path}")
    if not isinstance(payload["frame_dtype"], str) or not payload["frame_dtype"]:
        raise ValueError(f"camera statistics frame dtype is invalid: {path}")
    if effective_level is not None and (
        not np.isfinite(effective_level) or effective_level <= 0
    ):
        raise ValueError(f"camera statistics saturation level is invalid: {path}")
    if not np.isfinite(peak_raw):
        raise ValueError(f"camera statistics peak is invalid: {path}")
    if not np.isfinite(saturation_fraction) or not 0 <= saturation_fraction <= 1:
        raise ValueError(
            f"camera statistics saturation fraction is invalid: {path}"
        )


def _inspect_existing_point(
    point: ScanPoint,
    output_path: Path,
    expected_shape: tuple[int, int] | None,
    saturation_level: float | None,
) -> tuple[_ExistingPointInfo, tuple[int, int]]:
    average_npy = output_path / f"camera_average_{point.scan_label}.npy"
    if not average_npy.is_file():
        raise FileNotFoundError(
            f"camera average missing for {point.scan_label}: {average_npy}"
        )

    average: np.memmap | None = None
    raw_frames: np.memmap | None = None
    try:
        try:
            average = np.load(
                average_npy, mmap_mode="r", allow_pickle=False
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                f"cannot load camera average for {point.scan_label}: "
                f"{average_npy}"
            ) from error
        if average.ndim != 2:
            raise ValueError(
                f"camera average for {point.scan_label} must be two-dimensional"
            )
        if not np.issubdtype(average.dtype, np.number):
            raise ValueError(
                f"camera average for {point.scan_label} must be numeric"
            )
        current_shape = tuple(int(value) for value in average.shape)
        if expected_shape is not None and current_shape != expected_shape:
            raise ValueError(
                "camera average shape changed between scan points: "
                f"{expected_shape} -> {current_shape} ({point.scan_label})"
            )
        average_minimum = float(np.min(average))
        average_peak = float(np.max(average))
        if not np.isfinite(average_minimum) or not np.isfinite(average_peak):
            raise ValueError(
                f"camera average contains non-finite values: {point.scan_label}"
            )

        raw_path = output_path / f"camera_raw_{point.scan_label}.npy"
        raw_frames_npy = raw_path if raw_path.is_file() else None
        stats_path = output_path / f"camera_stats_{point.scan_label}.json"
        statistics = _load_point_statistics(stats_path)
        if statistics is not None:
            _validate_point_statistics(
                statistics, stats_path, point, current_shape
            )
        saved_level = (
            statistics.get("effective_saturation_level")
            if statistics is not None
            else None
        )
        acquisition_exposure_us = (
            float(statistics["exposure_us"])
            if statistics is not None
            else None
        )
        acquisition_frames_per_point = (
            int(statistics["frames_per_point"])
            if statistics is not None
            else None
        )

        if raw_frames_npy is not None:
            try:
                raw_frames = np.load(
                    raw_frames_npy, mmap_mode="r", allow_pickle=False
                )
            except (OSError, ValueError) as error:
                raise ValueError(
                    f"cannot load raw frames for {point.scan_label}: "
                    f"{raw_frames_npy}"
                ) from error
            if raw_frames.ndim != 3 or tuple(raw_frames.shape[1:]) != current_shape:
                raise ValueError(
                    f"raw camera stack shape is invalid for {point.scan_label}"
                )
            if statistics is not None and (
                str(raw_frames.dtype) != str(statistics["frame_dtype"])
                or int(raw_frames.shape[0])
                != int(statistics["frames_per_point"])
            ):
                raise ValueError(
                    "raw camera stack does not match camera statistics for "
                    f"{point.scan_label}"
                )
            if acquisition_frames_per_point is None:
                acquisition_frames_per_point = int(raw_frames.shape[0])
            raw_saturation_level = (
                saturation_level
                if saturation_level is not None
                else (float(saved_level) if saved_level is not None else None)
            )
            peak_raw, saturation_fraction = _saturation_statistics(
                raw_frames, raw_saturation_level
            )
            effective_level = _effective_saturation_level(
                raw_frames.dtype, raw_saturation_level
            )
            saturation_source = "raw_frames"
            saturation_is_exact = effective_level is not None
            peak_source = "raw_frames"
        else:
            compatible_statistics = statistics is not None and (
                saturation_level is None
                or (
                    saved_level is not None
                    and float(saved_level) == float(saturation_level)
                )
            )
            if compatible_statistics:
                peak_raw = float(statistics["peak_raw"])
                saturation_fraction = float(
                    statistics["saturation_fraction"]
                )
                saturation_source = "acquisition_sidecar"
                saturation_is_exact = saved_level is not None
                peak_source = "live_frames"
                effective_level = (
                    float(saved_level) if saved_level is not None else None
                )
            else:
                if saturation_level is None:
                    raise ValueError(
                        "--saturation-level is required when existing averages "
                        "have no raw frames or compatible camera_stats files"
                    )
                peak_raw = average_peak
                saturation_fraction = float(
                    np.count_nonzero(average >= saturation_level) / average.size
                )
                saturation_source = "average_threshold_estimate"
                saturation_is_exact = False
                peak_source = "average"
                effective_level = float(saturation_level)

        return (
            _ExistingPointInfo(
                point=point,
                average_npy=average_npy,
                average_tiff=(
                    output_path / f"camera_average_{point.scan_label}.tiff"
                ),
                raw_frames_npy=raw_frames_npy,
                saturation_fraction=float(saturation_fraction),
                peak_raw=float(peak_raw),
                saturation_fraction_source=saturation_source,
                saturation_fraction_is_exact=saturation_is_exact,
                peak_intensity_source=peak_source,
                average_peak=average_peak,
                acquisition_exposure_us=acquisition_exposure_us,
                acquisition_frames_per_point=acquisition_frames_per_point,
                effective_saturation_level=effective_level,
            ),
            current_shape,
        )
    finally:
        if raw_frames is not None:
            _close_memmap(raw_frames)
        if average is not None:
            _close_memmap(average)


def load_existing_acquired_points(
    points: Sequence[ScanPoint],
    acquisition_dir: str | Path,
    *,
    saturation_level: float | None,
    progress_fn: ProgressFn = print,
) -> list[AcquiredPoint]:
    """Memory-map averages left by a completed/interrupted hardware run."""
    if not points:
        raise ValueError("at least one scan point is required")
    if saturation_level is not None and saturation_level <= 0:
        raise ValueError("saturation_level must be positive")
    output_path = Path(acquisition_dir)
    if not output_path.is_dir():
        raise NotADirectoryError(
            f"existing acquisition directory not found: {output_path}"
        )

    inspected: list[_ExistingPointInfo] = []
    image_shape: tuple[int, int] | None = None
    point_count = len(points)
    for point_index, point in enumerate(points, start=1):
        point_info, current_shape = _inspect_existing_point(
            point, output_path, image_shape, saturation_level
        )
        if image_shape is None:
            image_shape = current_shape
        inspected.append(point_info)
        _report(
            progress_fn,
            f"[load {point_index}/{point_count}] Validated "
            f"{point_info.average_npy.name}",
        )

    acquired: list[AcquiredPoint] = []
    try:
        for point_info in inspected:
            average = np.load(
                point_info.average_npy, mmap_mode="r", allow_pickle=False
            )
            acquired.append(
                AcquiredPoint(
                    point=point_info.point,
                    average=average,
                    saturation_fraction=point_info.saturation_fraction,
                    peak_raw=point_info.peak_raw,
                    average_npy=point_info.average_npy,
                    average_tiff=point_info.average_tiff,
                    raw_frames_npy=point_info.raw_frames_npy,
                    saturation_fraction_source=(
                        point_info.saturation_fraction_source
                    ),
                    saturation_fraction_is_exact=(
                        point_info.saturation_fraction_is_exact
                    ),
                    peak_intensity_source=point_info.peak_intensity_source,
                    average_peak=point_info.average_peak,
                    acquisition_exposure_us=(
                        point_info.acquisition_exposure_us
                    ),
                    acquisition_frames_per_point=(
                        point_info.acquisition_frames_per_point
                    ),
                    effective_saturation_level=(
                        point_info.effective_saturation_level
                    ),
                )
            )
    except BaseException:
        _close_acquired_memmaps(acquired)
        raise
    return acquired


def _analyze_and_write_results(
    acquired: list[AcquiredPoint],
    output_path: Path,
    *,
    expected_spots: int,
    roi_xywh: tuple[int, int, int, int] | None,
    min_peak_distance_px: int,
    spot_radius_px: float | None,
    background_percentile: float,
    maximum_saturation_fraction: float,
    run_parameters: dict[str, object],
    progress_fn: ProgressFn,
) -> tuple[
    list[dict[str, float | str | bool]], dict[str, float | str | bool]
]:
    """Run the shared memory-bounded analysis and write derived outputs."""
    _report(progress_fn, "[analysis] Detecting one common set of spot centers")
    centers = detect_common_spots(
        [result.average for result in acquired],
        expected_spots,
        roi_xywh=roi_xywh,
        min_peak_distance_px=min_peak_distance_px,
        background_percentile=background_percentile,
        progress_fn=progress_fn,
    )
    _report(
        progress_fn,
        f"[analysis] Detected {len(centers)} spots; measuring scan points",
    )
    metric_rows, resolved_spot_radius = analyze_acquired_points(
        acquired,
        centers,
        spot_radius_px=spot_radius_px,
        roi_xywh=roi_xywh,
        background_percentile=background_percentile,
        progress_fn=progress_fn,
    )
    ranked_rows, best = rank_quality(
        metric_rows,
        maximum_saturation_fraction=maximum_saturation_fraction,
    )

    _report(progress_fn, "[output] Writing metrics, plots, and previews")
    _write_experimental_metrics(ranked_rows, output_path)
    _write_experimental_plot(
        ranked_rows, output_path / "experimental_metrics_vs_delta_z.png"
    )
    _write_spot_overlay(
        acquired,
        centers,
        resolved_spot_radius,
        output_path / "detected_spots.png",
        roi_xywh=roi_xywh,
        progress_fn=progress_fn,
    )
    _write_common_previews(
        acquired, output_path, progress_fn=progress_fn
    )

    selection_is_provisional = not all(
        result.saturation_fraction_is_exact for result in acquired
    )
    best_payload: dict[str, object] = {
        "delta_z_mm": float(best["delta_z_mm"]),
        "scan_label": str(best["scan_label"]),
        "quality_score": float(best["quality_score"]),
        "selection_is_provisional": selection_is_provisional,
        "selection": (
            "Highest equal-weight scan-normalized score from uniformity, "
            "spot sharpness, inverse halo, and inverse FWHM among points "
            "that did not exceed the configured saturation threshold."
        ),
    }
    if selection_is_provisional:
        best_payload["warning"] = (
            "Raw-frame saturation information was unavailable for at least "
            "one scan point. Thresholding an averaged image is only an "
            "estimate of per-frame saturation, so this selection is provisional."
        )
    _write_json_atomic(output_path / "best_delta_z.json", best_payload)

    sources = sorted(
        {result.saturation_fraction_source for result in acquired}
    )
    saturation_source = sources[0] if len(sources) == 1 else "mixed"
    scan_directory = acquired[0].point.bmp_path.parent
    parameters: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_version": 2,
        "analysis_memory_strategy": "streamed_float32_roi_local_spot_windows",
        "source_scan_directory": str(scan_directory.resolve()),
        "source_scan_parameters": _load_source_scan_parameters(scan_directory),
        "scan_count": len(acquired),
        "maximum_saturation_fraction": maximum_saturation_fraction,
        "saturation_fraction_source": saturation_source,
        "saturation_fraction_exact_for_all_points": (
            not selection_is_provisional
        ),
        "camera_roi_xywh": list(roi_xywh) if roi_xywh is not None else None,
        "background_scope": "camera_roi",
        "expected_spots": expected_spots,
        "min_peak_distance_px": min_peak_distance_px,
        "spot_radius_px": resolved_spot_radius,
        "background_percentile": background_percentile,
        "detected_spot_centers_yx": centers.tolist(),
        "quality_score_components": [
            "target_plane_uniformity",
            "mean_spot_sharpness",
            "inverse_background_halo",
            "inverse_mean_fwhm_px",
        ],
    }
    parameters.update(run_parameters)
    _write_json_atomic(
        output_path / "experimental_parameters.json", parameters
    )
    _report(
        progress_fn,
        "[done] Analysis complete: "
        f"best delta_z={float(best['delta_z_mm']):+.3f} mm",
    )
    best_result = dict(best)
    best_result["selection_is_provisional"] = selection_is_provisional
    return ranked_rows, best_result


def run_experimental_scan(
    points: list[ScanPoint],
    display,
    camera,
    output_dir: str | Path,
    *,
    exposure_us: float,
    frames_per_point: int,
    settle_seconds: float,
    correction: np.ndarray | None,
    correction_name: str | None,
    lut: int,
    save_raw_frames: bool,
    saturation_level: float | None,
    expected_spots: int,
    roi_xywh: tuple[int, int, int, int] | None,
    min_peak_distance_px: int,
    spot_radius_px: float | None,
    background_percentile: float,
    maximum_saturation_fraction: float,
    monitor_index: int,
    camera_index: int,
    sleep_fn=time.sleep,
    progress_fn: ProgressFn = print,
) -> tuple[
    list[dict[str, float | str | bool]], dict[str, float | str | bool]
]:
    """Run acquisition, fixed-ROI analysis, ranking, and output generation."""
    if not points:
        raise ValueError("at least one scan point is required")
    display_size_xy = tuple(int(value) for value in display.getSize())
    with Image.open(points[0].bmp_path) as first_bitmap:
        active_size_xy = tuple(int(value) for value in first_bitmap.size)

    acquired = acquire_scan_points(
        points,
        display,
        camera,
        output_dir,
        exposure_us=exposure_us,
        frames_per_point=frames_per_point,
        settle_seconds=settle_seconds,
        correction=correction,
        lut=lut,
        save_raw_frames=save_raw_frames,
        saturation_level=saturation_level,
        sleep_fn=sleep_fn,
        progress_fn=progress_fn,
    )
    try:
        return _analyze_and_write_results(
            acquired,
            Path(output_dir),
            expected_spots=expected_spots,
            roi_xywh=roi_xywh,
            min_peak_distance_px=min_peak_distance_px,
            spot_radius_px=spot_radius_px,
            background_percentile=background_percentile,
            maximum_saturation_fraction=maximum_saturation_fraction,
            progress_fn=progress_fn,
            run_parameters={
                "analysis_mode": "live_hardware_scan",
                "mechanics_during_scan": "fixed",
                "slm_connection": "Windows secondary monitor via slmpy",
                "monitor_index": monitor_index,
                "display_transport_shape_xy": list(display_size_xy),
                "slm_active_shape_xy": list(active_size_xy),
                "slm_interpolation_applied": False,
                "correction_bmp": correction_name,
                "lut": lut,
                "camera_backend": "Allied Vision Vimba X / vmbpy",
                "camera_index": camera_index,
                "exposure_us": exposure_us,
                "frames_per_point": frames_per_point,
                "settle_seconds": settle_seconds,
                "save_raw_frames": save_raw_frames,
                "saturation_level": saturation_level,
            },
        )
    finally:
        _close_acquired_memmaps(acquired)


def _common_acquisition_value(
    acquired: Sequence[AcquiredPoint], attribute: str
) -> object:
    values = [getattr(result, attribute) for result in acquired]
    first = values[0]
    if all(
        (value is None and first is None)
        or (value is not None and first is not None and value == first)
        for value in values
    ):
        return first
    return "mixed_or_unknown"


def run_existing_analysis(
    points: list[ScanPoint],
    acquisition_dir: str | Path,
    *,
    saturation_level: float | None,
    expected_spots: int,
    roi_xywh: tuple[int, int, int, int] | None,
    min_peak_distance_px: int,
    spot_radius_px: float | None,
    background_percentile: float,
    maximum_saturation_fraction: float,
    progress_fn: ProgressFn = print,
) -> tuple[
    list[dict[str, float | str | bool]], dict[str, float | str | bool]
]:
    """Analyze already saved NPY averages without opening SLM or camera."""
    output_path = Path(acquisition_dir)
    acquired = load_existing_acquired_points(
        points,
        output_path,
        saturation_level=saturation_level,
        progress_fn=progress_fn,
    )
    recovered_exposure = _common_acquisition_value(
        acquired, "acquisition_exposure_us"
    )
    recovered_frame_count = _common_acquisition_value(
        acquired, "acquisition_frames_per_point"
    )
    recovered_saturation_level = _common_acquisition_value(
        acquired, "effective_saturation_level"
    )
    try:
        return _analyze_and_write_results(
            acquired,
            output_path,
            expected_spots=expected_spots,
            roi_xywh=roi_xywh,
            min_peak_distance_px=min_peak_distance_px,
            spot_radius_px=spot_radius_px,
            background_percentile=background_percentile,
            maximum_saturation_fraction=maximum_saturation_fraction,
            progress_fn=progress_fn,
            run_parameters={
                "analysis_mode": "existing_averages",
                "mechanics_during_analysis": "not touched",
                "slm_connection": None,
                "monitor_index": None,
                "camera_backend": None,
                "camera_index": None,
                "exposure_us": recovered_exposure,
                "frames_per_point": recovered_frame_count,
                "settle_seconds": None,
                "correction_bmp": None,
                "lut": None,
                "saturation_level": (
                    saturation_level
                    if saturation_level is not None
                    else recovered_saturation_level
                ),
            },
        )
    finally:
        _close_acquired_memmaps(acquired)


class SecondaryMonitorSLM:
    """Lazy wrapper around the repository's wxPython secondary-monitor SLM."""

    def __init__(self, monitor_index: int):
        try:
            import slmpy
        except ImportError as error:
            raise RuntimeError(
                "wxPython/slmpy is unavailable; install wxPython on the "
                "Windows experiment computer"
            ) from error
        self._display = slmpy.SLMdisplay(
            monitor=monitor_index, isImageLock=True
        )

    def getSize(self):
        return self._display.getSize()

    def updateArray(self, frame):
        return self._display.updateArray(frame)

    def close(self):
        self._display.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Display an independently optimized delta-z scan on a secondary-"
            "monitor SLM, acquire Allied Vision frames, and rank real image "
            "quality."
        )
    )
    parser.add_argument("--scan-dir", default="delta_z_scan_outputs")
    parser.add_argument("--output-dir", default="delta_z_experiment")
    parser.add_argument(
        "--analyze-existing",
        metavar="ACQUISITION_DIR",
        help=(
            "Analyze camera_average_*.npy files from an interrupted or "
            "completed acquisition without opening the SLM or camera."
        ),
    )
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--exposure-us", type=float, default=50.0)
    parser.add_argument("--frames-per-point", type=int, default=16)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--correction-bmp", default="CAL_LSH0804730_785nm.bmp"
    )
    parser.add_argument("--lut", type=int, default=224)
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--save-raw-frames", action="store_true")
    parser.add_argument("--saturation-level", type=float)
    parser.add_argument("--expected-spots", type=int)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Restrict detection and metrics to a fixed CCD ROI.",
    )
    parser.add_argument("--min-peak-distance-px", type=int, default=8)
    parser.add_argument("--spot-radius-px", type=float)
    parser.add_argument("--background-percentile", type=float, default=10.0)
    parser.add_argument(
        "--maximum-saturation-fraction", type=float, default=0.001
    )
    return parser


def _resolve_correction_path(filename: str) -> Path:
    requested = Path(filename)
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(Path(__file__).resolve().parent / requested)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"SLM correction BMP not found: {filename}")


def _expected_spots_from_scan(
    scan_directory: Path, explicit_expected_spots: int | None
) -> int:
    if explicit_expected_spots is not None:
        if explicit_expected_spots <= 0:
            raise ValueError("expected_spots must be positive")
        return explicit_expected_spots
    parameters = _load_source_scan_parameters(scan_directory)
    if parameters is None or "target_array_size" not in parameters:
        raise ValueError(
            "expected spot count is unavailable; pass --expected-spots"
        )
    array_size = parameters["target_array_size"]
    if not isinstance(array_size, list) or len(array_size) != 2:
        raise ValueError("target_array_size in scan_parameters.json is invalid")
    expected_spots = int(array_size[0]) * int(array_size[1])
    if expected_spots <= 0:
        raise ValueError("target_array_size must contain positive values")
    return expected_spots


def _print_best_result(best: dict[str, float | str | bool]) -> None:
    provisional = bool(best.get("selection_is_provisional", False))
    prefix = "Provisional best delta_z" if provisional else "Best measured delta_z"
    print(
        f"{prefix}: {float(best['delta_z_mm']):+.3f} mm "
        f"(quality score {float(best['quality_score']):.4f})"
    )
    if provisional:
        print(
            "WARNING: raw-frame saturation data were unavailable; check "
            "the images for clipping before accepting this delta_z."
        )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    points = load_scan_points(args.scan_dir)
    scan_directory = Path(args.scan_dir)
    expected_spots = _expected_spots_from_scan(
        scan_directory, args.expected_spots
    )
    roi_xywh = tuple(args.roi) if args.roi is not None else None

    if args.analyze_existing is not None:
        _, best = run_existing_analysis(
            points,
            args.analyze_existing,
            saturation_level=args.saturation_level,
            expected_spots=expected_spots,
            roi_xywh=roi_xywh,
            min_peak_distance_px=args.min_peak_distance_px,
            spot_radius_px=args.spot_radius_px,
            background_percentile=args.background_percentile,
            maximum_saturation_fraction=args.maximum_saturation_fraction,
        )
        _print_best_result(best)
        return

    correction = None
    correction_name = None
    if not args.no_calibration:
        correction_path = _resolve_correction_path(args.correction_bmp)
        with Image.open(correction_path) as correction_bitmap:
            correction = np.asarray(
                correction_bitmap.convert("L"), dtype=np.uint8
            )
        correction_name = str(correction_path)

    from avt import VimbaCamera

    with ExitStack() as stack:
        display = SecondaryMonitorSLM(args.monitor)
        stack.callback(display.close)
        camera = VimbaCamera(cam_index=args.camera_index)
        stack.callback(camera.close)
        _, best = run_experimental_scan(
            points,
            display,
            camera,
            args.output_dir,
            exposure_us=args.exposure_us,
            frames_per_point=args.frames_per_point,
            settle_seconds=args.settle_seconds,
            correction=correction,
            correction_name=correction_name,
            lut=args.lut,
            save_raw_frames=args.save_raw_frames,
            saturation_level=args.saturation_level,
            expected_spots=expected_spots,
            roi_xywh=roi_xywh,
            min_peak_distance_px=args.min_peak_distance_px,
            spot_radius_px=args.spot_radius_px,
            background_percentile=args.background_percentile,
            maximum_saturation_fraction=args.maximum_saturation_fraction,
            monitor_index=args.monitor,
            camera_index=args.camera_index,
        )
    _print_best_result(best)


if __name__ == "__main__":
    main()
