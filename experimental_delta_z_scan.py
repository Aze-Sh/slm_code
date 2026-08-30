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


def _validate_background_percentile(background_percentile: float) -> None:
    if not 0 <= background_percentile < 100:
        raise ValueError("background_percentile must be in [0, 100)")


def _background_correct(
    image: np.ndarray, background_percentile: float
) -> np.ndarray:
    _validate_background_percentile(background_percentile)
    image_array = np.asarray(image, dtype=np.float64)
    if image_array.ndim != 2:
        raise ValueError("camera image must be two-dimensional")
    if not np.all(np.isfinite(image_array)):
        raise ValueError("camera image contains non-finite values")
    baseline = float(np.percentile(image_array, background_percentile))
    return np.clip(image_array - baseline, 0.0, None)


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
    images: list[np.ndarray],
    expected_count: int,
    *,
    roi_xywh: tuple[int, int, int, int] | None = None,
    min_peak_distance_px: int = 8,
    background_percentile: float = 10.0,
) -> np.ndarray:
    """Detect one common, fixed set of spot centers for the complete scan."""
    if not images:
        raise ValueError("at least one camera image is required")
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if min_peak_distance_px <= 0:
        raise ValueError("min_peak_distance_px must be positive")

    first_shape = tuple(np.asarray(images[0]).shape)
    if len(first_shape) != 2:
        raise ValueError("camera image must be two-dimensional")
    normalized_images = []
    for image in images:
        if tuple(np.asarray(image).shape) != first_shape:
            raise ValueError("all camera images must have the same shape")
        corrected = _background_correct(image, background_percentile)
        total = float(np.sum(corrected))
        if total <= 0:
            raise ValueError("camera image has no signal above background")
        normalized_images.append(corrected / total)
    reference = np.mean(normalized_images, axis=0)

    y_slice, x_slice = _roi_slices(first_shape, roi_xywh)
    roi_reference = reference[y_slice, x_slice]
    filter_size = 2 * min_peak_distance_px + 1
    local_maximum = ndimage.maximum_filter(
        roi_reference,
        size=filter_size,
        mode="constant",
        cval=-np.inf,
    )
    maxima_mask = (roi_reference == local_maximum) & (roi_reference > 0)
    labels, component_count = ndimage.label(maxima_mask)
    candidates = []
    for label_index in range(1, component_count + 1):
        component_coordinates = np.argwhere(labels == label_index)
        component_values = roi_reference[labels == label_index]
        best_coordinate = component_coordinates[int(np.argmax(component_values))]
        candidate_y = int(best_coordinate[0] + y_slice.start)
        candidate_x = int(best_coordinate[1] + x_slice.start)
        candidates.append(
            (float(np.max(component_values)), candidate_y, candidate_x)
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


def _spot_shape_metrics(
    signal: np.ndarray, mask: np.ndarray, y_grid: np.ndarray, x_grid: np.ndarray
) -> tuple[float, float, float]:
    values = signal[mask]
    if values.size == 0 or float(np.sum(values)) <= 0:
        return float("inf"), float("inf"), 0.0

    mask_y = y_grid[mask]
    mask_x = x_grid[mask]
    peak_index = int(np.argmax(values))
    peak_y = float(mask_y[peak_index])
    peak_x = float(mask_x[peak_index])
    half_maximum_area = int(np.count_nonzero(values >= 0.5 * values[peak_index]))
    equivalent_fwhm = 2.0 * np.sqrt(half_maximum_area / np.pi)

    distances = np.sqrt((mask_y - peak_y) ** 2 + (mask_x - peak_x) ** 2)
    order = np.argsort(distances)
    ordered_values = values[order]
    cumulative = np.cumsum(ordered_values)
    radius_index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    radius_index = min(radius_index, len(order) - 1)
    encircled_energy_radius_50 = float(distances[order[radius_index]])

    spot_sum = float(np.sum(values))
    sharpness = float(np.sum(np.square(values)) / spot_sum**2)
    return float(equivalent_fwhm), encircled_energy_radius_50, sharpness


def analyze_acquired_points(
    acquired: list[AcquiredPoint],
    centers: np.ndarray,
    *,
    spot_radius_px: float | None = None,
    roi_xywh: tuple[int, int, int, int] | None = None,
    background_percentile: float = 10.0,
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
    height, width = image_shape
    if np.any(center_array[:, 0] < 0) or np.any(center_array[:, 0] >= height):
        raise ValueError("spot center y coordinate lies outside the image")
    if np.any(center_array[:, 1] < 0) or np.any(center_array[:, 1] >= width):
        raise ValueError("spot center x coordinate lies outside the image")

    y_slice, x_slice = _roi_slices(image_shape, roi_xywh)
    analysis_mask = np.zeros(image_shape, dtype=bool)
    analysis_mask[y_slice, x_slice] = True
    y_grid, x_grid = np.indices(image_shape, dtype=np.float64)
    spot_masks = [
        ((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2 <= radius**2)
        & analysis_mask
        for center_y, center_x in center_array
    ]
    target_mask = np.logical_or.reduce(spot_masks)

    rows: list[dict[str, float | str]] = []
    for result in acquired:
        if tuple(np.asarray(result.average).shape) != image_shape:
            raise ValueError("all acquired camera images must have the same shape")
        signal = _background_correct(result.average, background_percentile)
        signal = np.where(analysis_mask, signal, 0.0)
        total_signal = float(np.sum(signal))
        if total_signal <= 0:
            raise ValueError(
                f"camera image has no signal above background: "
                f"{result.point.scan_label}"
            )

        spot_sums = np.asarray(
            [float(np.sum(signal[mask])) for mask in spot_masks]
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
        target_signal = float(np.sum(signal[target_mask]))
        target_efficiency = target_signal / total_signal
        background_halo = 1.0 - target_efficiency

        normalized_signal = signal / total_signal
        centroid_x = float(np.sum(x_grid * normalized_signal))
        centroid_y = float(np.sum(y_grid * normalized_signal))
        peak_flat_index = int(np.argmax(signal))
        peak_y, peak_x = divmod(peak_flat_index, width)
        centroid_peak_offset = float(
            np.hypot(centroid_x - peak_x, centroid_y - peak_y)
        )

        per_spot_metrics = [
            _spot_shape_metrics(signal, mask, y_grid, x_grid)
            for mask in spot_masks
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
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_spot_overlay(
    acquired: list[AcquiredPoint],
    centers: np.ndarray,
    radius: float,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    normalized_images = []
    for result in acquired:
        image = np.asarray(result.average, dtype=np.float64)
        scale = float(np.max(image))
        normalized_images.append(image / scale if scale > 0 else image)
    reference = np.mean(normalized_images, axis=0)

    figure, axis = plt.subplots(figsize=(8, 7))
    image_handle = axis.imshow(reference, cmap="gray")
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
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _load_source_scan_parameters(scan_directory: Path) -> dict[str, object] | None:
    parameters_path = scan_directory / "scan_parameters.json"
    if not parameters_path.is_file():
        return None
    with parameters_path.open(encoding="utf-8") as handle:
        return json.load(handle)


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
    )
    centers = detect_common_spots(
        [result.average for result in acquired],
        expected_spots,
        roi_xywh=roi_xywh,
        min_peak_distance_px=min_peak_distance_px,
        background_percentile=background_percentile,
    )
    metric_rows, resolved_spot_radius = analyze_acquired_points(
        acquired,
        centers,
        spot_radius_px=spot_radius_px,
        roi_xywh=roi_xywh,
        background_percentile=background_percentile,
    )
    ranked_rows, best = rank_quality(
        metric_rows,
        maximum_saturation_fraction=maximum_saturation_fraction,
    )

    output_path = Path(output_dir)
    _write_experimental_metrics(ranked_rows, output_path)
    _write_experimental_plot(
        ranked_rows, output_path / "experimental_metrics_vs_delta_z.png"
    )
    _write_spot_overlay(
        acquired,
        centers,
        resolved_spot_radius,
        output_path / "detected_spots.png",
    )
    best_payload = {
        "delta_z_mm": float(best["delta_z_mm"]),
        "scan_label": str(best["scan_label"]),
        "quality_score": float(best["quality_score"]),
        "selection": (
            "Highest equal-weight scan-normalized score from uniformity, "
            "spot sharpness, inverse halo, and inverse FWHM among "
            "unsaturated points."
        ),
    }
    (output_path / "best_delta_z.json").write_text(
        json.dumps(best_payload, indent=2), encoding="utf-8"
    )

    scan_directory = points[0].bmp_path.parent
    parameters = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_scan_directory": str(scan_directory.resolve()),
        "source_scan_parameters": _load_source_scan_parameters(scan_directory),
        "scan_count": len(points),
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
        "maximum_saturation_fraction": maximum_saturation_fraction,
        "camera_roi_xywh": list(roi_xywh) if roi_xywh is not None else None,
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
    (output_path / "experimental_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )
    return ranked_rows, best


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


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    points = load_scan_points(args.scan_dir)
    scan_directory = Path(args.scan_dir)
    expected_spots = _expected_spots_from_scan(
        scan_directory, args.expected_spots
    )

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
            roi_xywh=tuple(args.roi) if args.roi is not None else None,
            min_peak_distance_px=args.min_peak_distance_px,
            spot_radius_px=args.spot_radius_px,
            background_percentile=args.background_percentile,
            maximum_saturation_fraction=args.maximum_saturation_fraction,
            monitor_index=args.monitor,
            camera_index=args.camera_index,
        )
    print(
        "Best measured delta_z: "
        f"{float(best['delta_z_mm']):+.3f} mm "
        f"(quality score {float(best['quality_score']):.4f})"
    )


if __name__ == "__main__":
    main()
