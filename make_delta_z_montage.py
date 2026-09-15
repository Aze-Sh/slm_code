"""Build offline comparison sheets from a completed camera delta-z scan.

The script reads the memory-mapped ``camera_average_*.npy`` files produced by
``experimental_delta_z_scan.py``.  It never opens the SLM or camera.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


AVERAGE_NAME = re.compile(
    r"^camera_average_scan_(\d+)_delta_z_([mp])(\d+(?:\.\d+)?)mm\.npy$"
)


def load_crop(path: Path, roi_xywh: tuple[int, int, int, int]) -> np.ndarray:
    """Load only the requested ROI from a potentially large NPY average."""
    x, y, width, height = roi_xywh
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if array.ndim != 2 or x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError(f"invalid image or ROI for {path.name}")
        if x + width > array.shape[1] or y + height > array.shape[0]:
            raise ValueError(
                f"ROI is outside {path.name}: image shape={array.shape}"
            )
        return np.array(
            array[y : y + height, x : x + width],
            dtype=np.float32,
            copy=True,
        )
    finally:
        memory_map = getattr(array, "_mmap", None)
        if memory_map is not None and not memory_map.closed:
            memory_map.close()


def background_corrected_crop(
    path: Path,
    roi_xywh: tuple[int, int, int, int],
    background_percentile: float,
) -> tuple[np.ndarray, float]:
    image = load_crop(path, roi_xywh)
    background = float(np.percentile(image, background_percentile))
    image -= np.float32(background)
    np.maximum(image, 0, out=image)
    return image, background


def find_averages(folder: Path) -> list[tuple[int, float, Path]]:
    """Return scan-index ordered ``(index, delta_z_mm, path)`` records."""
    records: list[tuple[int, float, Path]] = []
    for path in folder.glob("camera_average_scan_*_delta_z_*mm.npy"):
        match = AVERAGE_NAME.match(path.name)
        if match is None:
            continue
        index = int(match.group(1))
        magnitude = float(match.group(3))
        delta_z_mm = magnitude if match.group(2) == "p" else -magnitude
        records.append((index, delta_z_mm, path))
    records.sort(key=lambda item: item[0])
    if not records:
        raise FileNotFoundError(
            f"no camera_average_scan_*_delta_z_*mm.npy files in {folder}"
        )
    if len({record[0] for record in records}) != len(records):
        raise ValueError("duplicate scan indices were found")
    return records


def auto_patch_radius(centers_yx: np.ndarray) -> int:
    """Choose a window below half the typical nearest-neighbour spacing."""
    if len(centers_yx) < 2:
        raise ValueError("at least two detected spots are required")
    difference = centers_yx[:, None, :] - centers_yx[None, :, :]
    distances = np.sqrt(np.sum(difference * difference, axis=2))
    np.fill_diagonal(distances, np.inf)
    median_nearest = float(np.median(np.min(distances, axis=1)))
    return max(8, min(40, int(math.floor(0.42 * median_nearest))))


def median_registered_spot(
    image: np.ndarray,
    centers_yx: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, int]:
    """Integer-register, peak-normalize, then median-combine all spot crops."""
    patches: list[np.ndarray] = []
    search_radius = max(2, min(6, radius // 5))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    distance = np.sqrt(yy * yy + xx * xx)
    outer_annulus = (distance >= 0.78 * radius) & (distance <= radius)

    for center_y, center_x in centers_yx:
        cy, cx = int(round(center_y)), int(round(center_x))
        sy0 = max(0, cy - search_radius)
        sy1 = min(image.shape[0], cy + search_radius + 1)
        sx0 = max(0, cx - search_radius)
        sx1 = min(image.shape[1], cx + search_radius + 1)
        search = image[sy0:sy1, sx0:sx1]
        if search.size == 0:
            continue
        local_y, local_x = np.unravel_index(int(np.argmax(search)), search.shape)
        cy, cx = sy0 + int(local_y), sx0 + int(local_x)
        if (
            cy - radius < 0
            or cx - radius < 0
            or cy + radius >= image.shape[0]
            or cx + radius >= image.shape[1]
        ):
            continue
        patch = np.array(
            image[
                cy - radius : cy + radius + 1,
                cx - radius : cx + radius + 1,
            ],
            dtype=np.float32,
            copy=True,
        )
        local_background = float(np.median(patch[outer_annulus]))
        patch -= np.float32(local_background)
        np.maximum(patch, 0, out=patch)
        peak = float(np.max(patch))
        if peak > 0 and np.isfinite(peak):
            patches.append(patch / np.float32(peak))

    if not patches:
        raise ValueError("no valid spot patches could be extracted")
    median_spot = np.median(np.stack(patches), axis=0).astype(np.float32)
    median_peak = float(np.max(median_spot))
    if median_peak > 0:
        median_spot /= np.float32(median_peak)
    return median_spot, len(patches)


def save_grid(
    panels: list[np.ndarray],
    labels: list[str],
    output: Path,
    *,
    columns: int,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    panel_inches: float,
) -> None:
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(panel_inches * columns, panel_inches * rows + 0.6),
    )
    axes = np.atleast_1d(axes).ravel()
    for axis, panel, label in zip(axes, panels, labels):
        axis.imshow(
            panel,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(label, fontsize=9)
        axis.axis("off")
    for axis in axes[len(panels) :]:
        axis.axis("off")
    figure.suptitle(title, fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _full_image_roi(path: Path) -> tuple[int, int, int, int]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if array.ndim != 2:
            raise ValueError(f"camera average must be two-dimensional: {path}")
        return 0, 0, int(array.shape[1]), int(array.shape[0])
    finally:
        memory_map = getattr(array, "_mmap", None)
        if memory_map is not None and not memory_map.closed:
            memory_map.close()


def build_montages(
    experiment_dir: Path,
    *,
    roi_xywh: tuple[int, int, int, int] | None = None,
    columns: int = 4,
    thumbnail_px: int = 650,
    patch_radius_px: int | None = None,
    ideal_distance_mm: float = 200.0,
) -> list[Path]:
    """Build one common-scale overview and two typical-spot comparisons."""
    folder = experiment_dir.resolve()
    if columns <= 0 or thumbnail_px <= 0:
        raise ValueError("columns and thumbnail_px must be positive")
    parameters_path = folder / "experimental_parameters.json"
    if not parameters_path.is_file():
        raise FileNotFoundError(
            f"missing {parameters_path.name}; first finish or re-run "
            "experimental_delta_z_scan.py --analyze-existing"
        )
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    centers = np.asarray(
        parameters["detected_spot_centers_yx"], dtype=np.float64
    )
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("detected_spot_centers_yx is invalid")
    records = find_averages(folder)
    if roi_xywh is None:
        roi_value = parameters.get("camera_roi_xywh")
        roi_xywh = (
            _full_image_roi(records[0][2])
            if roi_value is None
            else tuple(int(value) for value in roi_value)
        )
    x0, y0, _, _ = roi_xywh
    centers_roi = centers - np.array([y0, x0], dtype=np.float64)
    background_percentile = float(parameters.get("background_percentile", 10.0))

    # First streaming pass: find one robust display limit shared by all panels.
    upper_candidates = []
    for _, _, path in records:
        image, _ = background_corrected_crop(
            path, roi_xywh, background_percentile
        )
        upper_candidates.append(float(np.percentile(image, 99.995)))
    common_upper = max(upper_candidates)
    if not np.isfinite(common_upper) or common_upper <= 0:
        raise ValueError("all images contain no usable signal above background")

    radius = patch_radius_px or auto_patch_radius(centers_roi)
    full_panels: list[np.ndarray] = []
    spot_panels: list[np.ndarray] = []
    labels: list[str] = []
    for _, delta_z_mm, path in records:
        image, background = background_corrected_crop(
            path, roi_xywh, background_percentile
        )
        scaled = np.clip(image / np.float32(common_upper), 0, 1)
        thumbnail = Image.fromarray(
            np.rint(scaled * 255).astype(np.uint8), mode="L"
        )
        thumbnail.thumbnail(
            (thumbnail_px, thumbnail_px), Image.Resampling.LANCZOS
        )
        full_panels.append(np.asarray(thumbnail))
        median_spot, valid_count = median_registered_spot(
            image, centers_roi, radius
        )
        spot_panels.append(median_spot)
        labels.append(
            f"dz={delta_z_mm:+.1f} mm | "
            f"L2-pupil~{ideal_distance_mm + delta_z_mm:.1f} mm\n"
            f"median of {valid_count} spots | bg={background:.1f}"
        )
        print(
            f"Processed {path.name}: dz={delta_z_mm:+.3f} mm, "
            f"spots={valid_count}"
        )

    full_output = folder / "delta_z_full_array_montage.png"
    linear_output = folder / "delta_z_median_spot_montage.png"
    log_output = folder / "delta_z_median_spot_log_montage.png"
    save_grid(
        full_panels,
        labels,
        full_output,
        columns=columns,
        title=f"Full arrays: common linear scale, upper={common_upper:.2f}",
        cmap="gray",
        vmin=0,
        vmax=255,
        panel_inches=3.7,
    )
    save_grid(
        spot_panels,
        labels,
        linear_output,
        columns=columns,
        title=f"Median registered spot: linear 0-1, radius={radius} px",
        cmap="gray",
        vmin=0,
        vmax=1,
        panel_inches=3.0,
    )
    log_panels = [
        10 * np.log10(np.clip(panel, 1e-3, 1.0)) for panel in spot_panels
    ]
    save_grid(
        log_panels,
        labels,
        log_output,
        columns=columns,
        title="Median registered spot: logarithmic intensity, -30 to 0 dB",
        cmap="inferno",
        vmin=-30,
        vmax=0,
        panel_inches=3.0,
    )
    return [full_output, linear_output, log_output]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Make comparable delta-z camera montages from saved NPY averages."
        )
    )
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument(
        "--roi", type=int, nargs=4, metavar=("X", "Y", "WIDTH", "HEIGHT")
    )
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--thumbnail-px", type=int, default=650)
    parser.add_argument("--patch-radius-px", type=int)
    parser.add_argument("--ideal-distance-mm", type=float, default=200.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = build_montages(
        args.experiment_dir,
        roi_xywh=tuple(args.roi) if args.roi is not None else None,
        columns=args.columns,
        thumbnail_px=args.thumbnail_px,
        patch_radius_px=args.patch_radius_px,
        ideal_distance_mm=args.ideal_distance_mm,
    )
    print("\nCreated:")
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
