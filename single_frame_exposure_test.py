"""Display one existing 8x8 WGS phase and capture one raw AVT frame."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from experimental_delta_z_scan import (
    SecondaryMonitorSLM,
    _effective_saturation_level,
    _resolve_correction_path,
    _save_average_tiff,
    _save_preview_png,
    apply_slm_calibration,
    center_phase_on_display,
    load_scan_points,
)


def select_scan_point(scan_dir: str | Path, delta_z_mm: float):
    """Return the single manifest entry matching ``delta_z_mm``."""
    points = load_scan_points(scan_dir)
    matches = [
        point
        for point in points
        if np.isclose(point.delta_z_mm, delta_z_mm, atol=1e-9, rtol=0.0)
    ]
    if len(matches) != 1:
        available = ", ".join(f"{point.delta_z_mm:g}" for point in points)
        raise ValueError(
            f"delta_z={delta_z_mm:g} mm was not found exactly once; "
            f"available values: {available}"
        )
    return matches[0]


def _region_slices(
    image_shape: tuple[int, int],
    roi_xywh: tuple[int, int, int, int] | None,
) -> tuple[slice, slice]:
    height, width = image_shape
    if roi_xywh is None:
        return slice(0, height), slice(0, width)
    x, y, roi_width, roi_height = roi_xywh
    if x < 0 or y < 0 or roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI must contain non-negative x/y and positive size")
    if x + roi_width > width or y + roi_height > height:
        raise ValueError(
            f"ROI {roi_xywh} lies outside camera frame {width}x{height}"
        )
    return slice(y, y + roi_height), slice(x, x + roi_width)


def measure_frame(
    frame: np.ndarray,
    *,
    saturation_level: float | None,
    roi_xywh: tuple[int, int, int, int] | None,
) -> dict[str, object]:
    """Measure full-frame and optional ROI peak/saturation statistics."""
    image = np.asarray(frame)
    if image.ndim != 2:
        raise ValueError("camera frame must be a two-dimensional image")
    if image.size == 0:
        raise ValueError("camera frame must not be empty")
    effective_level = _effective_saturation_level(image.dtype, saturation_level)

    def region_statistics(
        values: np.ndarray, *, x_offset: int, y_offset: int
    ) -> dict[str, object]:
        flat_peak_index = int(np.argmax(values))
        local_y, local_x = np.unravel_index(flat_peak_index, values.shape)
        saturated_count = (
            int(np.count_nonzero(values >= effective_level))
            if effective_level is not None
            else 0
        )
        return {
            "peak": float(values[local_y, local_x]),
            "peak_x_px": int(local_x + x_offset),
            "peak_y_px": int(local_y + y_offset),
            "saturated_pixel_count": saturated_count,
            "saturation_fraction": float(saturated_count / values.size),
        }

    full = region_statistics(image, x_offset=0, y_offset=0)
    roi = None
    if roi_xywh is not None:
        y_slice, x_slice = _region_slices(tuple(image.shape), roi_xywh)
        roi = region_statistics(
            image[y_slice, x_slice],
            x_offset=int(x_slice.start),
            y_offset=int(y_slice.start),
        )

    return {
        "frame_dtype": str(image.dtype),
        "frame_shape_yx": [int(image.shape[0]), int(image.shape[1])],
        "effective_saturation_level": effective_level,
        "full_frame": full,
        "roi_xywh": list(roi_xywh) if roi_xywh is not None else None,
        "roi": roi,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Display one independently optimized 8x8 phase, acquire one raw "
            "AVT frame, and report peak and saturated-pixel counts."
        )
    )
    parser.add_argument("--scan-dir", default="delta_z_scan_outputs")
    parser.add_argument("--delta-z-mm", type=float, default=0.0)
    parser.add_argument("--output-dir", default="single_frame_tests")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--exposure-us", type=float, default=23.0)
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--correction-bmp", default="CAL_LSH0804730_785nm.bmp"
    )
    parser.add_argument("--lut", type=int, default=224)
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--saturation-level", type=float, default=255.0)
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.exposure_us <= 0:
        raise ValueError("exposure-us must be positive")
    if args.settle_seconds < 0:
        raise ValueError("settle-seconds must be non-negative")

    point = select_scan_point(args.scan_dir, args.delta_z_mm)
    with Image.open(point.bmp_path) as bitmap:
        phase = np.asarray(bitmap.convert("L"), dtype=np.uint8)

    correction = None
    correction_path = None
    if not args.no_calibration:
        correction_path = _resolve_correction_path(args.correction_bmp)
        with Image.open(correction_path) as correction_bitmap:
            correction = np.asarray(
                correction_bitmap.convert("L"), dtype=np.uint8
            )

    from avt import VimbaCamera

    with ExitStack() as stack:
        display = SecondaryMonitorSLM(args.monitor)
        stack.callback(display.close)
        camera = VimbaCamera(cam_index=args.camera_index)
        stack.callback(camera.close)

        calibrated_phase = apply_slm_calibration(
            phase, correction=correction, lut=args.lut
        )
        transport_frame = center_phase_on_display(
            calibrated_phase, tuple(int(value) for value in display.getSize())
        )
        display.updateArray(transport_frame)
        time.sleep(args.settle_seconds)
        camera_frame = np.asarray(camera.capture(args.exposure_us))
        try:
            actual_exposure_us = float(
                camera._cam.get_feature_by_name("ExposureTime").get()
            )
        except Exception:
            actual_exposure_us = None

    roi_xywh = tuple(args.roi) if args.roi is not None else None
    statistics = measure_frame(
        camera_frame,
        saturation_level=args.saturation_level,
        roi_xywh=roi_xywh,
    )
    statistics.update(
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scan_directory": str(Path(args.scan_dir).resolve()),
            "phase_bmp": str(point.bmp_path.resolve()),
            "delta_z_mm": point.delta_z_mm,
            "monitor_index": args.monitor,
            "camera_index": args.camera_index,
            "requested_exposure_us": args.exposure_us,
            "actual_exposure_us": actual_exposure_us,
            "settle_seconds": args.settle_seconds,
            "correction_bmp": (
                str(correction_path) if correction_path is not None else None
            ),
            "lut": args.lut,
        }
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_path = Path(args.output_dir) / f"{timestamp}_{point.scan_label}"
    run_path.mkdir(parents=True, exist_ok=False)
    np.save(run_path / "camera_raw.npy", camera_frame)
    _save_average_tiff(
        camera_frame, run_path / "camera_raw.tiff", camera_frame.dtype
    )
    _save_preview_png(camera_frame, run_path / "camera_preview.png")
    (run_path / "exposure_stats.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )

    full = statistics["full_frame"]
    print(f"Displayed phase: {point.bmp_path.name}")
    print(f"delta_z: {point.delta_z_mm:+.3f} mm")
    print(f"Requested exposure: {args.exposure_us:.3f} us")
    if actual_exposure_us is not None:
        print(f"Actual exposure: {actual_exposure_us:.3f} us")
    print(
        "Full frame: "
        f"peak={full['peak']:g} at "
        f"(x={full['peak_x_px']}, y={full['peak_y_px']}), "
        f"saturated pixels={full['saturated_pixel_count']} "
        f"({100.0 * full['saturation_fraction']:.8f}%)"
    )
    roi = statistics["roi"]
    if roi is not None:
        print(
            "Array ROI: "
            f"peak={roi['peak']:g} at "
            f"(x={roi['peak_x_px']}, y={roi['peak_y_px']}), "
            f"saturated pixels={roi['saturated_pixel_count']} "
            f"({100.0 * roi['saturation_fraction']:.8f}%)"
        )
    print(f"Saved raw frame and statistics to: {run_path.resolve()}")


if __name__ == "__main__":
    main()
