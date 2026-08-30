"""Offline WGS scan for axial SLM-image/objective-pupil conjugation error."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_functional
from PIL import Image

from WGS import WGS_phase_generate, angular_spectrum_propagate, circular_pupil_mask


COARSE_DELTA_Z_MM = (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


def _delta_z_label(delta_z_mm: float) -> str:
    sign = "m" if delta_z_mm < 0 else "p"
    return f"delta_z_{sign}{abs(delta_z_mm):07.3f}mm"


def _phase_to_screen(
    phase: np.ndarray, slm_resolution: tuple[int, int]
) -> np.ndarray:
    """Match the repository's existing centered crop and 8-bit encoding."""
    slm_width, slm_height = slm_resolution
    crop_size = min(slm_width, slm_height)
    height, width = phase.shape
    start_y = int(height / 2 - round(crop_size / 2))
    start_x = int(width / 2 - round(crop_size / 2))
    cropped = phase[
        start_y : start_y + crop_size,
        start_x : start_x + crop_size,
    ]
    encoded = np.around((cropped + np.pi) / (2 * np.pi) * 256).astype(
        np.uint8
    )
    screen = np.zeros((slm_height, slm_width), dtype=np.uint8)
    screen_start_y = int(slm_height / 2 - round(crop_size / 2))
    screen_start_x = int(slm_width / 2 - round(crop_size / 2))
    screen[
        screen_start_y : screen_start_y + crop_size,
        screen_start_x : screen_start_x + crop_size,
    ] = encoded
    return screen


def _simulate_intensity(
    init_amplitude: torch.Tensor,
    phase: torch.Tensor,
    *,
    delta_z_mm: float,
    wavelength_um: float,
    image_pixel_pitch_um: float,
    pupil_radius_mm: float,
) -> torch.Tensor:
    device = phase.device
    amplitude = init_amplitude.to(device)
    image_field = amplitude * torch.exp(1j * phase)
    if delta_z_mm == 0:
        pupil_field = image_field
    else:
        pupil_field = angular_spectrum_propagate(
            image_field,
            distance_mm=delta_z_mm,
            wavelength_um=wavelength_um,
            pixel_pitch_um=image_pixel_pitch_um,
        )
    pupil_mask = circular_pupil_mask(
        tuple(pupil_field.shape[-2:]),
        image_pixel_pitch_um,
        pupil_radius_mm,
        device=device,
        dtype=pupil_field.real.dtype,
    )
    pupil_field = pupil_field * pupil_mask
    target_field = torch.fft.fftshift(torch.fft.fft2(pupil_field))
    intensity = torch.abs(target_field) ** 2
    input_power = torch.sum(torch.square(torch.abs(image_field)))
    sample_count = image_field.shape[-2] * image_field.shape[-1]
    return intensity / (sample_count * input_power)


def _calculate_metrics(
    intensity: torch.Tensor,
    target_amplitude: torch.Tensor,
    halo_exclusion_radius_px: int,
) -> dict[str, float]:
    target = target_amplitude.to(intensity.device)
    target_mask = target != 0
    transmitted_power = torch.sum(intensity)
    target_values = intensity[target_mask]
    measured_distribution = target_values / torch.sum(target_values)
    desired_values = torch.square(torch.abs(target[target_mask]))
    desired_distribution = desired_values / torch.sum(desired_values)

    wgs_error = torch.sqrt(
        torch.sum(torch.square(measured_distribution - desired_distribution))
    )
    target_uniformity = torch.min(target_values) / torch.max(target_values)
    target_efficiency = torch.sum(target_values) / transmitted_power

    radius = halo_exclusion_radius_px
    if radius < 0:
        raise ValueError("halo_exclusion_radius_px must be non-negative")
    if radius == 0:
        expanded_target = target_mask
    else:
        expanded_target = torch_functional.max_pool2d(
            target_mask[None, None].to(intensity.dtype),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )[0, 0].bool()
    background_halo = torch.sum(intensity[~expanded_target]) / transmitted_power

    height, width = intensity.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=intensity.device, dtype=intensity.dtype),
        torch.arange(width, device=intensity.device, dtype=intensity.dtype),
        indexing="ij",
    )
    transmitted_distribution = intensity / transmitted_power
    centroid_x = torch.sum(x * transmitted_distribution)
    centroid_y = torch.sum(y * transmitted_distribution)
    peak_flat_index = int(torch.argmax(intensity).item())
    peak_y, peak_x = divmod(peak_flat_index, width)
    centroid_peak_offset = torch.sqrt(
        (centroid_x - peak_x) ** 2 + (centroid_y - peak_y) ** 2
    )

    return {
        "wgs_error": float(wgs_error.item()),
        "target_plane_uniformity": float(target_uniformity.item()),
        "background_halo": float(background_halo.item()),
        "target_efficiency": float(target_efficiency.item()),
        "pupil_transmission": float(transmitted_power.item()),
        "peak_intensity": float(torch.max(intensity).item()),
        "centroid_x_px": float(centroid_x.item()),
        "centroid_y_px": float(centroid_y.item()),
        "peak_x_px": float(peak_x),
        "peak_y_px": float(peak_y),
        "centroid_peak_offset_px": float(centroid_peak_offset.item()),
    }


def _write_metrics_plot(rows: list[dict[str, float | str]], path: Path) -> None:
    delta_z = [float(row["delta_z_mm"]) for row in rows]
    series = (
        ("wgs_error", "WGS error"),
        ("target_plane_uniformity", "Target uniformity (min/max)"),
        ("background_halo", "Background / halo"),
        ("target_efficiency", "Target efficiency"),
        ("peak_intensity", "Normalized peak intensity"),
        ("centroid_peak_offset_px", "Centroid-peak offset (px)"),
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


def run_delta_z_scan(
    *,
    init_amplitude: torch.Tensor,
    init_phase: torch.Tensor,
    target_amplitude: torch.Tensor,
    delta_z_values_mm: Iterable[float],
    output_dir: str | Path,
    loops: int,
    threshold: float,
    wavelength_um: float,
    image_pixel_pitch_um: float,
    pupil_radius_mm: float,
    slm_resolution: tuple[int, int],
    halo_exclusion_radius_px: int = 3,
) -> list[dict[str, float | str]]:
    """Run an independent WGS optimization and save outputs for every delta_z."""
    delta_z_values = [float(value) for value in delta_z_values_mm]
    if not delta_z_values:
        raise ValueError("at least one delta_z value is required")
    labels = [_delta_z_label(value) for value in delta_z_values]
    if len(set(labels)) != len(labels):
        raise ValueError("delta_z values produce duplicate output labels")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []

    for delta_z_mm in delta_z_values:
        label = _delta_z_label(delta_z_mm)
        phase = WGS_phase_generate(
            init_amplitude.clone(),
            init_phase.clone(),
            target_amplitude.clone(),
            Loop=loops,
            threshold=threshold,
            Plot=False,
            delta_z_mm=delta_z_mm,
            wavelength_um=wavelength_um,
            image_pixel_pitch_um=image_pixel_pitch_um,
            pupil_radius_mm=pupil_radius_mm,
        )
        phase_cpu = phase.detach().cpu().numpy()
        phase_filename = f"slm_phase_{label}.npy"
        bitmap_filename = f"slm_phase_{label}.bmp"
        np.save(output_path / phase_filename, phase_cpu)
        screen = _phase_to_screen(phase_cpu, slm_resolution)
        Image.fromarray(screen, mode="L").save(output_path / bitmap_filename)

        intensity = _simulate_intensity(
            init_amplitude,
            phase,
            delta_z_mm=delta_z_mm,
            wavelength_um=wavelength_um,
            image_pixel_pitch_um=image_pixel_pitch_um,
            pupil_radius_mm=pupil_radius_mm,
        )
        metrics = _calculate_metrics(
            intensity, target_amplitude, halo_exclusion_radius_px
        )
        rows.append(
            {
                "delta_z_mm": delta_z_mm,
                **metrics,
                "phase_file": phase_filename,
                "bmp_file": bitmap_filename,
            }
        )

    csv_path = output_path / "delta_z_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_metrics_plot(rows, output_path / "metrics_vs_delta_z.png")
    return rows


def build_delta_z_values(
    *,
    explicit_values_mm: Iterable[float] | None,
    start_mm: float | None,
    stop_mm: float | None,
    step_mm: float | None,
) -> list[float]:
    """Build an explicit, fine-range, or default coarse delta_z scan."""
    range_values = (start_mm, stop_mm, step_mm)
    if explicit_values_mm is not None and any(
        value is not None for value in range_values
    ):
        raise ValueError("explicit values and range scan cannot be combined")
    if explicit_values_mm is not None:
        values = [float(value) for value in explicit_values_mm]
        if not values:
            raise ValueError("explicit_values_mm must not be empty")
        return values

    if all(value is None for value in range_values):
        return list(COARSE_DELTA_Z_MM)
    if any(value is None for value in range_values):
        raise ValueError("start_mm, stop_mm and step_mm must be used together")
    assert start_mm is not None and stop_mm is not None and step_mm is not None
    if step_mm <= 0:
        raise ValueError("step_mm must be positive")
    if stop_mm < start_mm:
        raise ValueError("stop_mm must be >= start_mm")
    values = np.arange(start_mm, stop_mm + step_mm * 0.5, step_mm)
    return [float(value) for value in values if value <= stop_mm + 1e-9]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an offline WGS scan of SLM-image/objective-pupil delta_z."
    )
    parser.add_argument("--output-dir", default="delta_z_scan_outputs")
    parser.add_argument("--delta-z-mm", nargs="+", type=float)
    parser.add_argument("--scan-start-mm", type=float)
    parser.add_argument("--scan-stop-mm", type=float)
    parser.add_argument("--scan-step-mm", type=float)
    parser.add_argument("--loops", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--pupil-radius-mm", type=float)
    parser.add_argument("--halo-exclusion-radius-px", type=int, default=3)
    parser.add_argument("--seed", type=int, default=795)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    delta_z_values = build_delta_z_values(
        explicit_values_mm=args.delta_z_mm,
        start_mm=args.scan_start_mm,
        stop_mm=args.scan_stop_mm,
        step_mm=args.scan_step_mm,
    )
    output_dir = Path(args.output_dir).resolve()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repository_dir = Path(__file__).resolve().parent
    original_cwd = Path.cwd()
    try:
        os.chdir(repository_dir)
        from SLMGeneration import SLM_class

        slm = SLM_class()
    finally:
        os.chdir(original_cwd)
    slm.image_init(Plot=False)
    target_amplitude = slm.target_generate(Lattice_type="Rec", Plot=False)

    loops = args.loops if args.loops is not None else slm.Loop
    threshold = args.threshold if args.threshold is not None else slm.threshold
    pupil_radius_mm = (
        args.pupil_radius_mm
        if args.pupil_radius_mm is not None
        else slm.maskradius * 1e-3
    )
    image_pixel_pitch_um = slm.pixelpitch * abs(slm.magnification)

    rows = run_delta_z_scan(
        init_amplitude=torch.from_numpy(slm.initGaussianAmp),
        init_phase=torch.from_numpy(slm.initGaussianPhase),
        target_amplitude=torch.from_numpy(target_amplitude),
        delta_z_values_mm=delta_z_values,
        output_dir=output_dir,
        loops=loops,
        threshold=threshold,
        wavelength_um=slm.wavelength,
        image_pixel_pitch_um=image_pixel_pitch_um,
        pupil_radius_mm=pupil_radius_mm,
        slm_resolution=tuple(slm.SLMRes),
        halo_exclusion_radius_px=args.halo_exclusion_radius_px,
    )
    parameters = {
        "delta_z_values_mm": delta_z_values,
        "loops": loops,
        "threshold": threshold,
        "wavelength_um": slm.wavelength,
        "image_pixel_pitch_um": image_pixel_pitch_um,
        "pupil_radius_mm": pupil_radius_mm,
        "seed": args.seed,
    }
    (output_dir / "scan_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )

    print(f"Saved {len(rows)} independently optimized scan points to {output_dir}")


if __name__ == "__main__":
    main()
