import csv
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from WGS import WGS_phase_generate


def test_scan_independently_optimizes_and_writes_named_outputs(tmp_path):
    """Catches warm-started scan points or outputs that lose the delta_z value."""
    scan = importlib.import_module("delta_z_scan")
    generator = torch.Generator().manual_seed(795)
    init_amplitude = torch.rand((16, 16), generator=generator, dtype=torch.float64)
    init_phase = (
        torch.rand((16, 16), generator=generator, dtype=torch.float64)
        * 2
        * torch.pi
        - torch.pi
    )
    target_amplitude = torch.zeros((16, 16), dtype=torch.float64)
    target_amplitude[5, 5] = 1.0
    target_amplitude[10, 10] = 1.0

    rows = scan.run_delta_z_scan(
        init_amplitude=init_amplitude,
        init_phase=init_phase,
        target_amplitude=target_amplitude,
        delta_z_values_mm=[-1.0, 0.0, 1.0],
        output_dir=tmp_path,
        loops=2,
        threshold=0.01,
        wavelength_um=0.795,
        image_pixel_pitch_um=12.5,
        pupil_radius_mm=0.07,
        slm_resolution=(10, 8),
        halo_exclusion_radius_px=1,
    )

    assert [row["delta_z_mm"] for row in rows] == [-1.0, 0.0, 1.0]
    expected_labels = [
        "scan_000_delta_z_m001.000mm",
        "scan_001_delta_z_p000.000mm",
        "scan_002_delta_z_p001.000mm",
    ]
    for label in expected_labels:
        phase_path = tmp_path / f"slm_phase_{label}.npy"
        bitmap_path = tmp_path / f"slm_phase_{label}.bmp"
        assert phase_path.exists()
        assert bitmap_path.exists()
        with Image.open(bitmap_path) as bitmap:
            assert bitmap.mode == "L"
            assert bitmap.size == (10, 8)

    expected_positive_phase = WGS_phase_generate(
        init_amplitude,
        init_phase,
        target_amplitude,
        Loop=2,
        threshold=0.01,
        delta_z_mm=1.0,
        wavelength_um=0.795,
        image_pixel_pitch_um=12.5,
        pupil_radius_mm=0.07,
    ).cpu().numpy()
    saved_positive_phase = np.load(
        tmp_path / "slm_phase_scan_002_delta_z_p001.000mm.npy"
    )
    np.testing.assert_allclose(saved_positive_phase, expected_positive_phase)

    csv_path = tmp_path / "delta_z_metrics.csv"
    assert csv_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 3
    required_columns = {
        "delta_z_mm",
        "wgs_error",
        "target_plane_uniformity",
        "background_halo",
        "target_efficiency",
        "peak_intensity",
        "centroid_x_px",
        "centroid_y_px",
        "peak_x_px",
        "peak_y_px",
        "centroid_peak_offset_px",
    }
    assert required_columns <= set(csv_rows[0])
    bitmap_names = [row["bmp_file"] for row in rows]
    assert bitmap_names == sorted(bitmap_names)
    assert (tmp_path / "metrics_vs_delta_z.png").exists()


def test_phase_to_screen_uses_the_full_rectangular_slm_area():
    """Catches silently leaving the 248 extra horizontal LCOS pixels blank."""
    scan = importlib.import_module("delta_z_scan")
    phase = np.zeros((8, 10), dtype=np.float64)

    screen = scan._phase_to_screen(phase, slm_resolution=(6, 4))

    assert screen.shape == (4, 6)
    assert np.all(screen == 128)


def test_scan_refuses_to_mix_with_stale_hardware_outputs(tmp_path):
    """Catches old BMP files silently contaminating USB frame-memory order."""
    stale_bitmap = tmp_path / "slm_phase_scan_000_delta_z_m020.000mm.bmp"
    stale_bitmap.write_bytes(b"old scan")
    scan = importlib.import_module("delta_z_scan")
    field = torch.ones((8, 8), dtype=torch.float64)
    target = torch.zeros((8, 8), dtype=torch.float64)
    target[3, 3] = 1.0

    with pytest.raises(FileExistsError, match="fresh output directory"):
        scan.run_delta_z_scan(
            init_amplitude=field,
            init_phase=torch.zeros_like(field),
            target_amplitude=target,
            delta_z_values_mm=[0.0],
            output_dir=tmp_path,
            loops=1,
            threshold=0.01,
            wavelength_um=0.795,
            image_pixel_pitch_um=12.5,
            pupil_radius_mm=1.0,
            slm_resolution=(8, 8),
        )

    assert stale_bitmap.read_bytes() == b"old scan"


def test_scan_index_labels_sort_beyond_three_digits():
    """Catches scan_1000 sorting before scan_999 for very fine scans."""
    scan = importlib.import_module("delta_z_scan")

    labels = [
        scan._indexed_delta_z_label(index, 1001, float(index))
        for index in (998, 999, 1000)
    ]

    assert labels == sorted(labels)
    assert labels[-1].startswith("scan_1000_")


def test_scan_parameters_record_the_complete_physical_model():
    """Catches scan outputs that cannot be traced back to their optical model."""
    scan = importlib.import_module("delta_z_scan")
    slm = SimpleNamespace(
        objective_model="LD Plan 19X/0.65",
        objective_na=0.65,
        focallength=10500,
        wavelength=0.795,
        pixelpitch=12.5,
        magnification=1,
        spacing=[7.875, 7.875],
        arraysize=[8, 8],
        ImgResX=4096,
        ImgResY=4096,
        SLMRes=[1272, 1024],
    )

    parameters = scan._build_scan_parameters(
        slm=slm,
        delta_z_values_mm=[-5.0, 0.0, 5.0],
        loops=25,
        threshold=0.01,
        pupil_radius_mm=6.825,
        seed=795,
    )

    assert parameters["objective_model"] == "LD Plan 19X/0.65"
    assert parameters["objective_na"] == pytest.approx(0.65)
    assert parameters["objective_focal_length_um"] == 10500
    assert parameters["target_spacing_um"] == [7.875, 7.875]
    assert parameters["target_array_size"] == [8, 8]
    assert parameters["calculation_grid_shape_yx"] == [4096, 4096]
    assert parameters["slm_resolution_xy"] == [1272, 1024]
    assert parameters["slm_active_shape_yx"] == [1024, 1272]
    assert parameters["input_wavefront_model"] == "collimated"
    assert parameters["bmp_calibration_applied"] is False


def test_fine_scan_range_includes_both_endpoints():
    """Catches floating-point range construction that drops the final scan point."""
    scan = importlib.import_module("delta_z_scan")

    values = scan.build_delta_z_values(
        explicit_values_mm=None,
        start_mm=-1.0,
        stop_mm=1.0,
        step_mm=0.5,
    )

    assert values == [-1.0, -0.5, 0.0, 0.5, 1.0]


def test_explicit_scan_values_override_the_default_coarse_scan():
    scan = importlib.import_module("delta_z_scan")

    values = scan.build_delta_z_values(
        explicit_values_mm=[-0.5, 0.0, 0.5],
        start_mm=None,
        stop_mm=None,
        step_mm=None,
    )

    assert values == [-0.5, 0.0, 0.5]


def test_scan_range_rejects_a_nonpositive_step():
    scan = importlib.import_module("delta_z_scan")

    with pytest.raises(ValueError, match="step_mm must be positive"):
        scan.build_delta_z_values(
            explicit_values_mm=None,
            start_mm=-1.0,
            stop_mm=1.0,
            step_mm=0.0,
        )


def test_simulated_intensity_preserves_pupil_transmission_loss():
    """Catches per-image normalization that hides power clipped by the pupil."""
    scan = importlib.import_module("delta_z_scan")
    amplitude = torch.ones((16, 16), dtype=torch.float64)
    phase = torch.zeros((16, 16), dtype=torch.float64)

    small_pupil = scan._simulate_intensity(
        amplitude,
        phase,
        delta_z_mm=0.0,
        wavelength_um=0.795,
        image_pixel_pitch_um=12.5,
        pupil_radius_mm=0.04,
    )
    full_pupil = scan._simulate_intensity(
        amplitude,
        phase,
        delta_z_mm=0.0,
        wavelength_um=0.795,
        image_pixel_pitch_um=12.5,
        pupil_radius_mm=1.0,
    )

    assert torch.sum(small_pupil).item() < torch.sum(full_pupil).item()
    assert torch.sum(full_pupil).item() == pytest.approx(1.0)


def test_simulated_intensity_preserves_slm_active_area_loss():
    """Catches normalization that hides power missing outside the real LCOS area."""
    scan = importlib.import_module("delta_z_scan")
    amplitude = torch.ones((16, 16), dtype=torch.float64)
    phase = torch.zeros((16, 16), dtype=torch.float64)

    full_grid = scan._simulate_intensity(
        amplitude,
        phase,
        delta_z_mm=0.0,
        wavelength_um=0.795,
        image_pixel_pitch_um=12.5,
        pupil_radius_mm=1.0,
    )
    physical_slm = scan._simulate_intensity(
        amplitude,
        phase,
        delta_z_mm=0.0,
        wavelength_um=0.795,
        image_pixel_pitch_um=12.5,
        pupil_radius_mm=1.0,
        slm_active_shape=(8, 8),
    )

    assert torch.sum(full_grid).item() == pytest.approx(1.0)
    assert torch.sum(physical_slm).item() == pytest.approx(0.25)


def test_delta_z_labels_do_not_collide_below_point_one_mm():
    """Catches fine-scan phase files silently overwriting one another."""
    scan = importlib.import_module("delta_z_scan")

    labels = [scan._delta_z_label(value) for value in (0.0, 0.01, 0.04)]

    assert len(set(labels)) == 3


def test_explicit_and_range_scan_modes_cannot_be_combined():
    scan = importlib.import_module("delta_z_scan")

    with pytest.raises(ValueError, match="cannot be combined"):
        scan.build_delta_z_values(
            explicit_values_mm=[0.0],
            start_mm=-1.0,
            stop_mm=1.0,
            step_mm=0.5,
        )


def test_empty_scan_is_rejected_before_outputs_are_created(tmp_path):
    scan = importlib.import_module("delta_z_scan")
    field = torch.ones((4, 4), dtype=torch.float64)
    target = torch.zeros((4, 4), dtype=torch.float64)
    target[2, 2] = 1.0

    with pytest.raises(ValueError, match="at least one delta_z"):
        scan.run_delta_z_scan(
            init_amplitude=field,
            init_phase=torch.zeros_like(field),
            target_amplitude=target,
            delta_z_values_mm=[],
            output_dir=tmp_path / "outputs",
            loops=1,
            threshold=0.01,
            wavelength_um=0.795,
            image_pixel_pitch_um=12.5,
            pupil_radius_mm=0.02,
            slm_resolution=(4, 4),
        )

    assert not (tmp_path / "outputs").exists()
