import csv
import importlib

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def scan_module():
    return importlib.import_module("experimental_delta_z_scan")


def _write_scan_manifest(scan_dir, rows):
    with (scan_dir / "delta_z_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["delta_z_mm", "bmp_file"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_center_phase_on_1280_transport_without_interpolation(scan_module):
    """Catches slmpy stretching 1272 LCOS columns over 1280 transport pixels."""
    phase = np.arange(1024 * 1272, dtype=np.uint32)
    phase = np.remainder(phase, 256).astype(np.uint8).reshape(1024, 1272)

    frame = scan_module.center_phase_on_display(phase, (1280, 1024))

    assert frame.shape == (1024, 1280)
    np.testing.assert_array_equal(frame[:, 4:1276], phase)
    assert not frame[:, :4].any()
    assert not frame[:, 1276:].any()


def test_center_phase_rejects_a_smaller_display(scan_module):
    phase = np.zeros((1024, 1272), dtype=np.uint8)

    with pytest.raises(ValueError, match="does not fit"):
        scan_module.center_phase_on_display(phase, (1024, 768))


def test_calibration_applies_modular_correction_then_lut(scan_module):
    phase = np.array([[250, 128], [64, 1]], dtype=np.uint8)
    correction = np.array([[10, 128], [0, 255]], dtype=np.uint8)

    calibrated = scan_module.apply_slm_calibration(
        phase, correction=correction, lut=224
    )

    expected_modular_sum = np.array([[4, 0], [64, 0]], dtype=np.uint8)
    expected = (expected_modular_sum.astype(np.float64) / 256 * 224).astype(
        np.uint8
    )
    np.testing.assert_array_equal(calibrated, expected)


def test_load_scan_points_uses_csv_order_and_requires_every_bmp(
    scan_module, tmp_path
):
    names = [
        "slm_phase_scan_000_delta_z_m005.000mm.bmp",
        "slm_phase_scan_001_delta_z_p000.000mm.bmp",
    ]
    for name in names:
        Image.fromarray(np.zeros((4, 6), np.uint8), mode="L").save(
            tmp_path / name
        )
    _write_scan_manifest(
        tmp_path,
        [
            {"delta_z_mm": -5, "bmp_file": names[0]},
            {"delta_z_mm": 0, "bmp_file": names[1]},
        ],
    )

    points = scan_module.load_scan_points(tmp_path)

    assert [point.delta_z_mm for point in points] == [-5.0, 0.0]
    assert [point.bmp_path.name for point in points] == names
    assert [point.scan_label for point in points] == [
        "scan_000_delta_z_m005.000mm",
        "scan_001_delta_z_p000.000mm",
    ]

    (tmp_path / names[1]).unlink()
    with pytest.raises(FileNotFoundError, match=names[1]):
        scan_module.load_scan_points(tmp_path)


def test_load_scan_points_rejects_duplicate_delta_z(scan_module, tmp_path):
    names = [
        "slm_phase_scan_000_delta_z_p000.000mm.bmp",
        "slm_phase_scan_001_delta_z_p000.000mm.bmp",
    ]
    for name in names:
        Image.fromarray(np.zeros((4, 6), np.uint8), mode="L").save(
            tmp_path / name
        )
    _write_scan_manifest(
        tmp_path,
        [
            {"delta_z_mm": 0, "bmp_file": names[0]},
            {"delta_z_mm": 0, "bmp_file": names[1]},
        ],
    )

    with pytest.raises(ValueError, match="duplicate delta_z"):
        scan_module.load_scan_points(tmp_path)
