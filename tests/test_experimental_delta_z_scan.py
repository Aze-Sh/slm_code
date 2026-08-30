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


class FakeDisplay:
    def __init__(self, size_xy):
        self.size_xy = size_xy
        self.frames = []

    def getSize(self):
        return self.size_xy

    def updateArray(self, frame):
        self.frames.append(np.asarray(frame).copy())


class FakeCamera:
    def __init__(self, frames):
        self.frames = list(frames)
        self.exposures = []

    def capture(self, exposure_us):
        self.exposures.append(exposure_us)
        if not self.frames:
            raise RuntimeError("fake camera has no frames left")
        return self.frames.pop(0)


def _make_scan_points(scan_module, tmp_path, phase_values=(10, 20)):
    points = []
    for index, (delta_z, phase_value) in enumerate(
        zip((-5.0, 0.0), phase_values)
    ):
        sign = "m005.000" if delta_z < 0 else "p000.000"
        label = f"scan_{index:03d}_delta_z_{sign}mm"
        bitmap = tmp_path / f"slm_phase_{label}.bmp"
        Image.fromarray(
            np.full((4, 6), phase_value, dtype=np.uint8), mode="L"
        ).save(bitmap)
        points.append(scan_module.ScanPoint(delta_z, bitmap, label))
    return points


def test_acquisition_displays_in_order_averages_frames_and_saves_outputs(
    scan_module, tmp_path
):
    points = _make_scan_points(scan_module, tmp_path)
    display = FakeDisplay((8, 4))
    camera = FakeCamera(
        [
            np.ones((6, 6), dtype=np.uint8),
            np.full((6, 6), 3, dtype=np.uint8),
            np.full((6, 6), 5, dtype=np.uint8),
            np.full((6, 6), 7, dtype=np.uint8),
        ]
    )
    sleep_calls = []
    output_dir = tmp_path / "experiment"

    acquired = scan_module.acquire_scan_points(
        points,
        display,
        camera,
        output_dir,
        exposure_us=50,
        frames_per_point=2,
        settle_seconds=0.25,
        correction=None,
        lut=256,
        save_raw_frames=True,
        sleep_fn=sleep_calls.append,
    )

    assert [frame[0, 1] for frame in display.frames] == [10, 20]
    assert [frame[0, 0] for frame in display.frames] == [0, 0]
    assert sleep_calls == [0.25, 0.25]
    assert camera.exposures == [50, 50, 50, 50]
    assert np.all(acquired[0].average == 2)
    assert np.all(acquired[1].average == 6)
    for result in acquired:
        assert result.average_npy.is_file()
        assert result.average_tiff.is_file()
        assert result.raw_frames_npy is not None
        assert result.raw_frames_npy.is_file()
    np.testing.assert_array_equal(
        np.load(acquired[0].raw_frames_npy),
        np.stack(
            [
                np.ones((6, 6), dtype=np.uint8),
                np.full((6, 6), 3, dtype=np.uint8),
            ]
        ),
    )


def test_acquisition_rejects_mismatched_camera_shapes(scan_module, tmp_path):
    points = _make_scan_points(scan_module, tmp_path)[:1]
    camera = FakeCamera(
        [np.zeros((4, 4), dtype=np.uint8), np.zeros((5, 4), dtype=np.uint8)]
    )

    with pytest.raises(ValueError, match="camera frame shape changed"):
        scan_module.acquire_scan_points(
            points,
            FakeDisplay((8, 4)),
            camera,
            tmp_path / "experiment",
            exposure_us=50,
            frames_per_point=2,
            settle_seconds=0,
            correction=None,
            lut=256,
            save_raw_frames=False,
            sleep_fn=lambda _: None,
        )


@pytest.mark.parametrize("frames", [0, -1])
def test_acquisition_rejects_nonpositive_frame_count(
    scan_module, tmp_path, frames
):
    points = _make_scan_points(scan_module, tmp_path)

    with pytest.raises(ValueError, match="frames_per_point must be positive"):
        scan_module.acquire_scan_points(
            points,
            FakeDisplay((8, 4)),
            FakeCamera([]),
            tmp_path / "experiment",
            exposure_us=50,
            frames_per_point=frames,
            settle_seconds=0,
            correction=None,
            lut=256,
            save_raw_frames=False,
        )


def test_acquisition_refuses_nonempty_output_directory(scan_module, tmp_path):
    points = _make_scan_points(scan_module, tmp_path)
    output_dir = tmp_path / "experiment"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="fresh output directory"):
        scan_module.acquire_scan_points(
            points,
            FakeDisplay((8, 4)),
            FakeCamera([]),
            output_dir,
            exposure_us=50,
            frames_per_point=1,
            settle_seconds=0,
            correction=None,
            lut=256,
            save_raw_frames=False,
        )

    assert marker.read_text(encoding="utf-8") == "keep"
