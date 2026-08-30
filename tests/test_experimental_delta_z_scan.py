import csv
import importlib
import json
import sys

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


def _gaussian_grid(shape, centers, sigma, amplitudes, background=2.0):
    y, x = np.indices(shape, dtype=np.float64)
    image = np.full(shape, background, dtype=np.float64)
    for (center_y, center_x), amplitude in zip(centers, amplitudes):
        radius_squared = (y - center_y) ** 2 + (x - center_x) ** 2
        image += amplitude * np.exp(-radius_squared / (2 * sigma**2))
    return image


def _acquired_from_images(scan_module, tmp_path, images, saturation=None):
    results = []
    saturation = saturation or [0.0] * len(images)
    for index, (image, fraction) in enumerate(zip(images, saturation)):
        delta_z = float(index * 5)
        label = f"scan_{index:03d}_delta_z_p{delta_z:07.3f}mm"
        bitmap = tmp_path / f"slm_phase_{label}.bmp"
        Image.fromarray(np.zeros((4, 6), dtype=np.uint8), mode="L").save(
            bitmap
        )
        average_npy = tmp_path / f"camera_average_{label}.npy"
        average_tiff = tmp_path / f"camera_average_{label}.tiff"
        np.save(average_npy, image)
        Image.fromarray(np.asarray(image, dtype=np.uint16)).save(average_tiff)
        results.append(
            scan_module.AcquiredPoint(
                point=scan_module.ScanPoint(delta_z, bitmap, label),
                average=np.asarray(image, dtype=np.float64),
                saturation_fraction=fraction,
                peak_raw=float(np.max(image)),
                average_npy=average_npy,
                average_tiff=average_tiff,
                raw_frames_npy=None,
            )
        )
    return results


def test_common_spots_and_metrics_prefer_sharp_uniform_low_halo_image(
    scan_module, tmp_path
):
    expected_centers = np.array(
        [[20, 20], [20, 44], [44, 20], [44, 44]], dtype=int
    )
    sharp = _gaussian_grid(
        (64, 64), expected_centers, sigma=1.2, amplitudes=[100] * 4
    )
    blurred = _gaussian_grid(
        (64, 64),
        expected_centers,
        sigma=3.2,
        amplitudes=[100, 80, 60, 40],
    )
    acquired = _acquired_from_images(scan_module, tmp_path, [sharp, blurred])

    centers = scan_module.detect_common_spots(
        [point.average for point in acquired],
        expected_count=4,
        min_peak_distance_px=8,
        background_percentile=5,
    )
    rows, radius = scan_module.analyze_acquired_points(
        acquired,
        centers,
        spot_radius_px=6,
        background_percentile=5,
    )
    ranked_rows, best = scan_module.rank_quality(rows)

    assert radius == pytest.approx(6)
    np.testing.assert_allclose(centers, expected_centers, atol=1)
    assert rows[0]["mean_fwhm_px"] < rows[1]["mean_fwhm_px"]
    assert rows[0]["mean_encircled_energy_radius_50_px"] < rows[1][
        "mean_encircled_energy_radius_50_px"
    ]
    assert rows[0]["target_plane_uniformity"] > rows[1][
        "target_plane_uniformity"
    ]
    assert rows[0]["background_halo"] < rows[1]["background_halo"]
    assert rows[0]["mean_spot_sharpness"] > rows[1][
        "mean_spot_sharpness"
    ]
    assert ranked_rows[0]["quality_score"] > ranked_rows[1][
        "quality_score"
    ]
    assert best["delta_z_mm"] == pytest.approx(0.0)


def test_common_spot_detection_respects_camera_roi(scan_module):
    image = np.zeros((60, 80), dtype=np.float64)
    image[10, 10] = 1000
    image[30, 40] = 100
    image[30, 55] = 90

    centers = scan_module.detect_common_spots(
        [image],
        expected_count=2,
        roi_xywh=(30, 20, 35, 25),
        min_peak_distance_px=5,
        background_percentile=0,
    )

    np.testing.assert_array_equal(centers, np.array([[30, 40], [30, 55]]))


def test_quality_ranking_excludes_saturated_scan_point(scan_module):
    rows = [
        {
            "delta_z_mm": -5.0,
            "target_plane_uniformity": 1.0,
            "mean_spot_sharpness": 1.0,
            "background_halo": 0.0,
            "mean_fwhm_px": 1.0,
            "saturation_fraction": 0.1,
        },
        {
            "delta_z_mm": 0.0,
            "target_plane_uniformity": 0.8,
            "mean_spot_sharpness": 0.8,
            "background_halo": 0.1,
            "mean_fwhm_px": 2.0,
            "saturation_fraction": 0.0,
        },
    ]

    ranked, best = scan_module.rank_quality(
        rows, maximum_saturation_fraction=0.001
    )

    assert ranked[0]["quality_score"] > ranked[1]["quality_score"]
    assert ranked[0]["eligible_for_best"] is False
    assert ranked[1]["eligible_for_best"] is True
    assert best["delta_z_mm"] == pytest.approx(0.0)


def test_quality_ranking_rejects_an_all_saturated_scan(scan_module):
    row = {
        "delta_z_mm": 0.0,
        "target_plane_uniformity": 1.0,
        "mean_spot_sharpness": 1.0,
        "background_halo": 0.0,
        "mean_fwhm_px": 1.0,
        "saturation_fraction": 0.1,
    }

    with pytest.raises(ValueError, match="all scan points are saturated"):
        scan_module.rank_quality([row], maximum_saturation_fraction=0.001)


def test_cli_defaults_match_the_working_vimba_notebook(scan_module):
    assert "wx" not in sys.modules
    assert "vmbpy" not in sys.modules

    args = scan_module.build_parser().parse_args([])

    assert args.scan_dir == "delta_z_scan_outputs"
    assert args.output_dir == "delta_z_experiment"
    assert args.monitor == 1
    assert args.camera_index == 0
    assert args.exposure_us == pytest.approx(50)
    assert args.frames_per_point == 16
    assert args.settle_seconds == pytest.approx(1)
    assert args.correction_bmp == "CAL_LSH0804730_785nm.bmp"
    assert args.lut == 224
    assert args.no_calibration is False
    assert args.save_raw_frames is False
    assert "wx" not in sys.modules
    assert "vmbpy" not in sys.modules


def test_cli_parses_analysis_and_raw_frame_options(scan_module):
    args = scan_module.build_parser().parse_args(
        [
            "--no-calibration",
            "--save-raw-frames",
            "--roi",
            "100",
            "200",
            "800",
            "600",
            "--expected-spots",
            "64",
            "--min-peak-distance-px",
            "12",
            "--spot-radius-px",
            "8.5",
            "--saturation-level",
            "255",
        ]
    )

    assert args.no_calibration is True
    assert args.save_raw_frames is True
    assert args.roi == [100, 200, 800, 600]
    assert args.expected_spots == 64
    assert args.min_peak_distance_px == 12
    assert args.spot_radius_px == pytest.approx(8.5)
    assert args.saturation_level == pytest.approx(255)


def test_full_experimental_scan_writes_metrics_plots_and_best_delta_z(
    scan_module, tmp_path
):
    points = _make_scan_points(scan_module, tmp_path)
    centers = np.array([[20, 20], [20, 44], [44, 20], [44, 44]])
    sharp = _gaussian_grid(
        (64, 64), centers, sigma=1.2, amplitudes=[100] * 4
    )
    blurred = _gaussian_grid(
        (64, 64), centers, sigma=3.2, amplitudes=[100, 80, 60, 40]
    )
    output_dir = tmp_path / "experiment"

    rows, best = scan_module.run_experimental_scan(
        points,
        FakeDisplay((8, 4)),
        FakeCamera([sharp, blurred]),
        output_dir,
        exposure_us=50,
        frames_per_point=1,
        settle_seconds=0,
        correction=None,
        correction_name=None,
        lut=256,
        save_raw_frames=False,
        saturation_level=None,
        expected_spots=4,
        roi_xywh=None,
        min_peak_distance_px=8,
        spot_radius_px=6,
        background_percentile=5,
        maximum_saturation_fraction=0.001,
        monitor_index=1,
        camera_index=0,
        sleep_fn=lambda _: None,
    )

    assert len(rows) == 2
    assert best["delta_z_mm"] == pytest.approx(-5.0)
    expected_outputs = {
        "experimental_metrics.csv",
        "experimental_metrics_vs_delta_z.png",
        "detected_spots.png",
        "best_delta_z.json",
        "experimental_parameters.json",
    }
    assert expected_outputs <= {path.name for path in output_dir.iterdir()}
    with (output_dir / "best_delta_z.json").open(encoding="utf-8") as handle:
        saved_best = json.load(handle)
    assert saved_best["delta_z_mm"] == pytest.approx(-5.0)
    with (output_dir / "experimental_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        saved_rows = list(csv.DictReader(handle))
    assert len(saved_rows) == 2
    assert "quality_score" in saved_rows[0]
    with (output_dir / "experimental_parameters.json").open(
        encoding="utf-8"
    ) as handle:
        parameters = json.load(handle)
    assert parameters["display_transport_shape_xy"] == [8, 4]
    assert parameters["slm_active_shape_xy"] == [6, 4]
    assert parameters["exposure_us"] == pytest.approx(50)
    assert parameters["frames_per_point"] == 1
