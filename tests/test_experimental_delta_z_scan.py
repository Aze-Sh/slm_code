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
        assert result.average.dtype == np.float32
        assert isinstance(result.average, np.memmap)
        with Image.open(result.average_tiff) as image:
            assert image.mode == "L"
        assert (
            output_dir / f"camera_preview_{result.point.scan_label}.png"
        ).is_file()
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


def test_failed_raw_acquisition_never_exposes_partial_stack(
    scan_module, tmp_path
):
    points = _make_scan_points(scan_module, tmp_path)[:1]
    output_dir = tmp_path / "failed_raw"
    camera = FakeCamera(
        [np.zeros((4, 4), dtype=np.uint8), np.zeros((5, 4), dtype=np.uint8)]
    )

    with pytest.raises(ValueError, match="camera frame shape changed"):
        scan_module.acquire_scan_points(
            points,
            FakeDisplay((8, 4)),
            camera,
            output_dir,
            exposure_us=50,
            frames_per_point=2,
            settle_seconds=0,
            correction=None,
            lut=256,
            save_raw_frames=True,
            sleep_fn=lambda _: None,
            progress_fn=None,
        )

    assert not list(output_dir.glob("camera_raw_*.npy"))
    assert not list(output_dir.glob(".*.tmp.npy"))


def test_exact_stats_are_committed_before_preview_generation(
    scan_module, tmp_path, monkeypatch
):
    points = _make_scan_points(scan_module, tmp_path)[:1]
    point = points[0]
    output_dir = tmp_path / "preview_failure"

    def fail_preview(*args, **kwargs):
        raise OSError("preview disk failure")

    monkeypatch.setattr(scan_module, "_save_preview_png", fail_preview)
    with pytest.raises(OSError, match="preview disk failure"):
        scan_module.acquire_scan_points(
            points,
            FakeDisplay((8, 4)),
            FakeCamera([np.array([[0, 255]], dtype=np.uint8)]),
            output_dir,
            exposure_us=50,
            frames_per_point=1,
            settle_seconds=0,
            correction=None,
            lut=256,
            save_raw_frames=True,
            saturation_level=255,
            sleep_fn=lambda _: None,
            progress_fn=None,
        )

    assert (output_dir / f"camera_average_{point.scan_label}.npy").is_file()
    assert (output_dir / f"camera_raw_{point.scan_label}.npy").is_file()
    assert (output_dir / f"camera_stats_{point.scan_label}.json").is_file()


def test_acquisition_preserves_uint16_camera_tiff_range(scan_module, tmp_path):
    points = _make_scan_points(scan_module, tmp_path)[:1]
    source = np.array([[0, 4095], [1024, 2048]], dtype=np.uint16)

    acquired = scan_module.acquire_scan_points(
        points,
        FakeDisplay((8, 4)),
        FakeCamera([source]),
        tmp_path / "uint16_experiment",
        exposure_us=50,
        frames_per_point=1,
        settle_seconds=0,
        correction=None,
        lut=256,
        save_raw_frames=False,
        saturation_level=4095,
        sleep_fn=lambda _: None,
        progress_fn=None,
    )

    with Image.open(acquired[0].average_tiff) as image:
        assert image.mode in {"I;16", "I"}
        assert np.asarray(image).max() == 4095
    assert acquired[0].saturation_fraction == pytest.approx(0.25)


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


def _single_spot(shape, center, sigma_y, sigma_x):
    y, x = np.indices(shape, dtype=np.float64)
    center_y, center_x = center
    return np.exp(
        -0.5
        * (
            ((y - center_y) / sigma_y) ** 2
            + ((x - center_x) / sigma_x) ** 2
        )
    )


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


def test_gaussian_similarity_and_circularity_separate_shape_failures(
    scan_module,
):
    shape = (31, 31)
    center = (15, 15)
    y, x = np.indices(shape, dtype=np.float64)
    mask = (y - center[0]) ** 2 + (x - center[1]) ** 2 <= 15**2
    round_gaussian = _single_spot(shape, center, 2.5, 2.5)
    elliptical_gaussian = _single_spot(shape, center, 5.0, 2.0)
    radius = np.hypot(y - center[0], x - center[1])
    ring = np.exp(-0.5 * ((radius - 5.0) / 0.9) ** 2)

    round_metrics = scan_module._gaussian_round_metrics(
        round_gaussian, mask, 0, 0, *center
    )
    elliptical_metrics = scan_module._gaussian_round_metrics(
        elliptical_gaussian, mask, 0, 0, *center
    )
    ring_metrics = scan_module._gaussian_round_metrics(
        ring, mask, 0, 0, *center
    )

    assert round_metrics.gaussian_similarity > 0.99
    assert round_metrics.circularity > 0.99
    # Ellipticity is not counted twice: it remains a good Gaussian fit but
    # receives a low, separate sigma_minor/sigma_major score.
    assert elliptical_metrics.gaussian_similarity > 0.98
    assert elliptical_metrics.circularity < 0.5
    # A circular ring remains circular but is not Gaussian.
    assert ring_metrics.circularity > 0.99
    assert ring_metrics.gaussian_similarity < 0.5


def test_gaussian_similarity_penalizes_weak_ring_energy_monotonically(
    scan_module,
):
    shape = (41, 41)
    center = (20, 20)
    y, x = np.indices(shape, dtype=np.float64)
    radius = np.hypot(y - center[0], x - center[1])
    mask = radius <= 20
    core = _single_spot(shape, center, 2.5, 2.5)
    ring = np.exp(-0.5 * ((radius - 7.0) / 0.8) ** 2)
    core /= np.sum(core[mask])
    ring /= np.sum(ring[mask])

    similarities = []
    for ring_fraction in (0.0, 0.05, 0.10, 0.20):
        image = (1.0 - ring_fraction) * core + ring_fraction * ring
        metrics = scan_module._gaussian_round_metrics(
            image, mask, 0, 0, *center
        )
        assert metrics.valid
        assert metrics.circularity > 0.99
        similarities.append(metrics.gaussian_similarity)

    assert similarities[0] > 0.99
    assert all(
        first > second
        for first, second in zip(similarities, similarities[1:])
    )
    # Even a visually weak 5%-energy ring must create a measurable penalty;
    # 20% ring energy must not still look like an almost-perfect Gaussian.
    assert similarities[0] - similarities[1] > 0.03
    assert similarities[3] < 0.90


def test_gaussian_round_metrics_are_stable_under_amplitude_scaling(
    scan_module,
):
    shape = (31, 31)
    center = (15, 15)
    y, x = np.indices(shape, dtype=np.float64)
    mask = (y - center[0]) ** 2 + (x - center[1]) ** 2 <= 15**2
    image = _single_spot(shape, center, 3.5, 2.0)

    reference = scan_module._gaussian_round_metrics(
        image, mask, 0, 0, *center
    )
    scaled = scan_module._gaussian_round_metrics(
        137.0 * image, mask, 0, 0, *center
    )

    assert reference.valid and scaled.valid
    assert scaled.gaussian_similarity == pytest.approx(
        reference.gaussian_similarity, abs=1e-12
    )
    assert scaled.circularity == pytest.approx(
        reference.circularity, abs=1e-12
    )


def test_flat_background_is_not_mistaken_for_a_round_gaussian(scan_module):
    shape = (31, 31)
    center = (15, 15)
    y, x = np.indices(shape, dtype=np.float64)
    mask = (y - center[0]) ** 2 + (x - center[1]) ** 2 <= 15**2

    metrics = scan_module._gaussian_round_metrics(
        np.full(shape, 7.0), mask, 0, 0, *center
    )

    assert metrics.valid is False
    assert metrics.gaussian_similarity == pytest.approx(0.0)
    assert metrics.circularity == pytest.approx(0.0)


def test_missing_spot_has_finite_zero_shape_score_and_does_not_abort(
    scan_module, tmp_path
):
    image = np.zeros((48, 48), dtype=np.float64)
    image[2, 2] = 100.0  # signal exists, but not in the expected spot window
    acquired = _acquired_from_images(scan_module, tmp_path, [image])

    rows, _ = scan_module.analyze_acquired_points(
        acquired,
        np.asarray([[24, 24]]),
        spot_radius_px=6,
        background_percentile=0,
    )
    ranked, best = scan_module.rank_quality(rows)

    assert rows[0]["valid_spot_fraction"] == pytest.approx(0.0)
    assert rows[0]["mean_gaussian_similarity"] == pytest.approx(0.0)
    assert rows[0]["mean_spot_circularity"] == pytest.approx(0.0)
    assert rows[0]["gaussian_roundness_score"] == pytest.approx(0.0)
    assert np.isfinite(float(rows[0]["mean_fwhm_px"]))
    assert ranked[0]["quality_score"] == pytest.approx(0.0)
    assert best["quality_score"] == pytest.approx(0.0)


def test_analysis_does_not_allocate_full_coordinate_grids(
    scan_module, tmp_path, monkeypatch
):
    centers = np.array([[20, 20], [20, 44], [44, 20], [44, 44]])
    image = _gaussian_grid(
        (64, 64), centers, sigma=1.2, amplitudes=[100] * 4
    )
    acquired = _acquired_from_images(scan_module, tmp_path, [image])

    def reject_indices(*args, **kwargs):
        raise AssertionError("analysis must not allocate full x/y grids")

    monkeypatch.setattr(np, "indices", reject_indices)
    rows, _ = scan_module.analyze_acquired_points(
        acquired,
        centers,
        spot_radius_px=6,
        background_percentile=5,
    )

    assert len(rows) == 1


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


def test_common_spot_detection_crops_before_background_work(
    scan_module, monkeypatch
):
    image = np.zeros((600, 800), dtype=np.float32)
    image[230, 340] = 100
    seen_shapes = []
    original = scan_module._background_correct

    def recording_background(image_arg, percentile):
        seen_shapes.append(tuple(np.asarray(image_arg).shape))
        return original(image_arg, percentile)

    monkeypatch.setattr(
        scan_module, "_background_correct", recording_background
    )

    centers = scan_module.detect_common_spots(
        [image],
        expected_count=1,
        roi_xywh=(300, 200, 100, 80),
        min_peak_distance_px=5,
        background_percentile=0,
    )

    np.testing.assert_array_equal(centers, np.array([[230, 340]]))
    assert seen_shapes == [(80, 100)]


def test_spot_windows_scale_with_roi_and_radius_not_camera_frame(scan_module):
    centers = np.array([[2020, 3020], [2040, 3040]])

    target_mask, windows, origin_yx = scan_module._build_spot_windows(
        (4036, 5024),
        centers,
        radius=6,
        roi_xywh=(3000, 2000, 100, 80),
    )

    assert target_mask.shape == (80, 100)
    assert origin_yx == (2000, 3000)
    assert len(windows) == 2
    assert all(window.mask.size <= 13 * 13 for window in windows)


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
    assert args.analyze_existing is None
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
            "--analyze-existing",
            "delta_z_experiment_test",
        ]
    )

    assert args.no_calibration is True
    assert args.save_raw_frames is True
    assert args.roi == [100, 200, 800, 600]
    assert args.expected_spots == 64
    assert args.min_peak_distance_px == 12
    assert args.spot_radius_px == pytest.approx(8.5)
    assert args.saturation_level == pytest.approx(255)
    assert args.analyze_existing == "delta_z_experiment_test"


def test_cli_existing_analysis_does_not_touch_hardware(
    scan_module, tmp_path, monkeypatch
):
    points = _make_scan_points(scan_module, tmp_path)
    _write_scan_manifest(
        tmp_path,
        [
            {
                "delta_z_mm": point.delta_z_mm,
                "bmp_file": point.bmp_path.name,
            }
            for point in points
        ],
    )
    acquisition_dir = tmp_path / "existing"
    acquisition_dir.mkdir()
    calls = []

    def fake_existing(points_arg, directory_arg, **kwargs):
        calls.append((points_arg, directory_arg, kwargs))
        return [], {"delta_z_mm": -5.0, "quality_score": 0.75}

    def reject_hardware(*args, **kwargs):
        raise AssertionError("offline analysis must not initialize hardware")

    monkeypatch.setattr(scan_module, "run_existing_analysis", fake_existing)
    monkeypatch.setattr(scan_module, "_resolve_correction_path", reject_hardware)
    monkeypatch.setattr(scan_module, "SecondaryMonitorSLM", reject_hardware)

    scan_module.main(
        [
            "--scan-dir",
            str(tmp_path),
            "--analyze-existing",
            str(acquisition_dir),
            "--expected-spots",
            "4",
            "--saturation-level",
            "255",
        ]
    )

    assert len(calls) == 1
    assert calls[0][1] == str(acquisition_dir)


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


def test_load_existing_averages_uses_memory_maps_and_requires_all_points(
    scan_module, tmp_path
):
    points = _make_scan_points(scan_module, tmp_path)
    acquisition_dir = tmp_path / "existing"
    acquisition_dir.mkdir()
    arrays = [
        np.array([[0, 255], [5, 10]], dtype=np.float32),
        np.array([[1, 2], [3, 4]], dtype=np.float32),
    ]
    for point, array in zip(points, arrays):
        np.save(
            acquisition_dir / f"camera_average_{point.scan_label}.npy",
            array,
        )

    acquired = scan_module.load_existing_acquired_points(
        points,
        acquisition_dir,
        saturation_level=255,
        progress_fn=None,
    )

    assert all(isinstance(result.average, np.memmap) for result in acquired)
    assert acquired[0].peak_raw == pytest.approx(255)
    assert acquired[0].saturation_fraction == pytest.approx(0.25)
    assert acquired[1].saturation_fraction == pytest.approx(0)

    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    np.save(
        missing_dir / f"camera_average_{points[0].scan_label}.npy",
        arrays[0],
    )
    with pytest.raises(FileNotFoundError, match=points[1].scan_label):
        scan_module.load_existing_acquired_points(
            points,
            missing_dir,
            saturation_level=255,
            progress_fn=None,
        )


def test_existing_loader_uses_exact_acquisition_sidecar(scan_module, tmp_path):
    points = _make_scan_points(scan_module, tmp_path)[:1]
    acquisition_dir = tmp_path / "existing_with_stats"
    acquisition_dir.mkdir()
    point = points[0]
    np.save(
        acquisition_dir / f"camera_average_{point.scan_label}.npy",
        np.array([[0, 100], [5, 10]], dtype=np.float32),
    )
    (acquisition_dir / f"camera_stats_{point.scan_label}.json").write_text(
        json.dumps(
            {
                "scan_label": point.scan_label,
                "delta_z_mm": point.delta_z_mm,
                "frame_dtype": "uint8",
                "frame_shape_yx": [2, 2],
                "frames_per_point": 8,
                "exposure_us": 50,
                "effective_saturation_level": 255,
                "peak_raw": 255,
                "saturation_fraction": 0.125,
            }
        ),
        encoding="utf-8",
    )

    acquired = scan_module.load_existing_acquired_points(
        points,
        acquisition_dir,
        saturation_level=None,
        progress_fn=None,
    )

    assert acquired[0].peak_raw == pytest.approx(255)
    assert acquired[0].saturation_fraction == pytest.approx(0.125)
    assert acquired[0].saturation_fraction_is_exact is True
    assert acquired[0].saturation_fraction_source == "acquisition_sidecar"
    assert acquired[0].acquisition_exposure_us == pytest.approx(50)
    assert acquired[0].acquisition_frames_per_point == 8


def test_raw_recovery_prefers_saved_sensor_saturation_level(
    scan_module, tmp_path
):
    points = _make_scan_points(scan_module, tmp_path)[:1]
    point = points[0]
    acquisition_dir = tmp_path / "raw_with_stats"
    acquisition_dir.mkdir()
    raw = np.array(
        [
            [[0, 4095], [10, 20]],
            [[4095, 100], [10, 20]],
        ],
        dtype=np.uint16,
    )
    np.save(
        acquisition_dir / f"camera_average_{point.scan_label}.npy",
        raw.mean(axis=0, dtype=np.float32),
    )
    np.save(acquisition_dir / f"camera_raw_{point.scan_label}.npy", raw)
    (acquisition_dir / f"camera_stats_{point.scan_label}.json").write_text(
        json.dumps(
            {
                "scan_label": point.scan_label,
                "delta_z_mm": point.delta_z_mm,
                "frame_dtype": "uint16",
                "frame_shape_yx": [2, 2],
                "frames_per_point": 2,
                "exposure_us": 50,
                "effective_saturation_level": 4095,
                "peak_raw": 4095,
                "saturation_fraction": 0.25,
            }
        ),
        encoding="utf-8",
    )

    acquired = scan_module.load_existing_acquired_points(
        points,
        acquisition_dir,
        saturation_level=None,
        progress_fn=None,
    )

    assert acquired[0].saturation_fraction == pytest.approx(0.25)
    assert acquired[0].effective_saturation_level == pytest.approx(4095)
    assert acquired[0].saturation_fraction_is_exact is True


def test_existing_analysis_closes_memmaps_after_failure(
    scan_module, tmp_path, monkeypatch
):
    points = _make_scan_points(scan_module, tmp_path)
    acquisition_dir = tmp_path / "failed_analysis"
    acquisition_dir.mkdir()
    for point in points:
        np.save(
            acquisition_dir / f"camera_average_{point.scan_label}.npy",
            np.ones((16, 16), dtype=np.float32),
        )
    captured = []
    original_loader = scan_module.load_existing_acquired_points

    def recording_loader(*args, **kwargs):
        results = original_loader(*args, **kwargs)
        captured.extend(results)
        return results

    def fail_detection(*args, **kwargs):
        raise RuntimeError("intentional detection failure")

    monkeypatch.setattr(
        scan_module, "load_existing_acquired_points", recording_loader
    )
    monkeypatch.setattr(scan_module, "detect_common_spots", fail_detection)

    with pytest.raises(RuntimeError, match="intentional"):
        scan_module.run_existing_analysis(
            points,
            acquisition_dir,
            saturation_level=255,
            expected_spots=4,
            roi_xywh=None,
            min_peak_distance_px=8,
            spot_radius_px=6,
            background_percentile=5,
            maximum_saturation_fraction=0.001,
            progress_fn=None,
        )

    assert captured
    assert all(result.average._mmap.closed for result in captured)


def test_existing_scan_analysis_writes_results_without_hardware(
    scan_module, tmp_path
):
    points = _make_scan_points(scan_module, tmp_path)
    centers = np.array([[20, 20], [20, 44], [44, 20], [44, 44]])
    sharp = _gaussian_grid(
        (64, 64), centers, sigma=1.2, amplitudes=[100] * 4
    ).astype(np.float32)
    blurred = _gaussian_grid(
        (64, 64), centers, sigma=3.2, amplitudes=[100, 80, 60, 40]
    ).astype(np.float32)
    acquisition_dir = tmp_path / "existing"
    acquisition_dir.mkdir()
    for point, array in zip(points, [sharp, blurred]):
        np.save(
            acquisition_dir / f"camera_average_{point.scan_label}.npy",
            array,
        )

    rows, best = scan_module.run_existing_analysis(
        points,
        acquisition_dir,
        saturation_level=255,
        expected_spots=4,
        roi_xywh=None,
        min_peak_distance_px=8,
        spot_radius_px=6,
        background_percentile=5,
        maximum_saturation_fraction=0.001,
        progress_fn=None,
    )

    assert len(rows) == 2
    assert best["delta_z_mm"] == pytest.approx(-5.0)
    assert (acquisition_dir / "experimental_metrics.csv").is_file()
    assert (acquisition_dir / "best_delta_z.json").is_file()
    assert (acquisition_dir / "detected_spots.png").is_file()
    assert len(list(acquisition_dir.glob("camera_preview_*.png"))) == 2
    with (acquisition_dir / "experimental_parameters.json").open(
        encoding="utf-8"
    ) as handle:
        parameters = json.load(handle)
    assert parameters["analysis_mode"] == "existing_averages"
    assert (
        parameters["saturation_fraction_source"]
        == "average_threshold_estimate"
    )
    with (acquisition_dir / "best_delta_z.json").open(
        encoding="utf-8"
    ) as handle:
        best_payload = json.load(handle)
    assert best_payload["selection_is_provisional"] is True
