from types import SimpleNamespace

import numpy as np
import pytest

import single_frame_exposure_test as exposure_test


def test_select_scan_point_requires_exact_delta_z(monkeypatch):
    points = [
        SimpleNamespace(delta_z_mm=-10.0),
        SimpleNamespace(delta_z_mm=0.0),
        SimpleNamespace(delta_z_mm=10.0),
    ]
    monkeypatch.setattr(exposure_test, "load_scan_points", lambda _: points)

    assert exposure_test.select_scan_point("unused", 0.0) is points[1]
    with pytest.raises(ValueError, match="available values"):
        exposure_test.select_scan_point("unused", 5.0)


def test_measure_frame_reports_full_frame_and_roi_saturation():
    frame = np.zeros((5, 7), dtype=np.uint8)
    frame[0, 0] = 255
    frame[2, 3] = 200
    frame[3, 4] = 255

    result = exposure_test.measure_frame(
        frame, saturation_level=255, roi_xywh=(2, 1, 4, 3)
    )

    assert result["effective_saturation_level"] == pytest.approx(255)
    assert result["full_frame"] == {
        "peak": 255.0,
        "peak_x_px": 0,
        "peak_y_px": 0,
        "saturated_pixel_count": 2,
        "saturation_fraction": pytest.approx(2 / 35),
    }
    assert result["roi"] == {
        "peak": 255.0,
        "peak_x_px": 4,
        "peak_y_px": 3,
        "saturated_pixel_count": 1,
        "saturation_fraction": pytest.approx(1 / 12),
    }


def test_measure_frame_rejects_roi_outside_frame():
    with pytest.raises(ValueError, match="outside camera frame"):
        exposure_test.measure_frame(
            np.zeros((5, 7), dtype=np.uint8),
            saturation_level=255,
            roi_xywh=(5, 4, 3, 2),
        )
