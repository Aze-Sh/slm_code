import json

import numpy as np
from PIL import Image

import make_delta_z_montage as montage


def _gaussian_image(shape, centers, sigma):
    yy, xx = np.indices(shape)
    image = np.full(shape, 4.0, dtype=np.float32)
    for cy, cx in centers:
        image += 100 * np.exp(
            -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)
        )
    return image


def test_find_averages_uses_scan_order_and_decodes_sign(tmp_path):
    np.save(
        tmp_path / "camera_average_scan_001_delta_z_p005.500mm.npy",
        np.zeros((4, 4), dtype=np.float32),
    )
    np.save(
        tmp_path / "camera_average_scan_000_delta_z_m020.000mm.npy",
        np.zeros((4, 4), dtype=np.float32),
    )

    records = montage.find_averages(tmp_path)

    assert [(index, delta_z) for index, delta_z, _ in records] == [
        (0, -20.0),
        (1, 5.5),
    ]


def test_median_registered_spot_uses_all_valid_spots():
    centers = np.array([[20, 20], [20, 60], [60, 20], [60, 60]])
    image = _gaussian_image((81, 81), centers, sigma=2.0)

    median_spot, count = montage.median_registered_spot(
        image, centers, radius=12
    )

    assert count == 4
    assert median_spot.shape == (25, 25)
    assert np.max(median_spot) == 1.0
    assert median_spot[12, 12] == 1.0


def test_build_montages_writes_three_viewable_pngs(tmp_path):
    centers = [[20, 20], [20, 60], [60, 20], [60, 60]]
    parameters = {
        "camera_roi_xywh": [0, 0, 81, 81],
        "background_percentile": 10.0,
        "detected_spot_centers_yx": centers,
    }
    (tmp_path / "experimental_parameters.json").write_text(
        json.dumps(parameters), encoding="utf-8"
    )
    np.save(
        tmp_path / "camera_average_scan_000_delta_z_m005.000mm.npy",
        _gaussian_image((81, 81), centers, sigma=2.0),
    )
    np.save(
        tmp_path / "camera_average_scan_001_delta_z_p000.000mm.npy",
        _gaussian_image((81, 81), centers, sigma=3.0),
    )

    outputs = montage.build_montages(
        tmp_path, columns=2, thumbnail_px=64, patch_radius_px=12
    )

    assert len(outputs) == 3
    for output in outputs:
        assert output.is_file()
        with Image.open(output) as image:
            assert image.width > 0
            assert image.height > 0
