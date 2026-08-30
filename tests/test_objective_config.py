from pathlib import Path

import pytest
import torch

from SLMGeneration import SLM_class
from WGS import circular_pupil_mask


REPOSITORY_DIR = Path(__file__).resolve().parents[1]


def test_objective_config_generates_the_10_5_mm_target_sampling(monkeypatch):
    """Catches using the 200 mm relay lens as the Obj1 Fourier focal length."""
    monkeypatch.chdir(REPOSITORY_DIR)
    slm = SLM_class()
    assert slm.objective_model == "LD Plan 19X/0.65"
    assert slm.objective_na == pytest.approx(0.65)
    assert slm.focallength == 10500
    assert slm.wavelength == pytest.approx(0.795)
    assert slm.pixelpitch == pytest.approx(12.5)
    assert slm.magnification == pytest.approx(1.0)
    assert slm.spacing == pytest.approx([7.875, 7.875])
    assert slm.SLMRes == [1272, 1024]
    assert slm.zernike_aperture_radius == 6825
    slm.arraySizeBit = [8, 8]
    slm.arraysize = [2, 2]

    slm.image_init(Plot=False)
    target = slm.target_generate(Lattice_type="Rec", Plot=False)

    assert slm.Focalpitchx == pytest.approx(2.60859375)
    nonzero_y, nonzero_x = target.nonzero()
    assert len(nonzero_x) == 4
    assert sorted(set(nonzero_x)) == [126, 129]
    assert sorted(set(nonzero_y)) == [126, 129]
    assert (nonzero_x.max() - nonzero_x.min()) * slm.Focalpitchx == pytest.approx(
        7.82578125
    )


def test_objective_config_applies_the_na_limited_pupil(monkeypatch):
    """Catches confusing the 36 mm clear aperture with the effective pupil."""
    monkeypatch.chdir(REPOSITORY_DIR)
    slm = SLM_class()
    assert slm.maskradius == 6825
    mask = circular_pupil_mask(
        (1201, 1201),
        slm.pixelpitch * abs(slm.magnification),
        slm.maskradius * 1e-3,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    center = mask.shape[0] // 2

    assert mask[center, center + 480]
    assert not mask[center, center + 560]
