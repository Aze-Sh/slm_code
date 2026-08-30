import pytest
import torch

import WGS
from WGS import WGS_phase_generate


def _legacy_non_uniformity(amplitude_at_foci, target_amplitude, total_sites):
    intensity_at_foci = torch.square(amplitude_at_foci) / torch.sum(
        torch.square(amplitude_at_foci)
    )
    nonzero_intensity = torch.abs(intensity_at_foci[intensity_at_foci != 0])
    target_intensity = torch.square(target_amplitude) / torch.sum(
        torch.square(target_amplitude)
    )
    nonzero_target = torch.abs(target_intensity[target_intensity != 0])
    return (
        torch.sqrt(torch.sum(torch.square(nonzero_intensity - nonzero_target)))
        / total_sites
        / torch.mean(nonzero_target)
    )


def _legacy_wgs(init_amplitude, init_phase, target_amplitude, loops, threshold):
    slm_field = init_amplitude * torch.exp(1j * init_phase)
    target_amplitude = target_amplitude / torch.sqrt(
        torch.sum(torch.square(target_amplitude))
    )
    target_mask = (target_amplitude != 0) * 1
    total_sites = torch.count_nonzero(target_amplitude)
    weight = torch.abs(target_amplitude) / torch.sum(torch.abs(target_amplitude))
    previous_gain = torch.ones(1, dtype=init_amplitude.dtype)
    focal_phase = torch.zeros_like(target_amplitude)

    for count in range(loops):
        focal_field = torch.fft.fftshift(torch.fft.fft2(slm_field))
        focal_field = focal_field / torch.sqrt(
            torch.sum(torch.square(torch.abs(focal_field)))
        )
        focal_amplitude = torch.abs(focal_field)
        amplitude_at_foci = focal_amplitude * target_mask
        error = _legacy_non_uniformity(
            amplitude_at_foci, target_amplitude, total_sites
        )
        average = torch.sum(amplitude_at_foci) / total_sites * target_mask
        gain = torch.where(
            amplitude_at_foci != 0,
            average * weight / amplitude_at_foci * previous_gain,
            torch.zeros_like(average),
        )
        constrained_amplitude = target_amplitude * gain
        if error > threshold or count == 0:
            focal_phase = torch.angle(focal_field)
        constrained_field = constrained_amplitude * torch.exp(1j * focal_phase)
        slm_field = torch.fft.ifft2(torch.fft.ifftshift(constrained_field))
        slm_phase = torch.angle(slm_field)
        slm_field = init_amplitude * torch.exp(1j * slm_phase)
        previous_gain = gain

    return torch.angle(slm_field)


def test_zero_delta_z_is_exactly_the_legacy_wgs_path():
    """Catches any zero-distance propagation or refactor that changes legacy output."""
    generator = torch.Generator().manual_seed(795)
    init_amplitude = torch.rand((8, 8), generator=generator, dtype=torch.float64)
    init_phase = (
        torch.rand((8, 8), generator=generator, dtype=torch.float64) * 2 * torch.pi
        - torch.pi
    )
    target_amplitude = torch.zeros((8, 8), dtype=torch.float64)
    target_amplitude[2, 3] = 1.0
    target_amplitude[5, 4] = 1.0

    expected = _legacy_wgs(init_amplitude, init_phase, target_amplitude, 3, 0.01)
    actual = WGS_phase_generate(
        init_amplitude,
        init_phase,
        target_amplitude,
        Loop=3,
        threshold=0.01,
        delta_z_mm=0.0,
        wavelength_um=0.795,
        image_pixel_pitch_um=12.5,
    )

    torch.testing.assert_close(actual.cpu(), expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("distance_mm", [-3.0, 3.0])
def test_angular_spectrum_propagates_a_plane_wave_with_the_correct_sign(
    distance_mm,
):
    """Catches a wrong distance sign or incorrect spatial-frequency scaling."""
    size = 8
    wavelength_um = 0.795
    pitch_um = 12.5
    mode_x = 1
    mode_y = 2
    y, x = torch.meshgrid(
        torch.arange(size, dtype=torch.float64),
        torch.arange(size, dtype=torch.float64),
        indexing="ij",
    )
    field = torch.exp(
        2j * torch.pi * (mode_x * x / size + mode_y * y / size)
    )

    propagated = WGS.angular_spectrum_propagate(
        field,
        distance_mm=distance_mm,
        wavelength_um=wavelength_um,
        pixel_pitch_um=pitch_um,
    )

    wavelength_m = wavelength_um * 1e-6
    pitch_m = pitch_um * 1e-6
    distance_m = distance_mm * 1e-3
    fx = mode_x / (size * pitch_m)
    fy = mode_y / (size * pitch_m)
    longitudinal_frequency = (
        1.0 / wavelength_m**2 - fx**2 - fy**2
    ) ** 0.5
    expected_factor = torch.exp(
        torch.tensor(
            2j * torch.pi * distance_m * longitudinal_frequency,
            dtype=torch.complex128,
        )
    )

    torch.testing.assert_close(
        propagated, field * expected_factor, rtol=1e-11, atol=1e-11
    )


def test_nonzero_delta_z_is_part_of_each_wgs_iteration():
    """Catches a delta_z argument that is accepted but ignored by WGS."""
    generator = torch.Generator().manual_seed(17)
    init_amplitude = torch.rand((8, 8), generator=generator, dtype=torch.float64)
    init_phase = (
        torch.rand((8, 8), generator=generator, dtype=torch.float64) * 2 * torch.pi
        - torch.pi
    )
    target_amplitude = torch.zeros((8, 8), dtype=torch.float64)
    target_amplitude[1, 2] = 1.0
    target_amplitude[5, 6] = 0.7

    zero_phase = WGS_phase_generate(
        init_amplitude,
        init_phase,
        target_amplitude,
        Loop=2,
        delta_z_mm=0.0,
        wavelength_um=0.795,
        image_pixel_pitch_um=12.5,
        pupil_radius_mm=0.035,
    )
    shifted_phase = WGS_phase_generate(
        init_amplitude,
        init_phase,
        target_amplitude,
        Loop=2,
        delta_z_mm=5.0,
        wavelength_um=0.795,
        image_pixel_pitch_um=12.5,
        pupil_radius_mm=0.035,
    )

    wrapped_difference = torch.angle(torch.exp(1j * (shifted_phase - zero_phase)))
    assert torch.max(torch.abs(wrapped_difference)).item() > 1e-4
