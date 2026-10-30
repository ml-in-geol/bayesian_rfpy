from dataclasses import replace

import numpy as np
import pytest
from scipy.signal import fftconvolve

from bayesian_rf.deconvolution import multitaper_rf
from bayesian_rf.cps import IsotropicLayer, LayerCakeModel, forward_ps_receiver_function
from bayesian_rf.inversion import LayerCakeModelState
from bayesian_rf.inversion import MultitaperOperator, MultitaperPSStackForward, cps_impulse_masses


def source():
    t = np.arange(301) * 0.25 - 20
    return (1 - 2 * (np.pi * 0.3 * t) ** 2) * np.exp(-((np.pi * 0.3 * t) ** 2))


def test_operator_matches_direct_estimator_and_units():
    z = source()
    rng = np.random.default_rng(17)
    r, noise = rng.normal(size=z.size), rng.normal(size=60) * 0.03
    expected = multitaper_rf(z, r, 0.25, source_noise=noise).waveform()[1]
    for scale in (1.0, 1e-12):
        op = MultitaperOperator.from_source(z * scale, noise * scale, 0.25)
        np.testing.assert_allclose(op.apply(r * scale), expected, atol=1e-12)


def test_kernel_matrix_matches_finite_convolution_with_late_pulses():
    z = source()
    h = np.zeros(101)
    h[[0, 10, 29, 60]] = [0.5, 0.3, 0.25, -0.2]
    radial = fftconvolve(z, h)[: z.size]
    op = MultitaperOperator.from_source(z, np.zeros(60), 0.25)
    actual = op.kernel_matrix(np.arange(h.size) * 0.25) @ h
    expected = multitaper_rf(z, radial, 0.25, source_noise=np.zeros(60)).waveform()[1]
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_cps_pulse_replacement_and_homogeneous_model():
    z = source()
    op = MultitaperOperator.from_source(z, np.zeros(60), 0.25)
    homogeneous = LayerCakeModel((IsotropicLayer(10.0, 4.0, 2.0, 2.2), IsotropicLayer(0.0, 4.0, 2.0, 2.2)))
    outputs = []
    for alpha in (4.0, 8.0, 16.0):
        syn = forward_ps_receiver_function(
            homogeneous, 0.12, sampling_interval_s=0.125, npts=1024, delay_s=20.0, gaussian_alpha=alpha
        )
        times, masses = cps_impulse_masses(syn, 0.25)
        outputs.append(op.kernel_matrix(times) @ masses)
        scale = np.sum(syn.amplitudes) * syn.sampling_interval_s
        np.testing.assert_allclose(outputs[-1], scale * op.apply(z), atol=3e-5)
    np.testing.assert_allclose(outputs[0], outputs[2], atol=3e-5)
    with pytest.raises(ValueError):
        cps_impulse_masses(syn, 0.125)


def test_identical_slownesses_and_critical_halfspace():
    op = MultitaperOperator.from_source(source(), np.zeros(60), 0.25)
    forward = MultitaperPSStackForward((op, op), np.array([0.12, 0.12]))
    state = LayerCakeModelState((12.0, 15.0, 20.0), (2.2, 3.0, 3.7), 4.2, (1.8,) * 4)
    np.testing.assert_allclose(forward(state), replace(forward, exact=True)(state), atol=1e-12)
    assert np.all(np.isnan(forward(replace(state, halfspace_vs_km_s=5.5))))
    assert np.all(np.isnan(forward(replace(state, vs_km_s=(5.5, 3.0, 3.7)))))


def test_kks21_structure_is_inside_supported_domain():
    from bayesian_rf.inversion import LayerCakeInversionConfig, _is_valid

    vs = np.array([1.85, 2.8, 3.753, 4.334])
    vp = np.array([3.8, 4.5, 6.224, 7.668])
    config = LayerCakeInversionConfig(ray_parameter_s_per_km=0.12378725526113792, vpvs_bounds=(1.4, 2.4))
    state = LayerCakeModelState((10.0, 14.0, 24.0), tuple(vs[:3]), vs[-1], tuple(vp / vs))
    assert _is_valid(state, config)
