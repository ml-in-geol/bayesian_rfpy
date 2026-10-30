import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve

from bayesian_rf.deconvolution import spiking_rf, stack_diagnostics, tapered_source


@pytest.mark.parametrize("method", ["wiener", "spectral"])
def test_spiking_lag_polarity_and_unit_invariance(method):
    dt = 0.1
    t = np.arange(-20, 60 + dt / 2, dt)
    z = np.exp(-0.5 * (t / 0.35) ** 2)
    r = 0.6 * z + 0.35 * np.exp(-0.5 * ((t - 4) / 0.35) ** 2) - 0.2 * np.exp(-0.5 * ((t - 8) / 0.35) ** 2)
    kwargs = dict(method=method, source_start_s=-3.0, source_end_s=12.0, damping=0.001, gaussian_sigma_s=0.2)
    a = spiking_rf(z, r, t, **kwargs)
    b = spiking_rf(z * 1e-12, r * 1e-12, t, **kwargs)
    np.testing.assert_allclose(a.radial, b.radial, atol=1e-10)
    for lag, expected in [(0, 0.6), (4, 0.35), (8, -0.2)]:
        i = np.argmin(abs(a.lag_s - lag))
        assert abs(a.radial[i] - expected) < 0.03
    assert np.isclose(a.source_self[np.argmin(abs(a.lag_s))], 1.0)


def test_full_vertical_is_not_gated_source_self_rf():
    t = np.arange(-20.0, 60.25, 0.25)
    z = np.exp(-0.5 * (t / 0.4) ** 2) + 0.5 * np.exp(-0.5 * ((t - 20) / 0.4) ** 2)
    a = spiking_rf(z, z, t, source_start_s=-3, source_end_s=10)
    i = np.argmin(abs(a.lag_s - 20))
    assert a.vertical[i] > 0.45
    assert abs(a.source_self[i]) < 0.02
    np.testing.assert_allclose(a.radial, a.vertical)


def test_taper_retains_time_origin_and_does_not_mutate_input():
    t = np.arange(-10.0, 21.0)
    z = np.ones(len(t))
    s = tapered_source(z, t, -3, 10, taper_s=2)
    assert s[t == 0] == 1
    assert s[t == -3] == 0
    assert s[t == -2] == pytest.approx(0.5)
    assert np.all(s[t > 10] == 0)
    assert np.all(z == 1)
    with pytest.raises(ValueError):
        tapered_source(z, t, -30, 10)


def test_wiener_matches_explicit_convolution_least_squares():
    # Explicit rectangular convolution design checks both the Toeplitz RHS
    # and the absolute-amplitude (not energy) centroid time convention.
    t = np.arange(-5.0, 21.0)
    z = np.zeros(len(t))
    z[5:8] = [1.0, 0.5, -0.2]
    a = spiking_rf(
        z,
        z,
        t,
        source_start_s=-2,
        source_end_s=8,
        taper_s=0,
        damping=0.03,
        gaussian_sigma_s=0,
        min_lag=-5,
        max_lag=10,
    )
    short = z[(t >= -2) & (t <= 8)]
    n = len(short)
    design = toeplitz(np.r_[short, np.zeros(n - 1)], np.r_[short[0], np.zeros(n - 1)])
    centroid = np.dot(np.arange(n), abs(short)) / np.sum(abs(short))
    target = np.zeros(2 * n - 1)
    target[round(centroid)] = 1
    f = np.linalg.solve(design.T @ design + 0.03 * np.dot(short, short) * np.eye(n), design.T @ target)
    expected = fftconvolve(short, f)
    i0 = round(centroid)
    assert a.source_centroid_s == pytest.approx(-2 + centroid)
    for lag in range(-i0, min(10, len(expected) - i0 - 1) + 1):
        assert a.source_self[np.argmin(abs(a.lag_s - lag))] == pytest.approx(
            expected[i0 + lag] / expected[i0], abs=1e-10
        )


def test_p_only_ringing_is_not_scored_as_coherent_conversions():
    t = np.arange(-15.0, 30.25, 0.25)
    z = np.cos(2 * np.pi * 0.4 * t) * np.exp(-abs(t) / 3)
    z = np.tile(z, (8, 1))
    r = np.linspace(0.3, 0.8, 8)[:, None] * z
    score = stack_diagnostics(r, z, t)
    assert score["loo_raw_correlation"] == pytest.approx(1.0)
    assert score["loo_conversion_correlation"] == 0


def test_empty_source_rejected():
    with pytest.raises(ValueError):
        spiking_rf([], [], [])
