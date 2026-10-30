import numpy as np
import pytest
from scipy.signal import fftconvolve

from bayesian_rf.deconvolution import iterative_rf, multitaper_rf, waterlevel_rf


def pair():
    dt = 0.1
    t = np.arange(-20, 80, dt)
    z = np.exp(-0.5 * ((t - 2) / 0.35) ** 2)
    h = np.zeros(101)
    h[[0, 24, 48, 72]] = [0.6, 0.3, -0.2, 0.25]
    return z, fftconvolve(z, h)[: len(z)], dt


@pytest.mark.parametrize("method", ["waterlevel", "iterative", "multitaper"])
def test_lags_polarities_scaling_and_shared_normalization(method):
    z, r, dt = pair()

    def solve(scale):
        if method == "waterlevel":
            return waterlevel_rf(z * scale, r * scale, dt, waterlevel=0.001)
        if method == "iterative":
            return iterative_rf(z * scale, r * scale, dt, max_lag=10, improvement_tolerance=1e-10)
        return multitaper_rf(z * scale, r * scale, dt, source_noise=np.zeros(100))

    a, b = solve(1), solve(1e-12)
    t, h, self_rf = a.waveform()
    np.testing.assert_allclose(h, b.waveform()[1], atol=1e-8)
    assert self_rf[np.argmin(abs(t))] == pytest.approx(1.0)
    for lag, target in [(0, 0.6), (2.4, 0.3), (4.8, -0.2), (7.2, 0.25)]:
        assert h[np.argmin(abs(t - lag))] == pytest.approx(target, abs=0.065)


def test_iterative_error_decreases_and_reports_stopping():
    z, r, dt = pair()
    a = iterative_rf(z, r, dt, max_lag=10, improvement_tolerance=1e-9)
    assert a.diagnostics["stopped_by_tolerance"]
    assert np.all(np.diff(a.diagnostics["fractional_sse_trace"]) <= 1e-10)
    assert a.diagnostics["final_fractional_sse"] < 1e-4
    b = iterative_rf(z, r, dt, max_lag=10, max_iterations=1)
    assert not b.diagnostics["stopped_by_tolerance"]


def test_matched_vertical_resolution_and_no_radial_sign_flip():
    z, r, dt = pair()
    a = waterlevel_rf(z, -r, dt)
    b = multitaper_rf(z, -r, dt, source_noise=np.ones(100) * 0.01)
    common = np.minimum(a.self_transfer, b.self_transfer)
    t, ar, az = a.waveform(common_transfer=common)
    _, br, bz = b.waveform(common_transfer=common)
    np.testing.assert_allclose(az, bz, atol=1e-14)
    assert ar[np.argmin(abs(t))] < 0
    assert br[np.argmin(abs(t))] < 0
    with pytest.raises(ValueError):
        a.waveform(common_transfer=np.ones_like(common) * 2)


@pytest.mark.parametrize("method", [waterlevel_rf, iterative_rf])
def test_invalid_inputs(method):
    with pytest.raises(ValueError):
        method(np.zeros(20), np.ones(20), 0.25)
