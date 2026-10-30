import numpy as np
from scipy.signal import fftconvolve

from bayesian_rf.deconvolution import source_resolution


def test_resolution_matches_independent_deconvolution_without_changing_polarity():
    dt = 0.1
    t = np.arange(600) * dt
    source = np.exp(-0.5 * ((t - 10) / 0.6) ** 2)
    lags = np.arange(201) * dt
    rf = np.exp(-0.5 * ((lags - 2.4) / 0.25) ** 2) - 0.5 * np.exp(-0.5 * ((lags - 7.2) / 0.4) ** 2)
    response = fftconvolve(source, rf)[: len(source)] * dt
    operator = source_resolution(source, dt)
    times, model = operator.apply(rf, lags, max_lag=20)
    _, data = operator.deconvolve(response, max_lag=20)
    np.testing.assert_allclose(model, data, atol=1e-8)
    assert model[np.argmin(abs(times - 2.4))] > 0
    assert model[np.argmin(abs(times - 7.2))] < 0
    other = source_resolution(source * 1e-12, dt)
    _, tiny = other.deconvolve(response * 1e-12, max_lag=20)
    np.testing.assert_allclose(data, tiny, atol=1e-12)
    _, stack = operator.apply(np.stack([rf, 2 * rf]), lags, max_lag=20)
    np.testing.assert_allclose(stack[1], 2 * model, atol=1e-12)
