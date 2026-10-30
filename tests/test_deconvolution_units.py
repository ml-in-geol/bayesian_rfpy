from dataclasses import replace

import numpy as np
import pytest

from bayesian_rf.benchmarks import benchmark_config, make_benchmark
from bayesian_rf.deconvolution import _allowed_gaussians, _estimate_alpha, run_bayesian_deconvolution


def test_amplitude_proposals_are_independent_of_waveform_units():
    case = make_benchmark("ricker_sparse")
    lag = case.lag_times
    reference = _estimate_alpha(case.input_signal, case.observed_output, lag, 0.2, case.dt)
    tiny = _estimate_alpha(case.input_signal * 1e-12, case.observed_output * 1e-12, lag, 0.2, case.dt)
    assert tiny == pytest.approx(reference, rel=1e-12)


def test_sampled_rf_is_independent_of_common_input_scale():
    case = make_benchmark("impulse_gaussian")
    config = benchmark_config(case, iterations=3000, burnin=1500, save_every=100)
    results = []
    for scale in (1.0, 1e-12, 1e6):
        result = run_bayesian_deconvolution(
            case.input_signal * scale,
            case.observed_output * scale,
            case.noise_window * scale,
            case.dt,
            replace(config),
        )
        results.append(result)
        np.testing.assert_array_equal(result.parent, case.input_signal * scale)
    for result, scale in zip(results[1:], (1e-12, 1e6)):
        np.testing.assert_allclose(
            result.saved_receiver_functions, results[0].saved_receiver_functions, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(result.saved_sigma / scale, results[0].saved_sigma, rtol=1e-10)


def test_dimension_warmup_finishes_before_retained_samples():
    for burnin in (0, 1000, 125000, 1000000):
        warmup = burnin // 2
        assert _allowed_gaussians(max(1, warmup), 30, warmup) == 30
        assert _allowed_gaussians(burnin + 1, 30, warmup) == 30
