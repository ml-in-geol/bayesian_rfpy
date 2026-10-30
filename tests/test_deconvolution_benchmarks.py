"""Self-contained deconvolution checks: no external data, CPS or TauP calls."""

from dataclasses import replace
import json
import os
import sys

import numpy as np
import pytest

from bayesian_rf.benchmarks import (
    CASE_DESCRIPTIONS,
    benchmark_config,
    boxcar,
    convolution_matrix,
    evaluate_benchmark,
    gaussian,
    make_benchmark,
    relative_l2,
    ricker,
    ridge_reference,
    run_benchmark,
)
from bayesian_rf.deconvolution import _predict_daughter


def test_analytic_pulses_have_explicit_conventions():
    t = np.array([-1.0, 0.0, 1.0])
    np.testing.assert_allclose(gaussian(t, center=0, width=1), [np.exp(-0.5), 1, np.exp(-0.5)])
    np.testing.assert_array_equal(boxcar(t, start=0, stop=1), [0, 1, 0])
    wavelet = ricker(t, center=0, frequency=0.5)
    assert wavelet[1] == 1.0
    assert wavelet[0] == wavelet[-1] < 0.0
    fine_t = np.linspace(-8, 8, 16001)
    assert abs(np.sum(ricker(fine_t, center=0, frequency=0.5)) * 0.001) < 1e-12


@pytest.mark.parametrize("dt", [0.05, 0.1])
def test_shifted_impulse_preserves_kernel_amplitude_and_lag(dt):
    case = make_benchmark("impulse_gaussian", dt=dt, snr=np.inf)
    expected = np.zeros_like(case.clean_output)
    offset = round(2.0 / dt)
    expected[offset : offset + case.true_kernel.size] = case.true_kernel
    np.testing.assert_allclose(case.clean_output, expected, atol=1e-14)
    np.testing.assert_allclose(
        _predict_daughter(case.input_signal, case.true_kernel, dt), expected, atol=1e-14
    )
    assert case.times[np.argmax(case.clean_output)] == pytest.approx(5.0)


@pytest.mark.parametrize("name", CASE_DESCRIPTIONS)
def test_all_cases_use_full_linear_convolution_and_reproducible_noise(name):
    case = make_benchmark(name)
    other = make_benchmark(name)
    np.testing.assert_array_equal(case.observed_output, other.observed_output)
    np.testing.assert_array_equal(case.noise_window, other.noise_window)
    op = convolution_matrix(case.input_signal, case.true_kernel.size, case.dt)
    np.testing.assert_allclose(op @ case.true_kernel, case.clean_output, atol=1e-14)
    np.testing.assert_allclose(
        _predict_daughter(case.input_signal, case.true_kernel, case.dt), case.clean_output, atol=1e-14
    )
    assert case.input_signal.size == 2 * case.true_kernel.size - 1
    np.testing.assert_array_equal(case.input_signal[case.true_kernel.size :], 0)
    assert case.noise_sigma == pytest.approx(np.sqrt(np.mean(case.clean_output**2)) / 20.0)
    noisy = make_benchmark(name, seed=987)
    np.testing.assert_array_equal(noisy.true_kernel, case.true_kernel)
    assert not np.array_equal(noisy.observed_output, case.observed_output)
    noiseless = make_benchmark(name, snr=np.inf)
    np.testing.assert_array_equal(noiseless.clean_output, noiseless.observed_output)
    np.testing.assert_array_equal(noiseless.noise_window, 0)


def test_boxcar_is_an_out_of_basis_truth_not_a_gaussian_approximation():
    case = make_benchmark("gaussian_boxcar")
    np.testing.assert_array_equal(np.unique(case.true_kernel), [0, 0.5])
    assert np.sum(case.true_kernel) * case.dt == pytest.approx(1.0)


def test_resolution_pair_holds_truth_and_source_energy_fixed():
    narrow = make_benchmark("resolution_narrow")
    broad = make_benchmark("resolution_broad")
    np.testing.assert_array_equal(narrow.true_kernel, broad.true_kernel)
    for case in [narrow, broad]:
        assert np.dot(case.input_signal, case.input_signal) * case.dt == pytest.approx(1.0)
    assert not np.allclose(narrow.clean_output, broad.clean_output)


def test_ridge_reference_recovers_noiseless_identity_without_using_truth():
    case = make_benchmark("impulse_gaussian", snr=np.inf)
    reference, info = ridge_reference(case)
    assert relative_l2(reference, case.true_kernel) < 1e-8
    # Changing only the evaluation truth must not affect the data-driven reference.
    other, _ = ridge_reference(replace(case, true_kernel=np.zeros_like(case.true_kernel)))
    np.testing.assert_array_equal(reference, other)
    np.testing.assert_allclose(info["relative_singular_values"], 1)


def test_seeded_sampler_recovers_noisy_shifted_gaussian():
    case = make_benchmark("impulse_gaussian")
    config = benchmark_config(case, iterations=6000, burnin=3000, save_every=50)
    result = run_benchmark(case, config)
    metrics, arrays = evaluate_benchmark(case, result)
    assert result.saved_receiver_functions.shape == (60, 161)
    assert metrics["mean"]["kernel_relative_l2"] < 0.08
    assert metrics["map"]["kernel_relative_l2"] < 0.10
    assert metrics["mean"]["clean_output_relative_l2"] < 0.08
    best = max(result.map_components, key=lambda c: abs(c.amplitude))
    assert best.center == pytest.approx(3.0, abs=0.1)
    assert best.width == pytest.approx(0.4, abs=0.1)
    assert best.amplitude == pytest.approx(1.0, abs=0.15)
    assert np.isfinite(arrays["mean_prediction"]).all()
    assert np.all(arrays["kernel_upper"] >= arrays["kernel_lower"])


def test_short_sampler_is_reproducible():
    case = make_benchmark("impulse_gaussian")
    config = benchmark_config(case, iterations=30, burnin=10, save_every=10)
    first = run_benchmark(case, config)
    second = run_benchmark(case, config)
    np.testing.assert_array_equal(first.saved_receiver_functions, second.saved_receiver_functions)


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_DECONVOLUTION") != "1",
    reason="Set RUN_SLOW_DECONVOLUTION=1 for 100k-iteration recovery regressions.",
)
@pytest.mark.parametrize("name", ["ricker_sparse", "boxcar_sparse"])
def test_long_signed_pulse_recovery(name):
    case = make_benchmark(name, snr=20)
    result = run_benchmark(case, benchmark_config(case, iterations=100_000))
    metrics, _ = evaluate_benchmark(case, result)
    assert metrics["mean"]["kernel_relative_l2"] < 0.15
    assert metrics["map"]["kernel_relative_l2"] < 0.15
    assert metrics["mean"]["clean_output_relative_l2"] < 0.08
    # Signed peaks must remain correctly ordered, not merely fit the output.
    for center, sign in ((2.0, 1), (4.0, -1), (6.0, 1)):
        index = np.argmin(abs(result.lag_times - center))
        assert sign * result.mean_receiver_function[index] > 0.25


def test_cli_writes_reproducible_artifacts_without_seismic_inputs(tmp_path, monkeypatch, example_module):
    main = example_module("benchmarks").main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_deconvolution_benchmarks.py",
            "--cases",
            "impulse_gaussian",
            "--iterations",
            "300",
            "--burnin",
            "150",
            "--save-every",
            "50",
            "--snrs",
            "inf",
            "--output-dir",
            str(tmp_path),
        ],
    )
    main()
    records = json.loads((tmp_path / "summary.json").read_text())
    assert len(records) == 1
    record = records[0]
    assert record["snr"] == "inf"
    assert record["config"]["iterations"] == 300
    assert record["noise_sigma"] == 0
    assert (tmp_path / record["figure"]).stat().st_size > 1000
    assert (tmp_path / "summary.png").stat().st_size > 1000
    with np.load(next(tmp_path.glob("*.npz"))) as arrays:
        assert arrays["saved_kernels"].shape[0] == 3
        np.testing.assert_array_equal(arrays["clean_output"], arrays["observed_output"])


@pytest.mark.parametrize(
    "kwargs", [{"dt": 0}, {"dt": 0.3}, {"dt": 0.09}, {"snr": 0}, {"snr": -np.inf}, {"snr": np.nan}]
)
def test_invalid_experiment_settings_fail_early(kwargs):
    with pytest.raises(ValueError):
        make_benchmark("impulse_gaussian", **kwargs)


def test_invalid_sampler_settings_fail_before_work():
    case = make_benchmark("impulse_gaussian")
    for kwargs in (
        {"iterations": 0},
        {"burnin": -1},
        {"burnin": 100_000},
        {"save_every": 0},
        {"iterations": 10, "save_every": 10},
    ):
        with pytest.raises(ValueError):
            benchmark_config(case, **kwargs)
