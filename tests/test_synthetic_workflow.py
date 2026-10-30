"""Integration/regression tests, not a claim of MCMC convergence."""

from dataclasses import replace

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.parametrize("event", ["mt01", "mt02", "mt03"])
@pytest.mark.parametrize("noise", ["nonoise", "snr4", "snr2"])
def test_mars_synthetics_all_classical_methods(example_module, event, noise):
    workflow = example_module("synthetic_workflow")
    data, config = workflow.prepare(event, noise, max_lag=12.0)
    for method in ("waterlevel", "iterative", "multitaper"):
        lag, rf, _ = workflow.calculate(data, config, method)
        assert rf.shape == lag.shape
        assert np.all(np.isfinite(rf))
        assert np.linalg.norm(rf) > 0
        assert lag[0] == 0.0
        assert lag[-1] == 12.0


@pytest.mark.integration
def test_bayesian_synthetic_pipeline_preserves_standard_mode(example_module):
    workflow = example_module("synthetic_workflow")
    data, config = workflow.prepare()
    with pytest.warns(RuntimeWarning, match="unresolved"):
        lag, rf, result = workflow.calculate(data, config, "bayesian", iterations=40)
    assert config.component_separation_factor == 1.0
    assert np.isclose(np.diff(lag).mean(), 0.1)
    assert result.saved_receiver_functions.shape[0] == 20
    assert np.all(np.isfinite(rf))


@pytest.mark.integration
@pytest.mark.cps
@pytest.mark.parametrize("layers", [3, None])
def test_example_waveforms_deconvolve_then_invert(example_module, layers):
    workflow = example_module("synthetic_workflow")
    data, config = workflow.prepare(max_lag=12.0)
    lag, observed, _ = workflow.calculate(data, config)
    fit, settings = workflow.invert_multitaper(data, config, iterations=40, fixed_layers=layers)
    np.testing.assert_allclose(fit.observation.amplitudes, observed, atol=1e-12)
    np.testing.assert_allclose(fit.observation.lag_times_s, lag)
    assert np.all(np.isfinite(fit.best_prediction))
    assert fit.saved_vpvs_profiles.shape == fit.saved_vs_profiles.shape
    assert fit.saved_vs_profiles.shape[0] == 20
    assert np.all(
        (fit.saved_vs_profiles >= settings.vs_bounds_km_s[0])
        & (fit.saved_vs_profiles <= settings.vs_bounds_km_s[1])
    )
    if layers is not None:
        assert np.all(fit.saved_layer_counts == layers)


@pytest.mark.cps
def test_controlled_structural_inversion_improves_fit():
    """A displaced initial model must improve on a known CPS target."""
    from bayesian_rf.inversion import (
        LayerCakeInversionConfig,
        RFObservation,
        _forward_prediction,
        _initial_state,
        run_layer_cake_inversion,
    )

    config = LayerCakeInversionConfig(
        ray_parameter_s_per_km=0.06,
        fixed_layers=1,
        initial_layers=1,
        iterations=300,
        burnin=150,
        save_every=10,
        max_depth_km=60.0,
        initial_depth_km=22.0,
        initial_vs_km_s=3.2,
        initial_halfspace_vs_km_s=4.3,
        forward_sampling_interval_s=0.2,
        forward_npts=256,
        max_lag_s=15.0,
        random_seed=51,
    )
    lag = np.arange(0.0, 15.1, 0.2)
    template = RFObservation(lag, np.zeros(lag.size), np.full(lag.size, 0.015))
    initial = _initial_state(config)
    truth = replace(initial, thicknesses_km=(30.0,), vs_km_s=(3.6,), halfspace_vs_km_s=4.6)
    observed = replace(template, amplitudes=_forward_prediction(truth, template, config))
    initial_prediction = _forward_prediction(initial, observed, config)
    fit = run_layer_cake_inversion(observed, config)
    assert np.linalg.norm(fit.best_prediction - observed.amplitudes) < 0.6 * np.linalg.norm(
        initial_prediction - observed.amplitudes
    )
