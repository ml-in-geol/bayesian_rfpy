from dataclasses import replace

import numpy as np

from bayesian_rf.cps import IsotropicLayer, LayerCakeModel, forward_ps_receiver_function
from bayesian_rf.inversion import (
    LayerCakeInversionConfig,
    LayerCakeModelState,
    RFObservation,
    run_layer_cake_inversion,
)
from bayesian_rf.deconvolution import source_resolution
from bayesian_rf.inversion import ResolvedPSStackForward, match_cps_resolution


def _resolution(frequency=0.3):
    t = np.arange(301) * 0.25
    source = np.cos(2 * np.pi * frequency * (t - 20)) * np.exp(-0.5 * ((t - 20) / 3) ** 2)
    return source_resolution(source, 0.25)


def test_cps_gaussian_is_replaced_not_applied_twice():
    model = LayerCakeModel((IsotropicLayer(10.0, 4.0, 2.0, 2.2), IsotropicLayer(0.0, 4.0, 2.0, 2.2)))
    times = np.arange(-5.0, 20.01, 0.1)
    outputs = []
    resolution = _resolution()
    for alpha in (1.5, 2.828427, 5.0):
        synthetic = forward_ps_receiver_function(
            model, 0.12, sampling_interval_s=0.25, npts=512, delay_s=20.0, gaussian_alpha=alpha
        )
        outputs.append(match_cps_resolution(synthetic, resolution, times))
        # A homogeneous halfspace has only the direct surface R/Z pulse.
        np.testing.assert_allclose(
            outputs[-1][50], np.trapz(synthetic.amplitudes, synthetic.lag_times_s), atol=2e-5
        )
    np.testing.assert_allclose(outputs[0], outputs[1], atol=2e-5)
    np.testing.assert_allclose(outputs[1], outputs[2], atol=2e-5)


def test_mean_resolution_equals_average_for_identical_slownesses():
    forward = ResolvedPSStackForward(
        (_resolution(0.25), _resolution(0.4)), np.array([0.12, 0.12]), np.arange(-5.0, 20.01, 0.1)
    )
    state = LayerCakeModelState((12.0, 15.0, 20.0), (2.0, 2.8, 3.6), 4.3, (1.8, 1.8, 1.8, 1.8))
    np.testing.assert_allclose(forward(state), replace(forward, exact=True)(state), atol=1e-12)
    invalid = replace(state, halfspace_vs_km_s=5.5, vpvs=(1.8, 1.8, 1.8, 2.5))
    assert np.all(np.isnan(forward(invalid)))


def test_custom_forward_prediction_is_used_and_saved(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Default forward should not be called")

    monkeypatch.setattr("bayesian_rf.inversion.forward_ps_receiver_function", forbidden)
    times = np.arange(10.0)
    observed = np.sin(times)
    observation = RFObservation(times, observed, np.full(times.size, 0.1))
    config = LayerCakeInversionConfig(0.06, iterations=20, burnin=10, save_every=2)
    result = run_layer_cake_inversion(observation, config, forward_predictor=lambda state: observed / 2)
    np.testing.assert_allclose(result.best_prediction, observed)
    np.testing.assert_allclose(result.saved_predictions, np.tile(observed, (5, 1)))
    np.testing.assert_allclose(result.saved_scales, 2.0)
    assert result.saved_log_posteriors.shape == (5,)
    assert len(result.saved_models) == 5
