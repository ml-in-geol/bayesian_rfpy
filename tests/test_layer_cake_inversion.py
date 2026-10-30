import numpy as np

from bayesian_rf.cps import (
    IsotropicLayer,
    LayerCakeModel,
    forward_ps_receiver_function,
    forward_sp_receiver_function,
)
from bayesian_rf.inversion import (
    LayerCakeInversionConfig,
    LayerCakeModelState,
    RFObservation,
    _profile_amplitude_and_log_likelihood,
    _to_cps_model,
    load_text_rf_observation,
    run_joint_layer_cake_inversion,
    run_layer_cake_inversion,
)


def _synthetic_observation() -> RFObservation:
    true_model = LayerCakeModel(
        layers=(
            IsotropicLayer(30.0, 6.3, 3.6, 2.8),
            IsotropicLayer(0.0, 8.0, 4.6, 3.3),
        )
    )
    synthetic = forward_ps_receiver_function(
        true_model,
        0.06,
        sampling_interval_s=0.2,
        npts=128,
        delay_s=5.0,
        gaussian_alpha=1.5,
    )
    selected = (synthetic.lag_times_s >= 0.0) & (synthetic.lag_times_s <= 20.0)
    return RFObservation(
        synthetic.lag_times_s[selected],
        synthetic.amplitudes[selected],
        np.full(np.count_nonzero(selected), 0.03),
    )


def test_transdimensional_inversion_saves_profiles() -> None:
    observation = _synthetic_observation()
    config = LayerCakeInversionConfig(
        ray_parameter_s_per_km=0.06,
        iterations=30,
        burnin=10,
        save_every=5,
        forward_sampling_interval_s=0.2,
        forward_npts=128,
        max_depth_km=60.0,
        random_seed=7,
    )
    result = run_layer_cake_inversion(observation, config)

    assert result.saved_vs_profiles.shape[0] == 4
    assert result.saved_vs_profiles.shape[1] == result.depth_km.size
    assert np.all(np.isfinite(result.best_prediction))
    assert result.saved_vpvs_profiles.shape == result.saved_vs_profiles.shape
    assert np.all(
        (result.saved_vpvs_profiles >= config.vpvs_bounds[0])
        & (result.saved_vpvs_profiles <= config.vpvs_bounds[1])
    )


def test_fixed_layer_inversion_keeps_requested_count() -> None:
    observation = _synthetic_observation()
    config = LayerCakeInversionConfig(
        ray_parameter_s_per_km=0.06,
        iterations=30,
        burnin=10,
        save_every=5,
        min_layers=3,
        max_layers=3,
        fixed_layers=3,
        initial_layers=3,
        forward_sampling_interval_s=0.2,
        forward_npts=128,
        max_depth_km=60.0,
        random_seed=11,
    )
    result = run_layer_cake_inversion(observation, config)

    assert result.best_model.layer_count == 3
    assert np.all(result.saved_layer_counts == 3)
    assert len(result.best_model.vpvs) == 4
    assert result.metadata["parameterization"] == "fixed"


def test_cps_model_uses_layer_specific_vpvs() -> None:
    state = LayerCakeModelState(
        thicknesses_km=(10.0, 20.0),
        vs_km_s=(2.0, 3.0),
        halfspace_vs_km_s=4.0,
        vpvs=(1.4, 1.6, 1.8),
    )

    model = _to_cps_model(state)

    assert np.allclose([layer.vp_km_s for layer in model.layers], [2.8, 4.8, 7.2])


def test_joint_inversion_saves_ps_and_positive_lag_sp_predictions() -> None:
    true_model = LayerCakeModel(
        layers=(
            IsotropicLayer(30.0, 6.3, 3.6, 2.8),
            IsotropicLayer(0.0, 8.0, 4.6, 3.3),
        )
    )
    ps = forward_ps_receiver_function(
        true_model,
        0.06,
        sampling_interval_s=0.2,
        npts=128,
        delay_s=5.0,
        gaussian_alpha=1.5,
    )
    sp = forward_sp_receiver_function(
        true_model,
        0.10,
        sampling_interval_s=0.2,
        npts=128,
        delay_s=12.0,
        gaussian_alpha=1.5,
        positive_lag_convention=True,
    )
    ps_selected = (ps.lag_times_s >= 0.0) & (ps.lag_times_s <= 10.0)
    sp_selected = (sp.lag_times_s >= 0.0) & (sp.lag_times_s <= 10.0)
    ps_observation = RFObservation(
        ps.lag_times_s[ps_selected],
        ps.amplitudes[ps_selected],
        np.full(np.count_nonzero(ps_selected), 0.03),
    )
    sp_observation = RFObservation(
        sp.lag_times_s[sp_selected],
        sp.amplitudes[sp_selected],
        np.full(np.count_nonzero(sp_selected), 0.03),
    )
    config = LayerCakeInversionConfig(
        ray_parameter_s_per_km=0.06,
        sp_ray_parameter_s_per_km=0.10,
        iterations=20,
        burnin=8,
        save_every=4,
        min_layers=1,
        max_layers=1,
        fixed_layers=1,
        initial_layers=1,
        forward_sampling_interval_s=0.2,
        forward_npts=128,
        max_depth_km=60.0,
        max_lag_s=10.0,
        sp_max_lag_s=10.0,
        random_seed=9,
    )

    result = run_joint_layer_cake_inversion(ps_observation, sp_observation, config)

    assert result.metadata["inversion_mode"] == "joint_ps_sp"
    assert result.sp_observation is sp_observation
    assert result.best_sp_prediction is not None
    assert result.best_sp_prediction.shape == sp_observation.amplitudes.shape
    assert np.all(np.isfinite(result.best_sp_prediction))
    assert result.best_sp_scale is not None


def test_load_text_rf_observation_reads_supplied_uncertainty(tmp_path) -> None:
    path = tmp_path / "stack.txt"
    values = np.column_stack(
        (
            np.arange(4, dtype=float),
            np.array([0.0, 0.2, -0.1, 0.0]),
            np.full(4, 0.03),
        )
    )
    np.savetxt(path, values, header="Time RF Error")

    observation = load_text_rf_observation(path)

    assert np.allclose(observation.lag_times_s, values[:, 0])
    assert np.allclose(observation.amplitudes, values[:, 1])
    assert np.allclose(observation.standard_deviation, values[:, 2])


def test_rf_scale_lower_bound_prevents_dataset_dropout() -> None:
    observation = RFObservation(
        np.arange(4, dtype=float),
        np.array([0.0, 1.0, 0.0, -1.0]),
        np.full(4, 0.1),
    )

    _, scaled_prediction, scale = _profile_amplitude_and_log_likelihood(
        -observation.amplitudes,
        observation,
        min_lag_s=0.0,
        max_lag_s=3.0,
        likelihood_weight=1.0,
        scale_bounds=(0.05, 10.0),
    )

    assert scale == 0.05
    assert np.allclose(scaled_prediction, -0.05 * observation.amplitudes)
