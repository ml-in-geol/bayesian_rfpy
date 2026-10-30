import numpy as np
import pytest

from bayesian_rf.cps import (
    IsotropicLayer,
    LayerCakeModel,
    forward_ps_receiver_function,
    forward_sp_receiver_function,
)


def test_hrftn96_generates_a_finite_receiver_function() -> None:
    model = LayerCakeModel(
        layers=(
            IsotropicLayer(35.0, 6.4, 3.7, 2.8),
            IsotropicLayer(0.0, 8.0, 4.6, 3.3),
        )
    )
    receiver_function = forward_ps_receiver_function(
        model,
        ray_parameter_s_per_km=0.06,
        sampling_interval_s=0.1,
        npts=256,
        delay_s=5.0,
        gaussian_alpha=1.5,
    )

    assert receiver_function.lag_times_s.shape == (256,)
    assert receiver_function.lag_times_s[0] == -5.0
    assert np.all(np.isfinite(receiver_function.amplitudes))
    assert np.max(np.abs(receiver_function.amplitudes)) > 0.0


def test_sp_positive_lag_convention_reverses_time_and_polarity() -> None:
    model = LayerCakeModel(
        layers=(
            IsotropicLayer(35.0, 6.4, 3.7, 2.8),
            IsotropicLayer(0.0, 8.0, 4.6, 3.3),
        )
    )
    raw = forward_sp_receiver_function(
        model,
        ray_parameter_s_per_km=0.10,
        sampling_interval_s=0.1,
        npts=256,
        delay_s=12.0,
        gaussian_alpha=1.5,
    )
    displayed = forward_sp_receiver_function(
        model,
        ray_parameter_s_per_km=0.10,
        sampling_interval_s=0.1,
        npts=256,
        delay_s=12.0,
        gaussian_alpha=1.5,
        positive_lag_convention=True,
    )

    assert raw.incident_phase == "S"
    assert displayed.positive_lag_sp_convention
    assert np.allclose(displayed.lag_times_s, -raw.lag_times_s[::-1])
    assert np.allclose(displayed.amplitudes, -raw.amplitudes[::-1])


def test_reject_nonpropagating_incident_p_and_oversized_arrays():
    model = LayerCakeModel((IsotropicLayer(10.0, 8.0, 4.0, 3.0), IsotropicLayer(0.0, 6.0, 3.0, 2.7)))
    with pytest.raises(ValueError, match="propagate"):
        forward_ps_receiver_function(model, 0.13)
    with pytest.raises(ValueError, match="8192"):
        forward_ps_receiver_function(model, 0.06, npts=16384)
