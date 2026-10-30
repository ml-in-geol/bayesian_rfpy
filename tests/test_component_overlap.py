import numpy as np
import pytest

from bayesian_rf.deconvolution import (
    BayesianRFConfig,
    GaussianComponent,
    _components_overlap,
    _propose_birth,
    run_bayesian_deconvolution,
)


def test_default_overlap_rule_is_unchanged() -> None:
    components = [
        GaussianComponent(center=1.0, width=0.4, amplitude=1.0),
        GaussianComponent(center=1.7, width=0.4, amplitude=-0.5),
    ]

    assert _components_overlap(components)
    assert not _components_overlap(components, separation_factor=0.0)


def test_birth_can_use_all_centers_when_overlap_is_enabled() -> None:
    rng = np.random.default_rng(7)
    existing = [GaussianComponent(center=1.0, width=10.0, amplitude=1.0)]
    lag_times = np.array([0.0, 1.0, 2.0])

    assert _propose_birth(rng, existing, lag_times, (10.0, 10.0), 1.0) is None
    proposal = _propose_birth(
        rng,
        existing,
        lag_times,
        (10.0, 10.0),
        1.0,
        separation_factor=0.0,
    )
    assert proposal is not None
    assert proposal.center in lag_times


def test_negative_component_separation_factor_is_rejected() -> None:
    config = BayesianRFConfig(
        iterations=2,
        burnin=0,
        save_every=1,
        component_separation_factor=-0.1,
    )

    with pytest.raises(ValueError, match="component_separation_factor"):
        run_bayesian_deconvolution(
            parent=np.ones(8),
            daughter=np.ones(8),
            noise_window=np.ones(4),
            dt=1.0,
            config=config,
        )
