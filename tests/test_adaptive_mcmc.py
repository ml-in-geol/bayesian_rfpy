import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from bayesian_rf.inversion import CorrelatedRFLikelihood, sample_adaptive_metropolis


@pytest.mark.parametrize("prediction", [0.3, -0.3, 0.0])
def test_scale_marginalization_matches_quadrature(prediction):
    likelihood = CorrelatedRFLikelihood(np.array([0.4]), np.array([[0.09]]), (0.05, 2.0))
    actual, _, _ = likelihood.evaluate(np.array([prediction]))
    expected = quad(lambda scale: norm.pdf(0.4, scale * prediction, 0.3) / 1.95, 0.05, 2.0)[0]
    assert actual == pytest.approx(np.log(expected), abs=1e-10)


def test_conditional_amplitude_and_covariance():
    likelihood = CorrelatedRFLikelihood(np.array([1.0, 2.0]), np.array([[1.0, 0.5], [0.5, 1.0]]))
    logp, loc, sd = likelihood.evaluate(np.array([0.5, 1.0]))
    assert np.isfinite(logp)
    assert loc == pytest.approx(2.0)
    rng = np.random.default_rng(56)
    assert all(0.05 <= likelihood.sample_scale(loc, sd, rng) <= 5.0 for _ in range(100))
    with pytest.raises(ValueError):
        CorrelatedRFLikelihood(np.ones(2), np.array([[1.0, 0.5], [0.0, 1.0]]))


def test_adaptive_sampler_known_bounded_gaussian():
    def evaluate(x):
        return -0.5 * np.dot(x, x), x.copy()

    result = sample_adaptive_metropolis(
        evaluate,
        np.array([2.0, -2.0]),
        np.full(2, -5.0),
        np.full(2, 5.0),
        iterations=18000,
        burnin=6000,
        save_every=2,
        seed=56,
    )
    np.testing.assert_allclose(result["samples"].mean(axis=0), 0.0, atol=0.12)
    np.testing.assert_allclose(result["samples"].std(axis=0), 1.0, atol=0.12)
    np.testing.assert_allclose(result["samples"], result["auxiliary"])
    assert np.all(np.isfinite(result["log_posterior"]))
    assert np.all(result["best_parameters"] ** 2 < 0.03)


def test_adaptation_stops_before_retained_draws():
    def evaluate(x):
        return -0.5 * np.dot(x, x), x.copy()

    common = dict(
        initial=np.array([0.2, -0.3]),
        lower=np.full(2, -4.0),
        upper=np.full(2, 4.0),
        burnin=600,
        save_every=5,
        seed=19,
    )
    short = sample_adaptive_metropolis(evaluate, iterations=1200, **common)
    long = sample_adaptive_metropolis(evaluate, iterations=1500, **common)
    np.testing.assert_array_equal(short["proposal_covariance"], long["proposal_covariance"])
    assert short["proposal_scale"] == long["proposal_scale"]
    np.testing.assert_array_equal(short["samples"], long["samples"][: len(short["samples"])])
