import numpy as np

from bayesian_rf.diagnostics import diagnose_chains


def test_iid_chains_and_affine_invariance():
    x = np.random.default_rng(7).normal(size=(4, 2000, 3))
    a = diagnose_chains(x)
    assert a["all_passed"]
    assert np.max(a["rhat"]) < 1.01
    assert np.min(a["ess_bulk"]) > 5000
    b = diagnose_chains(x * 1e-15 + 3e-15)
    np.testing.assert_allclose(a["rhat"], b["rhat"])
    np.testing.assert_allclose(a["ess_bulk"], b["ess_bulk"])


def test_shift_and_scale_mismatches_rejected():
    x = np.random.default_rng(3).normal(size=(4, 2000))
    x[0] += 1
    assert not diagnose_chains(x)["all_passed"]
    y = np.random.default_rng(3).normal(size=(4, 2000))
    y[0] *= 3
    assert diagnose_chains(y)["rhat"][0] > 1.1


def test_autocorrelation_reduces_ess():
    x = np.random.default_rng(10).normal(size=(4, 5000))
    for i in range(1, x.shape[1]):
        x[:, i] += 0.9 * x[:, i - 1]
    result = diagnose_chains(x)
    expected = x.size * (1 - 0.9) / (1 + 0.9)
    assert 0.55 * expected < result["ess_bulk"][0] < 1.7 * expected


def test_constant_and_two_chain_runs_not_certified():
    assert not diagnose_chains(np.ones((4, 1000)))["all_passed"]
    assert not diagnose_chains(np.random.default_rng(7).normal(size=(2, 2000)))["all_passed"]
