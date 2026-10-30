"""Rank-normalized split R-hat and bulk/tail ESS for independent RF chains.

Follows Vehtari et al. (2021). Diagnostics can reject convergence, not prove
global exploration or validate a sampler's stationary distribution.
"""

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.special import ndtri
from scipy.stats import rankdata


def _split(x):
    half = x.shape[1] // 2
    return np.concatenate([x[:, :half], x[:, -half:]], axis=0)


def _rank_normalize(x):
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    ranks = rankdata(flat, axis=0, method="average")
    return ndtri((ranks - 0.375) / (len(flat) + 0.25)).reshape(shape)


def _rhat(x):
    n = x.shape[1]
    within = np.mean(np.var(x, axis=1, ddof=1), axis=0)
    between = n * np.var(np.mean(x, axis=1), axis=0, ddof=1)
    variance = (n - 1) / n * within + between / n
    return np.sqrt(np.divide(variance, within, out=np.full_like(within, np.inf), where=within > 0))


def _ess(x):
    chains, n, parameters = x.shape
    centered = x - x.mean(axis=1, keepdims=True)
    nfft = next_fast_len(2 * n)
    spectrum = rfft(centered, nfft, axis=1)
    acov = irfft(abs(spectrum) ** 2, nfft, axis=1)[:, :n] / n
    within = acov[:, 0].mean(axis=0) * n / (n - 1)
    variance = within * (n - 1) / n + np.var(x.mean(axis=1), axis=0, ddof=1)
    rho = 1 - np.divide(
        within[None, :] - acov.mean(axis=0),
        variance[None, :],
        out=np.ones((n, parameters)),
        where=variance[None, :] > 0,
    )
    rho[0] = 1.0
    result = np.zeros(parameters)
    for p in range(parameters):
        if within[p] <= 0:
            continue
        total, previous = 0.0, np.inf
        for i in range(0, n - 1, 2):
            pair = rho[i, p] + rho[i + 1, p]
            if pair < 0:
                break
            pair = min(pair, previous)
            total += pair
            previous = pair
        tau = max(1.0, -1 + 2 * total)
        result[p] = min(chains * n, chains * n / tau)
    return result


def diagnose_chains(values, *, rhat_limit=1.01, minimum_ess=400):
    """Input is (chain, draw[, parameter]); constant/stuck traces fail.

    Tail ESS is the smaller ESS of <=5% and <=95% quantile indicators.
    ESS is conservatively capped at the number of draws. At least four
    original chains are required for the returned convergence gate to pass.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim == 2:
        x = x[..., None]
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[1] < 8 or not np.all(np.isfinite(x)):
        raise ValueError("Need finite values from >=2 chains with >=8 draws each.")
    split = _split(x)
    rank = _rank_normalize(split)
    folded = _rank_normalize(abs(split - np.median(split, axis=(0, 1))))
    rhat = np.maximum(_rhat(rank), _rhat(folded))
    bulk = _ess(rank)
    q05, q95 = np.quantile(x, [0.05, 0.95], axis=(0, 1))
    tail = np.minimum(_ess(_split((x <= q05).astype(float))), _ess(_split((x <= q95).astype(float))))
    sd = np.std(x.reshape(-1, x.shape[-1]), axis=0, ddof=1)
    mcse = np.divide(sd, np.sqrt(bulk), out=np.full_like(sd, np.inf), where=bulk > 0)
    passed = (rhat < rhat_limit) & (bulk >= minimum_ess) & (tail >= minimum_ess) & (x.shape[0] >= 4)
    return dict(
        rhat=rhat,
        ess_bulk=bulk,
        ess_tail=tail,
        approximate_mcse_mean=mcse,
        passed=passed,
        all_passed=bool(np.all(passed)),
        chains=x.shape[0],
        draws_per_chain=x.shape[1],
    )
