"""Known-input, one-dimensional deconvolution benchmarks without seismic data.

The convention is y[n] = dt * sum_j x[n-j] h[j]. Times are arbitrary units;
the known input x is noiseless, and measurement noise is added only to y.
"""

from dataclasses import dataclass

import numpy as np

from .deconvolution import BayesianRFConfig, BayesianRFResult, run_bayesian_deconvolution


CASE_DESCRIPTIONS = {
    "impulse_gaussian": "Shifted unit-area impulse; recover one delayed Gaussian (sanity check).",
    "ricker_sparse": "Known Ricker input; recover three separated, signed Gaussian pulses.",
    "boxcar_sparse": "Known boxcar input; recover the same signed Gaussian pulse train.",
    "gaussian_boxcar": "Known Gaussian input; recover a discontinuous boxcar kernel (basis mismatch).",
    "gaussian_ricker": "Known Gaussian input; recover a signed Ricker kernel (non-Gaussian truth).",
    "resolution_narrow": "Narrow Gaussian input; resolve two nearby positive Gaussian pulses.",
    "resolution_broad": "Broad Gaussian input; recover the same pair with less input bandwidth.",
}


def gaussian(times: np.ndarray, *, center: float, width: float) -> np.ndarray:
    """Peak-normalized Gaussian; width is its standard deviation in time units."""
    if not np.isfinite(width) or width <= 0:
        raise ValueError("width must be positive and finite.")
    return np.exp(-0.5 * ((np.asarray(times) - center) / width) ** 2)


def ricker(times: np.ndarray, *, center: float, frequency: float) -> np.ndarray:
    """Peak-normalized Mexican-hat pulse: (1 - 2 a) exp(-a), a = (pi f t)^2."""
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("frequency must be positive and finite.")
    a = (np.pi * frequency * (np.asarray(times) - center)) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def boxcar(times: np.ndarray, *, start: float, stop: float) -> np.ndarray:
    """Unit-height boxcar on the half-open interval [start, stop)."""
    if not np.isfinite([start, stop]).all() or stop <= start:
        raise ValueError("Require finite start < stop.")
    times = np.asarray(times)
    return ((times >= start) & (times < stop)).astype(float)


@dataclass(frozen=True)
class DeconvolutionBenchmark:
    name: str
    description: str
    dt: float
    times: np.ndarray
    input_signal: np.ndarray
    lag_times: np.ndarray
    true_kernel: np.ndarray
    clean_output: np.ndarray
    observed_output: np.ndarray
    noise_window: np.ndarray
    noise_sigma: float
    requested_snr: float
    data_seed: int


def make_benchmark(
    name: str,
    *,
    dt: float = 0.1,
    snr: float = 20.0,
    seed: int = 1234,
) -> DeconvolutionBenchmark:
    """Create an independent, full linear-convolution experiment.

    SNR is RMS(clean output) / expected noise standard deviation, measured
    over the full recorded output. ``snr=np.inf`` disables noise. The source
    is zero-padded after convolution so the sampler sees the entire tail.
    Data generation deliberately does not use the sampler's forward routine.
    """
    if name not in CASE_DESCRIPTIONS:
        raise ValueError(f"Unknown benchmark {name!r}; choose from {tuple(CASE_DESCRIPTIONS)}.")
    if not np.isfinite(dt) or dt <= 0 or dt > 0.1 or not np.isclose(8.0 / dt, round(8.0 / dt)):
        raise ValueError("dt must be positive, <= 0.1, and divide the 8-unit kernel window.")
    if np.isnan(snr) or snr <= 0:
        raise ValueError("snr must be positive, or infinity for no noise.")
    lag_times = np.arange(round(8.0 / dt) + 1) * dt
    source_times = lag_times.copy()
    source = gaussian(source_times, center=2.0, width=0.25)
    if name == "impulse_gaussian":
        source = np.zeros_like(source_times)
        source[round(2.0 / dt)] = 1.0 / dt
        kernel = gaussian(lag_times, center=3.0, width=0.4)
    elif name in {"ricker_sparse", "boxcar_sparse"}:
        if name == "ricker_sparse":
            source = ricker(source_times, center=2.0, frequency=0.7)
        else:
            source = boxcar(source_times, start=1.5, stop=2.5)
        kernel = (
            gaussian(lag_times, center=2.0, width=0.25)
            - 0.65 * gaussian(lag_times, center=4.0, width=0.35)
            + 0.45 * gaussian(lag_times, center=6.0, width=0.3)
        )
    elif name == "gaussian_boxcar":
        kernel = 0.5 * boxcar(lag_times, start=3.0, stop=5.0)
    elif name == "gaussian_ricker":
        kernel = ricker(lag_times, center=4.0, frequency=0.5)
    else:
        source = gaussian(source_times, center=3.0, width=0.15 if name == "resolution_narrow" else 0.9)
        kernel = gaussian(lag_times, center=3.5, width=0.2) + gaussian(lag_times, center=4.5, width=0.2)
    # Equal input energy makes the resolution pair a bandwidth, not energy, comparison.
    source = source / np.sqrt(dt * np.dot(source, source)) if name != "impulse_gaussian" else source
    clean = np.convolve(source, kernel, mode="full") * dt
    source = np.pad(source, (0, kernel.size - 1))
    sigma = float(np.sqrt(np.mean(clean**2)) / snr)
    rng = np.random.default_rng(seed)
    observed = clean + rng.normal(0.0, sigma, clean.size)
    noise_window = rng.normal(0.0, sigma, clean.size)
    return DeconvolutionBenchmark(
        name,
        CASE_DESCRIPTIONS[name],
        dt,
        np.arange(clean.size) * dt,
        source,
        lag_times,
        kernel,
        clean,
        observed,
        noise_window,
        sigma,
        snr,
        seed,
    )


def benchmark_config(
    case: DeconvolutionBenchmark,
    *,
    iterations: int = 100_000,
    burnin: int | None = None,
    save_every: int = 100,
    seed: int = 42,
) -> BayesianRFConfig:
    """Generic finite-width pulse priors, not initialized from the true kernel."""
    burnin = iterations // 2 if burnin is None else burnin
    if iterations < 1 or not 0 <= burnin < iterations or save_every < 1:
        raise ValueError("Require iterations > burnin >= 0 and save_every >= 1.")
    if iterations // save_every - burnin // save_every < 2:
        raise ValueError("Save at least two posterior draws.")
    return BayesianRFConfig(
        min_lag=0.0,
        max_lag=float(case.lag_times[-1]),
        min_period=10.0 * case.dt,
        max_period=10.0,
        max_gaussian_width=1.0,
        target_sampling_rate=1.0 / case.dt,
        rf_output_sampling_rate=2.0 / case.dt,
        iterations=iterations,
        burnin=burnin,
        save_every=save_every,
        max_gaussians=20,
        noise_model="white",
        noise_move_probability=0.0,
        sigma_step_fraction=0.02,
        location_step=0.15,
        width_step=0.04,
        random_seed=seed,
    )


def run_benchmark(case: DeconvolutionBenchmark, config: BayesianRFConfig) -> BayesianRFResult:
    """Run the existing array-based sampler, without any seismic preprocessing."""
    return run_bayesian_deconvolution(
        case.input_signal,
        case.observed_output,
        case.noise_window,
        case.dt,
        config,
    )


def convolution_matrix(input_signal: np.ndarray, kernel_size: int, dt: float) -> np.ndarray:
    """Independent causal convolution operator, truncated to the recorded length."""
    x = np.asarray(input_signal, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or not 1 <= kernel_size <= x.size:
        raise ValueError("Require finite 1-D input and 1 <= kernel_size <= input size.")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite.")
    operator = np.zeros((x.size, kernel_size))
    for j in range(kernel_size):
        operator[j:, j] = x[: x.size - j] * dt
    return operator


def ridge_reference(case: DeconvolutionBenchmark) -> tuple[np.ndarray, dict]:
    """Independent Tikhonov reference, regularization selected by GCV, not truth.

    Minimize ||A h - y||^2 + lambda ||h||^2 using the SVD. This is a
    comparison method, not a substitute for the Bayesian posterior.
    """
    operator = convolution_matrix(case.input_signal, case.true_kernel.size, case.dt)
    u, s, vt = np.linalg.svd(operator, full_matrices=False)
    coefficients = u.T @ case.observed_output
    orthogonal = case.observed_output - u @ coefficients
    lambdas = s[0] ** 2 * np.logspace(-12, 2, 100)
    shrinkage = s[:, None] ** 2 / (s[:, None] ** 2 + lambdas)
    residual_squared = np.sum(((1.0 - shrinkage) * coefficients[:, None]) ** 2, axis=0)
    residual_squared += np.dot(orthogonal, orthogonal)
    n = operator.shape[0]
    gcv = (residual_squared / n) / (1.0 - np.sum(shrinkage, axis=0) / n) ** 2
    index = int(np.argmin(gcv))
    kernel = vt.T @ (s * coefficients / (s**2 + lambdas[index]))
    return kernel, {
        "method": "zero-order Tikhonov with generalized cross-validation",
        "regularization": float(lambdas[index]),
        "gcv": float(gcv[index]),
        "gcv_at_search_boundary": bool(index in {0, len(lambdas) - 1}),
        "relative_singular_values": (s / s[0]).tolist(),
    }


def relative_l2(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Relative Euclidean error, without post-hoc amplitude or time alignment."""
    return float(np.linalg.norm(estimate - truth) / max(np.linalg.norm(truth), np.finfo(float).tiny))


def evaluate_benchmark(case: DeconvolutionBenchmark, result: BayesianRFResult) -> tuple[dict, dict]:
    """Compare kernels and reconvolutions on the original data grid."""
    indices = np.searchsorted(result.lag_times, case.lag_times - 1.0e-10)
    if np.any(indices >= result.lag_times.size) or not np.allclose(result.lag_times[indices], case.lag_times):
        raise ValueError("RF output must contain the benchmark lag grid exactly.")
    samples = result.saved_receiver_functions[:, indices]
    mean = result.mean_receiver_function[indices]
    map_kernel = result.map_receiver_function[indices]
    operator = convolution_matrix(case.input_signal, case.true_kernel.size, case.dt)
    ridge, ridge_info = ridge_reference(case)
    lower, upper = np.quantile(samples, [0.05, 0.95], axis=0)
    metrics = {}
    arrays = {"kernel_lower": lower, "kernel_upper": upper, "ridge_kernel": ridge}
    for name, kernel in (("map", map_kernel), ("mean", mean), ("ridge", ridge)):
        prediction = operator @ kernel
        metrics[name] = {
            "kernel_relative_l2": relative_l2(kernel, case.true_kernel),
            "clean_output_relative_l2": relative_l2(prediction, case.clean_output),
            "observed_output_relative_l2": relative_l2(prediction, case.observed_output),
            "kernel_area": float(np.sum(kernel) * case.dt),
            "kernel_area_error": float(np.sum(kernel - case.true_kernel) * case.dt),
        }
        arrays[f"{name}_prediction"] = prediction
    midpoint = max(1, samples.shape[0] // 2)
    metrics["posterior"] = {
        "pointwise_90pct_truth_coverage_fraction": float(
            np.mean((case.true_kernel >= lower) & (case.true_kernel <= upper))
        ),
        "mean_90pct_interval_width": float(np.mean(upper - lower)),
        "split_mean_relative_difference": relative_l2(
            np.mean(samples[:midpoint], axis=0), np.mean(samples[midpoint:], axis=0)
        ),
        "median_noise_sigma": float(np.median(result.saved_sigma)),
        "saved_samples": int(samples.shape[0]),
        "acceptance_rate": result.acceptance_rate,
    }
    metrics["ridge_reference"] = ridge_info
    return metrics, arrays
