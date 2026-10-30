"""Bayesian, water-level, multitaper and iterative pulse-stripping deconvolution.

The Bayesian sampler is a preserved research implementation, not a certified
posterior sampler. See docs/limitations.md before using its uncertainty estimates.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Sequence
import warnings

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.linalg import cho_factor, cho_solve, solve_toeplitz
from scipy.signal import fftconvolve, resample
from scipy.signal.windows import dpss, tukey


@dataclass(frozen=True)
class GaussianComponent:
    center: float
    width: float
    amplitude: float


@dataclass
class BayesianRFConfig:
    min_lag: float = 0.0
    max_lag: float = 25.0
    min_period: float = 2.0
    max_period: float = 30.0
    freqmin: float = 1.0 / 30.0
    freqmax: float = 1.0 / 2.0
    pre_arrival: float = 20.0
    post_arrival: float = 55.0
    target_sampling_rate: float = 3.0
    # The inversion follows the paper's 3-Hz decimation, while the analytic
    # Gaussian RF model is evaluated more densely for output and plotting.
    rf_output_sampling_rate: float = 10.0
    # Kolb & Lekic (2014) use 2e6 iterations, a 1e6 iteration burn-in,
    # and save every 500th model. Dimensional warm-up is shortened when the
    # burn-in is shorter so all retained samples use the same component cap.
    iterations: int = 2_000_000
    burnin: int = 1_000_000
    save_every: int = 500
    max_gaussians: int = 30
    noise_model: str = "type3"
    omega0: float = 4.4
    noise_move_probability: float = 0.025
    location_step: float = 0.15
    width_step: float = 0.04
    amplitude_step_fraction: float = 0.1
    sigma_step_fraction: float = 0.0025
    lambda_step: float = 0.000125
    sigma_bounds_factor: tuple[float, float] = (1.0e-4, 3.0)
    lambda_bounds: tuple[float, float] = (0.02, 1.0)
    covariance_jitter: float = 1.0e-8
    covariance_cache_size: int = 16
    max_gaussian_width: float | None = None
    # Adjacent Gaussian centers must be separated by at least this factor
    # times the sum of their widths. 1.0 preserves the original RF prior;
    # 0.0 permits coincident, highly overlapping pulses for sharp signals.
    component_separation_factor: float = 1.0
    random_seed: int | None = 42

    def width_bounds(self) -> tuple[float, float]:
        """Return the Table 1 Gaussian-width prior in seconds.

        The standard prior is one tenth of the filter's period corners.
        ``max_gaussian_width`` is an optional application-specific tighter
        prior, for example 2 s for the supplied 2-s Mars synthetics.
        """
        minimum = self.min_period / 10.0
        maximum = self.max_period / 10.0
        if self.max_gaussian_width is not None:
            maximum = min(maximum, self.max_gaussian_width)
        if maximum < minimum:
            raise ValueError("max_gaussian_width must be at least min_period / 10.")
        return minimum, maximum


@dataclass
class BayesianRFResult:
    lag_times: np.ndarray
    mean_receiver_function: np.ndarray
    map_receiver_function: np.ndarray
    map_components: list[GaussianComponent]
    saved_receiver_functions: np.ndarray
    saved_sigma: np.ndarray
    saved_lambda: np.ndarray
    saved_counts: np.ndarray
    acceptance_rate: float
    best_log_likelihood: float
    parent: np.ndarray
    daughter: np.ndarray
    noise_window: np.ndarray
    metadata: dict[str, float | str]


@dataclass
class _State:
    components: list[GaussianComponent] = field(default_factory=list)
    sigma: float = 0.0
    lambda_value: float = 0.2
    log_likelihood: float = -np.inf
    receiver_function: np.ndarray | None = None


def _noise_correlation(
    lags: np.ndarray,
    lambda_value: float,
    noise_model: str,
    omega0: float,
) -> np.ndarray:
    if noise_model == "white":
        correlation = np.zeros_like(lags)
        correlation[0] = 1.0
        return correlation
    if noise_model == "type1":
        return np.exp(-lambda_value * lags)
    if noise_model == "type2":
        return np.exp(-((lambda_value * lags) ** 2))
    if noise_model == "type3":
        return np.exp(-lambda_value * lags) * np.cos(lambda_value * omega0 * lags)
    raise ValueError(f"Unsupported noise model: {noise_model}")


def _covariance_factor(
    npts: int,
    dt: float,
    lambda_value: float,
    config: BayesianRFConfig,
) -> np.ndarray:
    lags = np.arange(npts, dtype=np.float64) * dt
    first_column = _noise_correlation(lags, lambda_value, config.noise_model, config.omega0)
    toeplitz_indices = np.abs(np.subtract.outer(np.arange(npts), np.arange(npts)))
    correlation = first_column[toeplitz_indices]
    correlation.flat[:: npts + 1] += config.covariance_jitter

    jitter = config.covariance_jitter
    while True:
        try:
            factor, lower = cho_factor(correlation, overwrite_a=False, check_finite=False)
            return factor if lower else factor.T
        except np.linalg.LinAlgError:
            jitter *= 10.0
            if jitter > 1.0:
                raise
            correlation.flat[:: npts + 1] += jitter


def _receiver_function_from_components(
    lag_times: np.ndarray,
    components: Sequence[GaussianComponent],
) -> np.ndarray:
    if not components:
        return np.zeros_like(lag_times)

    centers = np.array([component.center for component in components])
    widths = np.array([component.width for component in components])
    amplitudes = np.array([component.amplitude for component in components])
    basis = np.exp(-0.5 * ((lag_times[:, None] - centers[None, :]) / widths[None, :]) ** 2)
    return basis @ amplitudes


def _predict_daughter(
    parent: np.ndarray,
    receiver_function: np.ndarray,
    dt: float,
) -> np.ndarray:
    return fftconvolve(parent, receiver_function, mode="full")[: len(parent)] * dt


def _estimate_alpha(
    parent: np.ndarray, daughter: np.ndarray, lag_times: np.ndarray, width_min: float, dt: float
) -> float:
    minimum_width_rf = np.exp(-0.5 * (lag_times / width_min) ** 2)
    minimum_width_rf /= max(np.max(np.abs(minimum_width_rf)), 1.0e-12)
    basis_trace = _predict_daughter(parent, minimum_width_rf, dt)
    scale = float(np.dot(basis_trace, basis_trace))
    if scale <= 0.0:
        raise ValueError("Parent waveform must have nonzero energy.")
    correlation = fftconvolve(daughter, basis_trace[::-1], mode="full")
    alpha = np.max(np.abs(correlation)) / scale
    return max(float(alpha), 1.0e-8)


def _estimate_initial_lambda(
    noise_window: np.ndarray,
    dt: float,
    config: BayesianRFConfig,
) -> float:
    noise = np.asarray(noise_window, dtype=np.float64)
    noise = noise - np.mean(noise)
    norm = np.dot(noise, noise)
    if norm <= 0.0:
        return 0.2

    max_lag = min(len(noise) - 1, int(round(10.0 / dt)))
    autocorr = np.array(
        [np.dot(noise[: len(noise) - lag], noise[lag:]) / norm for lag in range(max_lag + 1)],
        dtype=np.float64,
    )
    lags = np.arange(max_lag + 1, dtype=np.float64) * dt

    lambda_grid = np.linspace(config.lambda_bounds[0], config.lambda_bounds[1], 400)
    misfits = []
    for lambda_value in lambda_grid:
        model_corr = _noise_correlation(lags, lambda_value, config.noise_model, config.omega0)
        misfits.append(np.mean((model_corr - autocorr) ** 2))

    return float(lambda_grid[int(np.argmin(misfits))])


def _log_likelihood(
    parent: np.ndarray,
    daughter: np.ndarray,
    dt: float,
    lag_times: np.ndarray,
    components: Sequence[GaussianComponent],
    sigma: float,
    lambda_value: float,
    config: BayesianRFConfig,
    covariance_cache: OrderedDict[float, np.ndarray],
) -> tuple[float, np.ndarray]:
    receiver_function = _receiver_function_from_components(lag_times, components)
    prediction = _predict_daughter(parent, receiver_function, dt)
    residual = prediction - daughter

    # The covariance changes only when lambda moves. Reusing its Cholesky
    # factor is essential for the long chains required by the sampler.
    factor = covariance_cache.pop(lambda_value, None)
    if factor is None:
        factor = _covariance_factor(len(daughter), dt, lambda_value, config)
    covariance_cache[lambda_value] = factor
    while len(covariance_cache) > config.covariance_cache_size:
        covariance_cache.popitem(last=False)
    inverse_residual = cho_solve((factor, True), residual, check_finite=False)
    mahalanobis = float(residual @ inverse_residual) / (sigma**2)
    log_det = 2.0 * np.sum(np.log(np.diag(factor))) + 2.0 * len(daughter) * np.log(sigma)
    log_likelihood = -0.5 * (len(daughter) * np.log(2.0 * np.pi) + log_det + mahalanobis)
    return float(log_likelihood), receiver_function


def _components_overlap(
    components: Sequence[GaussianComponent],
    separation_factor: float = 1.0,
) -> bool:
    if separation_factor < 0.0:
        raise ValueError("component_separation_factor must be non-negative.")
    if len(components) < 2:
        return False
    ordered = sorted(components, key=lambda component: component.center)
    for left, right in zip(ordered[:-1], ordered[1:]):
        minimum_separation = separation_factor * (left.width + right.width)
        if abs(left.center - right.center) < minimum_separation:
            return True
    return False


def _clip_or_reject_sigma(sigma: float, sigma_reference: float, config: BayesianRFConfig) -> float | None:
    lower = max(sigma_reference * config.sigma_bounds_factor[0], 1.0e-12)
    upper = sigma_reference * config.sigma_bounds_factor[1]
    if lower <= sigma <= upper:
        return sigma
    return None


def _allowed_gaussians(iteration: int, max_gaussians: int, warmup_iterations: int | None = None) -> int:
    if warmup_iterations is not None:
        if iteration >= warmup_iterations:
            return max_gaussians
        # Finish dimensional warm-up before posterior retention even for
        # short runs; the original schedule assumes a million-step burn-in.
        schedule_end = 1000 * max_gaussians * (max_gaussians - 1) + 1
        iteration = iteration * max(1.0, schedule_end / warmup_iterations)
    for count in range(1, max_gaussians + 1):
        if iteration <= 1000 * count * (count + 1):
            return count
    return max_gaussians


def _propose_birth(
    rng: np.random.Generator,
    components: Sequence[GaussianComponent],
    lag_times: np.ndarray,
    width_bounds: tuple[float, float],
    alpha: float,
    separation_factor: float = 1.0,
) -> GaussianComponent | None:
    width = float(rng.uniform(*width_bounds))
    amplitude = float(rng.uniform(-1.5 * alpha, 1.5 * alpha))

    candidate_times = []
    for center in lag_times:
        if all(
            abs(center - component.center) >= separation_factor * (width + component.width)
            for component in components
        ):
            candidate_times.append(center)
    if not candidate_times:
        return None

    center = float(rng.choice(np.asarray(candidate_times)))
    return GaussianComponent(center=center, width=width, amplitude=amplitude)


def _apply_move(
    rng: np.random.Generator,
    state: _State,
    lag_times: np.ndarray,
    alpha: float,
    sigma_reference: float,
    config: BayesianRFConfig,
    iteration: int,
) -> tuple[_State, int]:
    width_bounds = config.width_bounds()
    proposed_components = list(state.components)
    sigma = state.sigma
    lambda_value = state.lambda_value

    allowed_gaussians = _allowed_gaussians(iteration, config.max_gaussians, config.burnin // 2)
    dimension_change = 0

    for move_index in range(3):
        if not proposed_components:
            move = "birth"
        else:
            weighted_moves = np.array(
                [
                    config.noise_move_probability,
                    (1.0 - config.noise_move_probability) / 6.0,
                    (1.0 - config.noise_move_probability) / 6.0,
                    (1.0 - config.noise_move_probability) / 6.0,
                    (1.0 - config.noise_move_probability) / 6.0,
                    (1.0 - config.noise_move_probability) / 6.0,
                    (1.0 - config.noise_move_probability) / 6.0,
                ]
            )
            moves = np.array(["lambda", "birth", "amplitude", "width", "location", "sigma", "death"])
            if dimension_change != 0:
                weighted_moves[moves == "birth"] = 0.0
                weighted_moves[moves == "death"] = 0.0
                weighted_moves = weighted_moves / weighted_moves.sum()
            move = str(rng.choice(moves, p=weighted_moves))

        if move == "birth":
            if len(proposed_components) >= allowed_gaussians:
                continue
            component = _propose_birth(
                rng,
                proposed_components,
                lag_times,
                width_bounds,
                alpha,
                config.component_separation_factor,
            )
            if component is None:
                continue
            proposed_components.append(component)
            dimension_change = 1
            continue

        if move == "death":
            if not proposed_components:
                continue
            proposed_components.pop(int(rng.integers(len(proposed_components))))
            dimension_change = -1
            continue

        if move == "sigma":
            proposal = sigma + rng.normal(scale=config.sigma_step_fraction * sigma_reference)
            clipped = _clip_or_reject_sigma(float(proposal), sigma_reference, config)
            if clipped is not None:
                sigma = clipped
            continue

        if move == "lambda":
            proposal = lambda_value + rng.normal(scale=config.lambda_step)
            lower, upper = config.lambda_bounds
            if lower <= proposal <= upper:
                lambda_value = float(proposal)
            continue

        if not proposed_components:
            continue

        index = int(rng.integers(len(proposed_components)))
        component = proposed_components[index]

        if move == "amplitude":
            proposed_components[index] = GaussianComponent(
                center=component.center,
                width=component.width,
                amplitude=float(
                    component.amplitude + rng.normal(scale=config.amplitude_step_fraction * alpha)
                ),
            )
            continue

        if move == "width":
            width = float(component.width + rng.normal(scale=config.width_step))
            if width_bounds[0] <= width <= width_bounds[1]:
                proposed_components[index] = GaussianComponent(
                    center=component.center,
                    width=width,
                    amplitude=component.amplitude,
                )
            continue

        if move == "location":
            center = float(component.center + rng.normal(scale=config.location_step))
            if lag_times[0] <= center <= lag_times[-1]:
                proposed_components[index] = GaussianComponent(
                    center=center,
                    width=component.width,
                    amplitude=component.amplitude,
                )

    proposed_components.sort(key=lambda component: component.center)
    return _State(components=proposed_components, sigma=sigma, lambda_value=lambda_value), dimension_change


def run_bayesian_deconvolution(
    parent: np.ndarray,
    daughter: np.ndarray,
    noise_window: np.ndarray,
    dt: float,
    config: BayesianRFConfig | None = None,
) -> BayesianRFResult:
    """Fit a Gaussian-pulse kernel with the legacy research sampler.

    This algorithm has known proposal/prior inconsistencies. Samples are NOT
    certified posterior draws; ``map_receiver_function`` retains the legacy
    name but selects the largest likelihood, not the largest posterior density.
    """
    warnings.warn(
        "Legacy Bayesian deconvolution has unresolved prior/proposal inconsistencies; "
        "do not interpret its sample spread as calibrated uncertainty. See docs/limitations.md.",
        RuntimeWarning,
        stacklevel=2,
    )
    config = config or BayesianRFConfig()
    if config.burnin < 0 or config.iterations <= config.burnin or config.save_every <= 0:
        raise ValueError("Require iterations > burnin >= 0 and save_every > 0.")
    if config.component_separation_factor < 0.0:
        raise ValueError("component_separation_factor must be non-negative.")
    original_parent = np.asarray(parent, dtype=np.float64)
    original_daughter = np.asarray(daughter, dtype=np.float64)
    original_noise = np.asarray(noise_window, dtype=np.float64)
    inputs = (original_parent, original_daughter, original_noise)
    if any(a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)) for a in inputs):
        raise ValueError("Waveforms must be nonempty, finite one-dimensional arrays.")
    if original_parent.shape != original_daughter.shape or dt <= 0:
        raise ValueError("Parent and daughter must share a shape and dt must be positive.")
    # A common internal scale preserves R/Z while removing dependence on m,
    # nm, or instrument counts in amplitude proposals and noise floors.
    data_scale = float(np.max(np.abs(original_parent)))
    if data_scale == 0.0:
        raise ValueError("Parent waveform must have nonzero energy.")
    parent, daughter, noise_window = (a / data_scale for a in inputs)
    rng = np.random.default_rng(config.random_seed)

    lag_times = np.arange(config.min_lag, config.max_lag + 0.5 * dt, dt, dtype=np.float64)
    # This follows the original THBD initialization: start by treating the
    # complete parent waveform as noise, then let the hierarchical model lower
    # the noise level once the receiver function explains coherent signal.
    sigma_reference = max(float(np.std(parent)), 1.0e-12)
    sigma0 = max(sigma_reference, 1.0e-12)
    lambda0 = _estimate_initial_lambda(noise_window, dt, config)
    width_min, _ = config.width_bounds()
    alpha = _estimate_alpha(parent, daughter, lag_times, width_min, dt)
    covariance_cache: OrderedDict[float, np.ndarray] = OrderedDict()

    current = _State(components=[], sigma=sigma0, lambda_value=lambda0)
    current.log_likelihood, current.receiver_function = _log_likelihood(
        parent,
        daughter,
        dt,
        lag_times,
        current.components,
        current.sigma,
        current.lambda_value,
        config,
        covariance_cache,
    )
    best = _State(
        components=list(current.components),
        sigma=current.sigma,
        lambda_value=current.lambda_value,
        log_likelihood=current.log_likelihood,
        receiver_function=current.receiver_function.copy(),
    )

    accepted = 0
    saved_components: list[list[GaussianComponent]] = []
    saved_sigma: list[float] = []
    saved_lambda: list[float] = []
    saved_counts: list[int] = []

    for iteration in range(1, config.iterations + 1):
        proposal, dimension_change = _apply_move(
            rng=rng,
            state=current,
            lag_times=lag_times,
            alpha=alpha,
            sigma_reference=sigma_reference,
            config=config,
            iteration=iteration,
        )

        if _components_overlap(proposal.components, config.component_separation_factor):
            accepted_proposal = False
        else:
            try:
                proposal.log_likelihood, proposal.receiver_function = _log_likelihood(
                    parent,
                    daughter,
                    dt,
                    lag_times,
                    proposal.components,
                    proposal.sigma,
                    proposal.lambda_value,
                    config,
                    covariance_cache,
                )
            except np.linalg.LinAlgError:
                accepted_proposal = False
            else:
                log_acceptance = proposal.log_likelihood - current.log_likelihood
                if dimension_change == 1:
                    k = len(current.components)
                    log_acceptance += np.log((k + 1.0) / (k + 2.0))
                elif dimension_change == -1 and current.components:
                    k = len(current.components)
                    log_acceptance += np.log((k + 1.0) / k)

                accepted_proposal = bool(np.log(rng.uniform()) < log_acceptance)

        if accepted_proposal:
            current = proposal
            accepted += 1
            if current.log_likelihood > best.log_likelihood:
                best = _State(
                    components=list(current.components),
                    sigma=current.sigma,
                    lambda_value=current.lambda_value,
                    log_likelihood=current.log_likelihood,
                    receiver_function=current.receiver_function.copy(),
                )

        if iteration > config.burnin and iteration % config.save_every == 0:
            saved_components.append(list(current.components))
            saved_sigma.append(current.sigma)
            saved_lambda.append(current.lambda_value)
            saved_counts.append(len(current.components))

    if not saved_components:
        raise ValueError("No posterior samples were saved. Increase iterations or lower burnin.")

    if config.rf_output_sampling_rate <= 0.0:
        raise ValueError("rf_output_sampling_rate must be positive.")
    output_dt = 1.0 / config.rf_output_sampling_rate
    output_lag_times = np.arange(
        config.min_lag,
        config.max_lag + 0.5 * output_dt,
        output_dt,
        dtype=np.float64,
    )
    saved_rf = np.vstack(
        [_receiver_function_from_components(output_lag_times, components) for components in saved_components]
    )
    return BayesianRFResult(
        lag_times=output_lag_times,
        mean_receiver_function=np.mean(saved_rf, axis=0),
        map_receiver_function=_receiver_function_from_components(output_lag_times, best.components),
        map_components=list(best.components),
        saved_receiver_functions=saved_rf,
        saved_sigma=np.asarray(saved_sigma) * data_scale,
        saved_lambda=np.asarray(saved_lambda),
        saved_counts=np.asarray(saved_counts),
        acceptance_rate=accepted / config.iterations,
        best_log_likelihood=best.log_likelihood - len(daughter) * np.log(data_scale),
        parent=original_parent,
        daughter=original_daughter,
        noise_window=original_noise,
        metadata={
            "dt": dt,
            "rf_output_dt": output_dt,
            "sigma0": sigma0 * data_scale,
            "lambda0": lambda0,
            "alpha": alpha,
            "data_scale": data_scale,
        },
    )


@dataclass
class ClassicalRF:
    spectrum: np.ndarray
    self_transfer: np.ndarray
    nfft: int
    dt: float
    diagnostics: dict = field(default_factory=dict)

    def waveform(self, *, sigma_s=0.25, min_lag=-5.0, max_lag=25.0, common_transfer=None):
        """Return R RF and Z self-RF, with positive Z-at-zero normalization.

        A supplied common transfer is a NONNEGATIVE unsmoothed zero-lag
        resolution transfer, bounded above by this method's self transfer.
        This equalizes the self pulse without sharpening any method. DPSS
        delay-dependent bias is not removed by zero-lag pulse matching.
        """
        if sigma_s < 0 or min_lag >= max_lag:
            raise ValueError("Invalid smoothing or lag range.")
        gaussian = np.exp(-0.5 * (2 * np.pi * rfftfreq(self.nfft, self.dt) * sigma_s) ** 2)
        if common_transfer is None:
            multiplier = gaussian
            transfer = self.self_transfer * gaussian
        else:
            common = np.asarray(common_transfer)
            if (
                common.shape != self.self_transfer.shape
                or np.any(common < 0)
                or np.any(common > self.self_transfer + 1e-12)
            ):
                raise ValueError("Common transfer must not sharpen this estimator.")
            multiplier = (
                np.divide(
                    common, self.self_transfer, out=np.zeros_like(common), where=self.self_transfer > 1e-15
                )
                * gaussian
            )
            transfer = common * gaussian
        normalization = float(irfft(transfer, self.nfft)[0])
        if normalization <= 0:
            raise ValueError("The vertical self RF must have a positive zero-lag spike.")
        lag = np.arange(round(min_lag / self.dt), round(max_lag / self.dt) + 1)
        if max(abs(lag)) >= self.nfft // 2:
            raise ValueError("Requested lags exceed the unaliased FFT interval.")
        rf = irfft(self.spectrum * multiplier, self.nfft) / normalization
        zrf = irfft(transfer, self.nfft) / normalization
        return lag * self.dt, rf[lag % self.nfft], zrf[lag % self.nfft]


def _inputs(source, response, dt, nfft):
    z, r = np.asarray(source, dtype=float), np.asarray(response, dtype=float)
    if z.ndim != 1 or z.shape != r.shape or len(z) < 8 or dt <= 0 or not np.all(np.isfinite([z, r])):
        raise ValueError("Require equally shaped finite source/response vectors and positive dt.")
    scale = np.max(abs(z))
    if scale == 0:
        raise ValueError("Source must have nonzero energy.")
    nfft = next_fast_len(4 * len(z)) if nfft is None else int(nfft)
    if nfft < 2 * len(z):
        raise ValueError("Require at least two record lengths of FFT padding.")
    return z / scale, r / scale, scale, nfft


def waterlevel_rf(source, response, dt, *, waterlevel=0.01, nfft=None):
    z, r, _, nfft = _inputs(source, response, dt, nfft)
    if not 0 < waterlevel <= 1:
        raise ValueError("Water level must be in (0,1].")
    taper = tukey(len(z), 0.1)
    zs, rs = rfft(z * taper, nfft), rfft(r * taper, nfft)
    power = abs(zs) ** 2
    denominator = np.maximum(power, waterlevel * power.max())
    return ClassicalRF(
        rs * zs.conj() / denominator,
        power / denominator,
        nfft,
        dt,
        dict(method="waterlevel", waterlevel=waterlevel),
    )


def iterative_rf(
    source,
    response,
    dt,
    *,
    fit_sigma_s=0.25,
    max_lag=25.0,
    max_iterations=10000,
    improvement_tolerance=1e-7,
    nfft=None,
):
    """Signed matching pursuit with exact finite-window coordinate updates.

    Fits Gaussian-filtered Z/R with impulse MASS coefficients at nonnegative
    lags. The returned spectrum is unsmoothed; waveform() applies output
    smoothing once. Improvement is fractional SSE, not percent. Reaching
    max_iterations is reported and is not claimed as optimizer convergence.
    """
    z, r, _, nfft = _inputs(source, response, dt, nfft)
    if (
        fit_sigma_s < 0
        or max_lag < 0
        or max_lag >= len(z) * dt
        or max_iterations < 1
        or improvement_tolerance <= 0
    ):
        raise ValueError("Invalid iterative deconvolution settings.")
    taper = tukey(len(z), 0.1)
    gaussian = np.exp(-0.5 * (2 * np.pi * rfftfreq(nfft, dt) * fit_sigma_s) ** 2)
    zf = irfft(rfft(z * taper, nfft) * gaussian, nfft)[: len(z)]
    rf = irfft(rfft(r * taper, nfft) * gaussian, nfft)[: len(r)]
    lags = np.arange(round(max_lag / dt) + 1)
    design = np.zeros((len(z), len(lags)))
    for k in lags:
        design[k:, k] = zf[: len(z) - k]
    gram = design.T @ design
    rhs = design.T @ rf
    norms = np.diag(gram)
    masses = np.zeros(len(lags))
    residual_dot = rhs.copy()
    response_power = float(np.dot(rf, rf))
    if response_power == 0:
        return ClassicalRF(
            np.zeros(nfft // 2 + 1, complex),
            np.ones(nfft // 2 + 1),
            nfft,
            dt,
            dict(method="iterative", iterations=0, stopped_by_tolerance=True, final_fractional_sse=0.0),
        )
    errors = []
    stopped = False
    sse = response_power
    for iteration in range(max_iterations):
        gains = np.divide(residual_dot**2, norms, out=np.zeros_like(norms), where=norms > 1e-20)
        k = int(np.argmax(gains))
        if gains[k] / response_power < improvement_tolerance:
            stopped = True
            break
        step = residual_dot[k] / norms[k]
        masses[k] += step
        residual_dot -= gram[:, k] * step
        sse -= gains[k]
        errors.append(max(0.0, float(sse / response_power)))
    residual = rf - design @ masses
    info = dict(
        method="iterative",
        iterations=len(errors),
        stopped_by_tolerance=stopped,
        improvement_tolerance=improvement_tolerance,
        fit_sigma_s=fit_sigma_s,
        final_fractional_sse=float(np.dot(residual, residual) / response_power),
        fractional_sse_trace=errors,
        spike_masses=masses.tolist(),
    )
    return ClassicalRF(rfft(masses, nfft), np.ones(nfft // 2 + 1), nfft, dt, info)


def multitaper_rf(
    source,
    response,
    dt,
    *,
    source_noise,
    time_bandwidth=2.5,
    tapers=3,
    noise_multiplier=1.0,
    numerical_floor=1e-6,
    nfft=None,
):
    """Full-window DPSS cross-spectrum / (source power + pre-P noise power).

    Unit-energy tapers keep noise spectra on a consistent scale even when
    noise and signal windows have different lengths. The same Z-noise-based
    denominator is used for R and Z; R is never normalized independently.
    A small explicit numerical floor handles zero-valued denoised pre-P noise.
    """
    z, r, scale, nfft = _inputs(source, response, dt, nfft)
    noise = np.asarray(source_noise, dtype=float) / scale
    if (
        noise.ndim != 1
        or len(noise) < 8
        or not np.all(np.isfinite(noise))
        or not 1 <= tapers <= int(2 * time_bandwidth - 1)
        or time_bandwidth >= min(len(z), len(noise)) / 2
        or noise_multiplier < 0
        or numerical_floor <= 0
    ):
        raise ValueError("Invalid DPSS, pre-event noise, or regularization settings.")
    windows, eigenvalues = dpss(len(z), time_bandwidth, Kmax=tapers, norm=2, return_ratios=True)
    noise_windows = dpss(len(noise), time_bandwidth, Kmax=tapers, norm=2)
    zs, rs = rfft(windows * z, nfft, axis=-1), rfft(windows * r, nfft, axis=-1)
    ns = rfft(noise_windows * (noise - noise.mean()), nfft, axis=-1)
    power, rpower = np.mean(abs(zs) ** 2, axis=0), np.mean(abs(rs) ** 2, axis=0)
    cross = np.mean(rs * zs.conj(), axis=0)
    noise_power = np.mean(abs(ns) ** 2, axis=0)
    denominator = np.maximum(power + noise_multiplier * noise_power, numerical_floor * power.max())
    coherence = np.divide(abs(cross) ** 2, power * rpower, out=np.zeros_like(power), where=power * rpower > 0)
    info = dict(
        method="multitaper",
        time_bandwidth=time_bandwidth,
        tapers=tapers,
        half_bandwidth_hz=time_bandwidth / (len(z) * dt),
        noise_multiplier=noise_multiplier,
        numerical_floor=numerical_floor,
        eigenvalues=eigenvalues.tolist(),
        coherence_squared=np.clip(coherence, 0, 1).tolist(),
    )
    return ClassicalRF(cross / denominator, power / denominator, nfft, dt, info)


def from_bayesian_samples(samples, lag_s, dt, nfft):
    """Convert RF density samples to impulse-mass spectra, without smoothing."""
    samples = np.asarray(samples)
    grid = np.arange(round(lag_s[-1] / dt) + 1) * dt
    masses = np.asarray([np.interp(grid, lag_s, row) * dt for row in np.atleast_2d(samples)])
    return rfft(masses, nfft, axis=-1)


@dataclass(frozen=True)
class RFResolution:
    dt: float
    nfft: int
    transfer: np.ndarray
    inverse_filter: np.ndarray
    normalization: float

    def _extract(self, array, min_lag, max_lag, output_sampling_rate):
        count = round(self.nfft * self.dt * output_sampling_rate)
        interpolated = resample(array, count, axis=-1)
        grid = np.arange(round(min_lag * output_sampling_rate), round(max_lag * output_sampling_rate) + 1)
        # The FFT period is preserved by resampling; explicitly enforce the
        # requested grid when its length is not an integer multiple of dt.
        periodic_t = np.arange(count) * self.nfft * self.dt / count
        target_t = np.mod(grid / output_sampling_rate, self.nfft * self.dt)
        rows = np.atleast_2d(interpolated)
        output = np.vstack([np.interp(target_t, periodic_t, row, period=self.nfft * self.dt) for row in rows])
        return grid / output_sampling_rate, output[0] if array.ndim == 1 else output

    def apply(self, receiver_functions, lag_times, *, min_lag=-5.0, max_lag=25.0, output_sampling_rate=10.0):
        """Filter RF densities, preserving polarity; normalize by Z self-RF."""
        functions = np.asarray(receiver_functions, dtype=float)
        times = np.asarray(lag_times, dtype=float)
        if functions.shape[-1] != times.size or np.any(np.diff(times) <= 0):
            raise ValueError("RF sample axis must match increasing lag times.")
        sample_indices = np.arange(int(np.floor(times[0] / self.dt)), int(np.ceil(times[-1] / self.dt)) + 1)
        rows = np.atleast_2d(functions)
        kernels = np.zeros((len(rows), self.nfft))
        for row, kernel in zip(rows, kernels):
            kernel[sample_indices % self.nfft] = (
                np.interp(sample_indices * self.dt, times, row, left=0.0, right=0.0) * self.dt
            )
        filtered = irfft(rfft(kernels, axis=-1) * self.transfer, self.nfft, axis=-1) / self.normalization
        if functions.ndim == 1:
            filtered = filtered[0]
        return self._extract(filtered, min_lag, max_lag, output_sampling_rate)

    def deconvolve(self, response, *, min_lag=-5.0, max_lag=25.0, output_sampling_rate=10.0):
        """Independent water-level RF using the same source inverse filter."""
        response = np.asarray(response, dtype=float)
        waveform = irfft(
            rfft(response * tukey(response.size, 0.1), self.nfft) * self.inverse_filter, self.nfft
        )
        return self._extract(waveform / self.normalization, min_lag, max_lag, output_sampling_rate)


def source_resolution(parent, dt, *, waterlevel=0.01, gaussian_sigma_s=0.25):
    """Source-dependent water-level resolution with a Gaussian low-pass.

    The transfer is |Z|^2/max(|Z|^2, waterlevel*max(|Z|^2)) times
    exp[-(2*pi*f*sigma)^2/2]. It is positive and imposes no arrival times.
    Applying it to a Bayesian RF is a linear posterior transformation,
    not a new inversion or an event-dependent radial sign correction.
    """
    source = np.asarray(parent, dtype=float)
    if source.ndim != 1 or source.size < 2 or dt <= 0 or not np.all(np.isfinite(source)):
        raise ValueError("Require a finite source array and positive dt.")
    if not 0 < waterlevel <= 1 or gaussian_sigma_s < 0:
        raise ValueError("Require 0 < waterlevel <= 1 and a nonnegative Gaussian width.")
    nfft = next_fast_len(4 * source.size)
    spectrum = rfft(source * tukey(source.size, 0.1), nfft)
    power = abs(spectrum) ** 2
    if power.max() == 0:
        raise ValueError("Source must have nonzero energy.")
    denominator = np.maximum(power, waterlevel * power.max())
    gaussian = np.exp(-0.5 * (2 * np.pi * rfftfreq(nfft, dt) * gaussian_sigma_s) ** 2)
    transfer = power / denominator * gaussian
    normalization = irfft(transfer, nfft)[0]
    return RFResolution(dt, nfft, transfer, spectrum.conj() / denominator * gaussian, normalization)


@dataclass(frozen=True)
class SpikingRF:
    lag_s: np.ndarray
    radial: np.ndarray
    vertical: np.ndarray
    source_self: np.ndarray
    source: np.ndarray
    source_centroid_s: float
    normalization: float


def tapered_source(vertical, time_s, start_s, end_s, *, taper_s=1.0):
    """Isolate a P source without shifting its time origin or changing R.

    ``taper_s`` is the duration of EACH cosine edge, not a fraction of the
    whole record. The returned array retains the original waveform length.
    """
    z = np.asarray(vertical, dtype=float)
    t = np.asarray(time_s, dtype=float)
    if (
        z.ndim != 1
        or t.shape != z.shape
        or len(t) < 3
        or not np.all(np.isfinite(z))
        or not np.all(np.isfinite(t))
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("Require finite, equally shaped waveforms and increasing times.")
    if not np.allclose(np.diff(t), t[1] - t[0]):
        raise ValueError("Waveform times must be uniformly sampled.")
    if not (t[0] <= start_s < end_s <= t[-1]) or taper_s < 0:
        raise ValueError("Source interval must be inside the record; taper must be nonnegative.")
    if 2 * taper_s > end_s - start_s:
        raise ValueError("Source interval must accommodate both taper edges.")
    weights = ((t >= start_s) & (t <= end_s)).astype(float)
    if taper_s:
        left = (t >= start_s) & (t < start_s + taper_s)
        right = (t > end_s - taper_s) & (t <= end_s)
        weights[left] = 0.5 * (1 - np.cos(np.pi * (t[left] - start_s) / taper_s))
        weights[right] = 0.5 * (1 - np.cos(np.pi * (end_s - t[right]) / taper_s))
    return z * weights


def spiking_rf(
    vertical,
    radial,
    time_s,
    *,
    method="wiener",
    source_start_s=None,
    source_end_s=None,
    taper_s=1.0,
    damping=0.01,
    gaussian_sigma_s=0.25,
    min_lag=-15.0,
    max_lag=30.0,
    filter_length_factor=1.0,
):
    """Apply one inverse filter to R, full Z, AND the isolated source.

    For ``wiener``, damping multiplies the zero-lag source autocorrelation
    on the Toeplitz diagonal; the target spike is at the absolute-amplitude
    centroid (Hannemann et al., 2017). For ``spectral``, damping is the
    water-level fraction of maximum power. They are not equivalent priors.

    All outputs share positive source-self normalization. Full-Z RF and
    source-self RF differ when the source is truncated: only the former
    reveals vertical coda left behind by the selected inverse filter.
    """
    z = np.asarray(vertical, dtype=float)
    r = np.asarray(radial, dtype=float)
    t = np.asarray(time_s, dtype=float)
    if (
        z.ndim != 1
        or len(z) < 3
        or t.shape != z.shape
        or r.shape != z.shape
        or not np.all(np.isfinite(r))
        or method not in ("wiener", "spectral")
        or not 0 < damping <= 1
        or gaussian_sigma_s < 0
        or filter_length_factor < 1
        or min_lag >= max_lag
    ):
        raise ValueError("Invalid waveforms or spiking-filter parameters.")
    start = t[0] if source_start_s is None else source_start_s
    end = t[-1] if source_end_s is None else source_end_s
    # With a full source, keep the established 10% whole-window taper.
    edge = 0.05 * (t[-1] - t[0]) if source_start_s is None and source_end_s is None else taper_s
    source = tapered_source(z, t, start, end, taper_s=edge)
    dt = t[1] - t[0]
    scale = np.max(np.abs(source))
    if not scale > 0:
        raise ValueError("Selected source must have nonzero energy.")
    source = source / scale
    z = z / scale * tukey(len(z), 0.1)
    r = r / scale * tukey(len(r), 0.1)
    indices = np.flatnonzero((t >= start) & (t <= end))
    short = source[indices]
    centroid = float(np.dot(np.arange(len(short)), abs(short)) / np.sum(abs(short)))
    spike = int(round(centroid))
    if method == "wiener":
        length = int(np.ceil(len(short) * filter_length_factor))
        ac = fftconvolve(short, short[::-1])[len(short) - 1 :]
        column = np.pad(ac, (0, length - len(ac)))
        column[0] *= 1 + damping
        rhs_indices = spike - np.arange(length)
        rhs = np.zeros(length)
        valid = (rhs_indices >= 0) & (rhs_indices < len(short))
        rhs[valid] = short[rhs_indices[valid]]
        inverse = solve_toeplitz(column, rhs, check_finite=False)
        nfft = next_fast_len(len(z) + length + 2 * round(max(abs(min_lag), abs(max_lag)) / dt))
        inv_spec = rfft(inverse, nfft)
        zero_index = int(indices[0] + spike)
    else:
        nfft = next_fast_len(4 * len(z) + 2 * round(max(abs(min_lag), abs(max_lag)) / dt))
        spectrum = rfft(source, nfft)
        power = abs(spectrum) ** 2
        inv_spec = spectrum.conj() / np.maximum(power, damping * power.max())
        zero_index = 0
    gaussian = np.exp(-0.5 * (2 * np.pi * rfftfreq(nfft, dt) * gaussian_sigma_s) ** 2)
    outputs = irfft(rfft(np.array([r, z, source]), nfft, axis=-1) * inv_spec * gaussian, nfft, axis=-1)
    norm = float(outputs[2, zero_index])
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Source inverse did not produce a positive target spike.")
    lags = np.arange(round(min_lag / dt), round(max_lag / dt) + 1)
    extracted = outputs[:, (lags + zero_index) % nfft] / norm
    return SpikingRF(lags * dt, *extracted, source * scale, float(t[indices[0]] + centroid * dt), norm)


def stack_diagnostics(radial, vertical, lag_s):
    """Score repeatability, not agreement with prescribed arrival times.

    The P-only control subtracts a scaled FULL-Z RF, not the gated-source
    self RF. This guards against ranking direct-P ringing as conversions.
    A selected score is a tuning diagnostic, not an unbiased validation.
    """
    r, z, t = np.asarray(radial), np.asarray(vertical), np.asarray(lag_s)
    if r.ndim != 2 or r.shape != z.shape or r.shape[0] < 3 or r.shape[1] != len(t):
        raise ValueError("Require matching arrays for at least three events.")
    zero = np.argmin(abs(t))
    amplitude = r[:, zero] / np.maximum(z[:, zero], 0.1)
    residual = r - amplitude[:, None] * z
    early = (t >= 1.5) & (t <= 10)
    noise = (t >= -12) & (t <= -4)
    coda = ((t >= 3) & (t <= 20)) | noise

    def correlations(rows):
        values = []
        for i in range(len(rows)):
            other = (rows.sum(axis=0) - rows[i]) / (len(rows) - 1)
            a, b = rows[i, early], other[early]
            reference = max(float(np.std(r[:, early])), np.finfo(float).tiny)
            informative = min(np.std(a), np.std(b)) > 1e-10 * reference
            values.append(float(np.corrcoef(a, b)[0, 1]) if informative else 0.0)
        return np.asarray(values)

    rms = lambda a: np.sqrt(np.mean(a**2, axis=-1))
    noise_ratio = float(np.median(rms(r[:, noise]) / np.maximum(rms(r[:, early]), 1e-15)))
    z_coda = float(np.median(rms(z[:, coda]) / np.maximum(abs(z[:, zero]), 0.1)))
    corr = correlations(residual)
    raw_corr = correlations(r)
    score = float(np.mean(corr) - 0.15 * np.log1p(noise_ratio) - 0.15 * np.log1p(z_coda))
    return dict(
        score=score,
        loo_conversion_correlation=float(np.mean(corr)),
        loo_raw_correlation=float(np.mean(raw_corr)),
        noise_ratio=noise_ratio,
        vertical_coda_ratio=z_coda,
        event_conversion_correlations=corr.tolist(),
    )
