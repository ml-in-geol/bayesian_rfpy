"""Transdimensional layer-cake inversion of Ps and joint Ps/Sp RFs with CPS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.linalg import cholesky, solve_triangular
from scipy.special import log_ndtr
from scipy.stats import truncnorm
from scipy.signal.windows import dpss
from .cps import CPSReceiverFunction
from .deconvolution import RFResolution

from .cps import (
    IsotropicLayer,
    LayerCakeModel,
    forward_ps_receiver_function,
    forward_sp_receiver_function,
)

if TYPE_CHECKING:
    from .deconvolution import BayesianRFResult


@dataclass(frozen=True)
class RFObservation:
    """A receiver function and independent per-sample uncertainty."""

    lag_times_s: np.ndarray
    amplitudes: np.ndarray
    standard_deviation: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.lag_times_s, dtype=np.float64)
        amplitudes = np.asarray(self.amplitudes, dtype=np.float64)
        standard_deviation = np.asarray(self.standard_deviation, dtype=np.float64)
        if times.ndim != 1 or amplitudes.shape != times.shape or standard_deviation.shape != times.shape:
            raise ValueError(
                "RF observation times, amplitudes, and standard deviations must be 1-D arrays of equal length."
            )
        if times.size < 4 or not np.all(np.diff(times) > 0.0):
            raise ValueError("RF observation times must be strictly increasing with at least four samples.")
        if not np.all(np.isfinite(amplitudes)) or not np.all(np.isfinite(standard_deviation)):
            raise ValueError("RF observation values must be finite.")
        if np.any(standard_deviation <= 0.0):
            raise ValueError("RF standard deviations must be positive.")

    @classmethod
    def from_deconvolution_result(
        cls,
        result: BayesianRFResult,
        *,
        statistic: str = "mean",
        uncertainty_floor_fraction: float = 0.05,
    ) -> RFObservation:
        """Construct a mean- or MAP-RF observation using posterior spread as uncertainty."""
        if uncertainty_floor_fraction <= 0.0:
            raise ValueError("uncertainty_floor_fraction must be positive.")
        if statistic not in {"mean", "map"}:
            raise ValueError("statistic must be 'mean' or 'map'.")
        amplitudes = result.mean_receiver_function if statistic == "mean" else result.map_receiver_function
        posterior_std = np.std(result.saved_receiver_functions, axis=0)
        floor = max(float(np.max(np.abs(amplitudes))) * uncertainty_floor_fraction, 1.0e-6)
        return cls(
            lag_times_s=result.lag_times.copy(),
            amplitudes=amplitudes.copy(),
            standard_deviation=np.maximum(posterior_std, floor),
        )


def load_rf_observation(
    path: str | Path,
    *,
    statistic: str = "mean",
    uncertainty_floor_fraction: float = 0.05,
) -> RFObservation:
    """Load an RF observation from an inversion NPZ or three-column text file."""
    if statistic not in {"mean", "map"}:
        raise ValueError("statistic must be 'mean' or 'map'.")
    path = Path(path)
    if path.suffix.lower() != ".npz":
        return load_text_rf_observation(path, uncertainty_floor_fraction=uncertainty_floor_fraction)
    with np.load(path) as arrays:
        lag_times = arrays["lag_times"]
        amplitudes = (
            arrays["mean_receiver_function"] if statistic == "mean" else arrays["map_receiver_function"]
        )
        posterior_std = np.std(arrays["saved_receiver_functions"], axis=0)
    floor = max(float(np.max(np.abs(amplitudes))) * uncertainty_floor_fraction, 1.0e-6)
    return RFObservation(lag_times, amplitudes, np.maximum(posterior_std, floor))


def load_text_rf_observation(
    path: str | Path,
    *,
    uncertainty_floor_fraction: float = 0.0,
) -> RFObservation:
    """Load ``time, amplitude[, standard deviation]`` columns from text."""
    if uncertainty_floor_fraction < 0.0:
        raise ValueError("uncertainty_floor_fraction cannot be negative.")
    values = np.loadtxt(Path(path), comments="#", ndmin=2)
    if values.shape[1] not in {2, 3}:
        raise ValueError("Text RF files must contain two or three numeric columns.")
    times = values[:, 0]
    amplitudes = values[:, 1]
    if values.shape[1] == 3:
        standard_deviation = values[:, 2]
    else:
        floor = max(float(np.max(np.abs(amplitudes))) * max(uncertainty_floor_fraction, 0.05), 1.0e-6)
        standard_deviation = np.full(times.shape, floor)
    if uncertainty_floor_fraction > 0.0:
        floor = max(float(np.max(np.abs(amplitudes))) * uncertainty_floor_fraction, 1.0e-6)
        standard_deviation = np.maximum(standard_deviation, floor)
    return RFObservation(times, amplitudes, standard_deviation)


def calculate_ps_sp_ray_parameters(
    model_path: str | Path,
    *,
    distance_deg: float,
    source_depth_km: float,
    planet_radius_km: float = 3389.5,
) -> tuple[float, float]:
    """Calculate direct P and S ray parameters in s/km with TauP."""
    if distance_deg <= 0.0 or source_depth_km < 0.0 or planet_radius_km <= 0.0:
        raise ValueError("Require positive distance/radius and non-negative source depth.")
    from .preprocessing import load_taup_model

    model = load_taup_model(model_path)
    kilometers_per_degree = np.pi * planet_radius_km / 180.0
    parameters: list[float] = []
    for phase in ("P", "S"):
        arrivals = model.get_travel_times(
            source_depth_in_km=source_depth_km,
            distance_in_degree=distance_deg,
            phase_list=[phase],
        )
        if not arrivals:
            raise ValueError(f"TauP found no direct {phase} arrival at {distance_deg:g} degrees.")
        direct = next((arrival for arrival in arrivals if arrival.name == phase), arrivals[0])
        parameters.append(float(direct.ray_param_sec_degree) / kilometers_per_degree)
    return parameters[0], parameters[1]


def load_nd_vs_profile(
    path: str | Path, *, max_depth_km: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read depth and Vs columns from a TauP/CPS-style ``.nd`` model file."""
    depths: list[float] = []
    velocities: list[float] = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            depth, _, vs = (float(fields[index]) for index in range(3))
        except ValueError:
            continue
        if max_depth_km is None or depth <= max_depth_km:
            depths.append(depth)
            velocities.append(vs)
    if not depths:
        raise ValueError(f"No depth/Vs rows were found in {path}.")
    return np.asarray(depths, dtype=np.float64), np.asarray(velocities, dtype=np.float64)


@dataclass(frozen=True)
class LayerCakeInversionConfig:
    """Priors and proposal widths for CPS-backed layer-cake inversion.

    ``fixed_layers=None`` selects reversible-jump sampling of the finite-layer
    count.  Supplying a positive value fixes that count and samples only the
    continuous model parameters.
    """

    ray_parameter_s_per_km: float
    sp_ray_parameter_s_per_km: float | None = None
    iterations: int = 20_000
    burnin: int = 10_000
    save_every: int = 20
    min_layers: int = 1
    max_layers: int = 12
    fixed_layers: int | None = None
    min_thickness_km: float = 1.0
    max_depth_km: float = 100.0
    vs_bounds_km_s: tuple[float, float] = (1.5, 5.5)
    vpvs_bounds: tuple[float, float] = (1.55, 2.00)
    initial_layers: int = 2
    initial_depth_km: float = 40.0
    initial_vs_km_s: float = 3.5
    initial_halfspace_vs_km_s: float = 4.5
    initial_vpvs: float = 1.75
    initial_thicknesses_km: tuple[float, ...] | None = None
    initial_layer_vs_km_s: tuple[float, ...] | None = None
    initial_layer_vpvs: tuple[float, ...] | None = None
    vs_step_km_s: float = 0.08
    thickness_step_km: float = 1.0
    vpvs_step: float = 0.015
    birth_vs_step_km_s: float = 0.08
    birth_vpvs_step: float = 0.03
    forward_sampling_interval_s: float = 0.1
    forward_npts: int = 512
    forward_delay_s: float = 5.0
    gaussian_alpha: float = 1.5
    min_lag_s: float = 0.0
    max_lag_s: float = 25.0
    profile_depth_interval_km: float = 0.5
    random_seed: int | None = 42
    sp_gaussian_alpha: float | None = None
    ps_likelihood_weight: float = 1.0
    sp_likelihood_weight: float = 1.0
    sp_min_lag_s: float | None = None
    sp_max_lag_s: float | None = None
    sp_positive_lag_convention: bool = True
    sp_component_polarity: int = 1
    ps_scale_bounds: tuple[float, float] = (0.0, np.inf)
    sp_scale_bounds: tuple[float, float] = (0.0, np.inf)

    def __post_init__(self) -> None:
        if self.ray_parameter_s_per_km <= 0.0:
            raise ValueError("ray_parameter_s_per_km must be positive.")
        if self.sp_ray_parameter_s_per_km is not None and self.sp_ray_parameter_s_per_km <= 0.0:
            raise ValueError("sp_ray_parameter_s_per_km must be positive when supplied.")
        if self.ps_likelihood_weight <= 0.0 or self.sp_likelihood_weight <= 0.0:
            raise ValueError("RF likelihood weights must be positive.")
        if self.sp_gaussian_alpha is not None and self.sp_gaussian_alpha <= 0.0:
            raise ValueError("sp_gaussian_alpha must be positive when supplied.")
        if self.sp_component_polarity not in {-1, 1}:
            raise ValueError("sp_component_polarity must be -1 or 1.")
        for bounds in (self.ps_scale_bounds, self.sp_scale_bounds):
            if bounds[0] < 0.0 or bounds[0] >= bounds[1]:
                raise ValueError("RF scale bounds must satisfy 0 <= lower < upper.")
        if self.iterations <= self.burnin or self.save_every <= 0:
            raise ValueError("iterations must exceed burnin and save_every must be positive.")
        if not 1 <= self.min_layers <= self.initial_layers <= self.max_layers:
            raise ValueError(
                "Require min_layers <= initial_layers <= max_layers and at least one finite layer."
            )
        if self.fixed_layers is not None:
            if self.fixed_layers < 1:
                raise ValueError("fixed_layers must be at least one when supplied.")
            if not self.min_layers <= self.fixed_layers <= self.max_layers:
                raise ValueError("fixed_layers must lie within min_layers and max_layers.")
            if self.initial_layers != self.fixed_layers:
                raise ValueError("initial_layers must equal fixed_layers in fixed-dimensional mode.")
        if (
            self.min_thickness_km <= 0.0
            or self.initial_depth_km < self.initial_layers * self.min_thickness_km
        ):
            raise ValueError("Initial depth must accommodate the requested finite layers.")
        if self.max_depth_km <= self.initial_depth_km:
            raise ValueError("max_depth_km must exceed initial_depth_km.")
        if self.vs_bounds_km_s[0] <= 0.0 or self.vs_bounds_km_s[0] >= self.vs_bounds_km_s[1]:
            raise ValueError("Invalid Vs bounds.")
        if self.vpvs_bounds[0] <= 1.0 or self.vpvs_bounds[0] >= self.vpvs_bounds[1]:
            raise ValueError("Invalid Vp/Vs bounds.")
        if self.profile_depth_interval_km <= 0.0:
            raise ValueError("profile_depth_interval_km must be positive.")
        initial_profiles = (
            self.initial_thicknesses_km,
            self.initial_layer_vs_km_s,
            self.initial_layer_vpvs,
        )
        if any(values is not None for values in initial_profiles):
            if any(values is None for values in initial_profiles):
                raise ValueError("All three initial layer profile tuples must be supplied together.")
            if not all(
                len(values) == self.initial_layers for values in initial_profiles if values is not None
            ):
                raise ValueError("Initial layer profile tuples must match initial_layers.")


@dataclass(frozen=True)
class LayerCakeModelState:
    """One physical model: finite layers followed by a halfspace."""

    thicknesses_km: tuple[float, ...]
    vs_km_s: tuple[float, ...]
    halfspace_vs_km_s: float
    vpvs: tuple[float, ...]

    @property
    def layer_count(self) -> int:
        """Number of finite layers, excluding the halfspace."""
        return len(self.thicknesses_km)


@dataclass
class LayerCakeInversionResult:
    """Saved posterior profiles and best-fitting layer-cake model."""

    depth_km: np.ndarray
    mean_vs_km_s: np.ndarray
    vs_lower_km_s: np.ndarray
    vs_upper_km_s: np.ndarray
    saved_vs_profiles: np.ndarray
    mean_vpvs: np.ndarray
    vpvs_lower: np.ndarray
    vpvs_upper: np.ndarray
    saved_vpvs_profiles: np.ndarray
    saved_layer_counts: np.ndarray
    best_model: LayerCakeModelState
    best_prediction: np.ndarray
    best_scale: float
    best_log_posterior: float
    acceptance_rate: float
    observation: RFObservation
    metadata: dict[str, float | int | str]
    sp_observation: RFObservation | None = None
    best_sp_prediction: np.ndarray | None = None
    best_sp_scale: float | None = None
    saved_predictions: np.ndarray | None = None
    saved_log_posteriors: np.ndarray | None = None
    saved_scales: np.ndarray | None = None
    saved_models: tuple[LayerCakeModelState, ...] = ()


def _density_from_vp(vp_km_s: float) -> float:
    """Simple crustal density relation used when density is not sampled."""
    return max(1.8, 0.32 * vp_km_s + 0.77)


def _to_cps_model(state: LayerCakeModelState) -> LayerCakeModel:
    finite_layers = tuple(
        IsotropicLayer(
            thickness_km=thickness,
            vp_km_s=vs * vpvs,
            vs_km_s=vs,
            density_g_cm3=_density_from_vp(vs * vpvs),
        )
        for thickness, vs, vpvs in zip(state.thicknesses_km, state.vs_km_s, state.vpvs[:-1], strict=True)
    )
    halfspace_vp = state.halfspace_vs_km_s * state.vpvs[-1]
    return LayerCakeModel(
        layers=(
            *finite_layers,
            IsotropicLayer(0.0, halfspace_vp, state.halfspace_vs_km_s, _density_from_vp(halfspace_vp)),
        ),
        title="Bayesian RF proposed layer-cake",
    )


def _is_valid(state: LayerCakeModelState, config: LayerCakeInversionConfig) -> bool:
    if not config.min_layers <= state.layer_count <= config.max_layers:
        return False
    if config.fixed_layers is not None and state.layer_count != config.fixed_layers:
        return False
    if len(state.vs_km_s) != state.layer_count or len(state.vpvs) != state.layer_count + 1:
        return False
    if any(thickness < config.min_thickness_km for thickness in state.thicknesses_km):
        return False
    if sum(state.thicknesses_km) > config.max_depth_km:
        return False
    lower_vs, upper_vs = config.vs_bounds_km_s
    if not all(lower_vs <= vs <= upper_vs for vs in (*state.vs_km_s, state.halfspace_vs_km_s)):
        return False
    return all(config.vpvs_bounds[0] <= ratio <= config.vpvs_bounds[1] for ratio in state.vpvs)


def _log_prior(state: LayerCakeModelState, config: LayerCakeInversionConfig) -> float:
    if not _is_valid(state, config):
        return -np.inf
    vs_width = config.vs_bounds_km_s[1] - config.vs_bounds_km_s[0]
    thickness_width = config.max_depth_km - config.min_thickness_km
    vpvs_width = config.vpvs_bounds[1] - config.vpvs_bounds[0]
    # A 1/(k + 1) dimension prior penalizes unnecessary interfaces.
    return (
        -np.log(state.layer_count + 1.0)
        - state.layer_count * np.log(thickness_width)
        - (state.layer_count + 1) * np.log(vs_width)
        - (state.layer_count + 1) * np.log(vpvs_width)
    )


def _profile_from_state(state: LayerCakeModelState, depth_km: np.ndarray) -> np.ndarray:
    profile = np.full_like(depth_km, state.halfspace_vs_km_s, dtype=np.float64)
    top = 0.0
    for thickness, vs in zip(state.thicknesses_km, state.vs_km_s, strict=True):
        profile[(depth_km >= top) & (depth_km < top + thickness)] = vs
        top += thickness
    return profile


def _vpvs_profile_from_state(state: LayerCakeModelState, depth_km: np.ndarray) -> np.ndarray:
    """Expand layer-wise Vp/Vs values onto a common posterior depth grid."""
    profile = np.full_like(depth_km, state.vpvs[-1], dtype=np.float64)
    top = 0.0
    for thickness, ratio in zip(state.thicknesses_km, state.vpvs[:-1], strict=True):
        profile[(depth_km >= top) & (depth_km < top + thickness)] = ratio
        top += thickness
    return profile


def _initial_state(config: LayerCakeInversionConfig) -> LayerCakeModelState:
    if config.initial_thicknesses_km is None:
        thickness = config.initial_depth_km / config.initial_layers
        thicknesses = (thickness,) * config.initial_layers
        velocities = (config.initial_vs_km_s,) * config.initial_layers
        vpvs = (config.initial_vpvs,) * (config.initial_layers + 1)
    else:
        thicknesses = config.initial_thicknesses_km
        velocities = config.initial_layer_vs_km_s
        vpvs = (*config.initial_layer_vpvs, config.initial_vpvs)
    state = LayerCakeModelState(
        thicknesses_km=thicknesses,
        vs_km_s=velocities,
        halfspace_vs_km_s=config.initial_halfspace_vs_km_s,
        vpvs=vpvs,
    )
    if not _is_valid(state, config):
        raise ValueError("Initial layer-cake model violates inversion bounds.")
    return state


def _forward_prediction(
    state: LayerCakeModelState,
    observation: RFObservation,
    config: LayerCakeInversionConfig,
    *,
    phase: str = "P",
) -> np.ndarray:
    model = _to_cps_model(state)
    if phase == "P":
        if any(config.ray_parameter_s_per_km * layer.vp_km_s >= 1.0 for layer in model.layers):
            return np.full(observation.amplitudes.shape, np.nan)
        synthetic = forward_ps_receiver_function(
            model,
            config.ray_parameter_s_per_km,
            sampling_interval_s=config.forward_sampling_interval_s,
            npts=config.forward_npts,
            delay_s=config.forward_delay_s,
            gaussian_alpha=config.gaussian_alpha,
        )
    elif phase == "S":
        if config.sp_ray_parameter_s_per_km is None:
            raise ValueError("Joint inversion requires sp_ray_parameter_s_per_km.")
        # Positive-lag Sp data require enough pre-S synthetic duration to span
        # the complete observed window before time reversal.
        sp_delay_s = max(
            config.forward_delay_s,
            float(np.max(observation.lag_times_s)) + config.forward_sampling_interval_s,
        )
        synthetic = forward_sp_receiver_function(
            model,
            config.sp_ray_parameter_s_per_km,
            sampling_interval_s=config.forward_sampling_interval_s,
            npts=config.forward_npts,
            delay_s=sp_delay_s,
            gaussian_alpha=config.sp_gaussian_alpha or config.gaussian_alpha,
            positive_lag_convention=config.sp_positive_lag_convention,
        )
    else:
        raise ValueError("phase must be 'P' or 'S'.")
    amplitudes = synthetic.amplitudes
    if phase == "S":
        amplitudes = config.sp_component_polarity * amplitudes
    return np.interp(observation.lag_times_s, synthetic.lag_times_s, amplitudes, left=0.0, right=0.0)


def _profile_amplitude_and_log_likelihood(
    prediction: np.ndarray,
    observation: RFObservation,
    *,
    min_lag_s: float,
    max_lag_s: float,
    likelihood_weight: float,
    scale_bounds: tuple[float, float],
) -> tuple[float, np.ndarray, float]:
    if not np.all(np.isfinite(prediction)):
        return -np.inf, prediction, np.nan
    selected = (observation.lag_times_s >= min_lag_s) & (observation.lag_times_s <= max_lag_s)
    if not np.any(selected):
        raise ValueError("The RF likelihood window does not contain any observations.")
    observed = observation.amplitudes[selected]
    predicted = prediction[selected]
    uncertainty = observation.standard_deviation[selected]
    weights = uncertainty**-2
    denominator = float(np.sum(weights * predicted**2))
    if denominator <= np.finfo(float).tiny:
        scale = scale_bounds[0]
    else:
        # Profile out a positive amplitude nuisance parameter, avoiding an
        # arbitrary mismatch between deconvolution and CPS amplitude conventions.
        unconstrained_scale = float(np.sum(weights * observed * predicted) / denominator)
        scale = float(np.clip(unconstrained_scale, *scale_bounds))
    residual = scale * predicted - observed
    log_likelihood = -0.5 * np.sum(residual**2 * weights + np.log(2.0 * np.pi * uncertainty**2))
    return float(likelihood_weight * log_likelihood), prediction * scale, scale


def _log_likelihood(
    state: LayerCakeModelState,
    observation: RFObservation,
    config: LayerCakeInversionConfig,
) -> tuple[float, np.ndarray, float]:
    prediction = _forward_prediction(state, observation, config, phase="P")
    return _profile_amplitude_and_log_likelihood(
        prediction,
        observation,
        min_lag_s=config.min_lag_s,
        max_lag_s=config.max_lag_s,
        likelihood_weight=config.ps_likelihood_weight,
        scale_bounds=config.ps_scale_bounds,
    )


def _joint_log_likelihood(
    state: LayerCakeModelState,
    ps_observation: RFObservation,
    sp_observation: RFObservation,
    config: LayerCakeInversionConfig,
) -> tuple[float, np.ndarray, float, np.ndarray, float]:
    ps_log_likelihood, ps_prediction, ps_scale = _log_likelihood(state, ps_observation, config)
    sp_raw_prediction = _forward_prediction(state, sp_observation, config, phase="S")
    sp_log_likelihood, sp_prediction, sp_scale = _profile_amplitude_and_log_likelihood(
        sp_raw_prediction,
        sp_observation,
        min_lag_s=config.sp_min_lag_s if config.sp_min_lag_s is not None else config.min_lag_s,
        max_lag_s=config.sp_max_lag_s if config.sp_max_lag_s is not None else config.max_lag_s,
        likelihood_weight=config.sp_likelihood_weight,
        scale_bounds=config.sp_scale_bounds,
    )
    return ps_log_likelihood + sp_log_likelihood, ps_prediction, ps_scale, sp_prediction, sp_scale


def _normal_log_pdf(value: float, standard_deviation: float) -> float:
    return -0.5 * (value / standard_deviation) ** 2 - np.log(standard_deviation * np.sqrt(2.0 * np.pi))


def _propose_birth(
    rng: np.random.Generator,
    state: LayerCakeModelState,
    config: LayerCakeInversionConfig,
) -> tuple[LayerCakeModelState | None, float]:
    if state.layer_count >= config.max_layers:
        return None, 0.0
    index = int(rng.integers(state.layer_count))
    thickness = state.thicknesses_km[index]
    lower_fraction = config.min_thickness_km / thickness
    upper_fraction = 1.0 - lower_fraction
    if lower_fraction >= upper_fraction:
        return None, 0.0
    fraction = float(rng.uniform(lower_fraction, upper_fraction))
    vs_perturbation = float(rng.normal(scale=config.birth_vs_step_km_s))
    vpvs_perturbation = float(rng.normal(scale=config.birth_vpvs_step))
    child_vs_1 = state.vs_km_s[index] + vs_perturbation
    child_vs_2 = state.vs_km_s[index] - vs_perturbation
    child_vpvs_1 = state.vpvs[index] + vpvs_perturbation
    child_vpvs_2 = state.vpvs[index] - vpvs_perturbation

    proposal = LayerCakeModelState(
        thicknesses_km=(
            *state.thicknesses_km[:index],
            fraction * thickness,
            (1.0 - fraction) * thickness,
            *state.thicknesses_km[index + 1 :],
        ),
        vs_km_s=(
            *state.vs_km_s[:index],
            child_vs_1,
            child_vs_2,
            *state.vs_km_s[index + 1 :],
        ),
        halfspace_vs_km_s=state.halfspace_vs_km_s,
        vpvs=(
            *state.vpvs[:index],
            child_vpvs_1,
            child_vpvs_2,
            *state.vpvs[index + 1 :],
        ),
    )
    if not _is_valid(proposal, config):
        return None, 0.0

    log_q_reverse_over_forward = (
        -np.log(upper_fraction - lower_fraction)
        - _normal_log_pdf(vs_perturbation, config.birth_vs_step_km_s)
        - _normal_log_pdf(vpvs_perturbation, config.birth_vpvs_step)
    )
    # Splitting maps (h, Vs, Vp/Vs) to two layer values; its determinant is
    # h * 2 * 2, with the fraction and two perturbations held fixed.
    log_jacobian = np.log(4.0 * thickness)
    return proposal, log_q_reverse_over_forward + log_jacobian


def _propose_death(
    rng: np.random.Generator,
    state: LayerCakeModelState,
    config: LayerCakeInversionConfig,
) -> tuple[LayerCakeModelState | None, float]:
    if state.layer_count <= config.min_layers:
        return None, 0.0
    index = int(rng.integers(state.layer_count - 1))
    thickness_1 = state.thicknesses_km[index]
    thickness_2 = state.thicknesses_km[index + 1]
    merged_thickness = thickness_1 + thickness_2
    fraction = thickness_1 / merged_thickness
    vs_perturbation = 0.5 * (state.vs_km_s[index] - state.vs_km_s[index + 1])
    vpvs_perturbation = 0.5 * (state.vpvs[index] - state.vpvs[index + 1])
    lower_fraction = config.min_thickness_km / merged_thickness
    upper_fraction = 1.0 - lower_fraction
    if not lower_fraction < fraction < upper_fraction:
        return None, 0.0

    proposal = LayerCakeModelState(
        thicknesses_km=(
            *state.thicknesses_km[:index],
            merged_thickness,
            *state.thicknesses_km[index + 2 :],
        ),
        vs_km_s=(
            *state.vs_km_s[:index],
            0.5 * (state.vs_km_s[index] + state.vs_km_s[index + 1]),
            *state.vs_km_s[index + 2 :],
        ),
        halfspace_vs_km_s=state.halfspace_vs_km_s,
        vpvs=(
            *state.vpvs[:index],
            0.5 * (state.vpvs[index] + state.vpvs[index + 1]),
            *state.vpvs[index + 2 :],
        ),
    )
    if not _is_valid(proposal, config):
        return None, 0.0

    log_q_reverse_over_forward = (
        np.log(upper_fraction - lower_fraction)
        + _normal_log_pdf(vs_perturbation, config.birth_vs_step_km_s)
        + _normal_log_pdf(vpvs_perturbation, config.birth_vpvs_step)
    )
    log_jacobian_inverse = -np.log(4.0 * merged_thickness)
    return proposal, log_q_reverse_over_forward + log_jacobian_inverse


def _propose_within_model(
    rng: np.random.Generator,
    state: LayerCakeModelState,
    config: LayerCakeInversionConfig,
) -> LayerCakeModelState | None:
    move = str(rng.choice(np.array(["vs", "thickness", "vpvs"]), p=np.array([0.55, 0.30, 0.15])))
    thicknesses = list(state.thicknesses_km)
    velocities = list(state.vs_km_s)
    halfspace_vs = state.halfspace_vs_km_s
    vpvs = list(state.vpvs)

    if move == "vs":
        index = int(rng.integers(state.layer_count + 1))
        if index == state.layer_count:
            halfspace_vs += rng.normal(scale=config.vs_step_km_s)
        else:
            velocities[index] += rng.normal(scale=config.vs_step_km_s)
    elif move == "thickness":
        index = int(rng.integers(state.layer_count))
        thicknesses[index] += rng.normal(scale=config.thickness_step_km)
    else:
        index = int(rng.integers(state.layer_count + 1))
        vpvs[index] += rng.normal(scale=config.vpvs_step)

    proposal = LayerCakeModelState(tuple(thicknesses), tuple(velocities), float(halfspace_vs), tuple(vpvs))
    return proposal if _is_valid(proposal, config) else None


def _run_layer_cake_inversion(
    ps_observation: RFObservation,
    config: LayerCakeInversionConfig,
    sp_observation: RFObservation | None,
    forward_predictor: Callable[[LayerCakeModelState], np.ndarray] | None = None,
) -> LayerCakeInversionResult:
    """Shared sampler implementation for Ps-only and joint Ps/Sp inversions."""
    if sp_observation is not None and config.sp_ray_parameter_s_per_km is None:
        raise ValueError("Joint inversion requires sp_ray_parameter_s_per_km in the configuration.")

    def evaluate(
        state: LayerCakeModelState,
    ) -> tuple[float, np.ndarray, float, np.ndarray | None, float | None]:
        if sp_observation is None:
            if forward_predictor is None:
                log_likelihood, prediction, scale = _log_likelihood(state, ps_observation, config)
            else:
                raw = np.asarray(forward_predictor(state), dtype=float)
                if raw.shape != ps_observation.amplitudes.shape:
                    raise ValueError("Custom Ps predictor must return the observation shape.")
                if not np.all(np.isfinite(raw)):
                    return -np.inf, raw, np.nan, None, None
                log_likelihood, prediction, scale = _profile_amplitude_and_log_likelihood(
                    raw,
                    ps_observation,
                    min_lag_s=config.min_lag_s,
                    max_lag_s=config.max_lag_s,
                    likelihood_weight=config.ps_likelihood_weight,
                    scale_bounds=config.ps_scale_bounds,
                )
            return log_likelihood, prediction, scale, None, None
        return _joint_log_likelihood(state, ps_observation, sp_observation, config)

    rng = np.random.default_rng(config.random_seed)
    current = _initial_state(config)
    current_log_likelihood, current_prediction, current_scale, current_sp_prediction, current_sp_scale = (
        evaluate(current)
    )
    current_log_posterior = current_log_likelihood + _log_prior(current, config)
    if not np.isfinite(current_log_posterior):
        raise ValueError("Initial model must have a finite log posterior.")

    best_state = current
    best_prediction = current_prediction.copy()
    best_scale = current_scale
    best_sp_prediction = None if current_sp_prediction is None else current_sp_prediction.copy()
    best_sp_scale = current_sp_scale
    best_log_posterior = current_log_posterior
    depth_km = np.arange(
        0.0, config.max_depth_km + 0.5 * config.profile_depth_interval_km, config.profile_depth_interval_km
    )
    saved_profiles: list[np.ndarray] = []
    saved_vpvs_profiles: list[np.ndarray] = []
    saved_layer_counts: list[int] = []
    saved_predictions: list[np.ndarray] = []
    saved_log_posteriors: list[float] = []
    saved_scales: list[float] = []
    saved_models: list[LayerCakeModelState] = []
    accepted = 0

    for iteration in range(1, config.iterations + 1):
        move = (
            str(rng.choice(np.array(["birth", "death", "within"])))
            if config.fixed_layers is None
            else "within"
        )
        log_move_correction = 0.0
        if move == "birth":
            proposal, log_move_correction = _propose_birth(rng, current, config)
        elif move == "death":
            proposal, log_move_correction = _propose_death(rng, current, config)
        else:
            proposal = _propose_within_model(rng, current, config)

        if proposal is not None:
            (
                proposal_log_likelihood,
                proposal_prediction,
                proposal_scale,
                proposal_sp_prediction,
                proposal_sp_scale,
            ) = evaluate(proposal)
            proposal_log_posterior = proposal_log_likelihood + _log_prior(proposal, config)
            log_acceptance = proposal_log_posterior - current_log_posterior + log_move_correction
            if np.log(rng.uniform()) < log_acceptance:
                current = proposal
                current_log_likelihood = proposal_log_likelihood
                current_log_posterior = proposal_log_posterior
                current_prediction = proposal_prediction
                current_scale = proposal_scale
                current_sp_prediction = proposal_sp_prediction
                current_sp_scale = proposal_sp_scale
                accepted += 1
                if current_log_posterior > best_log_posterior:
                    best_state = current
                    best_prediction = current_prediction.copy()
                    best_scale = current_scale
                    best_sp_prediction = (
                        None if current_sp_prediction is None else current_sp_prediction.copy()
                    )
                    best_sp_scale = current_sp_scale
                    best_log_posterior = current_log_posterior

        if iteration > config.burnin and iteration % config.save_every == 0:
            saved_profiles.append(_profile_from_state(current, depth_km))
            saved_vpvs_profiles.append(_vpvs_profile_from_state(current, depth_km))
            saved_layer_counts.append(current.layer_count)
            saved_predictions.append(current_prediction.copy())
            saved_log_posteriors.append(current_log_posterior)
            saved_scales.append(current_scale)
            saved_models.append(current)

    if not saved_profiles:
        raise ValueError("No posterior samples were saved. Increase iterations or lower burnin.")

    profiles = np.vstack(saved_profiles)
    vpvs_profiles = np.vstack(saved_vpvs_profiles)
    return LayerCakeInversionResult(
        depth_km=depth_km,
        mean_vs_km_s=np.mean(profiles, axis=0),
        vs_lower_km_s=np.quantile(profiles, 0.05, axis=0),
        vs_upper_km_s=np.quantile(profiles, 0.95, axis=0),
        saved_vs_profiles=profiles,
        mean_vpvs=np.mean(vpvs_profiles, axis=0),
        vpvs_lower=np.quantile(vpvs_profiles, 0.05, axis=0),
        vpvs_upper=np.quantile(vpvs_profiles, 0.95, axis=0),
        saved_vpvs_profiles=vpvs_profiles,
        saved_layer_counts=np.asarray(saved_layer_counts),
        best_model=best_state,
        best_prediction=best_prediction,
        best_scale=best_scale,
        best_log_posterior=best_log_posterior,
        acceptance_rate=accepted / config.iterations,
        observation=ps_observation,
        metadata={
            "ray_parameter_s_per_km": config.ray_parameter_s_per_km,
            "sp_ray_parameter_s_per_km": (
                config.sp_ray_parameter_s_per_km if config.sp_ray_parameter_s_per_km is not None else 0.0
            ),
            "gaussian_alpha": config.gaussian_alpha,
            "sp_gaussian_alpha": config.sp_gaussian_alpha or config.gaussian_alpha,
            "forward_sampling_interval_s": config.forward_sampling_interval_s,
            "likelihood_min_lag_s": config.min_lag_s,
            "likelihood_max_lag_s": config.max_lag_s,
            "sp_likelihood_min_lag_s": (
                config.sp_min_lag_s if config.sp_min_lag_s is not None else config.min_lag_s
            ),
            "sp_likelihood_max_lag_s": (
                config.sp_max_lag_s if config.sp_max_lag_s is not None else config.max_lag_s
            ),
            "inversion_mode": "joint_ps_sp" if sp_observation is not None else "ps",
            "ps_likelihood_weight": config.ps_likelihood_weight,
            "sp_likelihood_weight": config.sp_likelihood_weight if sp_observation is not None else 0.0,
            "sp_positive_lag_convention": int(config.sp_positive_lag_convention),
            "sp_component_polarity": config.sp_component_polarity,
            "ps_minimum_scale": config.ps_scale_bounds[0],
            "sp_minimum_scale": config.sp_scale_bounds[0],
            "parameterization": "rj" if config.fixed_layers is None else "fixed",
            "fixed_finite_layers": config.fixed_layers if config.fixed_layers is not None else 0,
            "vs_prior_lower_km_s": config.vs_bounds_km_s[0],
            "vs_prior_upper_km_s": config.vs_bounds_km_s[1],
            "vpvs_prior_lower": config.vpvs_bounds[0],
            "vpvs_prior_upper": config.vpvs_bounds[1],
        },
        sp_observation=sp_observation,
        best_sp_prediction=best_sp_prediction,
        best_sp_scale=best_sp_scale,
        saved_predictions=np.asarray(saved_predictions),
        saved_log_posteriors=np.asarray(saved_log_posteriors),
        saved_scales=np.asarray(saved_scales),
        saved_models=tuple(saved_models),
    )


def run_layer_cake_inversion(
    observation: RFObservation,
    config: LayerCakeInversionConfig,
    *,
    forward_predictor: Callable[[LayerCakeModelState], np.ndarray] | None = None,
) -> LayerCakeInversionResult:
    """Sample Ps structure, optionally using a resolution-matched predictor.

    The optional callable takes a model state and returns amplitudes on the
    observation grid. Nonfinite predictions reject physically invalid states.
    """
    return _run_layer_cake_inversion(observation, config, None, forward_predictor)


def run_joint_layer_cake_inversion(
    ps_observation: RFObservation,
    sp_observation: RFObservation,
    config: LayerCakeInversionConfig,
) -> LayerCakeInversionResult:
    """Jointly sample one layer cake against Ps and Sp receiver functions."""
    return _run_layer_cake_inversion(ps_observation, config, sp_observation)


def _log_normal_interval(lower, upper):
    if lower > 0:
        high, low = log_ndtr(-lower), log_ndtr(-upper)
    else:
        high, low = log_ndtr(upper), log_ndtr(lower)
    return high + np.log(-np.expm1(low - high))


@dataclass
class CorrelatedRFLikelihood:
    """Gaussian errors with a positive uniform amplitude analytically marginalized."""

    observed: np.ndarray
    covariance: np.ndarray
    scale_bounds: tuple[float, float] = (0.05, 5.0)

    def __post_init__(self):
        self.observed = np.asarray(self.observed, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)
        n = self.observed.size
        if (
            self.observed.ndim != 1
            or self.covariance.shape != (n, n)
            or not np.all(np.isfinite(self.observed))
            or not np.all(np.isfinite(self.covariance))
            or not np.allclose(self.covariance, self.covariance.T)
            or not 0 <= self.scale_bounds[0] < self.scale_bounds[1]
            or not np.all(np.isfinite(self.scale_bounds))
        ):
            raise ValueError("Invalid RF data, covariance, or scale prior.")
        self.factor = cholesky(self.covariance, lower=True)
        self.whitened_data = solve_triangular(self.factor, self.observed, lower=True)
        self.data_power = self.whitened_data @ self.whitened_data
        self.normalization = -0.5 * n * np.log(2 * np.pi) - np.log(self.factor.diagonal()).sum()

    def evaluate(self, prediction):
        """Return marginal log likelihood and conditional normal location/SD."""
        prediction = np.asarray(prediction, dtype=float)
        if prediction.shape != self.observed.shape or not np.all(np.isfinite(prediction)):
            raise ValueError("Prediction must match the finite observed RF.")
        white = solve_triangular(self.factor, prediction, lower=True, check_finite=False)
        a, b = float(white @ white), float(white @ self.whitened_data)
        if a < 1e-20:
            return self.normalization - 0.5 * self.data_power, 0.0, np.inf
        location, sd = b / a, 1 / np.sqrt(a)
        lo, hi = self.scale_bounds
        log_mass = _log_normal_interval((lo - location) / sd, (hi - location) / sd)
        log_likelihood = (
            self.normalization
            - 0.5 * (self.data_power - b * b / a)
            + 0.5 * np.log(2 * np.pi / a)
            + log_mass
            - np.log(hi - lo)
        )
        return float(log_likelihood), location, sd

    def sample_scale(self, location, sd, rng):
        lo, hi = self.scale_bounds
        if not np.isfinite(sd):
            return rng.uniform(lo, hi)
        return float(
            truncnorm.rvs(
                (lo - location) / sd, (hi - location) / sd, loc=location, scale=sd, random_state=rng
            )
        )


def sample_adaptive_metropolis(
    evaluate, initial, lower, upper, *, iterations=30000, burnin=15000, save_every=25, seed=42, progress=None
):
    """Evaluate physical vectors; callback returns (log density, numeric auxiliary).

    Bounds define the constant-volume affine coordinate transform only. The
    callback must reject any additional constraints and include its own prior.
    Warm-up covariance uses all states including repeated rejected states.
    Neither adaptation nor optimization continues into retained sampling.
    """
    initial, lower, upper = (np.asarray(x, dtype=float) for x in (initial, lower, upper))
    if (
        initial.ndim != 1
        or initial.shape != lower.shape
        or lower.shape != upper.shape
        or not np.all(np.isfinite([initial, lower, upper]))
        or np.any(upper <= lower)
        or np.any(initial < lower)
        or np.any(initial > upper)
        or not 0 <= burnin < iterations
        or save_every < 1
    ):
        raise ValueError("Invalid bounds, initial state, or MCMC schedule.")
    rng = np.random.default_rng(seed)
    width = upper - lower
    x = (initial - lower) / width
    current, aux = evaluate(initial)
    if not np.isfinite(current):
        raise ValueError("Initial log density must be finite.")
    aux = np.asarray(aux, dtype=float)
    dimension = x.size
    covariance = np.eye(dimension) * 0.02**2
    factor = cholesky(covariance, lower=True)
    mean, scatter, count = x.copy(), np.zeros_like(covariance), 1
    log_scale = 0.0
    best, best_x, best_aux = current, x.copy(), aux.copy()
    samples, log_values, saved_aux, warmup = [], [], [], []
    accepted = np.zeros(3, dtype=int)
    proposed = np.zeros(3, dtype=int)
    window_accepted = window_proposed = 0
    for step in range(1, iterations + 1):
        move = int(rng.choice(3, p=[0.65, 0.30, 0.05]))
        proposed[move] += 1
        if move == 0:
            proposal = x + np.exp(log_scale) * (2.38 / np.sqrt(dimension)) * (
                factor @ rng.normal(size=dimension)
            )
            window_proposed += 1
        elif move == 1:
            proposal = x.copy()
            index = rng.integers(dimension)
            proposal[index] += rng.normal(scale=0.02)
        else:
            proposal = rng.uniform(size=dimension)
        if np.all((proposal >= 0) & (proposal <= 1)):
            candidate, candidate_aux = evaluate(lower + width * proposal)
            if np.isfinite(candidate) and np.log(rng.uniform()) < candidate - current:
                x, current, aux = proposal, candidate, np.asarray(candidate_aux, dtype=float)
                accepted[move] += 1
                window_accepted += int(move == 0)
                if current > best:
                    best, best_x, best_aux = current, x.copy(), aux.copy()
        if step <= burnin:
            count += 1
            delta = x - mean
            mean += delta / count
            scatter += np.outer(delta, x - mean)
            if step >= 500 and step % 100 == 0:
                covariance = scatter / (count - 1) + np.eye(dimension) * 1e-5
                factor = cholesky(covariance, lower=True)
                if window_proposed:
                    log_scale = np.clip(
                        log_scale + 0.2 * (window_accepted / window_proposed - 0.234), -3.0, 2.0
                    )
                window_accepted = window_proposed = 0
            if step % save_every == 0:
                warmup.append(current)
        elif (step - burnin) % save_every == 0:
            samples.append(lower + width * x)
            log_values.append(current)
            saved_aux.append(aux.copy())
        if progress is not None and (step % 2500 == 0 or step == iterations):
            progress(step, current, best, float(accepted.sum() / step))
    if not samples:
        raise ValueError("Schedule retained no samples.")
    return dict(
        samples=np.array(samples),
        log_posterior=np.array(log_values),
        auxiliary=np.array(saved_aux),
        warmup_log_posterior=np.array(warmup),
        best_parameters=lower + width * best_x,
        best_log_posterior=best,
        best_auxiliary=best_aux,
        acceptance_by_move=accepted / np.maximum(proposed, 1),
        acceptance_rate=accepted.sum() / iterations,
        proposal_covariance=covariance,
        proposal_scale=np.exp(log_scale),
    )


def match_cps_resolution(
    synthetic: CPSReceiverFunction,
    resolution: RFResolution,
    lag_times: np.ndarray,
) -> np.ndarray:
    """Replace CPS's numerical source and Gaussian by the measured resolution.

    hrftn96 outputs an RF density convolved with a three-sample triangular
    source and exp[-(pi*f/alpha)^2]. Remove these known filters before applying
    the source-resolution transfer. No unstable division near Nyquist is used.
    The resolution's dt/normalization converts density to self-RF units.
    """
    dt = resolution.dt
    if not np.isclose(synthetic.sampling_interval_s, dt, rtol=1e-6):
        raise ValueError("CPS and resolution sampling intervals must agree.")
    if len(synthetic.amplitudes) > resolution.nfft:
        raise ValueError("Resolution FFT must be long enough to hold the CPS RF.")
    indices = np.rint(synthetic.lag_times_s / dt).astype(int)
    if not np.allclose(indices * dt, synthetic.lag_times_s, atol=1e-4):
        raise ValueError("CPS lags must lie on the resolution sampling grid.")
    density = np.zeros(resolution.nfft)
    density[indices % resolution.nfft] = synthetic.amplitudes
    frequencies = rfftfreq(resolution.nfft, dt)
    numerical_pulse = np.cos(np.pi * frequencies * dt) ** 2
    numerical_pulse *= np.exp(-((np.pi * frequencies / synthetic.gaussian_alpha) ** 2))
    if np.sum(abs(resolution.transfer[numerical_pulse <= 1e-4])) > 1e-4 * np.sum(abs(resolution.transfer)):
        raise ValueError(
            "CPS pulse suppresses resolved frequencies; increase its Gaussian alpha or sampling rate."
        )
    transfer = np.divide(
        resolution.transfer,
        numerical_pulse,
        out=np.zeros_like(numerical_pulse),
        where=numerical_pulse > 1e-4,
    )
    filtered = irfft(rfft(density) * transfer, resolution.nfft) * dt / resolution.normalization
    # Match the 10-Hz stack interpolation, not just its original 4-Hz samples.
    times, values = resolution._extract(filtered, lag_times[0], lag_times[-1], 10.0)
    return np.interp(lag_times, times, values)


@dataclass
class ResolvedPSStackForward:
    """Equal-weight mean of source-resolved CPS RFs.

    Sampling can use the mean ray parameter with the exact mean normalized
    resolution transfer. ``exact=True`` instead evaluates every event at its
    own ray parameter, for a post-run approximation check or direct fitting.
    """

    resolutions: tuple[RFResolution, ...]
    ray_parameters: np.ndarray
    lag_times: np.ndarray
    exact: bool = False
    npts: int = 512
    delay_s: float = 20.0
    gaussian_alpha: float = 2.82842712474619

    def __post_init__(self):
        self.ray_parameters = np.asarray(self.ray_parameters, dtype=float)
        if len(self.resolutions) != self.ray_parameters.size or not self.resolutions:
            raise ValueError("Require one resolution and positive ray parameter per event.")
        if np.any(self.ray_parameters <= 0):
            raise ValueError("Ray parameters must be positive.")
        first = self.resolutions[0]
        if any(r.dt != first.dt or r.nfft != first.nfft for r in self.resolutions):
            raise ValueError("Stack resolution grids must agree.")
        transfer = np.mean([r.transfer / r.normalization for r in self.resolutions], axis=0)
        self.mean_resolution = RFResolution(first.dt, first.nfft, transfer, first.inverse_filter, 1.0)

    def __call__(self, state: LayerCakeModelState) -> np.ndarray:
        # CPS otherwise silently moves the incident boundary to a shallower
        # layer when P cannot propagate in the proposed halfspace.
        vp = np.array([*state.vs_km_s, state.halfspace_vs_km_s]) * state.vpvs
        if max(self.ray_parameters) * max(vp) >= 1.0:
            return np.full(self.lag_times.shape, np.nan)
        model = _to_cps_model(state)
        ray_parameters = self.ray_parameters if self.exact else [np.mean(self.ray_parameters)]
        resolutions = self.resolutions if self.exact else [self.mean_resolution]
        predictions = []
        for ray_parameter, resolution in zip(ray_parameters, resolutions, strict=True):
            synthetic = forward_ps_receiver_function(
                model,
                float(ray_parameter),
                sampling_interval_s=resolution.dt,
                npts=self.npts,
                delay_s=self.delay_s,
                gaussian_alpha=self.gaussian_alpha,
            )
            predictions.append(match_cps_resolution(synthetic, resolution, self.lag_times))
        return np.mean(predictions, axis=0)


@dataclass
class MultitaperOperator:
    """Linear map from a radial record to its RF, conditional on observed Z.

    Source and noise are scaled together. ``matrix @ (R/source_scale)``
    reproduces classical.multitaper_rf(...).waveform(), including normalization.
    Unlike a stationary self-pulse filter, it retains the DPSS lag dependence.
    """

    source: np.ndarray
    source_scale: float
    dt: float
    lag_times: np.ndarray
    matrix: np.ndarray

    @classmethod
    def from_source(
        cls,
        source,
        source_noise,
        dt,
        *,
        sigma_s=0.25,
        time_bandwidth=2.5,
        tapers=3,
        noise_multiplier=1.0,
        numerical_floor=1e-6,
        min_lag=-5.0,
        max_lag=25.0,
    ):
        z = np.asarray(source, dtype=float)
        noise = np.asarray(source_noise, dtype=float)
        if (
            z.ndim != 1
            or z.size < 8
            or not np.all(np.isfinite(z))
            or noise.ndim != 1
            or noise.size < 8
            or not np.all(np.isfinite(noise))
            or not np.isfinite(dt)
            or dt <= 0
            or np.max(abs(z)) == 0
            or sigma_s < 0
            or noise_multiplier < 0
            or numerical_floor <= 0
            or not 1 <= tapers <= int(2 * time_bandwidth - 1)
            or time_bandwidth >= min(z.size, noise.size) / 2
            or min_lag >= max_lag
        ):
            raise ValueError("Invalid source, noise, or multitaper settings.")
        scale = float(np.max(abs(z)))
        z, noise = z / scale, noise / scale
        nfft = next_fast_len(4 * len(z))
        windows = dpss(len(z), time_bandwidth, Kmax=tapers, norm=2)
        noise_windows = dpss(len(noise), time_bandwidth, Kmax=tapers, norm=2)
        zs = rfft(windows * z, nfft, axis=-1)
        ns = rfft(noise_windows * (noise - noise.mean()), nfft, axis=-1)
        power = np.mean(abs(zs) ** 2, axis=0)
        denominator = np.maximum(
            power + noise_multiplier * np.mean(abs(ns) ** 2, axis=0), numerical_floor * power.max()
        )
        gaussian = np.exp(-0.5 * (2 * np.pi * rfftfreq(nfft, dt) * sigma_s) ** 2)
        normalization = irfft(power / denominator * gaussian, nfft)[0]
        kernels = irfft(zs.conj() / denominator * gaussian, nfft, axis=-1) / (tapers * normalization)
        lags = np.arange(round(min_lag / dt), round(max_lag / dt) + 1)
        if max(abs(lags)) >= nfft // 2:
            raise ValueError("Requested RF lags exceed the unaliased interval.")
        offsets = (lags[:, None] - np.arange(z.size)[None, :]) % nfft
        matrix = sum(kernel[offsets] * window[None, :] for kernel, window in zip(kernels, windows))
        return cls(z, scale, dt, lags * dt, matrix)

    def apply(self, response):
        response = np.asarray(response, dtype=float)
        if response.shape != self.source.shape or not np.all(np.isfinite(response)):
            raise ValueError("Response must have the source shape and finite values.")
        return self.matrix @ (response / self.source_scale)

    def kernel_matrix(self, kernel_lags_s):
        """Map impulse masses to RFs via non-wrapping Z * RF convolution."""
        lags = np.asarray(kernel_lags_s, dtype=float)
        samples = np.rint(lags / self.dt).astype(int)
        if not np.allclose(lags, samples * self.dt):
            raise ValueError("Impulse-response lags must lie on the source grid.")
        indices = np.arange(self.source.size)[:, None] - samples[None, :]
        valid = (indices >= 0) & (indices < self.source.size)
        design = self.source[np.clip(indices, 0, self.source.size - 1)] * valid
        return self.matrix @ design


def cps_impulse_masses(synthetic: CPSReceiverFunction, output_dt: float):
    """Remove CPS's known Gaussian/triangle on an oversampled numerical grid.

    Only frequencies through the coarser data Nyquist are retained. CPS must
    be sampled at least twice as fast, so triangle deconvolution is bounded
    away from its zero. Returned masses retain negative as well as positive
    lags; no causal truncation or extra Gaussian is imposed here.
    """
    fine_dt = synthetic.sampling_interval_s
    ratio = output_dt / fine_dt
    if ratio < 2 or not np.isclose(ratio, round(ratio)):
        raise ValueError("CPS must use an integer oversampling factor of at least two.")
    size = len(synthetic.amplitudes)
    coarse_size = size // round(ratio)
    if coarse_size * round(ratio) != size:
        raise ValueError("CPS record length must divide onto the data grid.")
    frequencies = rfftfreq(coarse_size, output_dt)
    pulse = np.cos(np.pi * frequencies * fine_dt) ** 2
    pulse *= np.exp(-((np.pi * frequencies / synthetic.gaussian_alpha) ** 2))
    if pulse.min() < 1e-5:
        raise ValueError("CPS Gaussian is too broad for stable pulse replacement.")
    spectrum = rfft(synthetic.amplitudes)[: frequencies.size] * fine_dt / pulse
    masses = irfft(spectrum, coarse_size)
    times = synthetic.lag_times_s[0] + np.arange(coarse_size) * output_dt
    return times, masses


@dataclass
class MultitaperPSStackForward:
    """Event-operator mean, with optional exact per-event CPS ray parameters.

    The fast sampling mode uses the mean slowness but the exact mean of the
    full event processing matrices. Exact mode tests this approximation using
    all event slownesses. Neither mode substitutes a Gaussian for the DPSS map.
    """

    operators: tuple[MultitaperOperator, ...]
    ray_parameters: np.ndarray
    exact: bool = False
    coarse_npts: int = 512
    delay_s: float = 20.0
    gaussian_alpha: float = 8.0

    def __post_init__(self):
        self.ray_parameters = np.asarray(self.ray_parameters, dtype=float)
        if not self.operators or self.ray_parameters.shape != (len(self.operators),):
            raise ValueError("Require one ray parameter per event.")
        if not np.all(np.isfinite(self.ray_parameters)) or np.any(self.ray_parameters <= 0):
            raise ValueError("Ray parameters must be finite and positive.")
        self.dt = self.operators[0].dt
        self.lag_times = self.operators[0].lag_times
        if any(o.dt != self.dt or not np.array_equal(o.lag_times, self.lag_times) for o in self.operators):
            raise ValueError("Event operators must share time grids.")
        if self.coarse_npts < 2 or self.coarse_npts & (self.coarse_npts - 1):
            raise ValueError("CPS coarse_npts must be a power of two.")
        self.kernel_lags = np.arange(self.coarse_npts) * self.dt - self.delay_s
        self.event_matrices = np.array([o.kernel_matrix(self.kernel_lags) for o in self.operators])
        self.mean_matrix = self.event_matrices.mean(axis=0)

    def __call__(self, state: LayerCakeModelState):
        # getmmxrayp in hrftn96 stops at the FIRST nonpropagating P layer,
        # even when the halfspace itself is valid, silently truncating models.
        vp = np.array([*state.vs_km_s, state.halfspace_vs_km_s]) * state.vpvs
        if max(self.ray_parameters) * max(vp) >= 1.0:
            return np.full(self.lag_times.shape, np.nan)
        model = _to_cps_model(state)
        parameters = self.ray_parameters if self.exact else [self.ray_parameters.mean()]
        matrices = self.event_matrices if self.exact else [self.mean_matrix]
        predictions = []
        for p, matrix in zip(parameters, matrices):
            synthetic = forward_ps_receiver_function(
                model,
                float(p),
                sampling_interval_s=self.dt / 2,
                npts=self.coarse_npts * 2,
                delay_s=self.delay_s,
                gaussian_alpha=self.gaussian_alpha,
            )
            times, masses = cps_impulse_masses(synthetic, self.dt)
            if not np.allclose(times, self.kernel_lags, atol=1e-5):
                raise ValueError("Unexpected CPS output grid.")
            predictions.append(matrix @ masses)
        return np.mean(predictions, axis=0)
