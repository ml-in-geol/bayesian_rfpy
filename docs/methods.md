# Methods and conventions

## Deconvolution

The Bayesian array convention is `response = dt * convolve(source, kernel)`;
Gaussian amplitudes are kernel densities. Classical spectra instead use impulse
masses. `ClassicalRF.waveform` smooths once and divides R by the positive Z
self-RF at zero lag. Never independently normalize or flip radial peaks.

| Function | Method | Main settings |
| --- | --- | --- |
| `run_bayesian_deconvolution` | Legacy Gaussian-pulse sampler | `BayesianRFConfig`, source/response/noise/dt |
| `waterlevel_rf` | Spectral water level | source-power floor fraction |
| `iterative_rf` | Ligorria & Ammon-style signed pulse stripping | fitting Gaussian, iterations, fractional-SSE tolerance |
| `multitaper_rf` | Equal-weight full-window DPSS | source-Z noise, time-bandwidth, taper count |
| `spiking_rf` | Wiener Toeplitz or source-gated spectral inverse | source window, damping, Gaussian |

Iterative deconvolution uses finite non-wrapping coordinate updates, not a
bit-for-bit port. Check `diagnostics['stopped_by_tolerance']`. Multitaper is
Park-Levin-style, NOT extended-time Helffrich multitaper; its sensitivity varies
with lag. Compare data and models at matched bandwidth and normalization.

`waveform(sigma_s=...)` takes Gaussian standard deviation in seconds. CPS uses
`gaussian_alpha` with transfer `exp[-(pi*f/alpha)^2]`, so
`sigma = 1/(sqrt(2)*alpha)` and Gaussian-only FWHM is `2*sqrt(log(2))/alpha`.
Source bandwidth and preprocessing also limit resolution. Bayesian Gaussian
widths are separate sampled quantities, defaulting to one tenth of the period
corners. `component_separation_factor=1` preserves standard non-overlap;
zero permits highly overlapping pulses, but does not fix sampler correctness.

## Preprocessing

`preprocessing.py` supports ObsPy-readable ZNE, SAC geometry, RTZ SAC,
free-surface transforms, and predicted synthetic P/S arrivals. All paths and
configuration are explicit. `preprocess_ps_data` assumes synthetic trace start
equals event origin. Real data require an explicit pick or
`data.preprocess_insight_event`, which computes the offset independently from
each product's start time and an absolute UTC catalog pick.

Physical deglitched displacement is differentiated once to velocity; denoised
velocity is not response-corrected again. Native counts need an inventory and
response removal before these physical-ZNE workflows. Missing directions or
non-direct-P picks require explicit overrides, not automatic relabelling.
Existing RF preprocessing uses zero-phase filters; this is not a first-motion
polarity-picking workflow. Verify gaps, orientation and signs independently.

## Inversion

```python
from bayesian_rf.inversion import (
    LayerCakeInversionConfig, load_text_rf_observation, run_layer_cake_inversion,
)
observation = load_text_rf_observation("prf.txt")  # time, RF[, standard deviation]
config = LayerCakeInversionConfig(
    ray_parameter_s_per_km=0.12,
    fixed_layers=3, initial_layers=3,
    vs_bounds_km_s=(1.0, 5.5), vpvs_bounds=(1.4, 2.4),
    min_lag_s=2.0, max_lag_s=20.0, gaussian_alpha=2.84,
    ps_scale_bounds=(0.05, 10.0),
    iterations=500_000, burnin=250_000, save_every=500,
)
result = run_layer_cake_inversion(observation, config)
```

`fixed_layers=None` enables RJ moves. N counts finite layers, excluding the
halfspace. Each finite layer and the halfspace has its own Vp/Vs; density is
an empirical function of Vp, not independently sampled. The generic default
Vp/Vs prior excludes KKS21's uppermost ratio; Mars examples explicitly widen
it to 1.4-2.4. Always inspect priors and sensitivity before interpretation.

`run_joint_layer_cake_inversion(prf, srf, config)` adds Sp data with its own
`sp_ray_parameter_s_per_km`. `calculate_ps_sp_ray_parameters` evaluates the
supplied planetary TauP model. CPS raw Sp has negative lag; the default joint
positive-lag convention compares `-RF_CPS(-t)`.

The standard likelihood uses independent sample errors and profiles a positive
amplitude scale for each dataset. This does not marginalize amplitude or
propagate a full deconvolution covariance. Optional `CorrelatedRFLikelihood`
and `sample_adaptive_metropolis` support correlated-error fixed-dimensional
experiments. Dense interpolation does not add independent information.

`MultitaperPSStackForward` applies the actual finite-window DPSS operator to
CPS kernels. `ResolvedPSStackForward` applies stationary source-resolution
transfers. Set `exact=True` to avoid their optional mean-slowness approximation.
Incident P must propagate in every layer: invalid models are rejected rather
than silently truncated by CPS. Large S slowness can make converted P evanescent
at depth, so Sp need not independently constrain every interface.

## References

- Kolb & Lekic (2014): https://doi.org/10.1093/gji/ggu079
- Ligorria & Ammon (1999): https://doi.org/10.1785/BSSA0890051395
- Park & Levin (2000): https://doi.org/10.1785/0119990122
- Herrmann (2013), CPS: https://doi.org/10.1785/0220110096
- Vehtari et al. (2021): https://arxiv.org/abs/1903.08008
