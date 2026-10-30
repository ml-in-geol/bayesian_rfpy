"""Calculate Mars synthetic RFs and optionally invert a multitaper RF for Vs.

Short runs are pipeline demonstrations, not converged posterior estimates.
Run from any directory after installing bayesian-rfpy with the plot extra.
"""

from dataclasses import asdict
import argparse
import json
from pathlib import Path

import numpy as np

from bayesian_rf import (
    BayesianRFConfig,
    MarsEvent,
    preprocess_ps_data,
    run_bayesian_deconvolution,
    iterative_rf,
    multitaper_rf,
    waterlevel_rf,
)
from bayesian_rf.inversion import (
    LayerCakeInversionConfig,
    MultitaperOperator,
    MultitaperPSStackForward,
    RFObservation,
    load_nd_vs_profile,
    run_layer_cake_inversion,
)

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = {
    "mt01": (10.18, 164.05, 18.4, 28.3),
    "mt02": (10.18, 164.05, 33.0, 28.3),
    "mt03": (-2.97, 171.90, 25.0, 37.0),
    "mt04": (13.50086804929426, -143.60079322153234, 33.0, 80.0),
    "mt05": (12.2608185206491, 179.45783163994645, 33.0, 44.0),
}


def prepare(event="mt01", noise="nonoise", data_root=ROOT / "example_synthetics", *, max_lag=20.0):
    if event in {"mt04", "mt05"} and noise != "nonoise":
        raise ValueError("Only distance-labelled no-noise MT04/MT05 files have verified geometry")
    lat, lon, depth, distance = GEOMETRY[event]
    suffix = {"mt04": "80deg_nonoise", "mt05": "44deg_nonoise"}.get(event, noise)
    path = Path(data_root) / "mars" / event / f"synth_{event}_{suffix}.mseed"
    metadata = MarsEvent(event, path, lat, lon, depth, distance_deg=distance)
    config = BayesianRFConfig(
        max_lag=max_lag,
        min_period=2.0,
        max_period=8.0,
        freqmin=1 / 8,
        freqmax=1 / 2,
        target_sampling_rate=4.0,
    )
    processed = preprocess_ps_data(path, metadata, Path(data_root) / "mars/insight_kks21_gp.nd", config)
    return processed, config


def calculate(processed, config, method="multitaper", *, sigma_s=0.5, iterations=100_000):
    z, r, dt = processed.parent, processed.daughter, processed.dt
    if method == "bayesian":
        config.iterations, config.burnin = iterations, iterations // 2
        config.save_every = max(1, iterations // 200)
        result = run_bayesian_deconvolution(z, r, processed.noise_window, dt, config)
        return result.lag_times, result.map_receiver_function, result
    if method == "waterlevel":
        result = waterlevel_rf(z, r, dt)
    elif method == "iterative":
        result = iterative_rf(z, r, dt, fit_sigma_s=sigma_s, max_lag=config.max_lag)
    elif method == "multitaper":
        # Multitaper stabilization uses noise on the source (Z), not radial noise.
        noise = z[: round(config.pre_arrival / dt)]
        result = multitaper_rf(z, r, dt, source_noise=noise)
    else:
        raise ValueError(f"Unknown method: {method}")
    lag, rf, _ = result.waveform(sigma_s=sigma_s, min_lag=0.0, max_lag=config.max_lag)
    return lag, rf, result


def invert_multitaper(
    processed, processing_config, *, iterations=20_000, fixed_layers=3, sigma_s=0.5, seed=42
):
    """Invert using the actual finite-window DPSS operator, not just a Gaussian."""
    z, dt = processed.parent, processed.dt
    operator = MultitaperOperator.from_source(
        z,
        z[: round(processing_config.pre_arrival / dt)],
        dt,
        sigma_s=sigma_s,
        min_lag=0.0,
        max_lag=processing_config.max_lag,
    )
    rf = operator.apply(processed.daughter)
    # An explicit illustrative error model, not posterior uncertainty from deconvolution.
    observation = RFObservation(operator.lag_times, rf, np.full(rf.shape, max(0.03 * np.max(abs(rf)), 0.01)))
    forward = MultitaperPSStackForward((operator,), np.array([processed.ray_parameter_s_per_km]))
    config = LayerCakeInversionConfig(
        ray_parameter_s_per_km=processed.ray_parameter_s_per_km,
        iterations=iterations,
        burnin=iterations // 2,
        save_every=max(1, iterations // 100),
        fixed_layers=fixed_layers,
        initial_layers=fixed_layers or 3,
        max_layers=6,
        initial_depth_km=45.0,
        max_depth_km=80.0,
        initial_vs_km_s=3.0,
        initial_halfspace_vs_km_s=4.0,
        initial_vpvs=1.7,
        vs_bounds_km_s=(1.0, 5.5),
        vpvs_bounds=(1.4, 2.4),
        min_lag_s=1.0,
        max_lag_s=processing_config.max_lag,
        ps_scale_bounds=(0.05, 10.0),
        random_seed=seed,
    )
    return run_layer_cake_inversion(observation, config, forward_predictor=forward), config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", choices=GEOMETRY, default="mt01")
    parser.add_argument("--noise", choices=("nonoise", "snr4", "snr2"), default="nonoise")
    parser.add_argument("--data-root", type=Path, default=ROOT / "example_synthetics")
    parser.add_argument(
        "--method", choices=("multitaper", "waterlevel", "iterative", "bayesian"), default="multitaper"
    )
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Currently requires --method multitaper for exact operator matching",
    )
    parser.add_argument("--layers", type=int, default=3, help="Finite layers; 0 selects RJ sampling")
    parser.add_argument("--output", type=Path, default=Path("results/synthetic_workflow"))
    args = parser.parse_args(argv)
    if args.invert and args.method != "multitaper":
        parser.error("Use --method multitaper for this operator-matched inversion example")
    if args.iterations < 4 or not 0 <= args.layers <= 6:
        parser.error("Require iterations >= 4 and 0 <= layers <= 6")
    processed, config = prepare(args.event, args.noise, args.data_root)
    lag, rf, result = calculate(processed, config, args.method, iterations=args.iterations)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "rf.npz",
        lag_times=lag,
        amplitudes=rf,
        source=processed.parent,
        response=processed.daughter,
        dt=processed.dt,
    )
    import matplotlib.pyplot as plt

    if args.method == "bayesian":
        from bayesian_rf.plotting import plot_deconvolution

        diagnostic = plot_deconvolution(result)
        diagnostic.savefig(args.output / "deconvolution.png", dpi=160)
        plt.close(diagnostic)
    fig, axes = plt.subplots(1, 2 if args.invert else 1, figsize=(11 if args.invert else 7, 5), squeeze=False)
    axes[0, 0].plot(lag, rf, color="black", label=f"{args.method} RF")
    axes[0, 0].set(xlabel="Lag (s)", ylabel="R/Z RF", title=f"{args.event} / {args.noise}")
    metadata = {
        "method": args.method,
        "processing": asdict(config),
        "seed": 42,
        "convergence_verified": False,
    }
    if args.invert:
        fit, inversion_config = invert_multitaper(
            processed, config, iterations=args.iterations, fixed_layers=args.layers or None
        )
        axes[0, 0].plot(fit.observation.lag_times_s, fit.best_prediction, label="Best sampled CPS fit")
        axes[0, 1].fill_betweenx(
            fit.depth_km, fit.vs_lower_km_s, fit.vs_upper_km_s, alpha=0.25, label="Retained 5-95%"
        )
        axes[0, 1].plot(fit.mean_vs_km_s, fit.depth_km, label="Retained mean")
        depth, vs = load_nd_vs_profile(args.data_root / "mars/insight_kks21_gp.nd", max_depth_km=80.0)
        axes[0, 1].plot(vs, depth, "k--", label="KKS21 input")
        axes[0, 1].set(xlabel="Vs (km/s)", ylabel="Depth (km)", title="Not convergence-certified")
        axes[0, 1].invert_yaxis()
        axes[0, 1].legend()
        np.savez_compressed(
            args.output / "inversion.npz",
            depth_km=fit.depth_km,
            vs_profiles=fit.saved_vs_profiles,
            vpvs_profiles=fit.saved_vpvs_profiles,
            layer_counts=fit.saved_layer_counts,
            predictions=fit.saved_predictions,
            log_posterior=fit.saved_log_posteriors,
            best_prediction=fit.best_prediction,
        )
        metadata["inversion"] = asdict(inversion_config)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig(args.output / "summary.png", dpi=160)
    plt.close(fig)
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
