"""Run synthetic, non-seismological benchmarks of the array deconvolver."""

import argparse
from dataclasses import asdict
import json
import os
import tempfile
from pathlib import Path
from time import perf_counter

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rfpy-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bayesian_rf.benchmarks import (
    CASE_DESCRIPTIONS,
    benchmark_config,
    evaluate_benchmark,
    make_benchmark,
    run_benchmark,
)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(CASE_DESCRIPTIONS), default=list(CASE_DESCRIPTIONS)
    )
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--burnin", type=int)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42], help="Sampler seeds; data stay fixed across chains."
    )
    parser.add_argument("--data-seed", type=int, default=1234)
    parser.add_argument(
        "--snrs", type=float, nargs="+", default=[20.0], help="RMS signal/noise ratios; inf disables noise."
    )
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("results/deconvolution_benchmarks"))
    return parser


def _save_figure(path, case, result, arrays, metrics):
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    fig.suptitle(f"{case.name} | {case.description}", fontsize=12)
    source_axis, data_axis, kernel_axis, fit_axis, residual_axis, spectrum_axis = axes.flat
    source_axis.plot(case.times, case.input_signal, color="#226477")
    source_axis.set(title="Known input x (noise-free)", xlabel="Time", ylabel="Amplitude")
    data_axis.plot(case.times, case.observed_output, color="0.55", lw=1, label="Observed output y")
    data_axis.plot(case.times, case.clean_output, color="black", lw=1, label="True x * h * dt")
    data_axis.set(title=f"Output | RMS SNR = {case.requested_snr:g}", xlabel="Time", ylabel="Amplitude")
    data_axis.legend(fontsize=8)

    samples = result.saved_receiver_functions
    low = min(float(samples.min()), float(case.true_kernel.min()))
    high = max(float(samples.max()), float(case.true_kernel.max()))
    margin = max(0.05, 0.05 * (high - low))
    edges = np.linspace(low - margin, high + margin, 161)
    density = np.array([np.histogram(column, bins=edges)[0] for column in samples.T]).T
    image = kernel_axis.imshow(
        np.log1p(density),
        origin="lower",
        aspect="auto",
        cmap="Greys",
        extent=(result.lag_times[0], result.lag_times[-1], edges[0], edges[-1]),
    )
    fig.colorbar(image, ax=kernel_axis, label="log(1 + posterior count)", pad=0.01)
    kernel_axis.plot(case.lag_times, case.true_kernel, color="black", ls="--", label="True kernel", lw=1.5)
    kernel_axis.plot(
        result.lag_times, result.mean_receiver_function, color="#009988", label="Posterior mean", lw=1.5
    )
    kernel_axis.plot(
        result.lag_times, result.map_receiver_function, color="#cc5544", label="Best sampled / MAP", lw=1
    )
    kernel_axis.plot(
        case.lag_times, arrays["ridge_kernel"], color="#4477aa", label="GCV ridge", lw=0.9, alpha=0.8
    )
    kernel_axis.set(title="Recovered kernel h", xlabel="Lag", ylabel="Kernel amplitude")
    kernel_axis.legend(fontsize=7)

    fit_axis.plot(case.times, case.observed_output, color="0.7", label="Observed", lw=1)
    fit_axis.plot(case.times, case.clean_output, color="black", label="Truth", lw=1.2)
    fit_axis.plot(case.times, arrays["mean_prediction"], color="#009988", label="Mean reconvolution", lw=1.2)
    fit_axis.plot(case.times, arrays["map_prediction"], color="#cc5544", label="MAP reconvolution", lw=1)
    fit_axis.set(
        title="Recovery test (no amplitude or timing adjustment)", xlabel="Time", ylabel="Output amplitude"
    )
    fit_axis.legend(fontsize=8)
    residual_axis.plot(
        case.times,
        arrays["mean_prediction"] - case.observed_output,
        color="#009988",
        label="Mean minus observed",
    )
    if case.noise_sigma > 0:
        residual_axis.axhspan(
            -2 * case.noise_sigma,
            2 * case.noise_sigma,
            color="0.8",
            alpha=0.4,
            label="+/- 2 true noise sigma",
        )
    residual_axis.axhline(0, color="0.5", lw=0.5)
    residual_axis.set(title="Residual, including model error", xlabel="Time", ylabel="Amplitude")
    residual_axis.legend(fontsize=8)
    singular = metrics["ridge_reference"]["relative_singular_values"]
    spectrum_axis.semilogy(np.arange(1, len(singular) + 1), singular, color="#226477")
    spectrum_axis.set(
        title="Convolution operator singular values", xlabel="Index (descending)", ylabel="s / largest s"
    )
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_summary(path, records):
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.55 * len(records))), constrained_layout=True)
    y = np.arange(len(records))
    labels = [f"{r['case']} | SNR {r['snr']} | seed {r['sampler_seed']}" for r in records]
    for ax, metric, title in zip(
        axes,
        ("kernel_relative_l2", "clean_output_relative_l2"),
        ("Kernel error", "Reconvolution error vs clean truth"),
        strict=True,
    ):
        for offset, name, color in (
            (-0.22, "map", "#cc5544"),
            (0.0, "mean", "#009988"),
            (0.22, "ridge", "#4477aa"),
        ):
            ax.barh(
                y + offset,
                [r["metrics"][name][metric] for r in records],
                height=0.22,
                color=color,
                label=name,
            )
        ax.set(
            yticks=y,
            yticklabels=labels if ax is axes[0] else [],
            title=title,
            xlabel="Relative L2 error (lower is better)",
        )
        ax.invert_yaxis()
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = _build_parser()
    args = parser.parse_args()
    # Validate every requested setting before any potentially long chain starts.
    cases = []
    for name in dict.fromkeys(args.cases):
        for snr in dict.fromkeys(args.snrs):
            try:
                case = make_benchmark(name, dt=args.dt, snr=snr, seed=args.data_seed)
                benchmark_config(
                    case, iterations=args.iterations, burnin=args.burnin, save_every=args.save_every
                )
            except ValueError as exc:
                parser.error(str(exc))
            cases.append(case)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for case in cases:
        for seed in dict.fromkeys(args.seeds):
            snr_tag = "inf" if np.isinf(case.requested_snr) else repr(case.requested_snr)
            stem = args.output_dir / f"{case.name}_snr{snr_tag.replace('.', 'p')}_seed{seed}"
            config = benchmark_config(
                case, iterations=args.iterations, burnin=args.burnin, save_every=args.save_every, seed=seed
            )
            print(f"Running {stem.name}: {config.iterations:,} iterations", flush=True)
            start = perf_counter()
            result = run_benchmark(case, config)
            elapsed = perf_counter() - start
            metrics, arrays = evaluate_benchmark(case, result)
            record = {
                "case": case.name,
                "description": case.description,
                "snr": "inf" if np.isinf(case.requested_snr) else case.requested_snr,
                "data_seed": case.data_seed,
                "sampler_seed": seed,
                "noise_sigma": case.noise_sigma,
                "elapsed_seconds": elapsed,
                "config": asdict(config),
                "metrics": metrics,
                "map_components": [asdict(c) for c in result.map_components],
                "figure": stem.with_suffix(".png").name,
            }
            np.savez_compressed(
                stem.with_suffix(".npz"),
                times=case.times,
                input_signal=case.input_signal,
                lag_times=case.lag_times,
                true_kernel=case.true_kernel,
                clean_output=case.clean_output,
                observed_output=case.observed_output,
                noise_window=case.noise_window,
                rf_lag_times=result.lag_times,
                mean_kernel=result.mean_receiver_function,
                map_kernel=result.map_receiver_function,
                saved_kernels=result.saved_receiver_functions,
                saved_sigma=result.saved_sigma,
                saved_counts=result.saved_counts,
                **arrays,
            )
            stem.with_suffix(".json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            _save_figure(stem.with_suffix(".png"), case, result, arrays, metrics)
            records.append(record)
            (args.output_dir / "summary.json").write_text(
                json.dumps(records, indent=2, allow_nan=False) + "\n"
            )
            print(
                f"  mean kernel error={metrics['mean']['kernel_relative_l2']:.3f}; clean-output error={metrics['mean']['clean_output_relative_l2']:.3f}; {elapsed:.1f}s",
                flush=True,
            )
    _save_summary(args.output_dir / "summary.png", records)


if __name__ == "__main__":
    main()
