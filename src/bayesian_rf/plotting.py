"""Optional matplotlib figures; imports remain lightweight without plotting."""

import numpy as np


def plot_deconvolution(result, *, source_label="Z component", response_label="R component", bins=120):
    """Return the standard four-row RF figure without writing files.

    For S-to-P or non-seismic problems, pass appropriate component labels.
    The legacy kernel's sample histogram is exploratory, not a calibrated
    posterior. The fourth row counts Gaussian pulses, NOT physical layers.
    """
    import matplotlib.pyplot as plt
    from .deconvolution import _predict_daughter, _receiver_function_from_components

    dt = float(result.metadata["dt"])
    times = np.arange(result.parent.size) * dt
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), constrained_layout=True)
    axes[0].plot(times, result.parent, label=source_label)
    axes[0].plot(times, result.daughter, label=response_label, alpha=0.8)
    axes[0].set(xlabel="Time from analysis-window start (s)", ylabel="Amplitude")
    samples = result.saved_receiver_functions
    lower, upper = float(samples.min()), float(samples.max())
    margin = max(0.05 * (upper - lower), np.finfo(float).eps)
    edges = np.linspace(lower - margin, upper + margin, bins + 1)
    density = np.array([np.histogram(column, edges)[0] for column in samples.T]).T
    mesh = axes[1].imshow(
        np.log1p(density),
        origin="lower",
        aspect="auto",
        cmap="Greys",
        extent=(result.lag_times[0], result.lag_times[-1], edges[0], edges[-1]),
    )
    fig.colorbar(mesh, ax=axes[1], label="log(1 + retained count)")
    axes[1].plot(result.lag_times, result.mean_receiver_function, label="Retained mean")
    axes[1].plot(result.lag_times, result.map_receiver_function, label="Best likelihood (legacy MAP)")
    axes[1].set(xlabel="RF lag (s)", ylabel="RF density", title="Exploratory sample distribution")
    grid = np.arange(result.lag_times[0], result.lag_times[-1] + 0.5 * dt, dt)
    kernel = _receiver_function_from_components(grid, result.map_components)
    reconstructed = _predict_daughter(result.parent, kernel, dt)
    axes[2].plot(times, result.daughter, color="black", label=f"Observed {response_label}")
    axes[2].plot(times, reconstructed, label="Source convolved with best RF", alpha=0.8)
    axes[2].set(xlabel="Time from analysis-window start (s)", ylabel="Amplitude")
    count_edges = np.arange(result.saved_counts.min() - 0.5, result.saved_counts.max() + 1.5)
    axes[3].hist(result.saved_counts, bins=count_edges, color="0.4")
    axes[3].set(xlabel="Number of Gaussian pulses (not structural layers)", ylabel="Retained count")
    for axis in axes[:3]:
        axis.legend(fontsize=8)
    return fig
