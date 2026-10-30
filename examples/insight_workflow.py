"""Compare an event's two processing products using absolute catalog picks."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from bayesian_rf import BayesianRFConfig, multitaper_rf
from bayesian_rf.data import preprocess_insight_event


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="S0235b")
    parser.add_argument(
        "--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "insight_data"
    )
    parser.add_argument("--back-azimuth", type=float)
    parser.add_argument("--p-arrival", help="Explicit UTC direct-P pick, if overriding the catalog")
    parser.add_argument("--output", type=Path, default=Path("results/insight_workflow"))
    args = parser.parse_args(argv)
    config = BayesianRFConfig(
        min_period=2.0, max_period=8.0, freqmin=1 / 8, freqmax=1 / 2, target_sampling_rate=4.0
    )
    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 4))
    settings = {"event": args.event, "processing": asdict(config), "method": "multitaper", "products": {}}
    for dataset in ("deglitched", "denoised"):
        try:
            data = preprocess_insight_event(
                args.data_root,
                args.event,
                dataset,
                config,
                back_azimuth=args.back_azimuth,
                p_arrival=args.p_arrival,
            )
        except FileNotFoundError as exc:
            print(exc)
            continue
        estimator = multitaper_rf(
            data.parent,
            data.daughter,
            data.dt,
            source_noise=data.parent[: round(config.pre_arrival / data.dt)],
        )
        lag, radial, vertical = estimator.waveform(sigma_s=0.25, min_lag=-5.0, max_lag=20.0)
        axis.plot(lag, radial, label=dataset)
        np.savetxt(
            args.output / f"{args.event}_{dataset}.txt",
            np.column_stack((lag, radial)),
            header="lag_s radial_rf; uncertainty not estimated",
        )
        np.savez_compressed(
            args.output / f"{args.event}_{dataset}.npz",
            lag_times=lag,
            radial_rf=radial,
            vertical_self_rf=vertical,
            source=data.parent,
            response=data.daughter,
            dt=data.dt,
        )
        settings["products"][dataset] = {
            "p_arrival_offset_s": data.p_arrival_offset_s,
            "back_azimuth_deg": data.back_azimuth,
            "waveform_units": "m/s",
            "gaussian_sigma_s": 0.25,
        }
    axis.set(xlabel="Lag from P (s)", ylabel="Radial RF", title=f"{args.event}: 2-8 s period band")
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output / f"{args.event}.png", dpi=160)
    plt.close(fig)
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")


if __name__ == "__main__":
    main()
