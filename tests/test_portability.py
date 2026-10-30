from pathlib import Path
import shutil

import numpy as np
import pytest

from bayesian_rf.cps import default_cps_root, hrftn96_path
from bayesian_rf.preprocessing import load_taup_model


def test_cps_sources_are_packaged_and_cache_is_platform_keyed(tmp_path, monkeypatch):
    monkeypatch.delenv("BAYESIAN_RF_HRFTN96", raising=False)
    monkeypatch.setenv("BAYESIAN_RF_CACHE", str(tmp_path / "cache with spaces"))
    source = default_cps_root()
    assert len(list(source.glob("*.f"))) == 7
    assert (source / "LICENSE").is_file()
    target = hrftn96_path()
    assert target.is_relative_to(tmp_path)
    assert not target.is_relative_to(source)


@pytest.mark.integration
def test_taup_build_does_not_write_next_to_input(tmp_path, monkeypatch):
    monkeypatch.setenv("BAYESIAN_RF_CACHE", str(tmp_path / "cache"))
    source = tmp_path / "input.nd"
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "example_synthetics/mars/insight_kks21_gp.nd", source
    )
    before = source.read_bytes()
    model = load_taup_model(source)
    assert model.get_travel_times(source_depth_in_km=18.4, distance_in_degree=28.3, phase_list=["P"])
    assert source.read_bytes() == before
    assert not source.with_suffix(".npz").exists()
    assert len(list((tmp_path / "cache").rglob("*.npz"))) == 1
    load_taup_model(source)
    assert len(list((tmp_path / "cache").rglob("*.npz"))) == 1


def test_standard_plot_preserves_reconvolution(tmp_path):
    from bayesian_rf.benchmarks import benchmark_config, make_benchmark, run_benchmark
    from bayesian_rf.plotting import plot_deconvolution
    import matplotlib.pyplot as plt

    case = make_benchmark("impulse_gaussian")
    with pytest.warns(RuntimeWarning, match="unresolved"):
        result = run_benchmark(case, benchmark_config(case, iterations=40, burnin=20, save_every=5))
    fig = plot_deconvolution(result, source_label="Source", response_label="Response")
    output = tmp_path / "figure.png"
    fig.savefig(output)
    assert output.stat().st_size > 1000
    assert np.all(np.isfinite(fig.axes[2].lines[1].get_ydata()))
    plt.close(fig)
