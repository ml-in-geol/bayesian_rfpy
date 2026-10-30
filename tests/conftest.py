import importlib.util
import os
from pathlib import Path
import shutil
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rfpy-matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture
def example_module():
    def load(name):
        spec = importlib.util.spec_from_file_location(
            f"rfpy_example_{name}", ROOT / "examples" / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


def pytest_collection_modifyitems(items):
    cps_files = {"test_cps_forward.py", "test_layer_cake_inversion.py", "test_stack_inversion.py"}
    real_functions = {
        "test_zne_preprocessing_uses_distinct_s0235b_trace_start",
        "test_zne_preprocessing_aligns_subsample_component_offsets",
        "test_sac_rtz_preprocessing_uses_explicit_catalog_pick",
        "test_free_surface_preprocessing_matches_paper_s0235b_geometry",
    }
    for item in items:
        if item.path.name in cps_files or (
            item.path.name == "test_multitaper_inversion.py"
            and item.name
            in {
                "test_cps_pulse_replacement_and_homogeneous_model",
                "test_identical_slownesses_and_critical_halfspace",
            }
        ):
            item.add_marker(pytest.mark.cps)
        if item.name in real_functions:
            item.add_marker(pytest.mark.insight)
        if item.name in {
            "test_sac_preprocessing_uses_geometry_and_returns_rtz_window",
            "test_mars_sp_preprocessing_reverses_direct_s_window",
        }:
            item.add_marker(pytest.mark.integration)
        if item.get_closest_marker("cps") and not (
            shutil.which(os.environ.get("FC", "gfortran")) or os.environ.get("BAYESIAN_RF_HRFTN96")
        ):
            item.add_marker(pytest.mark.skip(reason="CPS requires gfortran or BAYESIAN_RF_HRFTN96"))
        if (
            item.get_closest_marker("insight")
            and not (ROOT / "insight_data/deglitched/S0235b.mseed").exists()
        ):
            item.add_marker(pytest.mark.skip(reason="Optional local InSight waveforms not imported"))
        if item.get_closest_marker("integration") and not (ROOT / "example_synthetics/mars").is_dir():
            item.add_marker(pytest.mark.skip(reason="Synthetic waveforms are checkout-only inputs"))
