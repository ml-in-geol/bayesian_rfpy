"""Keep the instructional notebook runnable without a long sampling job."""

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples/S0235b_RF_demo.ipynb"


def test_notebook_has_valid_code_cells_and_both_products():
    if not NOTEBOOK.exists():
        pytest.skip("Notebook not available in this checkout")
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["metadata"]["rfpy_demo"]["datasets"] == ["deglitched", "denoised"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
            assert not any(output.get("output_type") == "error" for output in cell.get("outputs", []))


@pytest.mark.insight
def test_notebook_workflow_with_short_chains(tmp_path, monkeypatch):
    pytest.importorskip("IPython")
    if not NOTEBOOK.exists():
        pytest.skip("Notebook not available in this checkout")
    for name in ("deglitched", "denoised"):
        if not (ROOT / "insight_data" / name / "S0235b.mseed").exists():
            pytest.skip("Both S0235b products are needed")
    monkeypatch.setenv("RFPY_PROJECT_ROOT", str(ROOT))
    monkeypatch.setenv("RFPY_DEMO_OUTPUT", str(tmp_path))
    monkeypatch.setenv("RFPY_DEMO_ITERATIONS", "20")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    # This also checks that the notebook can find its inputs away from cwd.
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "__main__"}
    notebook = json.loads(NOTEBOOK.read_text())
    with pytest.warns(RuntimeWarning, match="unresolved prior/proposal"):
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                exec(compile("".join(cell["source"]), cell["id"], "exec"), namespace)

    settings = json.loads((tmp_path / "settings.json").read_text())
    assert settings["config"]["component_separation_factor"] == 1.0
    assert settings["convergence_verified"] is False
    assert settings["bayesian_posterior_validated"] is False
    assert len(list(tmp_path.glob("*.png"))) == 5
    assert set(namespace["processed"]) == {"deglitched", "denoised"}
    offsets = [namespace["processed"][n].p_arrival_offset_s for n in namespace["DATASETS"]]
    assert abs(offsets[0] - offsets[1]) == pytest.approx(38.219, abs=1e-3)
    for name in ("deglitched", "denoised"):
        with np.load(tmp_path / f"S0235b_{name}.npz") as arrays:
            assert arrays["native_bayesian_samples"].shape[0] == 10
            for method in ("bayesian_mean", "bayesian_best", "waterlevel", "multitaper"):
                assert np.isfinite(arrays[f"comparison_{method}"]).all()
                assert arrays[f"comparison_{method}"].shape == arrays["comparison_lag_s"].shape
            assert arrays["bayesian_reconstruction"].shape == arrays["r"].shape
