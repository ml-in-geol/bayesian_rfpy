# Bayesian RFpy

Receiver-function deconvolution and CPS-backed one-dimensional structural
inversion. Python 3.10-3.12, with no machine-specific paths or precompiled binaries.

**Research status:** the preserved Bayesian deconvolution kernel has known
prior/proposal inconsistencies. Its sample spread is not calibrated posterior
uncertainty. See [limitations](docs/limitations.md). The classical methods and
separate structural sampler are not that same kernel.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,plot]'
python -m pytest
```

Deconvolution needs no compiler. Structural inversion needs `gfortran`
(Homebrew `gcc` on macOS; `gfortran` on Debian/Ubuntu) or an existing CPS binary:

```bash
rfpy build-cps
# Or: export BAYESIAN_RF_HRFTN96=/path/to/hrftn96
```

Only seven required CPS Fortran files are bundled with upstream notices.
CPS builds and TauP caches go to `~/.cache/bayesian_rf`, not the installed
package or input directories. Override with `BAYESIAN_RF_CACHE`.
Linux/macOS are intended platforms; native Windows builds are not tested.

## API And Layout

```python
from bayesian_rf.deconvolution import (
    BayesianRFConfig, run_bayesian_deconvolution,
    waterlevel_rf, multitaper_rf, iterative_rf,
)
from bayesian_rf.inversion import (
    RFObservation, LayerCakeInversionConfig,
    run_layer_cake_inversion, run_joint_layer_cake_inversion,
)

# Physical, equally sampled waveform arrays:
estimate = multitaper_rf(z, r, dt, source_noise=z_noise)
lag, radial_rf, vertical_self_rf = estimate.waveform(sigma_s=0.5)
```

`iterative_rf` is Ligorria & Ammon-style signed pulse stripping. All methods
live in `deconvolution.py`, including standard/overlapping Bayesian pulses,
water level, full-window multitaper, and optional Wiener/source-gated inverses.
Structural inversion supports fixed or RJ layer counts, layer-specific Vs and
Vp/Vs, a halfspace, and joint Ps/Sp. See [methods](docs/methods.md).

```text
src/bayesian_rf/
    deconvolution.py   # all deconvolution methods and resolution operators
    inversion.py      # fixed/RJ, joint Ps/Sp, adaptive MCMC, matched operators
    preprocessing.py  # waveform IO, rotation, arrival-centred windows
    cps.py            # cached Fortran forward-model wrapper
    data.py           # MQS catalog and InSight import/load/preprocessing
    diagnostics.py    # multi-chain R-hat and ESS
    plotting.py       # optional four-row waveform/RF/reconstruction figure
    benchmarks.py     # analytic non-seismological cases
    cli.py            # rfpy build-cps / import-insight
examples/             # three portable workflows
example_synthetics/   # original Mars and Nakagawa inputs, preserved
insight_data/         # Quality A/B LF-family inventory and local waveforms
data/observed_rf_stacks/ # previously supplied observational inputs
tests/                # unit, integration and longer recovery tests
```

## Run Examples

For a step-by-step, executed comparison of both S0235b products and three
deconvolution methods, open [the Jupyter demonstration](examples/S0235b_RF_demo.ipynb):

```bash
python -m pip install -e '.[notebook]'
python -m jupyter lab examples/S0235b_RF_demo.ipynb
```

The notebook defaults to 100,000 Bayesian iterations per product, displays
ZNE/RTZ waveforms and matched-normalization RFs, and checks the native Bayesian
reconvolution. It is an exploratory demonstration, not a validated posterior.
Saved outputs are under `results/s0235b_notebook/`.

```bash
python examples/synthetic_workflow.py --event mt03 --noise snr2 --method iterative
python examples/synthetic_workflow.py --event mt01 --invert --layers 3 --iterations 20000
python examples/synthetic_workflow.py --event mt01 --invert --layers 0 --iterations 20000
python examples/insight_workflow.py --event S0235b
python examples/benchmarks.py --cases ricker_sparse boxcar_sparse --iterations 100000
```

Examples accept output directories; figures/settings are regenerated, not
committed. The synthetic inversion example matches the full multitaper operator
and overlays KKS21. Short runs demonstrate the pipeline, not convergence.
Other methods can calculate RFs with the same script, but structural comparison
requires an appropriately matched forward operator.

## Data And Tests

The InSight subset has 36 MQS V14 **Quality A/B LF-family** events (LF plus
broadband): 36 denoised and 34 deglitched. S0899d and S1097a lack deglitched
files. The manifest records availability, checksums, units and separate trace
start times. See [import and provenance](insight_data/README.md).

```bash
python -m pytest -m 'not insight and not slow'
RUN_SLOW_DECONVOLUTION=1 python -m pytest tests/test_deconvolution_benchmarks.py
```

Tests cover three classical methods across MT01-MT03/no-noise/SNR4/SNR2,
Bayesian smoke/recovery, synthetic-to-structure fixed/RJ pipelines, a known
CPS-model fit-improvement control, Ps/Sp signs, units and timestamp handling.
Short chains are regression tests, not KKS21 recovery or posterior-calibration
proof. Local-data tests skip absent inputs; CPS tests skip without a compiler
or explicit binary. GitHub Actions installs gfortran and runs synthetic tests.

Wheels/sdists exclude large research inputs. Local InSight waveforms are
git-ignored; the manifest remains portable. Before publishing, choose a project
license and confirm contributed-data permissions. See [third-party notes](THIRD_PARTY.md).
