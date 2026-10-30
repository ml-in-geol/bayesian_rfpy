# Scientific limitations and release status

## Preserved legacy Bayesian deconvolution

The cleanup preserves the standard Gaussian-pulse model and transition rules.
An earlier in-project audit found issues not resolved by longer chains:

- Amplitude random walks do not enforce the stated uniform prior bounds.
- Birth positions are on a discrete lag grid while location moves are
  continuous; the reverse proposal measure is not consistently accounted for.
- Non-overlap-conditioned births and compound move selection add proposal
  factors absent from the dimension-only acceptance factor.

Saved samples are therefore not validated posterior draws. The function now
emits a warning. `map_receiver_function` is the legacy name for the best
likelihood found, including warm-up, NOT a rigorously computed posterior MAP.
Means and percentile displays remain exploratory. The prior four-chain audit
did not pass convergence gates. Historical outputs and one-off analysis scripts
have been removed, not silently promoted into package validation results.

A corrected, explicitly derived transition kernel is a separate research task.
The structural RJ sampler is different, but also needs multi-chain and
simulation-based validation before calibrated uncertainty claims.

## Other limitations

Classical estimates can be biased by source noise, bandwidth, tapering and
regularization. Iterative stopping means only no sufficiently improving single
coordinate. Full-window DPSS can attenuate late conversions. Match the full
forward operator, not just an extra Gaussian, when inverting those RFs.

Profiled amplitude scales, independent-error likelihoods, finite windows,
informative priors, density assumptions and the flat isotropic layer-cake model
can all produce model-dependent structure. Attractive fits do not demonstrate
uniqueness or independently prove sensitivity to each interface.

## Tests and diagnostics

Unit tests protect signs, units, normalization, finite outputs, reproducible
seeds, and preprocessing. Integration tests calculate RFs from supplied Mars
waveforms and exercise fixed/RJ structure inversion. A known-CPS-target control
requires improvement from a displaced initial model. Short tests do not certify
KKS21 recovery, global convergence or calibrated coverage. Longer non-seismic
recovery regressions are opt-in.

`diagnose_chains` returns rank-normalized split/folded R-hat, bulk/tail ESS and
a gate requiring four chains. Passing cannot prove posterior correctness.
The implementation has not been independently cross-validated against ArviZ.

## Public release

Choose an owner-approved project license and verify contributed-data rights
before publication. No public GitHub repository or release was created by this
cleanup. The bundled CPS subset retains its separate upstream MIT license.
