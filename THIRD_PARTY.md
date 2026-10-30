# Licensing and provenance

## Project code

No license has been chosen on the project owner's behalf. Add an approved
root LICENSE before public distribution. Package metadata intentionally makes
no unapproved open-source license claim.

## CPS

Seven unmodified files from Robert B. Herrmann's CPS 3.30 are bundled under
his MIT license. `src/bayesian_rf/_cps/LICENSE` and `PROVENANCE.md` record the
April 25, 2024 archive hash and exact source paths. Upstream and license:
https://rbherrmann.github.io/ComputerProgramsSeismology/mitlicense.html.

## Research inputs

Dahmen waveforms: https://doi.org/10.5281/zenodo.11995107. MQS catalog:
https://doi.org/10.12686/a21. See `insight_data/README.md` for selection, units
and unresolved deglitched provenance. Raw data are not embedded in the wheel.

`example_synthetics` is preserved from user research inputs: Instaseis Mars
records, KKS21 ND model and Nakagawa SAC synthetics. Confirm redistribution
rights and citations before publishing them. Original generation scripts are
provenance, not installed commands; they need external databases.

`data/observed_rf_stacks` preserves earlier input stacks provided by DK /
Jung-Hun Song, including the Sp time/polarity convention note. These are not
new results; confirm permission before public sharing.

The journal PDF formerly at the repository root is not bundled with the
package. Cite https://doi.org/10.1093/gji/ggu079 instead.
