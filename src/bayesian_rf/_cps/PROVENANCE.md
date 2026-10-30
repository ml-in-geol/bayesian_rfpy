# Computer Programs in Seismology source provenance

This directory vendors the narrow CPS 3.30 source subset required by the
optional `bayesian_rf.cps` forward-model backend.

- Upstream: Robert B. Herrmann, Computer Programs in Seismology.
- Pinned archive: `NP330.Apr-25-2024.tgz`.
- Download URL: `https://www.eas.slu.edu/eqc/eqc_cps/Download/NP330.Apr-25-2024.tgz`.
- SHA-256: `08354f06e35ee0477f2a0123f934f6123ce2aa70e25d5c1405420a3021f28c48`.
- License: MIT, as stated by the upstream CPS project site.

The unmodified files are `VOLVI/src/hrftn96.f` and `SUBS/{sacsubf,igetmod,
mchdep,lgstr,mnmarg,mgtarg}.f`. They support isotropic P and S incidence.
No binary, object, graphics library, full CPS tree, or third-party readline
source is bundled. Compilation writes to a platform/content-keyed user cache.

License notice verified against the author's primary site:
https://rbherrmann.github.io/ComputerProgramsSeismology/mitlicense.html
