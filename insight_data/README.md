# Quality A/B LF-family InSight data

Only MQS V14 preferred-origin Quality A/B events of type LOW_FREQUENCY or
BROADBAND enter this inventory: 36 events, comprising 14 A and 22 B. There are
36 denoised and 34 deglitched products. S0899d and S1097a deglitched data were
absent from the supplied collection and are explicitly marked missing.

```text
manifest.json          # relative paths, metadata, checksums and availability
catalogs/              # supplied full MQS V14 XML, local/git-ignored
denoised/S0235b.mseed   # original extensionless archive file, renamed only
deglitched/S0235b.mseed # original *_DISP_ZNE.mseed, renamed only
legacy_sac/S0235b/     # prior RTZ SAC export for regression tests only
```

Original waveforms are copied byte-for-byte and never modified at their source.
Raw native-UVW counts, manual alternatives and low-quality events are not
substituted. Legacy SAC may differ from the denoised archive and is not used
by the standard InSight workflow.

## Import on another computer

```bash
rfpy import-insight \
  --catalog /path/to/events_extended_multiorigin_v14_2023-01-01.xml \
  --deglitched /path/to/deglitched_marsquakes \
  --denoised /path/to/11995107/waveforms \
  --output insight_data
```

This is an offline importer, not a downloader. It records missing products and
refuses to overwrite different waveform bytes. Use a fresh destination when
changing selections to avoid stale unlisted files from a previous import.

## Units and provenance

- Denoised: Dahmen et al. (2024), 20 sps physical velocity ZNE, XB.ELYDL.03;
  https://doi.org/10.5281/zenodo.11995107.
- Deglitched: user-supplied physical displacement ZNE miniSEED. Its full
  processing provenance and redistribution terms remain to be documented.
- Catalog: supplied MQS V14; https://doi.org/10.12686/a21. Full XML is metadata;
  only A/B LF-family waveforms are selected.

`load_insight_waveforms(..., velocity=True)` differentiates deglitched
displacement once and leaves denoised velocity unchanged. Set `velocity=False`
to inspect native units. No instrument response is removed a second time.
Every trace's separate original start/end time is recorded.

`preprocess_insight_event` uses absolute picks, checks full window coverage,
then rotates and filters. Missing directions or non-direct-P picks require
explicit overrides; no automatic Cerberus Fossae assumption is made. The
example uses a 2-8 s band. Neither dataset is independently polarity-flipped.

Raw waveforms and XML are git-ignored and excluded from Python distributions.
The manifest has no personal absolute paths. Verify attribution and sharing
conditions before distributing the data; a code license is not a data license.
