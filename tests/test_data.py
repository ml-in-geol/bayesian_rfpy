from pathlib import Path
import json

import numpy as np
from obspy import Stream, Trace, UTCDateTime
import pytest

from bayesian_rf.data import (
    import_insight_data,
    load_insight_waveforms,
    load_mqs_catalog,
    preprocess_insight_event,
)
from bayesian_rf import BayesianRFConfig


def make_catalog(path):
    events = []
    for index, quality, family in [
        (1, "A", "LOW_FREQUENCY"),
        (2, "B", "BROADBAND"),
        (3, "C", "LOW_FREQUENCY"),
        (4, "A", "HIGH_FREQUENCY"),
    ]:
        events.append(
            f"""<event publicID="event{index}">
        <description><text>S{index:04d}a</text><type>earthquake name</type></description>
        <mars:type>type#{family}</mars:type><preferredOriginID>origin{index}</preferredOriginID>
        <origin publicID="origin{index}"><time><value>2020-01-01T00:00:00Z</value></time>
        <latitude><value>10</value></latitude><longitude><value>164</value></longitude>
        <mars:locationQuality>quality#{quality}</mars:locationQuality>
        <mars:distance><mars:value>28</mars:value></mars:distance>
        <arrival><pickID>pick{index}</pickID><phase>P</phase></arrival></origin>
        <pick publicID="pick{index}"><time><value>2020-01-01T00:00:40Z</value></time></pick>
        </event>"""
        )
    path.write_text(
        """<q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2"
    xmlns="http://quakeml.org/xmlns/bed/1.2" xmlns:mars="http://quakeml.org/xmlns/bed/1.2/mars">
    <eventParameters>"""
        + "".join(events)
        + "</eventParameters></q:quakeml>"
    )


def make_stream(path, start=0.0):
    t = np.arange(2400) / 20
    traces = [
        Trace(
            (np.sin(2 * np.pi * 0.25 * t) * np.exp(-0.5 * ((t - 40 + start) / 5) ** 2)).astype("float64"),
            header=dict(channel=f"BH{c}", sampling_rate=20.0, starttime=UTCDateTime("2020-01-01") + start),
        )
        for c in "ZNE"
    ]
    Stream(traces).write(str(path), format="MSEED")


@pytest.fixture
def imported(tmp_path):
    catalog = tmp_path / "catalog.xml"
    make_catalog(catalog)
    dg, dn, out = (tmp_path / name for name in ("dg", "dn", "out"))
    dg.mkdir()
    dn.mkdir()
    make_stream(dg / "deglitched_S0001a_DISP_ZNE.mseed", start=2.0)
    make_stream(dn / "S0001a")
    make_stream(dn / "S0002a")
    make_stream(dn / "S0003a")
    manifest = import_insight_data(catalog, dg, dn, out)
    return out, manifest, catalog, dg, dn


def test_import_selects_only_ab_lf_and_preserves_bytes(imported):
    out, manifest, catalog, dg, dn = imported
    assert [event["name"] for event in manifest["events"]] == ["S0001a", "S0002a"]
    assert len(load_mqs_catalog(catalog, qualities=None)) == 3
    assert (out / "denoised/S0001a.mseed").read_bytes() == (dn / "S0001a").read_bytes()
    assert manifest["events"][1]["products"]["deglitched"]["status"] == "missing"
    assert not (out / "denoised/S0003a.mseed").exists()
    assert import_insight_data(catalog, dg, dn, out) == manifest
    with pytest.raises(FileNotFoundError):
        load_insight_waveforms(out, "S0002a", "deglitched")
    (out / "denoised/S0001a.mseed").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Checksum"):
        load_insight_waveforms(out, "S0001a", "denoised")
    with pytest.raises(FileExistsError):
        import_insight_data(catalog, dg, dn, out)


def test_preprocessing_uses_each_products_absolute_start(imported):
    out = imported[0]
    config = BayesianRFConfig(
        pre_arrival=10.0, post_arrival=15.0, freqmin=1 / 8, freqmax=1 / 2, target_sampling_rate=4.0
    )
    deglitched = preprocess_insight_event(out, "S0001a", "deglitched", config)
    denoised = preprocess_insight_event(out, "S0001a", "denoised", config)
    assert deglitched.p_arrival_offset_s == 38.0
    assert denoised.p_arrival_offset_s == 40.0
    assert deglitched.parent.shape == denoised.parent.shape
    long_window = BayesianRFConfig(pre_arrival=100.0)
    with pytest.raises(ValueError, match="full requested"):
        preprocess_insight_event(out, "S0001a", "denoised", long_window)


def test_no_implicit_pick_or_back_azimuth_assumptions(imported):
    out, manifest = imported[:2]
    manifest["events"][0]["back_azimuth_deg"] = None
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="back azimuth"):
        preprocess_insight_event(out, "S0001a", "denoised", BayesianRFConfig())


def test_repository_manifest_contains_only_ab_lf():
    path = Path(__file__).resolve().parents[1] / "insight_data/manifest.json"
    if not path.exists():
        pytest.skip("Repository manifest not included in installed sdist")
    manifest = json.loads(path.read_text())
    assert len(manifest["events"]) == 36
    assert {e["quality"] for e in manifest["events"]} == {"A", "B"}
    assert {e["family"] for e in manifest["events"]} == {"LOW_FREQUENCY", "BROADBAND"}
    names = {e["name"] for e in manifest["events"]}
    for kind in ("denoised", "deglitched"):
        assert {p.stem for p in (path.parent / kind).glob("*.mseed")} <= names
