from pathlib import Path

import pytest

from bayesian_rf.data import catalog_back_azimuth, load_mqs_catalog


CATALOG = Path(__file__).resolve().parents[1] / (
    "insight_data/catalogs/events_extended_multiorigin_v14_2023-01-01.xml"
)


@pytest.mark.skipif(not CATALOG.exists(), reason="Local MQS V14 catalog is not installed")
def test_v14_quality_ab_lf_family_inventory() -> None:
    events = load_mqs_catalog(CATALOG)

    assert len(events) == 36
    assert sum(event.quality == "A" for event in events) == 14
    assert sum(event.quality == "B" for event in events) == 22
    assert {event.family for event in events} == {"LOW_FREQUENCY", "BROADBAND"}


@pytest.mark.skipif(not CATALOG.exists(), reason="Local MQS V14 catalog is not installed")
def test_catalog_uses_direct_and_named_p_like_phases() -> None:
    events = {event.name: event for event in load_mqs_catalog(CATALOG)}

    assert events["S0235b"].parent_phase == "P"
    assert events["S1000a"].parent_phase == "Pdiff"
    assert events["S0976a"].parent_phase == "PP"
    assert events["S1097a"].parent_phase == "x1"
    assert catalog_back_azimuth(events["S0235b"]) == pytest.approx(77.073, abs=0.01)
