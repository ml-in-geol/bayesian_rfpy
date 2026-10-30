from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib
import json
import shutil

from obspy import UTCDateTime, read
from obspy.geodetics.base import gps2dist_azimuth


_NAMESPACES = {
    "bed": "http://quakeml.org/xmlns/bed/1.2",
    "mars": "http://quakeml.org/xmlns/bed/1.2/mars",
}


@dataclass(frozen=True)
class MarsquakeCatalogEvent:
    name: str
    family: str
    quality: str
    origin_time: UTCDateTime
    distance_deg: float | None
    latitude_deg: float | None
    longitude_deg: float | None
    parent_phase: str | None
    parent_arrival: UTCDateTime | None


def _suffix(value: str | None) -> str:
    return "" if value is None else value.rsplit("#", 1)[-1]


def load_mqs_catalog(
    path: str | Path,
    *,
    families: tuple[str, ...] = ("LOW_FREQUENCY", "BROADBAND"),
    qualities: tuple[str, ...] | None = ("A", "B"),
    parent_phase_priority: tuple[str, ...] = ("P", "Pdiff", "PP", "x1"),
) -> list[MarsquakeCatalogEvent]:
    """Read preferred-origin metadata from the Mars-specific MQS QuakeML."""
    root = ET.parse(path).getroot()
    parameters = root.find("bed:eventParameters", _NAMESPACES)
    if parameters is None:
        raise ValueError("MQS eventParameters element was not found.")

    records: list[MarsquakeCatalogEvent] = []
    for event in parameters.findall("bed:event", _NAMESPACES):
        name = event.findtext(
            "bed:description[bed:type='earthquake name']/bed:text",
            namespaces=_NAMESPACES,
        )
        family = _suffix(event.findtext("mars:type", namespaces=_NAMESPACES))
        preferred_origin_id = event.findtext("bed:preferredOriginID", namespaces=_NAMESPACES)
        origin = next(
            (
                candidate
                for candidate in event.findall("bed:origin", _NAMESPACES)
                if candidate.attrib.get("publicID") == preferred_origin_id
            ),
            None,
        )
        if name is None or origin is None or family not in families:
            continue

        quality = _suffix(origin.findtext("mars:locationQuality", namespaces=_NAMESPACES))
        if qualities is not None and quality not in qualities:
            continue

        arrivals = {
            arrival.findtext("bed:pickID", namespaces=_NAMESPACES): arrival.findtext(
                "bed:phase", namespaces=_NAMESPACES
            )
            for arrival in origin.findall("bed:arrival", _NAMESPACES)
        }
        phase_picks: dict[str, UTCDateTime] = {}
        for pick in event.findall("bed:pick", _NAMESPACES):
            phase = arrivals.get(pick.attrib.get("publicID"))
            time = pick.findtext("bed:time/bed:value", namespaces=_NAMESPACES)
            if phase is not None and time is not None and phase not in phase_picks:
                phase_picks[phase] = UTCDateTime(time)
        parent_phase = next((phase for phase in parent_phase_priority if phase in phase_picks), None)

        def optional_float(query: str) -> float | None:
            value = origin.findtext(query, namespaces=_NAMESPACES)
            return None if value is None else float(value)

        origin_time = origin.findtext("bed:time/bed:value", namespaces=_NAMESPACES)
        if origin_time is None:
            raise ValueError(f"Preferred origin for {name} has no origin time.")
        records.append(
            MarsquakeCatalogEvent(
                name=name,
                family=family,
                quality=quality,
                origin_time=UTCDateTime(origin_time),
                distance_deg=optional_float("mars:distance/mars:value"),
                latitude_deg=optional_float("bed:latitude/bed:value"),
                longitude_deg=optional_float("bed:longitude/bed:value"),
                parent_phase=parent_phase,
                parent_arrival=None if parent_phase is None else phase_picks[parent_phase],
            )
        )
    return records


def catalog_back_azimuth(
    event: MarsquakeCatalogEvent,
    *,
    station_latitude_deg: float = 4.5024,
    station_longitude_deg: float = 135.6234,
) -> float:
    """Return station-to-event back azimuth from a catalogued location."""
    if event.latitude_deg is None or event.longitude_deg is None:
        raise ValueError(f"{event.name} has no catalogued latitude/longitude.")
    _, _, back_azimuth = gps2dist_azimuth(
        event.latitude_deg,
        event.longitude_deg,
        station_latitude_deg,
        station_longitude_deg,
    )
    return float(back_azimuth % 360.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_verified(source: Path, destination: Path) -> str:
    checksum = _sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) != checksum:
            raise FileExistsError(f"Refusing to overwrite different data: {destination}")
    else:
        shutil.copy2(source, destination)
        if _sha256(destination) != checksum:
            raise OSError(f"Copy verification failed: {destination}")
    return checksum


def import_insight_data(
    catalog: str | Path,
    deglitched_dir: str | Path,
    denoised_dir: str | Path,
    destination: str | Path,
    *,
    qualities: tuple[str, ...] = ("A", "B"),
) -> dict:
    """Copy available LF-family originals, recording missing products explicitly.

    LF-family means LOW_FREQUENCY plus BROADBAND, defaulting to Quality A/B.
    Events must be classified by the supplied catalog. Raw native-UVW counts
    and locally re-exported SAC files are not substituted for physical ZNE.
    Existing different files are never overwritten. Input files are unchanged.
    """
    catalog, deglitched_dir, denoised_dir, destination = map(
        Path, (catalog, deglitched_dir, denoised_dir, destination)
    )
    for path in (deglitched_dir, denoised_dir):
        if not path.is_dir():
            raise NotADirectoryError(path)
    events = {event.name: event for event in load_mqs_catalog(catalog, qualities=qualities)}
    names = set(events)
    catalog_relative = Path("catalogs") / catalog.name
    catalog_hash = _copy_verified(catalog, destination / catalog_relative)
    records = []
    for name in sorted(names):
        event = events.get(name)
        record = {
            "name": name,
            "in_mqs_v14_lf_family": event is not None,
            "family": event.family if event else None,
            "family_source": "MQS_V14" if event else "Dahmen_2024_revised_LF_family",
            "quality": event.quality if event else None,
            "origin_time": str(event.origin_time) if event else None,
            "distance_deg": event.distance_deg if event else None,
            "parent_phase": event.parent_phase if event else None,
            "parent_arrival_utc": str(event.parent_arrival) if event and event.parent_arrival else None,
            "back_azimuth_deg": (
                catalog_back_azimuth(event)
                if event and event.latitude_deg is not None and event.longitude_deg is not None
                else None
            ),
            "products": {},
        }
        for kind, directory, filename, units in (
            ("deglitched", deglitched_dir, f"deglitched_{name}_DISP_ZNE.mseed", "m"),
            ("denoised", denoised_dir, name, "m/s"),
        ):
            source = directory / filename
            product = {"status": "missing", "units": units, "source_filename": filename}
            if source.is_file():
                stream = read(str(source), headonly=True)
                components = {tr.stats.channel[-1] for tr in stream}
                if not {"Z", "N", "E"}.issubset(components):
                    raise ValueError(f"Expected physical ZNE components in {source}")
                relative = Path(kind) / f"{name}.mseed"
                product.update(
                    {
                        "status": "available",
                        "path": relative.as_posix(),
                        "sha256": _copy_verified(source, destination / relative),
                        "bytes": source.stat().st_size,
                        "traces": [
                            {
                                "id": tr.id,
                                "start_utc": str(tr.stats.starttime),
                                "end_utc": str(tr.stats.endtime),
                                "npts": tr.stats.npts,
                                "sampling_rate_hz": tr.stats.sampling_rate,
                            }
                            for tr in stream
                        ],
                    }
                )
            record["products"][kind] = product
        records.append(record)
    manifest = {
        "schema_version": 1,
        "selection": {"families": ["LOW_FREQUENCY", "BROADBAND"], "qualities": list(qualities)},
        "catalog": {"path": catalog_relative.as_posix(), "sha256": catalog_hash},
        "denoised_source": "https://doi.org/10.5281/zenodo.11995107",
        "deglitched_source": "User-supplied physical displacement ZNE miniSEED; processing history external",
        "events": records,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def load_insight_manifest(directory: str | Path) -> dict:
    """Read the portable inventory without requiring the waveform files."""
    manifest = json.loads((Path(directory) / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported InSight manifest schema.")
    return manifest


def load_insight_waveforms(
    directory: str | Path,
    event: str,
    dataset: str,
    *,
    velocity: bool = True,
    verify_checksum: bool = True,
):
    """Load an event and metadata; optionally put both products in velocity.

    Deglitched displacement is differentiated once. Denoised physical velocity
    is never response-corrected or differentiated again. No filtering, sign
    flips, rotation, or guessed picks are performed here.
    """
    if dataset not in {"deglitched", "denoised"}:
        raise ValueError("dataset must be deglitched or denoised")
    directory = Path(directory).resolve()
    record = next((e for e in load_insight_manifest(directory)["events"] if e["name"] == event), None)
    if record is None:
        raise KeyError(event)
    product = record["products"][dataset]
    if product["status"] != "available":
        raise FileNotFoundError(f"No {dataset} data available for {event}")
    path = (directory / product["path"]).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Manifest waveform path must stay inside insight_data")
    if verify_checksum and _sha256(path) != product["sha256"]:
        raise ValueError(f"Checksum mismatch for {path}")
    stream = read(str(path))
    if velocity and product["units"] == "m":
        stream.detrend("linear").differentiate(method="gradient")
    return stream, record


def preprocess_insight_event(directory, event, dataset, config, *, back_azimuth=None, p_arrival=None):
    """Prepare a direct-P RF using absolute time, never a reused trace offset.

    Missing/ambiguous picks and directions require explicit overrides. PP,
    Pdiff and x1 are not silently treated as direct P arrivals.
    """
    from .preprocessing import preprocess_zne_ps_data

    stream, record = load_insight_waveforms(directory, event, dataset)
    if p_arrival is None:
        if record["parent_phase"] != "P" or record["parent_arrival_utc"] is None:
            raise ValueError(f"{event} has no catalogued direct P; provide p_arrival explicitly")
        p_arrival = record["parent_arrival_utc"]
    if back_azimuth is None:
        back_azimuth = record["back_azimuth_deg"]
    if back_azimuth is None:
        raise ValueError(f"{event} has no catalogued back azimuth; provide one explicitly")
    pick = UTCDateTime(p_arrival)
    if any(
        tr.stats.starttime > pick - config.pre_arrival or tr.stats.endtime < pick + config.post_arrival
        for tr in stream
    ):
        raise ValueError("Waveforms do not cover the full requested P window")
    return preprocess_zne_ps_data(
        stream,
        p_arrival_offset_s=float(pick - stream[0].stats.starttime),
        back_azimuth=float(back_azimuth),
        config=config,
    )
