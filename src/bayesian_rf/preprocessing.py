"""Waveform input, RTZ/free-surface transforms, and arrival-centred windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence
import hashlib
import os
import tempfile
import shutil

import numpy as np
from obspy import Stream, Trace, read
from obspy.geodetics.base import gps2dist_azimuth
from obspy.signal.rotate import rotate_ne_rt
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
from scipy.signal.windows import tukey

if TYPE_CHECKING:
    from .deconvolution import BayesianRFConfig


@dataclass(frozen=True)
class MarsEvent:
    name: str
    stream_path: Path
    source_latitude: float
    source_longitude: float
    source_depth_km: float
    station_latitude: float = 4.50
    station_longitude: float = 135.62
    distance_deg: float = 0.0


@dataclass
class ProcessedRFData:
    parent: np.ndarray
    daughter: np.ndarray
    noise_window: np.ndarray
    dt: float
    p_arrival_time: float
    back_azimuth: float
    sampling_rate: float
    transverse: np.ndarray | None = None
    p_arrival_offset_s: float | None = None
    vertical: np.ndarray | None = None
    radial: np.ndarray | None = None
    longitudinal: np.ndarray | None = None
    sv: np.ndarray | None = None
    waveform_time_axis: np.ndarray | None = None
    phase: str = "P"
    ray_parameter_s_per_degree: float | None = None
    ray_parameter_s_per_km: float | None = None
    transform_vp_km_s: float | None = None
    transform_vs_km_s: float | None = None
    transform_ray_parameter_s_per_degree: float | None = None
    transform_correlation: float | None = None


def load_taup_model(model_path: str | Path) -> TauPyModel:
    model_path = Path(model_path)
    # TauP ships common Earth models such as iasp91 by name. Local Mars
    # models, in contrast, are compiled from their accompanying .nd files.
    if not model_path.exists() and not model_path.suffix:
        return TauPyModel(str(model_path))
    if model_path.suffix == ".npz":
        return TauPyModel(str(model_path))

    # Content-addressed, writable cache: never write next to installed inputs.
    import obspy

    digest = hashlib.sha256(model_path.read_bytes() + obspy.__version__.encode()).hexdigest()[:20]
    cache = Path(os.environ.get("BAYESIAN_RF_CACHE", Path.home() / ".cache" / "bayesian_rf"))
    directory = cache / "taup" / digest
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / "model.npz"
    if not npz_path.exists():
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            source = Path(temporary) / ("model" + model_path.suffix)
            shutil.copyfile(model_path, source)
            build_taup_model(str(source), output_folder=temporary, verbose=False)
            os.replace(source.with_suffix(".npz"), npz_path)
    return TauPyModel(str(npz_path))


def read_stream(path: str | Path) -> Stream:
    return read(str(path))


def _sac_float(trace: Trace, key: str) -> float | None:
    """Return a finite SAC header value, treating SAC's unset sentinel as missing."""
    value = getattr(getattr(trace.stats, "sac", None), key, None)
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) and value > -12344.0 else None


def _sac_component(stream: Stream, suffix: str) -> Trace:
    """Select exactly one SAC component ending in ``suffix`` (N, E, or Z)."""
    matches = [trace for trace in stream if trace.stats.channel.upper().endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one *{suffix} component, found {len(matches)}.")
    return matches[0]


def _align_zne_stream(stream: Stream) -> Stream:
    """Put Z/N/E components on their exact common sample grid.

    Some response-corrected miniSEED exports differ by a millisecond between
    component start times even though they share a 20 Hz nominal sample rate.
    Rotation requires sample-aligned arrays, so only those offset inputs are
    interpolated; already aligned streams retain their original samples.
    """
    traces = [_sac_component(stream, suffix).copy() for suffix in ("Z", "N", "E")]
    sampling_rates = np.asarray([trace.stats.sampling_rate for trace in traces])
    if not np.allclose(sampling_rates, sampling_rates[0]):
        raise ValueError("Z/N/E components must have matching sampling rates.")

    common_start = max(trace.stats.starttime for trace in traces)
    common_end = min(trace.stats.endtime for trace in traces)
    sampling_rate = float(sampling_rates[0])
    npts = int(np.floor(float(common_end - common_start) * sampling_rate)) + 1
    if npts < 4:
        raise ValueError("Z/N/E components do not share a usable time interval.")

    starts_aligned = all(trace.stats.starttime == common_start for trace in traces)
    if starts_aligned:
        for trace in traces:
            trace.data = trace.data[:npts]
            trace.stats.npts = npts
    else:
        for trace in traces:
            trace.interpolate(
                sampling_rate=sampling_rate,
                method="lanczos",
                starttime=common_start,
                npts=npts,
                a=8,
            )
    return Stream(traces=traces)


def preprocess_sac_ps_data(
    stream_or_paths: Stream | Sequence[str | Path],
    *,
    config: BayesianRFConfig,
    taup_model: str | Path = "iasp91",
    phase: str = "P",
) -> ProcessedRFData:
    """Prepare SAC N/E/Z data for Ps deconvolution using its embedded geometry.

    SAC ``baz`` is the station-to-event back azimuth required by
    :func:`obspy.signal.rotate.rotate_ne_rt`.  The direct-P travel time is
    calculated from SAC ``evdp`` and ``gcarc``; SAC's ``b`` and ``o`` offsets
    place that arrival correctly relative to the trace start time.
    """
    if isinstance(stream_or_paths, Stream):
        stream = stream_or_paths.copy()
    else:
        stream = Stream()
        for path in stream_or_paths:
            stream += read_stream(path)
    if len(stream) < 3:
        raise ValueError("SAC Ps preprocessing requires N, E, and Z components.")

    north = _sac_component(stream, "N")
    east = _sac_component(stream, "E")
    vertical = _sac_component(stream, "Z")
    for trace in (north, east, vertical):
        if not hasattr(trace.stats, "sac"):
            raise ValueError("SAC Ps preprocessing requires SAC headers on every component.")

    source_depth_km = _sac_float(vertical, "evdp")
    distance_deg = _sac_float(vertical, "gcarc")
    back_azimuth = _sac_float(vertical, "baz")
    if source_depth_km is None or distance_deg is None or back_azimuth is None:
        raise ValueError("SAC headers evdp, gcarc, and baz are required for Ps preprocessing.")

    stream.detrend("demean")
    stream.taper(max_percentage=0.02)
    stream.filter(
        "bandpass",
        freqmin=config.freqmin,
        freqmax=config.freqmax,
        corners=2,
        zerophase=True,
    )
    target_sampling_rate = config.target_sampling_rate or (6.0 * config.freqmax)
    if not np.isclose(stream[0].stats.sampling_rate, target_sampling_rate):
        stream.interpolate(sampling_rate=target_sampling_rate, method="lanczos", a=8)

    north = _sac_component(stream, "N")
    east = _sac_component(stream, "E")
    vertical = _sac_component(stream, "Z")
    if not (north.stats.starttime == east.stats.starttime == vertical.stats.starttime):
        raise ValueError("SAC N/E/Z components must have matching start times.")
    if not (north.stats.delta == east.stats.delta == vertical.stats.delta):
        raise ValueError("SAC N/E/Z components must have matching sampling intervals.")

    model = load_taup_model(taup_model)
    arrivals = model.get_travel_times(
        source_depth_in_km=source_depth_km,
        distance_in_degree=distance_deg,
        phase_list=[phase],
    )
    if not arrivals:
        raise ValueError(f"No {phase} arrival was found for the SAC event geometry.")
    p_arrival_time = float(arrivals[0].time)

    # SAC sample time is reference time + b, while origin time is reference
    # time + o. An unset o conventionally means the SAC reference time.
    begin_offset_s = _sac_float(vertical, "b")
    origin_offset_s = (_sac_float(vertical, "o") or 0.0) - (begin_offset_s or 0.0)
    p_arrival_offset_s = origin_offset_s + p_arrival_time
    radial, transverse = rotate_ne_rt(north.data, east.data, back_azimuth)

    def trimmed_trace(data: np.ndarray) -> Trace:
        trace = Trace(data=np.asarray(data, dtype=np.float64), header=vertical.stats)
        trace.trim(
            starttime=vertical.stats.starttime + p_arrival_offset_s - config.pre_arrival,
            endtime=vertical.stats.starttime + p_arrival_offset_s + config.post_arrival,
            pad=False,
        )
        return trace

    vertical_trace = trimmed_trace(vertical.data)
    radial_trace = trimmed_trace(radial)
    transverse_trace = trimmed_trace(transverse)
    npts = min(len(vertical_trace.data), len(radial_trace.data), len(transverse_trace.data))
    if npts < 4:
        raise ValueError("The requested P-referenced window is not covered by the SAC traces.")

    parent = np.asarray(vertical_trace.data[:npts], dtype=np.float64)
    daughter = np.asarray(radial_trace.data[:npts], dtype=np.float64)
    transverse_data = np.asarray(transverse_trace.data[:npts], dtype=np.float64)
    dt = float(vertical_trace.stats.delta)
    noise_end = max(1, int(round(config.pre_arrival / dt)))
    parent -= np.mean(parent[:noise_end])
    daughter -= np.mean(daughter[:noise_end])
    transverse_data -= np.mean(transverse_data[:noise_end])
    noise_window = daughter[:noise_end].copy()

    return ProcessedRFData(
        parent=parent,
        daughter=daughter,
        noise_window=noise_window,
        dt=dt,
        p_arrival_time=p_arrival_time,
        back_azimuth=back_azimuth,
        sampling_rate=float(vertical_trace.stats.sampling_rate),
        transverse=transverse_data,
        p_arrival_offset_s=p_arrival_offset_s,
    )


def preprocess_sac_rtz_ps_data(
    stream_or_paths: Stream | Sequence[str | Path],
    *,
    p_arrival_offset_s: float,
    config: BayesianRFConfig,
) -> ProcessedRFData:
    """Prepare already-rotated SAC R/T/Z data for Ps deconvolution.

    ``p_arrival_offset_s`` is measured from the trace start time. This is
    useful for catalogued phase picks when SAC headers do not contain an
    origin-time or arrival-time marker. The radial component is used as the
    daughter trace and the transverse component is retained for diagnostics.
    """
    if p_arrival_offset_s <= 0.0:
        raise ValueError("p_arrival_offset_s must be positive.")
    if isinstance(stream_or_paths, Stream):
        stream = stream_or_paths.copy()
    else:
        stream = Stream()
        for path in stream_or_paths:
            stream += read_stream(path)
    if len(stream) < 3:
        raise ValueError("SAC RTZ preprocessing requires R, T, and Z components.")

    radial = _sac_component(stream, "R")
    transverse = _sac_component(stream, "T")
    vertical = _sac_component(stream, "Z")
    for trace in (radial, transverse, vertical):
        if not hasattr(trace.stats, "sac"):
            raise ValueError("SAC RTZ preprocessing requires SAC headers on every component.")
    back_azimuth = _sac_float(vertical, "baz")
    if back_azimuth is None:
        raise ValueError("SAC header baz is required for RTZ preprocessing metadata.")

    stream.detrend("demean")
    stream.taper(max_percentage=0.02)
    stream.filter(
        "bandpass",
        freqmin=config.freqmin,
        freqmax=config.freqmax,
        corners=2,
        zerophase=True,
    )
    target_sampling_rate = config.target_sampling_rate or (6.0 * config.freqmax)
    if not np.isclose(stream[0].stats.sampling_rate, target_sampling_rate):
        stream.interpolate(sampling_rate=target_sampling_rate, method="lanczos", a=8)

    radial = _sac_component(stream, "R")
    transverse = _sac_component(stream, "T")
    vertical = _sac_component(stream, "Z")
    if not (radial.stats.starttime == transverse.stats.starttime == vertical.stats.starttime):
        raise ValueError("SAC R/T/Z components must have matching start times.")
    if not (radial.stats.delta == transverse.stats.delta == vertical.stats.delta):
        raise ValueError("SAC R/T/Z components must have matching sampling intervals.")

    def trimmed_trace(data: np.ndarray) -> Trace:
        trace = Trace(data=np.asarray(data, dtype=np.float64), header=vertical.stats)
        trace.trim(
            starttime=vertical.stats.starttime + p_arrival_offset_s - config.pre_arrival,
            endtime=vertical.stats.starttime + p_arrival_offset_s + config.post_arrival,
            pad=False,
        )
        return trace

    vertical_trace = trimmed_trace(vertical.data)
    radial_trace = trimmed_trace(radial.data)
    transverse_trace = trimmed_trace(transverse.data)
    npts = min(len(vertical_trace.data), len(radial_trace.data), len(transverse_trace.data))
    if npts < 4:
        raise ValueError("The requested P-referenced window is not covered by the SAC traces.")

    parent = np.asarray(vertical_trace.data[:npts], dtype=np.float64)
    daughter = np.asarray(radial_trace.data[:npts], dtype=np.float64)
    transverse_data = np.asarray(transverse_trace.data[:npts], dtype=np.float64)
    dt = float(vertical_trace.stats.delta)
    noise_end = max(1, int(round(config.pre_arrival / dt)))
    parent -= np.mean(parent[:noise_end])
    daughter -= np.mean(daughter[:noise_end])
    transverse_data -= np.mean(transverse_data[:noise_end])
    noise_window = daughter[:noise_end].copy()

    return ProcessedRFData(
        parent=parent,
        daughter=daughter,
        noise_window=noise_window,
        dt=dt,
        p_arrival_time=p_arrival_offset_s,
        back_azimuth=back_azimuth,
        sampling_rate=float(vertical_trace.stats.sampling_rate),
        transverse=transverse_data,
        p_arrival_offset_s=p_arrival_offset_s,
    )


def preprocess_zne_ps_data(
    stream_or_path: Stream | str | Path,
    *,
    p_arrival_offset_s: float,
    back_azimuth: float,
    config: BayesianRFConfig,
) -> ProcessedRFData:
    """Prepare Z/N/E waveform data for Ps deconvolution with an explicit P pick.

    This path is intended for formats such as miniSEED that do not retain SAC
    event headers. ``p_arrival_offset_s`` is relative to the input trace start
    time, and ``back_azimuth`` is the station-to-event angle in degrees used
    for the N/E-to-R/T rotation.
    """
    if p_arrival_offset_s <= 0.0:
        raise ValueError("p_arrival_offset_s must be positive.")
    if not 0.0 <= back_azimuth <= 360.0:
        raise ValueError("back_azimuth must be within [0, 360] degrees.")
    stream = stream_or_path.copy() if isinstance(stream_or_path, Stream) else read_stream(stream_or_path)
    if len(stream) < 3:
        raise ValueError("Z/N/E Ps preprocessing requires N, E, and Z components.")
    absolute_p_arrival = stream[0].stats.starttime + p_arrival_offset_s
    stream = _align_zne_stream(stream)
    p_arrival_offset_s = float(absolute_p_arrival - stream[0].stats.starttime)

    stream.detrend("demean")
    stream.taper(max_percentage=0.02)
    stream.filter(
        "bandpass",
        freqmin=config.freqmin,
        freqmax=config.freqmax,
        corners=2,
        zerophase=True,
    )
    target_sampling_rate = config.target_sampling_rate or (6.0 * config.freqmax)
    if not np.isclose(stream[0].stats.sampling_rate, target_sampling_rate):
        stream.interpolate(sampling_rate=target_sampling_rate, method="lanczos", a=8)

    north = _sac_component(stream, "N")
    east = _sac_component(stream, "E")
    vertical = _sac_component(stream, "Z")
    if not (north.stats.starttime == east.stats.starttime == vertical.stats.starttime):
        raise ValueError("Filtered Z/N/E components must have matching start times.")
    if not (north.stats.delta == east.stats.delta == vertical.stats.delta):
        raise ValueError("Filtered Z/N/E components must have matching sampling intervals.")
    radial, transverse = rotate_ne_rt(north.data, east.data, back_azimuth)

    def trimmed_trace(data: np.ndarray) -> Trace:
        trace = Trace(data=np.asarray(data, dtype=np.float64), header=vertical.stats)
        trace.trim(
            starttime=vertical.stats.starttime + p_arrival_offset_s - config.pre_arrival,
            endtime=vertical.stats.starttime + p_arrival_offset_s + config.post_arrival,
            pad=False,
        )
        return trace

    vertical_trace = trimmed_trace(vertical.data)
    radial_trace = trimmed_trace(radial)
    transverse_trace = trimmed_trace(transverse)
    npts = min(len(vertical_trace.data), len(radial_trace.data), len(transverse_trace.data))
    if npts < 4:
        raise ValueError("The requested P-referenced window is not covered by the waveform data.")

    parent = np.asarray(vertical_trace.data[:npts], dtype=np.float64)
    daughter = np.asarray(radial_trace.data[:npts], dtype=np.float64)
    transverse_data = np.asarray(transverse_trace.data[:npts], dtype=np.float64)
    dt = float(vertical_trace.stats.delta)
    noise_end = max(1, int(round(config.pre_arrival / dt)))
    parent -= np.mean(parent[:noise_end])
    daughter -= np.mean(daughter[:noise_end])
    transverse_data -= np.mean(transverse_data[:noise_end])
    noise_window = daughter[:noise_end].copy()

    return ProcessedRFData(
        parent=parent,
        daughter=daughter,
        noise_window=noise_window,
        dt=dt,
        p_arrival_time=p_arrival_offset_s,
        back_azimuth=back_azimuth,
        sampling_rate=float(vertical_trace.stats.sampling_rate),
        transverse=transverse_data,
        p_arrival_offset_s=p_arrival_offset_s,
    )


def _free_surface_psv_transform(
    radial: np.ndarray,
    vertical: np.ndarray,
    *,
    vp_km_s: float,
    vs_km_s: float,
    ray_parameter_s_per_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return incident P and SV wavefields from radial and vertical motion.

    This is the Kennett (1991) free-surface transformation in the radial-
    vertical plane. ``radial`` follows ObsPy's ``rotate_ne_rt`` convention
    (positive away from the source), whereas the Kennett matrix uses radial
    positive toward the source. The output is ordered as longitudinal P and
    SV.
    """
    if vp_km_s <= 0.0 or vs_km_s <= 0.0 or ray_parameter_s_per_km <= 0.0:
        raise ValueError("Surface velocities and ray parameter must be positive.")
    if np.asarray(radial).shape != np.asarray(vertical).shape:
        raise ValueError("Radial and vertical arrays must have matching shapes.")

    p = float(ray_parameter_s_per_km)
    vp_squared = vp_km_s**2
    vs_squared = vs_km_s**2
    p_squared = p**2
    q_p_squared = 1.0 / vp_squared - p_squared
    q_s_squared = 1.0 / vs_squared - p_squared
    if q_p_squared <= 0.0 or q_s_squared <= 0.0:
        raise ValueError("Ray parameter produces an evanescent P or SV wave at the surface.")

    q_p = np.sqrt(q_p_squared)
    q_s = np.sqrt(q_s_squared)
    numerator = 1.0 - 2.0 * vs_squared * p_squared
    p_vertical = -numerator / (2.0 * vp_km_s * q_p)
    p_radial = p * vs_squared / vp_km_s
    sv_radial = numerator / (2.0 * vs_km_s * q_s)
    sv_vertical = p * vs_km_s

    radial = -np.asarray(radial, dtype=np.float64)
    vertical = np.asarray(vertical, dtype=np.float64)
    longitudinal = -sv_vertical * radial - p_vertical * vertical
    sv = sv_radial * radial + p_radial * vertical
    return longitudinal, sv


def preprocess_zne_free_surface_ps_data(
    stream_or_path: Stream | str | Path,
    *,
    p_arrival_offset_s: float,
    back_azimuth: float,
    vp_km_s: float,
    vs_km_s: float,
    ray_parameter_s_per_degree: float,
    planet_radius_km: float,
    source_window_s: float,
    source_taper_s: float,
    config: BayesianRFConfig,
    differentiate_to_velocity: bool = True,
) -> ProcessedRFData:
    """Prepare the Kennett-transformed P/SV data used by paper-style Ps RFs.

    The source and response are restricted to a tapered window beginning at P,
    as in the S0235b Bayesian processing in Knapmeyer-Endrun et al. (2021).
    A noise window preceding P is retained only to initialize the hierarchical
    noise model; it is not included in the deconvolution likelihood.
    """
    if p_arrival_offset_s <= 0.0:
        raise ValueError("p_arrival_offset_s must be positive.")
    if not 0.0 <= back_azimuth <= 360.0:
        raise ValueError("back_azimuth must be within [0, 360] degrees.")
    if planet_radius_km <= 0.0:
        raise ValueError("planet_radius_km must be positive.")
    if source_window_s <= 0.0 or source_taper_s < 0.0:
        raise ValueError("source_window_s must be positive and source_taper_s non-negative.")
    if source_taper_s * 2.0 > source_window_s:
        raise ValueError("source_taper_s cannot exceed half of source_window_s.")

    stream = stream_or_path.copy() if isinstance(stream_or_path, Stream) else read_stream(stream_or_path)
    if len(stream) < 3:
        raise ValueError("Free-surface Ps preprocessing requires N, E, and Z components.")
    north = _sac_component(stream, "N")
    east = _sac_component(stream, "E")
    vertical = _sac_component(stream, "Z")
    if not (north.stats.starttime == east.stats.starttime == vertical.stats.starttime):
        raise ValueError("Z/N/E components must have matching start times.")
    if not (north.stats.delta == east.stats.delta == vertical.stats.delta):
        raise ValueError("Z/N/E components must have matching sampling intervals.")

    stream.detrend("linear")
    stream.taper(max_percentage=0.02)
    if differentiate_to_velocity:
        stream.differentiate(method="gradient")
        stream.detrend("demean")
    stream.filter(
        "bandpass",
        freqmin=config.freqmin,
        freqmax=config.freqmax,
        corners=2,
        zerophase=True,
    )
    target_sampling_rate = config.target_sampling_rate or (6.0 * config.freqmax)
    if not np.isclose(stream[0].stats.sampling_rate, target_sampling_rate):
        stream.interpolate(sampling_rate=target_sampling_rate, method="lanczos", a=8)

    north = _sac_component(stream, "N")
    east = _sac_component(stream, "E")
    vertical = _sac_component(stream, "Z")
    radial, transverse = rotate_ne_rt(north.data, east.data, back_azimuth)
    kilometers_per_degree = np.pi * planet_radius_km / 180.0
    ray_parameter_s_per_km = ray_parameter_s_per_degree / kilometers_per_degree
    longitudinal, sv = _free_surface_psv_transform(
        radial,
        vertical.data,
        vp_km_s=vp_km_s,
        vs_km_s=vs_km_s,
        ray_parameter_s_per_km=ray_parameter_s_per_km,
    )

    def trim_data(data: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
        trace = Trace(data=np.asarray(data, dtype=np.float64), header=vertical.stats)
        trace.trim(
            starttime=vertical.stats.starttime + p_arrival_offset_s + start_s,
            endtime=vertical.stats.starttime + p_arrival_offset_s + end_s,
            pad=False,
        )
        return np.asarray(trace.data, dtype=np.float64)

    waveform_start_s = -config.pre_arrival
    waveform_end_s = config.post_arrival
    waveform_vertical = trim_data(vertical.data, waveform_start_s, waveform_end_s)
    waveform_radial = trim_data(radial, waveform_start_s, waveform_end_s)
    waveform_transverse = trim_data(transverse, waveform_start_s, waveform_end_s)
    waveform_longitudinal = trim_data(longitudinal, waveform_start_s, waveform_end_s)
    waveform_sv = trim_data(sv, waveform_start_s, waveform_end_s)
    npts = min(
        waveform_vertical.size,
        waveform_radial.size,
        waveform_transverse.size,
        waveform_longitudinal.size,
        waveform_sv.size,
    )
    if npts < 4:
        raise ValueError("The requested P-referenced waveform window is not covered by the data.")
    waveform_vertical = waveform_vertical[:npts]
    waveform_radial = waveform_radial[:npts]
    waveform_transverse = waveform_transverse[:npts]
    waveform_longitudinal = waveform_longitudinal[:npts]
    waveform_sv = waveform_sv[:npts]
    dt = float(vertical.stats.delta)
    waveform_time_axis = np.arange(npts, dtype=np.float64) * dt + waveform_start_s
    noise_selector = waveform_time_axis < 0.0
    source_selector = (waveform_time_axis >= 0.0) & (waveform_time_axis <= source_window_s)
    if noise_selector.sum() < 4 or source_selector.sum() < 4:
        raise ValueError("The selected noise or source window is too short after trimming.")

    noise_window = waveform_sv[noise_selector].copy()
    noise_window -= np.mean(noise_window)
    parent = waveform_longitudinal[source_selector].copy()
    daughter = waveform_sv[source_selector].copy()
    source_taper_fraction = 0.0 if source_taper_s == 0.0 else min(1.0, 2.0 * source_taper_s / source_window_s)
    source_taper = tukey(parent.size, alpha=source_taper_fraction)
    parent *= source_taper
    daughter *= source_taper

    return ProcessedRFData(
        parent=parent,
        daughter=daughter,
        noise_window=noise_window,
        dt=dt,
        p_arrival_time=p_arrival_offset_s,
        back_azimuth=back_azimuth,
        sampling_rate=float(vertical.stats.sampling_rate),
        transverse=0.5 * waveform_transverse,
        p_arrival_offset_s=p_arrival_offset_s,
        vertical=waveform_vertical,
        radial=waveform_radial,
        longitudinal=waveform_longitudinal,
        sv=waveform_sv,
        waveform_time_axis=waveform_time_axis,
    )


def preprocess_ps_data(
    stream_or_path: Stream | str | Path,
    event: MarsEvent,
    model_path: str | Path,
    config: BayesianRFConfig,
) -> ProcessedRFData:
    stream = stream_or_path.copy() if isinstance(stream_or_path, Stream) else read_stream(stream_or_path)
    stream = stream.copy()
    stream.detrend("demean")
    stream.taper(max_percentage=0.02)
    stream.filter(
        "bandpass",
        freqmin=config.freqmin,
        freqmax=config.freqmax,
        corners=2,
        zerophase=True,
    )

    target_sampling_rate = config.target_sampling_rate or (6.0 * config.freqmax)
    if not np.isclose(stream[0].stats.sampling_rate, target_sampling_rate):
        stream.interpolate(sampling_rate=target_sampling_rate, method="lanczos", a=8)

    model = load_taup_model(model_path)
    arrivals = model.get_travel_times(
        source_depth_in_km=event.source_depth_km,
        distance_in_degree=event.distance_deg,
        phase_list=["P"],
    )
    if not arrivals:
        raise ValueError("No P arrival was found for the supplied event geometry.")
    p_arrival_time = float(arrivals[0].time)
    ray_parameter_s_per_degree = float(arrivals[0].ray_param_sec_degree)
    ray_parameter_s_per_km = ray_parameter_s_per_degree / (np.pi * 3389.5 / 180.0)

    _, _, back_azimuth = gps2dist_azimuth(
        event.source_latitude,
        event.source_longitude,
        event.station_latitude,
        event.station_longitude,
    )

    north = stream.select(channel="*N")[0]
    east = stream.select(channel="*E")[0]
    vertical = stream.select(channel="*Z")[0]
    radial, _ = rotate_ne_rt(north.data, east.data, back_azimuth)

    parent_trace = Trace(data=vertical.data.astype(np.float64, copy=True), header=vertical.stats)
    daughter_trace = Trace(data=np.asarray(radial, dtype=np.float64), header=vertical.stats)

    start_time = vertical.stats.starttime + p_arrival_time - config.pre_arrival
    end_time = vertical.stats.starttime + p_arrival_time + config.post_arrival
    parent_trace.trim(starttime=start_time, endtime=end_time, pad=False)
    daughter_trace.trim(starttime=start_time, endtime=end_time, pad=False)

    if len(parent_trace.data) != len(daughter_trace.data):
        npts = min(len(parent_trace.data), len(daughter_trace.data))
        parent_trace.data = parent_trace.data[:npts]
        daughter_trace.data = daughter_trace.data[:npts]

    parent = np.asarray(parent_trace.data, dtype=np.float64)
    daughter = np.asarray(daughter_trace.data, dtype=np.float64)
    dt = float(parent_trace.stats.delta)

    noise_end = max(1, int(round(config.pre_arrival / dt)))
    noise_window = daughter[:noise_end].copy()
    parent -= np.mean(parent[:noise_end])
    daughter -= np.mean(daughter[:noise_end])
    noise_window -= np.mean(noise_window)

    return ProcessedRFData(
        parent=parent,
        daughter=daughter,
        noise_window=noise_window,
        dt=dt,
        p_arrival_time=p_arrival_time,
        back_azimuth=float(back_azimuth),
        sampling_rate=float(parent_trace.stats.sampling_rate),
        phase="P",
        ray_parameter_s_per_degree=ray_parameter_s_per_degree,
        ray_parameter_s_per_km=ray_parameter_s_per_km,
    )


def preprocess_sp_data(
    stream_or_path: Stream | str | Path,
    event: MarsEvent,
    model_path: str | Path,
    config: BayesianRFConfig,
    *,
    phase: str = "S",
    vp_km_s: float = 3.8,
    vs_km_s: float = 1.85,
    planet_radius_km: float = 3389.5,
    source_window_before_s: float = 9.0,
    source_window_after_s: float = 2.0,
    source_taper_s: float = 1.0,
    differentiate_to_velocity: bool = False,
    optimize_transform: bool = False,
    decorrelation_window_s: float = 5.0,
) -> ProcessedRFData:
    """Prepare a positive-display-lag S-to-P deconvolution from Z/N/E data.

    Physical Sp conversions precede the parent S phase, so the transformed P
    and SV windows are reversed before deconvolution. The Kennett incident-P
    row has the opposite polarity to CPS ``hrftn96``; reversal therefore
    directly produces the common ``-RF_CPS(-t)`` convention used by
    :func:`forward_sp_receiver_function`.

    ``phase`` may be any S-like TauP phase present in the waveform, such as
    ``S`` or ``ScS``. A short interval after the phase is retained so that the
    complete direct-S wavelet lies inside the reversed source window.
    """
    if not phase.upper().endswith("S"):
        raise ValueError("phase must be an S-like TauP phase.")
    if planet_radius_km <= 0.0:
        raise ValueError("planet_radius_km must be positive.")
    if source_window_before_s <= 0.0 or source_window_after_s < 0.0:
        raise ValueError("Require a positive pre-S window and a non-negative post-S window.")
    source_window_s = source_window_before_s + source_window_after_s
    if source_taper_s < 0.0 or 2.0 * source_taper_s > source_window_s:
        raise ValueError("source_taper_s must be between zero and half the source-window length.")
    if decorrelation_window_s <= 0.0:
        raise ValueError("decorrelation_window_s must be positive.")

    stream = stream_or_path.copy() if isinstance(stream_or_path, Stream) else read_stream(stream_or_path)
    stream = stream.copy()
    if len(stream) < 3:
        raise ValueError("S-to-P preprocessing requires N, E, and Z components.")
    stream.detrend("linear")
    stream.taper(max_percentage=0.02)
    if differentiate_to_velocity:
        stream.differentiate(method="gradient")
        stream.detrend("demean")
    stream.filter(
        "bandpass",
        freqmin=config.freqmin,
        freqmax=config.freqmax,
        corners=2,
        zerophase=True,
    )
    target_sampling_rate = config.target_sampling_rate or (6.0 * config.freqmax)
    if not np.isclose(stream[0].stats.sampling_rate, target_sampling_rate):
        stream.interpolate(sampling_rate=target_sampling_rate, method="lanczos", a=8)

    north = _sac_component(stream, "N")
    east = _sac_component(stream, "E")
    vertical = _sac_component(stream, "Z")
    if not (north.stats.starttime == east.stats.starttime == vertical.stats.starttime):
        raise ValueError("Z/N/E components must have matching start times.")
    if not (north.stats.delta == east.stats.delta == vertical.stats.delta):
        raise ValueError("Z/N/E components must have matching sampling intervals.")

    model = load_taup_model(model_path)
    arrivals = model.get_travel_times(
        source_depth_in_km=event.source_depth_km,
        distance_in_degree=event.distance_deg,
        phase_list=[phase],
    )
    if not arrivals:
        raise ValueError(f"No {phase} arrival was found for the supplied event geometry.")
    arrival = arrivals[0]
    arrival_time = float(arrival.time)
    ray_parameter_s_per_degree = float(arrival.ray_param_sec_degree)
    ray_parameter_s_per_km = ray_parameter_s_per_degree / (np.pi * planet_radius_km / 180.0)

    _, _, back_azimuth = gps2dist_azimuth(
        event.source_latitude,
        event.source_longitude,
        event.station_latitude,
        event.station_longitude,
    )
    radial, transverse = rotate_ne_rt(north.data, east.data, back_azimuth)
    transform_vp = vp_km_s
    transform_vs = vs_km_s
    transform_ray_parameter_s_per_degree = ray_parameter_s_per_degree
    transform_correlation = None
    if optimize_transform:
        dt = float(vertical.stats.delta)
        first = max(0, int(round(arrival_time / dt)))
        last = min(len(vertical.data), first + max(4, int(round(decorrelation_window_s / dt))) + 1)
        radial_window = np.asarray(radial[first:last], dtype=np.float64)
        vertical_window = np.asarray(vertical.data[first:last], dtype=np.float64)
        best: tuple[float, float, float, float] | None = None
        kilometers_per_degree = np.pi * planet_radius_km / 180.0
        for candidate_vp in np.linspace(2.5, 5.0, 26):
            for candidate_vpvs in np.linspace(1.5, 2.5, 21):
                candidate_vs = candidate_vp / candidate_vpvs
                for candidate_p_degree in np.linspace(3.0, 14.0, 45):
                    candidate_p_km = candidate_p_degree / kilometers_per_degree
                    try:
                        candidate_p, candidate_sv = _free_surface_psv_transform(
                            radial_window,
                            vertical_window,
                            vp_km_s=float(candidate_vp),
                            vs_km_s=float(candidate_vs),
                            ray_parameter_s_per_km=float(candidate_p_km),
                        )
                    except ValueError:
                        continue
                    correlation = float(abs(np.corrcoef(candidate_p, candidate_sv)[0, 1]))
                    if np.isfinite(correlation) and (best is None or correlation < best[0]):
                        best = (
                            correlation,
                            float(candidate_vp),
                            float(candidate_vs),
                            float(candidate_p_degree),
                        )
        if best is None:
            raise ValueError("No propagating free-surface transform was found in the decorrelation grid.")
        transform_correlation, transform_vp, transform_vs, transform_ray_parameter_s_per_degree = best

    transform_ray_parameter_s_per_km = transform_ray_parameter_s_per_degree / (
        np.pi * planet_radius_km / 180.0
    )
    longitudinal, sv = _free_surface_psv_transform(
        radial,
        vertical.data,
        vp_km_s=transform_vp,
        vs_km_s=transform_vs,
        ray_parameter_s_per_km=transform_ray_parameter_s_per_km,
    )

    def trim_data(data: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
        trace = Trace(data=np.asarray(data, dtype=np.float64), header=vertical.stats)
        trace.trim(
            starttime=vertical.stats.starttime + arrival_time + start_s,
            endtime=vertical.stats.starttime + arrival_time + end_s,
            pad=False,
        )
        return np.asarray(trace.data, dtype=np.float64)

    start_s = -source_window_before_s
    end_s = source_window_after_s
    source_longitudinal = trim_data(longitudinal, start_s, end_s)
    source_sv = trim_data(sv, start_s, end_s)
    source_vertical = trim_data(vertical.data, start_s, end_s)
    source_radial = trim_data(radial, start_s, end_s)
    source_transverse = trim_data(transverse, start_s, end_s)
    npts = min(
        source_longitudinal.size,
        source_sv.size,
        source_vertical.size,
        source_radial.size,
        source_transverse.size,
    )
    if npts < 4:
        raise ValueError("The requested S-referenced source window is not covered by the data.")

    taper_fraction = 0.0 if source_taper_s == 0.0 else min(1.0, 2.0 * source_taper_s / source_window_s)
    source_taper = tukey(npts, alpha=taper_fraction)
    parent = source_sv[:npts][::-1].copy() * source_taper
    daughter = source_longitudinal[:npts][::-1].copy() * source_taper
    parent -= np.mean(parent)
    daughter -= np.mean(daughter)

    # The supplied synthetics begin well before any body-wave arrival. This
    # quiet segment is preferable to post-S data, which can contain S coda.
    dt = float(vertical.stats.delta)
    noise_npts = min(len(longitudinal), max(4, int(round(config.pre_arrival / dt))))
    noise_window = np.asarray(longitudinal[:noise_npts], dtype=np.float64).copy()
    noise_window -= np.mean(noise_window)
    display_start_s = -source_window_after_s
    waveform_time_axis = np.arange(npts, dtype=np.float64) * dt + display_start_s

    return ProcessedRFData(
        parent=parent,
        daughter=daughter,
        noise_window=noise_window,
        dt=dt,
        p_arrival_time=arrival_time,
        back_azimuth=float(back_azimuth),
        sampling_rate=float(vertical.stats.sampling_rate),
        transverse=source_transverse[:npts][::-1].copy(),
        p_arrival_offset_s=arrival_time,
        vertical=source_vertical[:npts][::-1].copy(),
        radial=source_radial[:npts][::-1].copy(),
        longitudinal=source_longitudinal[:npts][::-1].copy(),
        sv=source_sv[:npts][::-1].copy(),
        waveform_time_axis=waveform_time_axis,
        phase=phase,
        ray_parameter_s_per_degree=ray_parameter_s_per_degree,
        ray_parameter_s_per_km=ray_parameter_s_per_km,
        transform_vp_km_s=transform_vp,
        transform_vs_km_s=transform_vs,
        transform_ray_parameter_s_per_degree=transform_ray_parameter_s_per_degree,
        transform_correlation=transform_correlation,
    )
