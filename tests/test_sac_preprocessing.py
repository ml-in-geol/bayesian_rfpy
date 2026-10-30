from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
from obspy import read

from bayesian_rf import (
    BayesianRFConfig,
    MarsEvent,
    preprocess_sp_data,
    preprocess_sac_ps_data,
    preprocess_sac_rtz_ps_data,
    preprocess_zne_free_surface_ps_data,
    preprocess_zne_ps_data,
)
from bayesian_rf.preprocessing import _free_surface_psv_transform
from bayesian_rf.deconvolution import _noise_correlation


def test_white_noise_correlation_is_identity_kernel() -> None:
    correlation = _noise_correlation(np.arange(5.0), 0.2, "white", 4.4)
    assert np.array_equal(correlation, [1.0, 0.0, 0.0, 0.0, 0.0])


def test_free_surface_transform_handles_obspy_radial_polarity() -> None:
    vp = 3.8
    vs = 1.85
    p = 0.20
    q_p = np.sqrt(1.0 / vp**2 - p**2)
    q_s = np.sqrt(1.0 / vs**2 - p**2)
    numerator = 1.0 - 2.0 * vs**2 * p**2
    p_vertical = -numerator / (2.0 * vp * q_p)
    p_radial = p * vs**2 / vp
    sv_radial = numerator / (2.0 * vs * q_s)
    sv_vertical = p * vs
    vertical = np.array([1.0, -0.5, 0.25])

    # ObsPy radial polarity is opposite to the radial basis of the transform.
    pure_sv_radial = (p_vertical / sv_vertical) * vertical
    longitudinal, sv_wavefield = _free_surface_psv_transform(
        pure_sv_radial,
        vertical,
        vp_km_s=vp,
        vs_km_s=vs,
        ray_parameter_s_per_km=p,
    )
    assert np.allclose(longitudinal, 0.0, atol=1.0e-12)
    assert np.any(np.abs(sv_wavefield) > 0.0)

    pure_p_radial = (p_radial / sv_radial) * vertical
    p_wavefield, sv = _free_surface_psv_transform(
        pure_p_radial,
        vertical,
        vp_km_s=vp,
        vs_km_s=vs,
        ray_parameter_s_per_km=p,
    )
    assert np.any(np.abs(p_wavefield) > 0.0)
    assert np.allclose(sv, 0.0, atol=1.0e-12)


def test_sac_preprocessing_uses_geometry_and_returns_rtz_window() -> None:
    input_dir = ROOT / "example_synthetics/Nakagawa_10s_example"
    paths = [input_dir / f"II.MSEY.BX{component}.sem.sac" for component in ("N", "E", "Z")]
    config = BayesianRFConfig(
        min_lag=0.0,
        max_lag=90.0,
        min_period=10.0,
        max_period=20.0,
        freqmin=1.0 / 20.0,
        freqmax=1.0 / 10.0,
        pre_arrival=30.0,
        post_arrival=90.0,
        target_sampling_rate=1.0,
    )

    processed = preprocess_sac_ps_data(paths, config=config, taup_model="iasp91")

    assert np.isclose(processed.back_azimuth, 326.501, atol=1.0e-3)
    assert np.isclose(processed.p_arrival_time, 535.5239806133364, atol=1.0e-6)
    assert processed.transverse is not None
    assert processed.parent.shape == processed.daughter.shape == processed.transverse.shape
    assert 119 <= processed.parent.size <= 122
    assert np.isclose(processed.dt, 1.0)
    assert np.all(np.isfinite(processed.parent))
    assert np.all(np.isfinite(processed.daughter))
    assert np.all(np.isfinite(processed.transverse))


def test_zne_preprocessing_uses_distinct_s0235b_trace_start() -> None:
    config = BayesianRFConfig(
        min_lag=0.0,
        max_lag=25.0,
        min_period=1.5,
        max_period=7.0,
        freqmin=1.0 / 7.0,
        freqmax=1.0 / 1.5,
        pre_arrival=20.0,
        post_arrival=55.0,
        target_sampling_rate=4.0,
    )

    processed = preprocess_zne_ps_data(
        str(ROOT / "insight_data/deglitched/S0235b.mseed"),
        p_arrival_offset_s=219.242,
        back_azimuth=77.07291,
        config=config,
    )

    assert np.isclose(processed.back_azimuth, 77.07291)
    assert np.isclose(processed.p_arrival_offset_s, 219.242)
    assert processed.transverse is not None
    assert processed.parent.shape == processed.daughter.shape == processed.transverse.shape
    assert 299 <= processed.parent.size <= 302
    assert np.isclose(processed.dt, 0.25)
    assert np.all(np.isfinite(processed.parent))
    assert np.all(np.isfinite(processed.daughter))
    assert np.all(np.isfinite(processed.transverse))


def test_zne_preprocessing_aligns_subsample_component_offsets() -> None:
    stream = read(str(ROOT / "insight_data/deglitched/S0235b.mseed"))
    stream.select(channel="*N")[0].stats.starttime += 0.001
    config = BayesianRFConfig(
        min_lag=0.0,
        max_lag=5.0,
        min_period=2.0,
        max_period=8.0,
        freqmin=1.0 / 8.0,
        freqmax=1.0 / 2.0,
        pre_arrival=10.0,
        post_arrival=10.0,
        target_sampling_rate=4.0,
    )

    processed = preprocess_zne_ps_data(
        stream,
        p_arrival_offset_s=219.242,
        back_azimuth=77.07291,
        config=config,
    )

    assert processed.parent.shape == processed.daughter.shape == processed.transverse.shape
    assert np.all(np.isfinite(processed.parent))
    assert np.all(np.isfinite(processed.daughter))


def test_sac_rtz_preprocessing_uses_explicit_catalog_pick() -> None:
    input_dir = ROOT / "insight_data/legacy_sac/S0235b"
    paths = [input_dir / f"S0235b.XB.ELYDL.03.BH{component}.SAC" for component in ("R", "T", "Z")]
    config = BayesianRFConfig(
        min_lag=0.0,
        max_lag=25.0,
        min_period=2.0,
        max_period=30.0,
        freqmin=1.0 / 30.0,
        freqmax=1.0 / 2.0,
        pre_arrival=20.0,
        post_arrival=55.0,
        target_sampling_rate=3.0,
    )

    processed = preprocess_sac_rtz_ps_data(paths, p_arrival_offset_s=241.023, config=config)

    assert np.isclose(processed.back_azimuth, 77.07291, atol=1.0e-4)
    assert np.isclose(processed.p_arrival_offset_s, 241.023)
    assert processed.transverse is not None
    assert processed.parent.shape == processed.daughter.shape == processed.transverse.shape
    assert 224 <= processed.parent.size <= 227
    assert np.isclose(processed.dt, 1.0 / 3.0)
    assert np.all(np.isfinite(processed.parent))
    assert np.all(np.isfinite(processed.daughter))
    assert np.all(np.isfinite(processed.transverse))


def test_free_surface_preprocessing_matches_paper_s0235b_geometry() -> None:
    config = BayesianRFConfig(
        min_lag=0.0,
        max_lag=15.0,
        min_period=1.0 / 0.8,
        max_period=10.0,
        freqmin=0.1,
        freqmax=0.8,
        pre_arrival=20.0,
        post_arrival=25.0,
        target_sampling_rate=4.8,
    )

    processed = preprocess_zne_free_surface_ps_data(
        str(ROOT / "insight_data/deglitched/S0235b.mseed"),
        p_arrival_offset_s=219.242,
        back_azimuth=74.0,
        vp_km_s=3.0,
        vs_km_s=1.5,
        ray_parameter_s_per_degree=6.4,
        planet_radius_km=3389.5,
        source_window_s=8.0,
        source_taper_s=1.0,
        config=config,
    )

    assert np.isclose(processed.sampling_rate, 4.8)
    assert np.isclose(processed.dt, 1.0 / 4.8)
    assert np.isclose(processed.back_azimuth, 74.0)
    assert processed.parent.shape == processed.daughter.shape
    assert 38 <= processed.parent.size <= 40
    assert processed.vertical is not None
    assert processed.radial is not None
    assert processed.longitudinal is not None
    assert processed.sv is not None
    assert processed.waveform_time_axis is not None
    assert np.all(np.isfinite(processed.parent))
    assert np.all(np.isfinite(processed.daughter))


def test_mars_sp_preprocessing_reverses_direct_s_window() -> None:
    config = BayesianRFConfig(
        min_lag=0.0,
        max_lag=8.0,
        pre_arrival=20.0,
        target_sampling_rate=3.0,
    )
    event = MarsEvent(
        name="mt01_nonoise",
        stream_path=ROOT / "example_synthetics/mars/mt01/synth_mt01_nonoise.mseed",
        source_latitude=10.18,
        source_longitude=164.05,
        source_depth_km=18.4,
        distance_deg=28.3,
    )

    processed = preprocess_sp_data(
        event.stream_path,
        event,
        ROOT / "example_synthetics/mars/insight_kks21_gp.nd",
        config,
    )

    assert processed.phase == "S"
    assert np.isclose(processed.p_arrival_time, 394.250346863517)
    assert np.isclose(processed.ray_parameter_s_per_degree, 13.166747318719725)
    assert np.isclose(processed.ray_parameter_s_per_km, 0.22256942064547375)
    assert processed.parent.shape == processed.daughter.shape
    assert 33 <= processed.parent.size <= 35
    assert np.isclose(processed.dt, 1.0 / 3.0)
    assert processed.waveform_time_axis is not None
    assert np.isclose(processed.waveform_time_axis[0], -2.0)
    assert np.all(np.isfinite(processed.parent))
    assert np.all(np.isfinite(processed.daughter))
