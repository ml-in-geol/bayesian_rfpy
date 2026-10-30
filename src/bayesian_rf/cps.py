"""Thin Python interface to CPS ``hrftn96`` layer-cake RF synthetics.

The wrapper deliberately executes the CPS program instead of binding its
program-level Fortran internals. This preserves CPS's supported MODEL96 and
SAC interfaces while keeping the inversion layer independent of CPS build
details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
import subprocess
import tempfile
import hashlib
import os
import platform

import numpy as np
from obspy import read


class CPSForwardError(RuntimeError):
    """Raised when CPS cannot build or evaluate a receiver-function model."""


@dataclass(frozen=True)
class IsotropicLayer:
    """A constant-velocity isotropic MODEL96 layer in km, km/s, and g/cm^3."""

    thickness_km: float
    vp_km_s: float
    vs_km_s: float
    density_g_cm3: float
    qp: float = 0.0
    qs: float = 0.0
    etap: float = 0.0
    etas: float = 0.0
    frefp_hz: float = 1.0
    frefs_hz: float = 1.0

    def __post_init__(self) -> None:
        if self.thickness_km < 0.0:
            raise ValueError("Layer thickness must be non-negative.")
        if self.vs_km_s <= 0.0 or self.vp_km_s <= self.vs_km_s:
            raise ValueError("Require positive Vs and Vp > Vs for an isotropic solid layer.")
        if self.density_g_cm3 <= 0.0:
            raise ValueError("Layer density must be positive.")


@dataclass(frozen=True)
class LayerCakeModel:
    """An isotropic 1-D Earth model for CPS ``hrftn96``.

    The final layer is treated as the halfspace; its thickness is ignored by
    the forward calculation and is conventionally set to zero.
    """

    layers: tuple[IsotropicLayer, ...]
    title: str = "Bayesian RF layer-cake model"

    def __post_init__(self) -> None:
        if not 2 <= len(self.layers) <= 200:
            raise ValueError("CPS hrftn96 requires between 2 and 200 model layers.")

    def write_model96(self, path: str | Path) -> Path:
        """Write the model in the CPS isotropic MODEL96 text format."""
        path = Path(path)
        header = [
            "MODEL.01",
            self.title[:80],
            "ISOTROPIC",
            "KGS",
            "FLAT EARTH",
            "1-D",
            "CONSTANT VELOCITY",
            "LINE08",
            "LINE09",
            "LINE10",
            "LINE11",
            " H(KM)      VP(KM/S)    VS(KM/S)    RHO(G/CC)  QP        QS        ETAP      ETAS      FREFP     FREFS",
        ]
        rows = [
            (
                f"{layer.thickness_km:10.4f} {layer.vp_km_s:10.4f} {layer.vs_km_s:10.4f} "
                f"{layer.density_g_cm3:10.4f} {layer.qp:10.4f} {layer.qs:10.4f} "
                f"{layer.etap:10.4f} {layer.etas:10.4f} {layer.frefp_hz:10.4f} {layer.frefs_hz:10.4f}"
            )
            for layer in self.layers
        ]
        path.write_text("\n".join([*header, *rows, ""]) + "\n")
        return path


@dataclass(frozen=True)
class CPSReceiverFunction:
    """A receiver function synthesized by CPS ``hrftn96``."""

    lag_times_s: np.ndarray
    amplitudes: np.ndarray
    sampling_interval_s: float
    ray_parameter_s_per_km: float
    gaussian_alpha: float
    model: LayerCakeModel = field(repr=False)
    incident_phase: str = "P"
    positive_lag_sp_convention: bool = False


def default_cps_root() -> Path:
    """Return the small, platform-independent CPS source bundle."""
    return Path(__file__).resolve().parent / "_cps"


def hrftn96_path(cps_root: str | Path | None = None) -> Path:
    """Locate an explicit executable or a content/platform-keyed build cache."""
    external = os.environ.get("BAYESIAN_RF_HRFTN96")
    if external:
        return Path(external).expanduser().resolve()
    root = Path(cps_root) if cps_root is not None else default_cps_root()
    if (root / "bin" / "hrftn96").is_file():
        return (root / "bin" / "hrftn96").resolve()
    sources = _cps_sources(root)
    digest = hashlib.sha256((platform.system() + platform.machine() + "-O2-legacy-v1").encode())
    for source in sources:
        digest.update(source.read_bytes())
    cache = Path(os.environ.get("BAYESIAN_RF_CACHE", Path.home() / ".cache" / "bayesian_rf"))
    return (cache / "cps" / digest.hexdigest()[:20] / "hrftn96").resolve()


def _cps_sources(root: Path) -> list[Path]:
    names = ("hrftn96", "sacsubf", "igetmod", "mchdep", "lgstr", "mnmarg", "mgtarg")
    if (root / "hrftn96.f").is_file():
        return [root / f"{name}.f" for name in names]
    return [root / "VOLVI/src/hrftn96.f", *[root / "SUBS" / f"{name}.f" for name in names[1:]]]


def build_hrftn96(cps_root: str | Path | None = None) -> Path:
    """Compile the narrow CPS target needed for isotropic Ps RF synthetics."""
    root = Path(cps_root or default_cps_root())
    executable = hrftn96_path(cps_root)
    if executable.is_file() and os.access(executable, os.X_OK):
        return executable
    if os.environ.get("BAYESIAN_RF_HRFTN96"):
        raise CPSForwardError(f"BAYESIAN_RF_HRFTN96 is not executable: {executable}")
    compiler = shutil.which(os.environ.get("FC", "gfortran"))
    if compiler is None:
        raise CPSForwardError(
            "Install gfortran or set BAYESIAN_RF_HRFTN96 to an existing hrftn96 executable."
        )
    executable.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=executable.parent) as temporary:
        target = Path(temporary) / "hrftn96"
        command = [
            compiler,
            "-O2",
            "-std=legacy",
            "-frecord-marker=4",
            *[str(p.resolve()) for p in _cps_sources(root)],
            "-o",
            str(target),
        ]
        completed = subprocess.run(
            command, cwd=temporary, text=True, capture_output=True, check=False, timeout=180
        )
        if completed.returncode != 0 or not target.is_file():
            raise CPSForwardError(f"Could not build CPS hrftn96.\n{completed.stdout}\n{completed.stderr}")
        os.replace(target, executable)
    return executable


def _forward_receiver_function(
    model: LayerCakeModel,
    ray_parameter_s_per_km: float,
    *,
    incident_phase: str,
    sampling_interval_s: float = 0.1,
    npts: int = 512,
    delay_s: float = 5.0,
    gaussian_alpha: float = 1.0,
    positive_lag_sp_convention: bool = False,
    cps_root: str | Path | None = None,
) -> CPSReceiverFunction:
    """Predict a Ps or Sp RF for a proposed layer-cake model using CPS.

    ``gaussian_alpha`` follows CPS's ``-ALP`` convention, where the Gaussian
    filter's approximate corner frequency is ``alpha / pi``. ``npts`` is
    rounded up internally by CPS to a power of two.
    """
    phase = incident_phase.upper()
    if phase not in {"P", "S"}:
        raise ValueError("incident_phase must be 'P' or 'S'.")
    if positive_lag_sp_convention and phase != "S":
        raise ValueError("positive_lag_sp_convention is only valid for incident S.")
    if ray_parameter_s_per_km <= 0.0:
        raise ValueError("ray_parameter_s_per_km must be positive.")
    if sampling_interval_s <= 0.0 or npts < 4 or npts > 8192 or delay_s < 0.0 or gaussian_alpha <= 0.0:
        raise ValueError(
            "Require positive dt, 4 <= npts <= 8192, non-negative delay, and positive gaussian_alpha."
        )
    if phase == "P" and any(ray_parameter_s_per_km * layer.vp_km_s >= 1.0 for layer in model.layers):
        raise ValueError("Incident P must propagate in every layer; CPS would silently truncate this model.")

    executable = build_hrftn96(cps_root)
    with tempfile.TemporaryDirectory(prefix="bayesian-rf-cps-") as directory:
        workdir = Path(directory)
        model_path = model.write_model96(workdir / "model.mod")
        command = [
            str(executable),
            f"-{phase}",
            "-2",
            "-D",
            str(delay_s),
            "-DT",
            str(sampling_interval_s),
            "-NSAMP",
            str(npts),
            "-RAYP",
            str(ray_parameter_s_per_km),
            "-ALP",
            str(gaussian_alpha),
            "-M",
            # CPS stores command-line arguments in a 50-character buffer.
            # The working directory is isolated, so a local filename avoids
            # truncating otherwise valid temporary paths.
            model_path.name,
        ]
        completed = subprocess.run(
            command, cwd=workdir, text=True, capture_output=True, check=False, timeout=60
        )
        output_path = workdir / "hrftn96.sac"
        if completed.returncode != 0 or not output_path.is_file():
            raise CPSForwardError(
                "CPS hrftn96 failed to create a receiver function.\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        trace = read(str(output_path))[0]
        begin_time = float(getattr(trace.stats.sac, "b", -delay_s))
        lag_times = trace.times().astype(np.float64) + begin_time
        amplitudes = np.asarray(trace.data, dtype=np.float64)

    if positive_lag_sp_convention:
        # Common Sp plotting convention: reverse time and polarity so
        # conversions preceding S are displayed at positive lag.
        lag_times = -lag_times[::-1]
        amplitudes = -amplitudes[::-1]

    return CPSReceiverFunction(
        lag_times_s=lag_times,
        amplitudes=amplitudes,
        sampling_interval_s=float(trace.stats.delta),
        ray_parameter_s_per_km=ray_parameter_s_per_km,
        gaussian_alpha=gaussian_alpha,
        model=model,
        incident_phase=phase,
        positive_lag_sp_convention=positive_lag_sp_convention,
    )


def forward_ps_receiver_function(
    model: LayerCakeModel,
    ray_parameter_s_per_km: float,
    *,
    sampling_interval_s: float = 0.1,
    npts: int = 512,
    delay_s: float = 5.0,
    gaussian_alpha: float = 1.0,
    cps_root: str | Path | None = None,
) -> CPSReceiverFunction:
    """Predict a P-to-S RF for a proposed layer-cake model using CPS."""
    return _forward_receiver_function(
        model,
        ray_parameter_s_per_km,
        incident_phase="P",
        sampling_interval_s=sampling_interval_s,
        npts=npts,
        delay_s=delay_s,
        gaussian_alpha=gaussian_alpha,
        cps_root=cps_root,
    )


def forward_sp_receiver_function(
    model: LayerCakeModel,
    ray_parameter_s_per_km: float,
    *,
    sampling_interval_s: float = 0.1,
    npts: int = 512,
    delay_s: float = 15.0,
    gaussian_alpha: float = 1.0,
    positive_lag_convention: bool = False,
    cps_root: str | Path | None = None,
) -> CPSReceiverFunction:
    """Predict an S-to-P RF for a proposed layer-cake model using CPS.

    CPS returns Sp conversions at negative lag because the converted P arrives
    before the parent S. With ``positive_lag_convention=True``, both time and
    amplitude are reversed to match the convention used by
    ``data/bayesian_marsquake_rfs/Srf_data.txt``.
    """
    return _forward_receiver_function(
        model,
        ray_parameter_s_per_km,
        incident_phase="S",
        sampling_interval_s=sampling_interval_s,
        npts=npts,
        delay_s=delay_s,
        gaussian_alpha=gaussian_alpha,
        positive_lag_sp_convention=positive_lag_convention,
        cps_root=cps_root,
    )
