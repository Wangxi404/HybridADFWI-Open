"""Reproducible Marmousi-II models, following ADFWI's elastic example.

Original SEG-Y arrays use (x, z), velocities km/s and density g/cm³.
Prepared arrays use (z, x) and SI units, always NumPy float64.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


BASE_URL = "https://www.agl.uh.edu/downloads/"
SOURCE_HASHES = {
    "vp_marmousi-ii.segy.gz": "2c7a0c4170c95a15620c2146573711c61a4cef8dc27338ef6c211a81ffa68920",
    "vs_marmousi-ii.segy.gz": "c88bec68fc4bd7c6e624737a20ecd3241c90e10512103e28e61b344fb5baed7d",
    "density_marmousi-ii.segy.gz": "9832e20ce1f79470fa8c100e79d66566cf8c1c162baa4289f37e7ee0b96f1e67",
}
PRESETS = {
    "smoke": {"nx": 101, "nz": 39, "dx": 90.0, "dz": 90.0, "ox": 5000.0, "oz": 0.0},
    "paper": {"nx": 200, "nz": 78, "dx": 45.0, "dz": 45.0, "ox": 5000.0, "oz": 0.0},
}
EXAMPLE_URL = (
    "https://github.com/liufeng2317/ADFWI/blob/bv1.1/"
    "examples/elastic/Iso-elastic-Marmousi2-shotTop-recTop/01_forward.ipynb"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_sources(raw_dir: str | Path) -> dict:
    """Fetch the original files and enforce the recorded source checksums.

    Existing files are verified too. A changed remote archive is an explicit
    error rather than silently changing the experiment's model.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for filename, expected in SOURCE_HASHES.items():
        path = raw_dir / filename
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            request = urllib.request.Request(BASE_URL + filename, headers={"User-Agent": "HybridADFWI-model-preparation"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                if _sha256(temporary) != expected:
                    raise ValueError(f"Checksum mismatch for downloaded {filename}; source data changed")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {path}; expected {expected}, got {actual}")
        records[filename] = {"url": BASE_URL + filename, "sha256": actual, "bytes": path.stat().st_size}
    return records


def _read_resample(path: Path, z: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Use a bounded linear interpolator to avoid elastic/water overshoot."""
    import obspy

    traces = obspy.read(str(path), format="SEGY")
    raw_xz = np.stack([trace.data for trace in traces]).astype(np.float64)
    shape = raw_xz.shape
    # Extents and unit conversion are the same as ADFWI's velocityDemo loader.
    source_x = np.linspace(0.0, 17000.0, shape[0])
    source_z = np.linspace(0.0, 3500.0, shape[1])
    interpolator = RegularGridInterpolator((source_x, source_z), raw_xz, method="linear", bounds_error=True)
    zz, xx = np.meshgrid(z, x, indexing="ij")
    result_zx = interpolator(np.stack((xx, zz), axis=-1)) * 1000.0
    return np.ascontiguousarray(result_zx, dtype=np.float64), shape


def _smooth_solid(field: np.ndarray, water: np.ndarray, sigma_grid: tuple) -> np.ndarray:
    """Smooth below the water layer without averaging zero Vs into solids."""
    if np.all(field == field.flat[0]) or not any(sigma_grid):
        return field.copy()
    solid = (~water).astype(np.float64)
    weights = gaussian_filter(solid, sigma_grid, mode="nearest")
    smoothed = gaussian_filter(field * solid, sigma_grid, mode="nearest") / np.maximum(weights, np.finfo(np.float64).tiny)
    smoothed[water] = field[water]
    return np.ascontiguousarray(smoothed, dtype=np.float64)


def prepare_marmousi(
    output: str | Path,
    raw_dir: str | Path,
    preset: str = "smoke",
    smooth_sigma_m: float = 150.0,
    density_mode: str = "constant2450",
) -> Path:
    """Write true/initial fields and a provenance sidecar for a model preset.

    ``paper`` uses ADFWI's nominal 200×78, 45 m elastic Marmousi-II crop.
    ``smoke`` covers approximately the same crop on a 90 m grid; it is a
    correctness/startup dataset, not a production-resolution benchmark.
    Density defaults to the constant 2450 kg/m³ used in that ADFWI example.
    Use ``source`` to retain the original elastic density field instead.
    """
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; choose {tuple(PRESETS)}")
    if smooth_sigma_m < 0 or not np.isfinite(smooth_sigma_m):
        raise ValueError("smooth_sigma_m must be finite and nonnegative")
    if density_mode not in ("constant2450", "source"):
        raise ValueError("density_mode must be 'constant2450' or 'source'")
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("output must have a .npz extension")
    source_records = download_sources(raw_dir)
    grid = dict(PRESETS[preset])
    x = grid["ox"] + np.arange(grid["nx"], dtype=np.float64) * grid["dx"]
    z = grid["oz"] + np.arange(grid["nz"], dtype=np.float64) * grid["dz"]
    fields = {}
    raw_shapes = {}
    for key, prefix in (("vp", "vp"), ("vs", "vs"), ("rho", "density")):
        fields[key], raw_shapes[key] = _read_resample(Path(raw_dir) / f"{prefix}_marmousi-ii.segy.gz", z, x)
    if density_mode == "constant2450":
        fields["rho"] = np.full_like(fields["rho"], 2450.0)
    water = fields["vs"] == 0.0
    if not all(np.isfinite(value).all() for value in fields.values()):
        raise ValueError("Nonfinite model values")
    if np.any(fields["vp"] <= 0) or np.any(fields["vs"] < 0) or np.any(fields["rho"] <= 0):
        raise ValueError("Nonphysical Marmousi-II model values")
    if np.any(fields["vp"] ** 2 < 2.0 * fields["vs"] ** 2):
        raise ValueError("Marmousi-II crop has negative first Lamé parameter")
    sigma_grid = (smooth_sigma_m / grid["dz"], smooth_sigma_m / grid["dx"])
    prepared = {}
    for key, value in fields.items():
        prepared[key + "_true"] = value
        prepared[key + "_initial"] = _smooth_solid(value, water, sigma_grid)
    metadata = {
        "name": "Marmousi-II", "preset": preset,
        "upstream_example": EXAMPLE_URL, "sources": source_records,
        "source_array_axes": "x,z", "prepared_array_axes": "z,x",
        "source_shapes_xz": raw_shapes,
        "source_extent_m": {"x": [0, 17000], "z": [0, 3500]},
        "grid": grid, "x_extent_m": [float(x[0]), float(x[-1])], "z_extent_m": [float(z[0]), float(z[-1])],
        "dtype": "float64", "units": {"vp": "m/s", "vs": "m/s", "rho": "kg/m^3"},
        "source_to_SI_multiplier": 1000.0,
        "resampling": "bilinear on exact nominal grid; no extrapolation",
        "density_mode": density_mode, "initial_smoothing_sigma_m": smooth_sigma_m,
        "smoothing": "Gaussian normalized within solid; true water cells restored for all fields",
        "water_cells": int(water.sum()),
        "note": "ADFWI notebook samples linspace across nx*dx; here coordinates use (nx-1)*dx to match solver grid spacing exactly.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **prepared, water_mask=water, x=x, z=z, **grid,
                        metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    metadata["prepared_sha256"] = _sha256(output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def load_model(path: str | Path) -> dict:
    """Return ordinary arrays/scalars, with no live NPZ file handle."""
    with np.load(path, allow_pickle=False) as data:
        return {key: value.item() if value.ndim == 0 else value.copy() for key, value in data.items()}
