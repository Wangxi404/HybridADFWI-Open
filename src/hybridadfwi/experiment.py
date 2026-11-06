"""Shared inputs for both differentiation strategies; no optimizer or filtering."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "559f581149343b1ed3dd2168caae595690bd601d"
COMPONENTS = ("txx", "tzz", "txz", "vx", "vz")


def verify_upstream():
    manifest = json.loads((ROOT / "upstream.lock.json").read_text())
    if manifest["commit"] != UPSTREAM_COMMIT:
        raise RuntimeError("Unexpected upstream commit lock")
    for relative, expected in manifest["files"].items():
        path = ROOT / "vendor" / "ADFWI" / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Upstream source differs from lock: " + relative)
    return manifest["commit"]


def load_upstream():
    vendor = ROOT / "vendor" / "ADFWI"
    if not (vendor / "ADFWI").is_dir():
        raise RuntimeError("Run git submodule update --init first")
    sys.path.insert(0, str(vendor))
    from ADFWI.model import IsotropicElasticModel
    from ADFWI.propagator import ElasticPropagator
    from ADFWI.survey import Source, Receiver, Survey
    return IsotropicElasticModel, ElasticPropagator, Source, Receiver, Survey


def build(config, truth=False, trainable=True, nt=None):
    Model, Propagator, Source, Receiver, Survey = load_upstream()
    with np.load(config["model"]) as f:
        key = "true" if truth else "initial"
        arrays = {p: np.array(f[p + "_" + key], dtype=np.float64) for p in ("vp", "vs", "rho")}
        dx, dz = float(f["dx"]), float(f["dz"])
        ox, oz = float(f["ox"]), float(f["oz"])
    nz, nx = arrays["vp"].shape
    if dx != dz:
        raise ValueError("This controlled preset requires equal dx/dz")
    nt = config["nt"] if nt is None else nt
    active = config["parameters"] if trainable else []
    model = Model(ox=ox, oz=oz, nx=nx, nz=nz, dx=dx, dz=dz, **arrays,
                  vp_grad="vp" in active, vs_grad="vs" in active, rho_grad="rho" in active,
                  auto_update_rho=False, auto_update_vp=False, free_surface=config["free_surface"],
                  abc_type="PML", nabc=config["nabc"], device=config["device"], dtype=torch.float64)
    src = Source(nt, config["dt"], config["f0"])
    rec = Receiver(nt, config["dt"])
    t = np.arange(nt, dtype=np.float64) * config["dt"]
    a = np.pi * config["f0"] * (t - 1.5 / config["f0"])
    wavelet = (1 - 2*a*a) * np.exp(-a*a)
    z = min(nz - 5, max(2, int(round(config["depth_m"] / dz))))
    xs = np.rint(np.linspace(8, nx - 9, config["shots"] + 2)[1:-1]).astype(int)
    for x in xs:
        src.add_source(int(x), z, wavelet, src_type="mt", src_mt=np.diag([1., 0., 1.]))
    for x in np.unique(np.rint(np.linspace(5, nx - 6, config["receivers"])).astype(int)):
        rec.add_receiver(int(x), z, "vz")
    prop = Propagator(model, Survey(src, rec), device=config["device"], dtype=torch.float64)
    if arrays["vp"].max() * config["dt"] / dx > .5:
        raise ValueError("Conservative CFL guard exceeded")
    return model, prop


def objective(predicted, observed, scale, dt, kind="l2"):
    """Fixed observed-data scale, identical differentiable objective in both arms."""
    residual = (predicted - observed) / scale
    value = .5 * residual.square().mean()
    if kind == "l2_time_difference":
        # A second data operation exercises AD-derived receiver sources.
        d = torch.diff(residual, dim=1)
        value = value + .05 * d.square().mean()
    elif kind != "l2":
        raise ValueError(kind)
    return value
