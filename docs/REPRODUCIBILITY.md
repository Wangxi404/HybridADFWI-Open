# Reproducing HybridADFWI

HybridADFWI compares two ways to differentiate the **same ADFWI forward model**. Native ADFWI already uses automatic differentiation. HybridADFWI replaces the temporal AD graph with an external discrete adjoint, while retaining AD for the data objective and static material mapping. SPECFEM and AxiSEM3D adapters are separate projects.

## Supported configuration

| Item | Supported benchmark path |
| --- | --- |
| Physics | 2-D isotropic elastic PSV |
| Discretization | Pinned ADFWI FD8 staggered grid, split PML, free surface on/off |
| Precision | Float64 fields/materials/data/gradients; native float32-generated FD coefficients shared by both methods |
| Derivatives | First-order Vp, Vs and density; fixed source, geometry and damping |
| Memory modes | Segmented checkpoint replay or true full-history `--no-checkpoint` |
| Objective | Observed-RMS-normalized waveform L2, optionally a time-difference term |
| Devices | CPU; CUDA for NVIDIA measurements. MPS is not a benchmark device option. |

Higher-order derivatives, source/geometry gradients, anisotropy, other stencil orders, other boundary formulations and a full optimizer/inversion workflow are outside the validated scope. Module hooks require serial use of the same upstream modules in one process.

## Source installation and CPU verification

Use Python 3.9 or newer and a PyTorch installation appropriate for the target device. The archived GPU run used Python 3.9.21, PyTorch 2.6.0+cu118, NumPy 2.0.1 and SciPy 1.13.1. This records the measurement environment; the dependency ranges in `pyproject.toml` are not an exact environment lock.

```bash
git clone --recurse-submodules https://github.com/Wangxi404/HybridADFWI-Open.git
cd HybridADFWI-Open
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

The public repository is `HybridADFWI-Open`; the project and Python package remain HybridADFWI and `hybridadfwi`. Reproduction requires only the release checkout and its pinned upstream source; the original development history is not required. For an existing clone run `git submodule update --init --recursive`. Execute from the repository root. The pinned submodule and `upstream.lock.json` must remain available: the current package expects a source checkout, and a standalone wheel is not a self-contained solver distribution. The submodule contains upstream examples and may require substantial disk space.

The existing tests use small synthetic models and mocked GPU queries: **no external data, CUDA device or network is required**. They exercise bitwise forward agreement, all five receiver outputs, all three material gradients, multiple checkpoint counts, both free-surface settings, repeated receivers, an independent Vs directional difference, full-history/no-replay behavior, and cooperative resource monitoring.

`h5py`, `segyio` and `ipython` are included because the pinned upstream import path loads utilities/view modules that require them even for this benchmark. The larger upstream optional dependency collection is not needed for these tests.

## Marmousi-II CPU example

Review [the source and data terms](DATA_PROVENANCE.md), then prepare the smaller model locally:

```bash
python scripts/prepare_marmousi.py --preset smoke
python scripts/benchmark.py run --preset smoke --device cpu \
  --repeats 1 --threads 1 --output runs/smoke
```

`smoke` uses a 39×101 model at 90 m spacing and 192 steps. It checks startup and correctness; its timing is not the reported GPU result. Preparation downloads three checksum-verified compressed SEG-Y sources, crops/resamples them, and records units, model hash and smoothing provenance. Defaults preserve water cells and use constant density 2450 kg/m³. `--density-mode source` is a different model configuration and must be reported as such.

The preparation step reads the much larger source arrays, so its RAM use is not set by the small output grid. Keep several GiB of host memory available for preparation. `--help` describes local download/output overrides. Downloads and run outputs stay ignored by Git.

## Archived GPU configurations

The formal checkpointed comparison uses a 78×200 model at 45 m, 2500 time steps, nine shots, float64 and 50 segments on an RTX A5000. Prepare the model and launch only when resources are available:

```bash
python scripts/prepare_marmousi.py --preset paper
python scripts/run_when_idle.py --min-gpu-gib 18 --min-ram-gib 12 \
  --evidence results/resource_gate.json -- \
  python scripts/benchmark.py run --preset paper --device cuda \
  --repeats 3 --threads 1 --output runs/marmousi2-fd8-f64
```

The idle gate requires Linux, `nvidia-smi`, at least 18 GiB free GPU memory and 12 GiB available host RAM for these commands. These are launch thresholds, not universal memory requirements or reservations. Unknown GPU compute tasks block launch; concurrent CPU work can still alter wall-clock time. Each measurement uses a fresh process and includes per-worker resource-audit records. No existing tasks are stopped.

The matched no-checkpoint comparison uses **one shot in both methods**, retaining the original grid and 2500 steps:

```bash
python scripts/run_when_idle.py --min-gpu-gib 18 --min-ram-gib 12 \
  --evidence results/no-checkpoint-gate.json -- \
  python scripts/benchmark.py run --preset paper --shots 1 --device cuda \
  --no-checkpoint --repeats 3 --threads 1 --output runs/no-checkpoint-1shot
```

`--no-checkpoint` bypasses the checkpoint callback. Setting only `--segments 1` still invokes upstream checkpointing. At nine shots native AD ran out of memory on the A5000; that capacity attempt is not a completed runtime or gradient comparison. Its separate capacity utility is documented in the root README. Do not compare timing across different shot counts or CPU-load conditions.

## Outputs and interpretation

New run folders contain the protocol, `REPORT.md`, `summary.json`, per-worker JSON/NPZ/logs, input/source hashes and resource evidence. A valid comparison requires bitwise-equal forward `vz`, finite nonzero unprocessed kernels and relative L2 gradient error ≤1e-8. CUDA memory refers to PyTorch allocator peaks; CPU RSS is sampled and can miss brief peaks. Performance depends on hardware and contention.

Read the preserved [checkpointed report](RESULTS_20261106.md), [no-checkpoint report](NO_CHECKPOINT_RESULTS_20261106.md), [experiment protocol](EXPERIMENT.md) and [evidence guide](../evidence/README.md) before reusing results. Historical absolute paths in those records describe the original host; use fresh generated protocols for a new checkout. Model binaries excluded from the release can be regenerated as described in [data provenance](DATA_PROVENANCE.md).
