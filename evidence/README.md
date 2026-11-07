# Experiment evidence

## Known-Vs Vp-only control, 2026-11-07

`gradient-detail-20261107-vponly` contains a fresh native AD/Hybrid pair with
only Vp blurred and trainable. True Vs and the designated constant-density
truth are fixed in the actual propagation. Both full-physical Vp and actual
coordinate gradients pass the original 10⁻⁸ gate. Independent field, pressure,
source and complete-gradient checks are included. `vponly_independent_review`
contains the pre-run array/material audit. See [VP_ONLY.md](../docs/VP_ONLY.md)
for the unchanged-target proof, higher initial loss, qualitative negative
result and single-run memory outcome. Numerical correctness does not imply
a finer geological image.

## Gradient-detail experiments, 2026-11-07

These are separate functional experiments, not replacements for the archived
time/memory benchmarks. See the [figure and parameter report](../docs/GRADIENT_DETAIL.md).

| Bundle | What it establishes |
|---|---|
| `gradient_detail_baseline_audit.json` | Why the old gradient display is dominated by water and edges |
| `pml-profile-control-20261107` | Single-factor fixed-profile control on the old nine-shot case |
| `gradient-detail-models-20261107` | Source checksums, ROI/preparation rules and bounded FD8 dispersion analysis; no input arrays |
| `gradient-detail-pilot-20261107` | Shallow pressure/VJP/static-coordinate numerical pilot |
| `gradient-detail-full-depth-central-20261107` | AD/Hybrid numerical agreement, **rejected** late-time forward-quality case |
| `gradient-detail-top-absorbing-control-20261107` | Unmuted trace control; 200 m PML still amplifies late arrivals |
| `gradient-detail-thick-pml-control-20261107` | Unmuted 600 m PML trace control over the declared 3.5 s record |
| `gradient-detail-reflection-candidate-20261107` | Verified declared-coordinate derivatives and quality screen; auxiliary full-physical gate failed, original `summary.passed=false` retained |

Public arrays contain derivatives, geometry and model-free receiver evidence.
Worker files containing evaluated model inputs are excluded. Each bundle keeps
its executed source/version and declared scope; passing a discrete derivative
check does not override a rejected forward-quality screen.

## Original benchmarks

`local-smoke-20261106` and `local-composite-20261106` validate the real, coarsened Marmousi-II startup case on CPU. The composite target includes a time-difference term. They are not the formal GPU performance comparison.

The Marmousi-II JSON sidecars record source URLs/checksums, units, crop, resampling and initialization. Raw data and complete local run outputs are ignored by Git. Formal GPU bundles include raw kernels for each repetition, exact run protocol, individual timings/memory, runtime versions, hashes and process-audit samples. The original run folders retain full traces and observations.

Prepared Marmousi-II `model_input.npz` copies are excluded from the release tree. They remain local research inputs and can be regenerated from the official AGL downloads; see [data provenance and reconstruction](../docs/DATA_PROVENANCE.md). Historical manifests preserve their original hashes. The project code license does not relicense these third-party inputs. Historical paths and host metadata are retained as measurement provenance, not portable replay paths.
