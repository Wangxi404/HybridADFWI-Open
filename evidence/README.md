# Experiment evidence

`local-smoke-20261106` and `local-composite-20261106` validate the real, coarsened Marmousi-II startup case on CPU. The composite target includes a time-difference term. They are not the formal GPU performance comparison.

The Marmousi-II JSON sidecars record source URLs/checksums, units, crop, resampling and initialization. Raw data and complete local run outputs are ignored by Git. Formal GPU bundles include raw kernels for each repetition, exact run protocol, individual timings/memory, runtime versions, hashes and process-audit samples. The original run folders retain full traces and observations.

Prepared Marmousi-II `model_input.npz` copies are excluded from the release tree. They remain local research inputs and can be regenerated from the official AGL downloads; see [data provenance and reconstruction](../docs/DATA_PROVENANCE.md). Historical manifests preserve their original hashes. The project code license does not relicense these third-party inputs. Historical paths and host metadata are retained as measurement provenance, not portable replay paths.
