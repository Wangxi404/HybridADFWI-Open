# Experiment 01: ADFWI PSV vs HybridADFWI

Owner: Sid. Initial protocol: 2026-11-06.

Question: with the *same* existing high-performance forward code and discretization, what changes when temporal AD is replaced by an explicit discrete adjoint, while retaining data-domain AD?

## Controlled factors

Both arms use the pinned upstream ADFWI FD8 PML kernel, identical true/initial Marmousi-II models, waveform and moment tensor, geometry, free-surface mode, receiver component, loss, precision, active parameters and checkpoint segmentation. Vp, Vs and density kernels are reported before processing. No optimization step is included.

The identical external padding dtype correction is enabled in both arms. Native float32-generated FD constants are preserved, rather than silently replacing the baseline stencil. Thus “float64” describes materials, state, data and gradient storage/arithmetic, with this stated coefficient provenance.

`paper`: model (nz,nx)=(78,200), dx=dz=45m, nt=2500, dt=.003s, f0=5Hz, 9 simultaneous shots, 100 receivers, source/receiver depth 540m, PML20, 50 checkpoint segments, all Vp/Vs/rho active. Explosive in-plane moment tensor. Depth differs from the original surface-survey notebook and is explicitly fixed across arms. The first run tests raw gradient calculation, not inversion convergence.

`smoke`: real Marmousi-II resampled to (39,101),90m, nt192,dt.004s, one shot,25receivers,PML10. This is a software correctness and startup case; its timing is not the paper's production result.

The fixed scale in the normalized objective is the RMS of the common true-model observations, computed outside measurement. Gradients are with respect to physical SI-valued parameters. A second target adds a discrete time-difference term to exercise receiver-domain AD.

## Hybrid sequence

1. Keep static parameter-to-material transformations differentiable; they scale with the model, not time.
2. Run the original segmented propagation under no-grad and save segment boundary states.
3. Make receiver data a new leaf; compute data VJP from the chosen target.
4. Recompute each segment with original native steps under no-grad. Run the explicitly transposed updates and accumulate coefficient sensitivities.
5. Apply one static material-map VJP to return Vp/Vs/rho kernels.

The original source remains fixed. The data VJP is injected through the transpose of the native sampling operator with the native time convention; it is not simply another physical source waveform without these conventions.

### No-checkpoint extension

With `--no-checkpoint`, both arms bypass the checkpoint callback rather than merely setting one segment. Native AD records the complete temporal graph. Hybrid retains the original forward's five required field histories in five preallocated contiguous GPU tensors; its compiled native numerical loop is externally instrumented only to write these buffers. The vendor file is unchanged. Reverse consumes those histories directly, with zero checkpoint states, no recomputation and no CPU offload. Allocation and writes are part of measured forward time. Three short warmup evaluations exercise profiling before measurement.

The original nine-shot problem is attempted separately as a capacity test. CUDA OOM records a failed attempt and allocator peak, never a completed runtime or kernel. If needed, both arms reduce to the same one-shot survey while retaining the original grid, precision and full2500steps; this matched case yields the complete no-checkpoint runtime/memory/kernel comparison. Its times must not be compared directly to the prior nine-shot checkpointed experiment.

## Validation

Unit tests compare all five recorded components, multiple checkpoint counts, both free-surface modes, multiple shots, repeated receiver locations and all three parameter gradients with upstream AD. An independent Vs central difference tests model-space differentiation. The benchmark requires bitwise identical forward `vz` and nonzero, finite kernels with relative L2 error <=1e-8.

## Measurement

Each arm/repetition is a new process. Warm up the same grid with a short waveform, release that work, then build identical measured inputs. Alternate AB/BA. CUDA synchronization bounds each timing phase. Total includes forward, objective/data VJP, checkpoint replay, explicit or AD reverse and static parameter VJP; excludes initialization, data generation, warmup and storage. Run three paired repetitions and report individual values, mean and sample standard deviation.

CUDA: reset PyTorch peak statistics immediately before measurement; report allocated and reserved absolute peaks, baseline allocation and incremental peak. These are allocator metrics, not total `nvidia-smi` process memory. CPU: sample this worker's RSS every5ms only during measurement; report baseline, sampled absolute peak and increment. Sampling may miss short peaks. Host RAM and GPU RAM are separate metrics.

Only launch after GPU idle gate (three checks,5s apart), with >=18GiB free GPU and >=12GiB free host RAM. Unknown compute tasks block. Use one CPU thread and nice10; never stop or reprioritize other work. Record host load/affinity because concurrent CPU tasks can materially affect GPU wall-clock time. The gate is cooperative, not a scheduler reservation; later noncooperating jobs remain a potential source of interference.

Each child is also checked before launch and audited every3s by its parent. Unknown concurrent GPU compute or monitoring failure invalidates that measurement and stops later measurements after the current child finishes. Sampling cannot exclude sub-interval activity. CPU RSS can retain allocator memory from the method's excluded warmup; absolute peak and baseline should be read together, and the increment is not a pure propagation allocation measure.

## Artifacts and interpretation

Retain protocol, source/data hashes, runtime versions, resource-gate evidence, per-arm logs, raw traces/kernels, raw timings and report. A failing gradient check blocks performance claims. This first implementation supports only the pinned isotropic FD8 split-PML path and first-order model derivatives; it does not establish adaptation cost for arbitrary solvers, all ADFWI physics, or a completed inversion experiment.
