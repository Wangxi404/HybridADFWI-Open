# ADFWI vs HybridADFWI

Result: PASS

Same Marmousi-II model, native FD8 forward, float64 states and raw unprocessed kernels.

| Method | Mean seconds | Peak process RSS (MiB) | Peak CUDA allocated (MiB) |
|---|---:|---:|---:|
| autograd | 1.939514 | 453.72 | N/A |
| hybrid | 1.377543 | 381.08 | N/A |

| Repeat | Forward bitwise equal | Parameter | Relative L2 kernel error | Cosine |
|---:|---|---|---:|---:|
| 0 | True | vp | 5.0612e-16 | 1.0000000000000002 |
| 0 | True | vs | 5.9750e-16 | 1.0 |
| 0 | True | rho | 7.4480e-16 | 1.0 |

Full protocol, individual phase times, hashes and limitations are in the JSON files.
CPU RSS includes the Python/PyTorch runtime; CUDA allocated/reserved are recorded separately.
The upstream float32-generated FD constants are shared. Both paths use the same external dtype-preserving padding adapter.
Warmup, observations and output storage are excluded. No optimizer, gradient clipping, normalization or illumination preconditioning.
