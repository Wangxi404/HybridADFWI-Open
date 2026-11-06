# ADFWI vs HybridADFWI

Result: PASS

Same Marmousi-II model, native FD8 forward, float64 states and raw unprocessed kernels.

| Method | Mean seconds | Peak process RSS (MiB) | Peak CUDA allocated (MiB) |
|---|---:|---:|---:|
| autograd | 2.133894 | 435.42 | N/A |
| hybrid | 1.448987 | 387.00 | N/A |

| Repeat | Forward bitwise equal | Parameter | Relative L2 kernel error | Cosine |
|---:|---|---|---:|---:|
| 0 | True | vp | 5.6032e-16 | 1.0 |
| 0 | True | vs | 7.1530e-16 | 1.0000000000000002 |
| 0 | True | rho | 9.0589e-16 | 1.0 |

Full protocol, individual phase times, hashes and limitations are in the JSON files.
CPU RSS includes the Python/PyTorch runtime; CUDA allocated/reserved are recorded separately.
The upstream float32-generated FD constants are shared. Both paths use the same external dtype-preserving padding adapter.
Warmup, observations and output storage are excluded. No optimizer, gradient clipping, normalization or illumination preconditioning.
