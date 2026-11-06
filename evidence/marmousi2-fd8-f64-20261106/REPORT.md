# ADFWI vs HybridADFWI

Result: PASS

Same Marmousi-II model, native FD8 forward, float64 states and raw unprocessed kernels.

| Method | Mean seconds | Peak process RSS (MiB) | Peak CUDA allocated (MiB) |
|---|---:|---:|---:|
| autograd | 144.623399 | 1045.34 | 2163.37 |
| hybrid | 54.010579 | 839.32 | 1940.81 |

| Repeat | Forward bitwise equal | Parameter | Relative L2 kernel error | Cosine |
|---:|---|---|---:|---:|
| 0 | True | vp | 5.0106e-12 | 1.0000000000000002 |
| 0 | True | vs | 3.5763e-12 | 1.0 |
| 0 | True | rho | 3.5689e-12 | 1.0 |
| 1 | True | vp | 5.0093e-12 | 1.0 |
| 1 | True | vs | 3.5763e-12 | 1.0 |
| 1 | True | rho | 3.5685e-12 | 1.0 |
| 2 | True | vp | 5.0092e-12 | 1.0 |
| 2 | True | vs | 3.5763e-12 | 1.0 |
| 2 | True | rho | 3.5684e-12 | 0.9999999999999998 |

Full protocol, individual phase times, hashes and limitations are in the JSON files.
CPU RSS includes the Python/PyTorch runtime; CUDA allocated/reserved are recorded separately.
The upstream float32-generated FD constants are shared. Both paths use the same external dtype-preserving padding adapter.
Warmup, observations and output storage are excluded. No optimizer, gradient clipping, normalization or illumination preconditioning.
