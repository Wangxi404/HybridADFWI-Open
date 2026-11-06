# ADFWI vs HybridADFWI

Result: PASS

Same Marmousi-II model, native FD8 forward, float64 states and raw unprocessed kernels.

Checkpointing: disabled in both arms; full AD tape versus original-forward history; zero replay

| Method | Mean seconds | Peak process RSS (MiB) | Peak CUDA allocated (MiB) |
|---|---:|---:|---:|
| autograd | 148.997716 | 2787.04 | 4269.69 |
| hybrid | 62.643152 | 829.43 | 2378.62 |

| Repeat | Forward bitwise equal | Parameter | Relative L2 kernel error | Cosine |
|---:|---|---|---:|---:|
| 0 | True | vp | 9.0554e-13 | 1.0 |
| 0 | True | vs | 5.8159e-14 | 0.9999999999999998 |
| 0 | True | rho | 2.6906e-13 | 1.0 |
| 1 | True | vp | 9.0580e-13 | 1.0 |
| 1 | True | vs | 5.8143e-14 | 0.9999999999999998 |
| 1 | True | rho | 2.6902e-13 | 1.0 |
| 2 | True | vp | 9.0569e-13 | 1.0 |
| 2 | True | vs | 5.8160e-14 | 0.9999999999999998 |
| 2 | True | rho | 2.6905e-13 | 1.0 |

Full protocol, individual phase times, hashes and limitations are in the JSON files.
CPU RSS includes the Python/PyTorch runtime; CUDA allocated/reserved are recorded separately.
The upstream float32-generated FD constants are shared. Both paths use the same external dtype-preserving padding adapter.
Warmup, observations and output storage are excluded. No optimizer, gradient clipping, normalization or illumination preconditioning.
