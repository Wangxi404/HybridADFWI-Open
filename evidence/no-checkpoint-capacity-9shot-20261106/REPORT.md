# Unchanged full-size no-checkpoint capacity test

Both arms bypass temporal checkpointing. Common inputs and reference observations are unchanged.

| Method | Status | Completed gradient seconds | Peak CUDA allocated (GiB) |
|---|---|---:|---:|
| autograd | cuda_oom | N/A (OOM) | 23.350554 |
| hybrid | ok | 62.348164591006935 | 20.829915 |

An OOM attempt has no completed gradient time or kernel; its failure elapsed time is only in the individual diagnostic JSON.
Use a matched smaller shot count in both arms for a complete no-checkpoint comparison if needed.
