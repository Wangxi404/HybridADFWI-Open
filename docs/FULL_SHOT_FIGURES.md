# Full-shot inversion figure contract

The figures must answer whether the **actual accepted Vp trajectory** improves
the data fit and recovers structure from the blurred initial model. A visually
interesting derivative is not evidence of a recovered model.

| Output | Question | Evidence and scope |
| --- | --- | --- |
| `model_recovery` | What structure was recovered? | True, initial and current physical Vp use one colour scale. True-minus-initial, true-minus-current and the accumulated model change use one signed scale. |
| `gradient_direction_update` | What computation caused the latest change? | Raw all-shot coordinate derivatives at the first/last accepted-update anchors, optimizer direction and actual accepted Vp change are separate panels. Both derivatives share a declared linear P99 scale; no smoothing or depth gain. |
| `data_fit_and_model_error` | Does better fit coincide with better recovery? | Full-shot data objectives are normalized within each frequency stage. Vp RMSE is calculated offline on the fully active region, for every accepted model. |

Run the offline reader after at least one **accepted** full-shot update:

```bash
python scripts/plot_multishot_fwi.py \
  --run RUN_DIRECTORY \
  --truth-posthoc OBSERVATION_MODEL_NPZ \
  --output docs/figures/fullshot
```

The offline reader needs no PyTorch. A tested Python 3.9 visualization stack
is recorded in `requirements-fwi-visualization.txt`; it can use a separate
environment from the numerical worker. The job pipeline accepts
`--plot-python /path/to/visualization/python` for that interpreter.

This plotting program can read true Vp for evaluation. The optimizer cannot:
its training container physically omits unknown true Vp. True-model RMSE must
never select a step length, frequency-stage transition, epoch count or stop.
Known true Vs and designated density remain explicit priors.

Every iteration first evaluates all declared independent shots at the same
current model, accumulates batch **sums**, divides by the full shot count once,
then permits one accepted update. Full-shot line-search trials are provisional
models. Raw gradient panels use the shot **sum**; the optimizer uses its
declared mean. Changing frequency bands changes the objective: loss values
across stage boundaries are not directly comparable.

The derivative used for accepted update K is evaluated at model K−1. It is
not the gradient at the displayed current model K. Frequency stages are
labelled on the derivative panels; a common colour scale does not turn them
into derivatives of the same objective. The trajectory CSV also retains
post-hoc spatial-derivative errors/cosines to distinguish structural recovery
from a reduction in average velocity error.

The source manifest records array hashes, all scaling limits, saturated
fractions, active-region counts and per-accepted-model metrics. The only image
transform is the declared physical coordinate view. Numerical data are not
filtered to improve appearance. PNG is a 300 dpi repository preview; PDF and
SVG preserve editable text. Current report-sized plates are not certified as
a journal's final production dimensions.

Before publishing actual figures, retain the source preflight, rendered
alignment JSON (1.5 pt tolerance), final-PDF glyph audit and collision audit.
Inspect **every panel and the complete exported PDF**, including tick labels;
an alignment or glyph pass alone cannot detect all raster/vector clipping.
An incomplete run must be labelled with its actual accepted count and status.
An epoch cap is not evidence of convergence, and more epochs do not guarantee
geological resolution.
