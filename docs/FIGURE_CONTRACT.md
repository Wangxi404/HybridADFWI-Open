# Scientific figure contract

The repository gallery answers six questions, using Python/matplotlib throughout. These are research-output illustrations, not inversion or resolution claims. All plotted fields and measurements come from the documented solver, model preparation, or archived experiment records.

| Plate | Scientific question and evidence roles | Data and display contract |
|---|---|---|
| Model | What did both methods receive? Three columns show true Vp/Vs/rho; the second row shows the actual prepared initial fields. | Paper grid, shared true/initial scale within each parameter; no display smoothing; constant benchmark density is shown honestly. |
| Kernels | Does the hybrid reverse reproduce native AD? Each parameter row shows native AD, hybrid, and their signed difference. | Archived 9-shot checkpoint repeat 0; first two columns share a symmetric 99.5th-percentile absolute scale and symlog normalization (linear core 0.001 × that limit); difference is divided by that parameter's raw native maximum, with its own explicitly labelled scale. All errors use unmodified arrays. All repeats remain in the performance plate. |
| Agreement | Does every material derivative agree in every repeat? Two protocol panels show all three parameter errors and all repeats. | Raw relative L2 errors; positive-only log scale; benchmark tolerance 1e−8. |
| Resources | What was measured, and how stable were the three repeats? Time and CUDA allocated memory are paired by repetition; kernel error labels report the maximum over all parameters and repetitions; a separate agreement plate shows every value. | Both archived paired protocols shown separately; all three repeats, means and sample SD, no significance test or pooled cross-protocol estimate. |
| Wavefields | What does the native solver actually propagate through this Marmousi crop? A 2×3 sequence shows native vertical velocity at fixed times. | A new single-shot CPU smoke-grid run, using existing native-history instrumentation; no reconstructed or synthetic wavefront. Shared signed amplitude scale and physical coordinates; display clipping is quantified. |
| Data derivative | What connects simulation and explicit adjoint? True-model observation, initial-model prediction, signed residual, and actual PyTorch data VJP. | Same CPU run and fixed observed RMS normalization; observations/predictions share one scale; residual and VJP have separate labelled scales. |

Export: PNG previews at 300 dpi plus PDF/SVG with editable text, 7 pt or larger main text and a 5 pt rendered glyph floor. Image panels use nearest-neighbour display. Panel labels, comparable final axes geometry, rendered PDF fonts and collisions are audited; QA artifacts live separately in `docs/figure-qa/`.

Marmousi-II attribution: Allied Geophysical Laboratories, University of Houston; Gary S. Martin, Robert Wiley and Kurt J. Marfurt. Figure outputs do not relicense the input models. The raw SEG-Y and prepared input arrays remain excluded from version control. Source URLs, model preparation metadata, file hashes, simulation configuration, clipping fractions and numerical summaries are recorded in the generated figure manifest.
