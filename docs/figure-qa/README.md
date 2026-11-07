# Figure QA

Final scientific panels are in [the gallery](../VISUAL_REPORT.md). This folder contains only layout and render diagnostics, not alternate scientific outputs.

All six final plates passed the rendered plot-area alignment gate at 1.5 pt tolerance, the PDF text audit at a 5 pt floor (actual minimum 7 pt), and the rendered collision audit with **zero failures and zero warnings**. [summary.json](summary.json) records hashes of the audited final PDFs and PNGs. Each `*.alignment.json` contains measured axes rectangles and group comparisons; `*.alignment.svg` only visualizes that geometry. `*.collision.json` and `*.text.json` refer to the final PDFs. Clean collision audits do not emit marked PDFs.

The source preflight has no failures. Its three reviewed warnings reflect this delivery contract: PNG previews rather than TIFF submission masters, 300 dpi rather than the skill's 600 dpi default, and report/gallery page widths rather than fixed 89/183 mm journal columns. The figures must be relaid out and rechecked before use at a smaller journal width. PDF and SVG text remains editable.

The native CPU illustration validated bitwise receiver equality between the existing instrumented history loop and the pinned uninstrumented native loop. The standard CPU suite passed **21 tests** (17 upstream/dependency warnings) in 5.89 s; numerical core and vendor source were unchanged. No archived GPU benchmark was rerun.

The final PNGs were inspected panel by panel for signed-scale consistency, physical coordinates, clipping disclosure, readable text, source/receiver locations, complete data extent, and footer clearance. The images are nearest-neighbour displays of actual model/grid values; the receiver gathers use actual rounded receiver positions. No gradient smoothing is applied. Repeat and color-scale definitions are in the [visual report](../VISUAL_REPORT.md) and [manifest](../figures/manifest.json).

| Figure / panel | Evidence role | Center / spread | Replicate unit | Labels and scale | Alignment / collisions / visual review |
| --- | --- | --- | --- | --- | --- |
| models / a | true Vp | deterministic field; no interval applicable | one actual model/CPU shot | shared true/initial per parameter | PASS / PASS / inspected |
| models / b | true Vs | deterministic field; no interval applicable | one actual model/CPU shot | shared true/initial per parameter | PASS / PASS / inspected |
| models / c | true constant density | deterministic field; no interval applicable | one actual model/CPU shot | shared true/initial per parameter | PASS / PASS / inspected |
| models / d | actual smoothed initial Vp | deterministic field; no interval applicable | one actual model/CPU shot | shared true/initial per parameter | PASS / PASS / inspected |
| models / e | actual smoothed initial Vs | deterministic field; no interval applicable | one actual model/CPU shot | shared true/initial per parameter | PASS / PASS / inspected |
| models / f | actual constant initial density | deterministic field; no interval applicable | one actual model/CPU shot | shared true/initial per parameter | PASS / PASS / inspected |
| wavefields / a | native pre-update v_z at 0.384 s | deterministic field; no interval applicable | one actual model/CPU shot | fixed common signed scale | PASS / PASS / inspected |
| wavefields / b | native pre-update v_z at 0.640 s | deterministic field; no interval applicable | one actual model/CPU shot | fixed common signed scale | PASS / PASS / inspected |
| wavefields / c | native pre-update v_z at 0.960 s | deterministic field; no interval applicable | one actual model/CPU shot | fixed common signed scale | PASS / PASS / inspected |
| wavefields / d | native pre-update v_z at 1.280 s | deterministic field; no interval applicable | one actual model/CPU shot | fixed common signed scale | PASS / PASS / inspected |
| wavefields / e | native pre-update v_z at 1.792 s | deterministic field; no interval applicable | one actual model/CPU shot | fixed common signed scale | PASS / PASS / inspected |
| wavefields / f | native pre-update v_z at 2.304 s | deterministic field; no interval applicable | one actual model/CPU shot | fixed common signed scale | PASS / PASS / inspected |
| data_adjoint_source / a | true-model receiver observation | deterministic field; no interval applicable | one actual model/CPU shot | labelled in panel / report | PASS / PASS / inspected |
| data_adjoint_source / b | initial-model receiver prediction | deterministic field; no interval applicable | one actual model/CPU shot | labelled in panel / report | PASS / PASS / inspected |
| data_adjoint_source / c | signed data residual | deterministic field; no interval applicable | one actual model/CPU shot | labelled in panel / report | PASS / PASS / inspected |
| data_adjoint_source / d | actual AD receiver VJP | deterministic field; no interval applicable | one actual model/CPU shot | labelled in panel / report | PASS / PASS / inspected |
| gradients / a | Vp native raw gradient | deterministic field; no interval applicable | archived repeat 0 | shared within row; symlog AD/Hybrid, linear differences | PASS / PASS / inspected |
| gradients / b | Vp hybrid raw gradient | deterministic field; no interval applicable | archived repeat 0 | shared within row; symlog AD/Hybrid, linear differences | PASS / PASS / inspected |
| gradients / c | Vp raw signed method difference | deterministic field; no interval applicable | archived repeat 0 | shared within row; symlog AD/Hybrid, linear differences | PASS / PASS / inspected |
| gradients / d | Vs native raw gradient | deterministic field; no interval applicable | archived repeat 0 | shared within row; symlog AD/Hybrid, linear differences | PASS / PASS / inspected |
| gradients / e | Vs hybrid raw gradient | deterministic field; no interval applicable | archived repeat 0 | shared within row; symlog AD/Hybrid, linear differences | PASS / PASS / inspected |
| gradients / f | Vs raw signed method difference | deterministic field; no interval applicable | archived repeat 0 | shared within row; symlog AD/Hybrid, linear differences | PASS / PASS / inspected |
| gradients / g | rho native raw gradient | deterministic field; no interval applicable | archived repeat 0 | shared within row; symlog AD/Hybrid, linear differences | PASS / PASS / inspected |
| gradients / h | rho hybrid raw gradient | deterministic field; no interval applicable | archived repeat 0 | shared within row; symlog AD/Hybrid, linear differences | PASS / PASS / inspected |
| gradients / i | rho raw signed method difference | deterministic field; no interval applicable | archived repeat 0 | shared within row; symlog AD/Hybrid, linear differences | PASS / PASS / inspected |
| agreement / a | all checkpointed parameter errors | all raw repeat values; no aggregate spread | three fresh-process repeats | labelled in panel / report | PASS / PASS / inspected |
| agreement / b | all no-checkpoint parameter errors | all raw repeat values; no aggregate spread | three fresh-process repeats | labelled in panel / report | PASS / PASS / inspected |
| performance / a | checkpointed complete gradient time | raw; mean ± sample SD also shown | three fresh-process repeats | labelled in panel / report | PASS / PASS / inspected |
| performance / b | no-checkpoint complete gradient time | raw; mean ± sample SD also shown | three fresh-process repeats | labelled in panel / report | PASS / PASS / inspected |
| performance / c | checkpointed allocated memory | raw; mean ± sample SD also shown | three fresh-process repeats | labelled in panel / report | PASS / PASS / inspected |
| performance / d | no-checkpoint allocated memory | raw; mean ± sample SD also shown | three fresh-process repeats | labelled in panel / report | PASS / PASS / inspected |
