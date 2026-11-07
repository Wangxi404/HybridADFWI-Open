# Independent full-shot review evidence

These reports describe independent source, numerical-array and control-flow checks. Their individual scope and limitations remain authoritative. Tiny analytic control-flow tests do not substitute for actual wave-solver derivative checks, and an accepted update does not certify completion or geological recovery.

Public JSON paths are logical identifiers. `<TEMP>` denotes an ephemeral diagnostic directory; `env:NAME/site-packages/...` retains only the environment and relative module path. They are not literal executable paths. Original private report bytes are retained locally under the Git-ignored `.cache/private-review/multishot_fwi_independent_review/`. Every public report records `original_private_report_sha256`; original numerical values and numerical-source SHA fields are unchanged. Portable scripts have their own current hash in `portable_public_reproducer`, while existing original-reader hash fields still identify the originally executed script.

The scripts infer their repository root from their installed location and accept `--repo-root`. Output defaults to a new system temporary directory; `--output` must name a new directory. The two loop tests also accept `--skill-root`, which must point to `skills/differentiate-seismic-solver` in a SeismicDiffSkill checkout. They use NumPy and CPU PyTorch; the policy test additionally uses psutil and a POSIX process-group API. No GPU is needed.

The path/CLI portability edits were checked by syntax parsing and `--help`; the historical numerical reports were not regenerated. The newly requested process-group policy check was actually rerun with harmless owned sleep sessions. No production solver, GPU task, numerical environment, or frozen numerical source was changed by publication cleanup.

From the repository root:

```sh
python evidence/multishot_fwi_independent_review/independent_hybrid_full_loop.py --repo-root . --skill-root ../SeismicDiffSkill/skills/differentiate-seismic-solver
python evidence/multishot_fwi_independent_review/independent_foreign_gpu_policy.py --repo-root .
```

The loop test replaces only its forward/gradient engine with a small analytic CPU oracle to exercise the actual optimizer orchestration. The policy test creates two inert sleep sessions and signals only those sessions. `foreign_gpu_policy_receipt.json` retains the historical process-only check; `foreign_gpu_group_policy_receipt.json` verifies the later `start_new_session=True` / own-group termination implementation.

`audit_hybrid_actual_gradient.py --repo-root .` reads only the completed first nine-shot gradient keys, map, fixed priors and metadata. `hybrid_initial9_gradient_audit.json` verifies the 2+2+2+2+1 batch SUM, one global /9, actual fixed-field hashes and the P/1000 coordinate chain. It does not load pressure/window payload or certify the still-pending full-model FD, Armijo update or inversion completion. Its valid open ledger prefix is explicitly recorded as INCOMPLETE.

`audit_diagnostic_cache_import.py --repo-root .` tests strict import guards and producer preservation with tiny synthetic artifacts. `audit_transition_bindings.py --hybrid-root .` independently checks the actual v2-to-v3 lineage, all original and consumer hashes, retained failed gate and unchanged cached gradient arrays. Imported blocks are explicitly old executions; they do not qualify the new quarter-step FD or an accepted update. The original coarse FD failure remains a failure; the two original errors do not establish formal second-order convergence. These metadata tests additionally need pytest and CPU PyTorch.

`audit_hybrid_fd_summaries.py --repo-root .` independently recomputes the two refined all-nine scalar FD quotients from cached per-shot loss summaries and the audited gradient, and checks same-point/execution provenance. `hybrid_v3_refined_fd_summary_audit.json` records both finite-h checks passing the original 1e-5 tolerance, while explicitly recording nonmonotone errors and no formal convergence-order claim. Pressure-to-objective recomputation and the accepted update are separate scopes.

`audit_hybrid_first_accepted.py --repo-root .` verifies the immutable first accepted q/Vp/actual-update arrays, physical projection/cap, global-gradient Armijo calculation, full-shot ledger ordering and both refined FD pairs before acceptance. It also directly compares exactly two same-q1 archived B2/B9 pressure traces as uint64; no all-nine, initial-point, unarchived GPU wavefield/cotensor or completed-inversion claim is made. Models and pressure remain private; only numeric summaries and hashes are published in `hybrid_v3_first_accepted_audit.json`.
