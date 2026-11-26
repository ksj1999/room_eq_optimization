# Robust Multi-Position Parametric Room Equalization

This document records the original research plan. The [project record](PROJECT_RECORD.md) describes the completed implementation, experiments, and results.

## Project objective

Design one shared parametric audio equalizer for multiple listening positions as a reproducible Python research project for CSE 5403 (Algorithms for Nonlinear Optimization). Implement a custom constrained generalized Gauss–Newton solver and evaluate accuracy, robustness, convergence, and spatial generalization using measured room impulse responses (RIRs). Deliver the code, tests, measured results, figures, and technical report.

## Context and contribution
Multi-position room correction, parametric IIR filters, peak-based initialization, and smooth maximum objectives are established techniques. The proposed course-level contribution is their rigorous integration: generalized Gauss–Newton for a mean/worst-seat objective, temperature continuation, theoretical lower bounds, and spatial generalization from sparse measurements. Treat improvements as hypotheses, not established findings.

## Data
- Primary spatial benchmark: MeshRIR single-source dataset, reported 3,969 positions on a 5 cm grid, 48 kHz, approximately 1 GB. Repository: https://github.com/sh01k/MeshRIR ; paper: https://arxiv.org/abs/2106.10801 . Verify actual download link, archive structure, coordinates, sampling rate and license from repository before use.
- Secondary controlled benchmark: Arni, Aalto Acoustic Labs, https://doi.org/10.5281/zenodo.6985104 . Proposal reports 132,037 RIRs, 5,342 absorption configurations, five receivers and repeated sweeps, CC BY 4.0. Archives are large; verify sizes and download only one archive initially. Use 10–20 configurations if practical. These are configurations of one room, not independent rooms.
- Optional development alternative: dEchorate, https://sofacoustics.org/data/database/dechorate/ ; https://doi.org/10.1186/s13636-021-00229-0 . Verify source/receiver mapping and choose spatially separated receivers with a fixed source.
- Never mix different source positions into one shared-EQ problem. Record exact source files, checksums where practical, license, receiver coordinates and split IDs. Cache derived spectra; do not commit large raw data.
- Start with synthetic RIRs for unit tests only; clearly label synthetic outputs. Real measured data is required for final scientific results.

## Mathematical model
For seat m and frequency bin j, let h[m,j] = 20 log10(max(|FFT(RIR_m)[j]|, epsilon)). Let T[j] be a fixed target in dB, and u[j;theta] the shared EQ magnitude in dB. Corrected response is h+u. Use a cascade of K stable peaking biquads, with parameters theta_k=(gain_dB, log(center_frequency_Hz), log(Q)). Compute u as the sum of individual filter dB responses. Use standard RBJ peaking EQ coefficients and verify against scipy.signal.sosfreqz.

Use normalized positive frequency weights w_j (sum = 1). Residual r_m[j] = sqrt(w_j)*(h[m,j]+u[j]-T[j]); seat loss e_m = 0.5*||r_m||². Report seat RMSE = sqrt(2*e_m). Use a common level reference derived from training data; do not independently normalize seats unless explicitly studying shape-only correction. Document smoothing, FFT length, window, epsilon and target. Use log-spaced bins; avoid double-counting perceptual weighting.

Objectives:
1. No EQ: u=0.
2. Single seat: minimize e_center + lambda R.
3. Mean: minimize mean(e_m) + lambda R.
4. Robust: F = alpha*mean(e_m) + (1-alpha)*tau*log(mean(exp(e_m/tau))) + lambda R.

Compute log-sum-exp stably using scipy.special.logsumexp. alpha in [0,1]; tau > 0 in squared-dB loss units. Use R = 0.5*sum(gain_k²) initially; optionally add a documented smooth boost penalty. Distinguish hard constraints from penalties.

## Generalized Gauss–Newton derivation
Let J_m = derivative of r_m with respect to theta, a_m = J_m.T @ r_m, p = softmax(e/tau), q_m = alpha/M + (1-alpha)*p_m.

Gradient: g = sum_m q_m*a_m + lambda*grad(R).

Approximate Hessian: B = sum_m q_m*J_m.T@J_m + ((1-alpha)/tau)*Cov_p(a_m) + lambda*Hessian(R), where Cov_p(a) = sum p_m*a_m*a_m.T - mean_p(a)*mean_p(a). This drops second derivatives of residuals while retaining outer-loss curvature. For the quadratic gain regularizer the Hessian is PSD. The residual Jacobian is identical across seats when weights and shared EQ are identical; exploit this if useful.

Solve a damped bound-constrained quadratic subproblem or use projected damped steps with sufficient-decrease line search. Do not assume clipping an unconstrained direction guarantees descent. Provide a projected-gradient fallback, adaptive damping, stopping criteria based on projected gradient and objective change, iteration/evaluation budgets, and failure diagnostics. Never explicitly invert B. Verify derivatives with central finite differences. Record accepted steps, damping and projected gradient norms.

## Constraints and initialization
- Initial band: 50–500 Hz; extend toward 16 kHz only after validation and below Nyquist.
- K=6 nominal; K=2,4,8 optional ablation.
- Gain bounds [-12,+6] dB (original proposal allowed +12; conservative revision).
- Q bounds [0.3,10]. Center frequencies within chosen band.
- Monitor maximum combined EQ boost on a dense frequency grid. If imposing a hard combined boost cap, use a solver supporting that constraint and enforce it consistently across comparisons; otherwise explicitly report it as a penalty/diagnostic, not a hard constraint.
- Peak initialization: detect persistent positive deviations from target across training seats, choose separated peaks, moderate Q and conservative negative nonzero gains. Compare seeded random starts and near-flat starts with nonzero gains. Exact zero gain makes frequency/Q derivatives vanish.
- Temperature continuation: start with a larger tau and decrease geometrically to a fixed final tau; warm-start each stage. Compare with direct optimization at the same final tau under matched total evaluation budgets. Choose schedules from training-only pilot runs.

## Theoretical benchmarks
For shared EQ, corrected differences between seats equal original differences at every frequency. Do not claim reduction of pointwise spatial variance. Shared EQ can reduce errors relative to a target, not eliminate spatial differences.

For unrestricted per-bin correction and unregularized mean squared error, u* = T - mean_m(h_m). The minimum mean squared error is sum_j w_j*Var_m(h_m[j]). This is a lower bound on practical mean-error EQ performance.

For the exact mean/max objective, solve the convex epigraph relaxation over unrestricted u and scalar t: minimize alpha*mean(e_m(u))+(1-alpha)*t subject to e_m(u)<=t for all m. Optionally use CVXPY. Compare only like-for-like objectives; do not call the exact-max bound an optimality gap for a different smooth/regularized objective without justification. Report relaxation gap separately from solver convergence.

## Solvers and fair comparisons
- Custom damped Gauss–Newton for mean loss and generalized Gauss–Newton for robust loss.
- SciPy L-BFGS-B for the identical scalar objective and box bounds.
- SciPy least_squares(method='trf') for bounded mean least squares, with regularizer residuals if needed. It is not a direct baseline for the robust scalar objective; use another suitable scalar constrained solver for that objective if needed.
- Optional multistart and differential evolution on small cases. Multistart is a strategy, not a distinct local algorithm. Standard SciPy LM does not support box bounds.
- Match objective, bounds, starting points, stopping tolerances and budgets as closely as possible. Count function/Jacobian evaluations and wall time; disclose differences.

## Experiments in priority order
1. Real-data smoke test: one fixed source, several seats, no EQ versus mean EQ. Validate response and filters.
2. No EQ vs single-seat vs mean vs robust EQ on identical training/test splits. Sweep alpha (e.g. 0, .25, .5, .75, 1) with fixed tau; report actual maximum seat error, not only smooth surrogate.
3. Solver comparison: custom vs reference solvers, 5 seeded starts initially. Compare final matched objective, runtime, evaluations, projected gradient and failure rate.
4. Fixed temperature vs continuation; peak-informed vs random initialization. Separate ablations and match total budgets.
5. MeshRIR spatial generalization: choose one horizontal plane if verified, train on 1, 3, 5, 10, 20 positions (adjust as needed), test on a fixed disjoint set of unseen positions. Compare spatially spread vs random selection if time permits. Use multiple seeded splits and avoid tuning on test positions. Plot spatial error heatmaps and performance vs measurement count.
6. Optional Arni: leave-one-receiver-out across configurations; use repeated sweeps to assess actual measurement variability. Optional filter-count and noise sensitivity studies.

Report mean of seat RMSE and pooled RMSE separately, worst-seat RMSE, 95th percentile seat RMSE, maximum absolute deviation, maximum combined boost, runtime, evaluations, convergence failures, and lower-bound gap where valid. Report training and held-out metrics separately. Use paired comparisons, median/IQR across splits and configurations; avoid treating neighboring spatial points as independent rooms. Negative results are valid.

## Implementation structure
```text
README.md
pyproject.toml
configs/{smoke,main}.yaml
src/room_eq/{data,preprocess,biquad,objectives,initialization,metrics}.py
src/room_eq/solvers/{gauss_newton,baselines}.py
scripts/{download_data,run_experiments,make_figures,make_audio}.py
tests/
results/{tables,figures,configs}/
report/
```
Use Python, NumPy, SciPy, pandas, matplotlib, pytest; optional JAX for derivative verification, CVXPY for convex bounds, SoundFile for audio. CPU only. Pin dependencies, use deterministic seeds, expose CLI commands and log configuration and data provenance. Keep modules small and testable. No unnecessary neural networks or UI.

## Testing and acceptance criteria
- Biquad magnitude agrees with independent reference; zero gain is unity; poles inside unit circle; cascade agrees with SOS response.
- Analytical gradient/Jacobian agree with finite differences at representative nondegenerate points; generalized Hessian symmetric and PSD up to numerical tolerance for quadratic regularization.
- Stable log-sum-exp at extreme loss/temperature values; alpha=1 matches mean loss.
- Verify spatial variance invariance and analytical mean lower bound numerically.
- Solver respects bounds, decreases objective on accepted steps, and handles failed steps and zero gradients.
- At least one real dataset subset loads reproducibly; no train/test leakage.
- Produce actual before/after curves, convergence plots, mean/worst tradeoff, held-out results, and solver summary table. Save raw run records, not just aggregated figures.
- One documented command reproduces principal results from cached data. Report limitations and unsuccessful runs.

## Optional demonstration
Convolve a dry licensed or self-recorded audio clip with measured RIRs, with and without the optimized EQ. Use consistent playback level normalization and avoid clipping. Export center, worst and unseen-position examples. Audio is supplementary, not evidence of subjective listening preference. Optional Streamlit dashboard only after experiments are complete.

## Literature to verify and cite
- Cecchi et al., Room Response Equalization—A Review (2018), https://doi.org/10.3390/app8010016 .
- Vairetti et al., An Automatic Design Procedure for Low-order IIR Parametric Equalizers (2018), https://doi.org/10.17743/jaes.2018.0049 .
- Nercessian, Neural Parametric Equalizer Matching Using Differentiable Biquads (2020); locate original DAFx paper.
- Prawda, Schlecht, Välimäki, Arni dataset (2022), DOI above.
- MeshRIR and dEchorate dataset papers above.
- Search specifically for minimax/multi-position parametric equalization, robust IIR design, generalized Gauss–Newton and continuation. Verify bibliographic details and exact overlap before making novelty claims.

## Planned deliverables

The original proposal targeted an 8–12 page report, a 15–20 minute presentation, a Python package, and a reproducible results bundle. The planned sequence was dataset validation and a measured-data smoke test, followed by the mathematical core, solver comparison, spatial experiments, and ablations. Synthetic responses were reserved for unit tests. The [README](README.md) describes the current setup and reproduction commands.
