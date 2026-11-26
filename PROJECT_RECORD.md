# Multi Position Room Equalization Project Record

This CSE 5403 project tested whether one six-filter parametric equalizer could improve measured room responses at unseen listening positions, and whether a smooth worst-seat objective or a custom generalized Gauss–Newton solver offered an advantage. Shared EQ substantially reduced error on the spatial holdout. Gauss–Newton reached stationarity in fewer calls in the matched benchmark. The tested robust objective helped worst-seat training error more often than worst-seat holdout error.

## Original plan and completed work

The [project specification](PROJECT_SPEC.md) gives the original research plan, mathematical formulation, and acceptance criteria. The implemented scope used MeshRIR as the measured benchmark and covered all of the following experiments.

| Planned work | Executed protocol | Evidence |
|---|---|---|
| Validate measured data and a basic EQ fit | Fit six peaking filters to nine positions and score twelve disjoint positions on the same plane. | [Smoke summary](results/smoke/summary.json), [response plot](results/smoke/responses.png) |
| Compare custom and reference solvers | Compare projected damped generalized Gauss–Newton and SciPy L-BFGS-B on mean and robust objectives, using five matched starts, identical box bounds, and a projected-gradient tolerance of 0.0001. | [Solver summary](results/main/solver_summary.csv), [raw fits](results/main/runs.jsonl), [convergence plot](results/main/convergence.png) |
| Evaluate generalization from sparse measurements | Compare no EQ with single-seat, mean, and smooth worst-seat designs. Use three nested training selections at each of 3, 10, and 20 positions and a fixed 126-position holdout at least 25 cm from training. Select each fit by its training objective. | [Split manifest](results/main/manifest.json), [scores](results/main/scores.csv), [spatial map](results/main/spatial_map.png), [alpha sweep](results/main/tradeoff.png) |
| Test temperature continuation | Select a schedule on training-only pilot subsets, then compare it against direct fitting on the same five starts and final temperature under a shared evaluation cap. | [Pilot ranking](results/ablation/schedule_ranking.csv), [stage traces](results/ablation/continuation_runs.jsonl) |
| Test peak-informed initialization | Use the median training response to place cuts at separated positive peaks. Compare one peak start with five random-center starts that share its gains and Q, plus a near-flat reference. Run 70 fits across the matched split and nine spatial training splits. | [Per-split summary](results/initialization/summary.csv), [raw fits](results/initialization/runs.jsonl) |
| Deliver the report and presentation | Build the 11-page technical report and 12-slide deck with 17 minutes of timed notes. Verify the submission package in a freshly extracted workspace. | [Report](report/room_equalization.pdf), [slides](presentation/room_equalization_slides.pdf), [speaker notes](presentation/speaker_notes.txt) |

The optional second dataset, wider frequency bands, additional filter counts, exact-max relaxation, and listening demonstration remain available as extensions.

## Data and analysis conventions

- MeshRIR S1-M3969 release 1.2 contains 3,969 single-source measurements at 48 kHz. The experiments use the middle horizontal plane. The local archive is `data/raw/S1-M3969_npy.zip` and its published MD5 is `2cb598eb44bb9905560c545db7af3432`. The loader reads measurements directly from the ZIP. [Dataset release and attribution](https://zenodo.org/records/10852693)
- The model uses six peaking biquads over 50–500 Hz and 128 uniformly weighted log-frequency samples. Gains are bounded to −12 to +6 dB per filter and Q to 0.3–10. The primary robust setting uses `alpha = 0.5` and `tau = 2 dB²`. [Smoke configuration](configs/smoke.yaml), [main configuration](configs/main.yaml)
- The matched benchmark uses a common reference from its nine training positions. The spatial experiment uses the center receiver as a common reference across every training count. That receiver also supplies the single-seat baseline. Pooled RMSE weights squared error across seats and frequencies before taking the square root. Worst-seat RMSE is the largest individual seat RMSE.
- The shared correction preserves pointwise differences across seats. The unrestricted mean-error floor is the square root of the frequency-weighted spatial variance. For the matched training split, the floor was 3.614 dB and the selected L-BFGS-B mean fit reached 4.289 dB pooled RMSE. [Derivation and benchmark](report/room_equalization.pdf)

## Results

| Finding | Measured result |
|---|---|
| Smoke test | Held-out pooled RMSE fell from 6.85 to 4.62 dB and worst-seat RMSE from 8.13 to 6.03 dB. |
| Matched solver benchmark | All 20 fits reached the shared tolerance. Median joint objective and gradient calls for Gauss–Newton versus L-BFGS-B were 116 versus 399 for mean loss and 129 versus 353 for robust loss. L-BFGS-B reached the lowest observed objective for both losses. Gauss–Newton also computed curvature, so these are call counts for this benchmark. |
| Spatial holdout | With 20 training positions, median pooled and worst-seat RMSE across three selections were 7.73 and 9.02 dB without EQ, 4.93 and 6.17 dB for single-seat EQ, 5.10 and 6.32 dB for mean EQ, and 5.13 and 6.36 dB for robust EQ. |
| Robust versus mean EQ | At the primary setting, robust EQ reduced worst-seat RMSE in 7 of 9 training comparisons and 2 of 9 holdout comparisons. The paired median changes were −0.036 dB in training and +0.064 dB on the holdout. [Paired results](results/ablation/paired_spatial.csv) |
| Temperature continuation | The pilot selected 32 → 8 → 2 dB². Continuation lowered the final objective on 1 of 5 matched starts and used more calls on all five. Median calls were 233 for continuation and 129 for fixed temperature. All ten endpoints reached the shared tolerance. [Continuation results](results/ablation/continuation_runs.csv) |
| Peak-informed starts | Against the median random-center start on each spatial split, peak placement improved the initial objective on 9 of 9 splits and the final objective on 4 of 9. It used a median of 18 more evaluations in the paired comparison. All 70 fits reached stationarity. |

## Result audit and interpretation

- The audit recomputed saved objectives and acoustic metrics from the measured responses and filter parameters using SciPy SOS frequency responses. The reported values agreed within 0.0000001. All 16 tests passed. A fresh extraction also reproduced all ten checked result tables apart from wall-clock timings and rebuilt both PDFs.
- The 126 holdout positions are disjoint from the spatial training sets. Four positions overlap the **separate** matched solver benchmark. Receiver IDs 1788 and 2164 were in the solver training split. IDs 1779 and 1780 were in its test split. Rescoring the saved spatial filters over the remaining 122 positions left the primary robust result at 2 of 9 wins and a +0.064 dB median worst-seat change. Treat the published split as a holdout for the spatial fits. [Receiver IDs](results/main/manifest.json)
- Rescoring the saved filters on 1,024 frequency points preserved the primary robust result at 2 of 9 wins, with a +0.070 dB median worst-seat change. These differences describe one room, source, measurement plane, and fixed test block. The nine comparisons reuse test positions and use nested training draws.
- The largest combined boost among reported fits was +12.10 dB, even with the +6 dB per-filter bound. Combined boost is a diagnostic. Playback with the fitted filters requires attention to signal headroom. [Reported boosts](results/main/scores.csv)

## Reproduce and navigate

Use Python 3.13, uv, and curl. Start at the project root. The data command downloads about 996 MB on a fresh machine and verifies the published checksum. The reproduction command uses the verified local archive and writes the result tables, run traces, and figures under `results/`.

```powershell
uv sync --locked --python 3.13 --cache-dir .uv-cache
.\.venv\Scripts\room-eq-data.exe
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\reproduce.py
```

For the code, begin with [the biquad response and Jacobian](src/room_eq/biquad.py), [the objective and curvature](src/room_eq/objectives.py), and [the solvers](src/room_eq/solvers.py). The run entry points are [the main experiments](src/room_eq/experiments.py), [the continuation ablation](src/room_eq/ablation.py), and [the initialization ablation](src/room_eq/peak_ablation.py). The [README](README.md) includes the document build commands.
