# Step 51: pose-filter comparison (offline, logged raw poses)

Static run `20261027T033608-9db092df`, movement run `20261027T042028-839c66ed` (1007 frames, 731 valid, 30 source switches inside segments, 1 stops, 462 moving frames, 34 still). Array state yaw 0 / scale 1 in both runs (checked from `diagnostics.array`). Evaluation points: the 16 panel corners, projected with the run's K; scene distance d = 2.003 m (mean of the two runs' median camera-to-tag distances, 2.126 and 1.880).

- Acceptance check: offline `PoseSmoother` (EMA 0.4, the reset rule below) reproduces the logged `render_T_CW` with max abs difference 0.00e+00 (rad / m).
- Reset rule: first valid pose after an invalid frame, and dt > 0.2 s. Frames with no raw pose produce no output.
- Split: selection on the first 50% of the movement run (by time) plus the whole static run; movement numbers below are on the second 50%.
- Measured KF-fixed measurement noise (static run, raw minus 2 s moving median), sigma = [0.00267, 0.00189, 0.0026, 0.00077, 0.00103, 0.00087] (c in m, rotation vector in rad).
- KF-cov scale: mean NIS by s_R at q = 1e-05: {0.03: 6.39, 0.06: 7.35, 0.12: 6.74, 0.25: 4.16, 0.5: 2.18, 1: 1.13, 2: 0.58, 4: 0.30}; s_R = 0.25 (closest to 6).
- dt-aware EMA (w = 1 - exp(-dt/tau)) vs fixed-weight EMA: largest relative difference on J, E_dyn or lag over the weight grid 11.1% (reported separately in results.json). Median dt 93 ms.
- Lag: delay tau minimising sum ||p_f(t) - p_ref(t - tau)||^2 over moving frames (reference speed > 30 px/s), tau in -100..400 ms, 5 ms steps, parabola refined; frames whose whole tau range lies outside the segment are dropped. Positive = the filtered point is late.

## dt-aware EMA against the fixed-weight EMA (test half)

| w | tau s | J EMA | J EMA-dt | E_dyn EMA | E_dyn EMA-dt | lag EMA ms | lag EMA-dt ms |
|---|---|---|---|---|---|---|---|
| 0.2 | 0.416 | 0.279 | 0.283 | 63.33 | 61.27 | 271 | 256 |
| 0.4 | 0.182 | 0.337 | 0.351 | 31.92 | 30.33 | 127 | 121 |
| 0.6 | 0.101 | 0.412 | 0.429 | 15.84 | 14.87 | 58 | 54 |

Settings marked * lie outside the specified grids (KF q < 1e-4, 1 euro beta > 10); they were added because the best settings sat on the grid edges.
Test half of the movement run: 9 source switches and 1 stops, so the overshoot column rests on that many events.

## Equal jitter (static J <= EMA 0.4's 0.337 px; best E_dyn on the tuning half)

| filter | settings | static J px | rot std deg | pos std mm | E_dyn px | lag ms | motion noise px | overshoot px med/max | switch step px |
|---|---|---|---|---|---|---|---|---|---|
| raw | w=1 | 0.587 | 0.050 | 2.37 | 4.04 | 0 | 4.02 | 1.08 / 1.08 | 43.03 |
| EMA 0.4 | w=0.4 | 0.337 | 0.028 | 1.34 | 31.92 | 127 | 1.14 | 4.30 / 4.30 | 26.61 |
| EMA (best at equal J) | w=0.4 | 0.337 | 0.028 | 1.34 | 31.92 | 127 | 1.14 | 4.30 / 4.30 | 26.61 |
| 1euro | fmin=0.1, beta=10 | 0.336 | 0.028 | 1.37 | 4.94 | 12 | 3.45 | 3.43 / 3.43 | 40.35 |
| KF-fixed (lowest J; target not reached) | q=1e-05* | 0.411 | 0.042 | 1.80 | 10.04 | 5 | 6.76 | 1.65 / 1.65 | 49.47 |
| KF-cov (lowest J; target not reached) | q=1e-06, sr=0.25* | 0.359 | 0.032 | 1.45 | 10.83 | 3 | 6.89 | 6.77 / 6.77 | 55.55 |

## Equal lag (lag <= EMA 0.4's tuning-half lag 129 ms; lowest static J)

| filter | settings | static J px | rot std deg | pos std mm | E_dyn px | lag ms | motion noise px | overshoot px med/max | switch step px |
|---|---|---|---|---|---|---|---|---|---|
| raw | w=1 | 0.587 | 0.050 | 2.37 | 4.04 | 0 | 4.02 | 1.08 / 1.08 | 43.03 |
| EMA 0.4 | w=0.4 | 0.337 | 0.028 | 1.34 | 31.92 | 127 | 1.14 | 4.30 / 4.30 | 26.61 |
| EMA | w=0.4 | 0.337 | 0.028 | 1.34 | 31.92 | 127 | 1.14 | 4.30 / 4.30 | 26.61 |
| 1euro | fmin=0.3, beta=0.5 | 0.280 | 0.024 | 1.09 | 27.50 | 104 | 1.58 | 8.09 / 8.09 | 28.45 |
| KF-fixed | q=1e-05* | 0.411 | 0.042 | 1.80 | 10.04 | 5 | 6.76 | 1.65 / 1.65 | 49.47 |
| KF-cov | q=1e-06, sr=0.25* | 0.359 | 0.032 | 1.45 | 10.83 | 3 | 6.89 | 6.77 / 6.77 | 55.55 |

## Knee of each front (both axes scaled by EMA 0.4's values)

| filter | settings | static J px | rot std deg | pos std mm | E_dyn px | lag ms | motion noise px | overshoot px med/max | switch step px |
|---|---|---|---|---|---|---|---|---|---|
| EMA | w=0.5 | 0.374 | 0.031 | 1.50 | 22.70 | 82 | 1.50 | 3.25 / 3.25 | 29.99 |
| 1euro | fmin=0.05, beta=5 | 0.296 | 0.025 | 1.16 | 7.14 | 22 | 3.09 | 5.05 / 5.05 | 38.29 |
| KF-fixed | q=1e-05* | 0.411 | 0.042 | 1.80 | 10.04 | 5 | 6.76 | 1.65 / 1.65 | 49.47 |
| KF-cov | q=1e-06, sr=0.25* | 0.359 | 0.032 | 1.45 | 10.83 | 3 | 6.89 | 6.77 / 6.77 | 55.55 |

## Savitzky-Golay window sensitivity (test half, equal-jitter settings)

| filter | settings | E_dyn px (W=5) | E_dyn px (W=7) | lag ms (W=5) | lag ms (W=7) |
|---|---|---|---|---|---|
| EMA 0.4 | w=0.4 | 31.92 | 31.86 | 127 | 128 |
| EMA | w=0.4 | 31.92 | 31.86 | 127 | 128 |
| 1euro | fmin=0.1, beta=10 | 4.94 | 6.33 | 12 | 12 |
| KF-fixed | q=1e-05 | 10.04 | 11.36 | 5 | 4 |
| KF-cov | q=1e-06, sr=0.25 | 10.83 | 12.30 | 3 | 2 |

## Scene-distance sensitivity (equal-jitter settings re-run with d x 0.5 / 1 / 2)

| setting | static J px | E_dyn test px | lag test ms |
|---|---|---|---|
| 1euro d x0.5 | 0.320 | 6.33 | 18 |
| 1euro d x1.0 | 0.336 | 4.94 | 12 |
| 1euro d x2.0 | 0.362 | 4.22 | 7 |
| KF-fixed d x0.5 | 0.428 | 10.04 | 5 |
| KF-fixed d x1.0 | 0.411 | 10.04 | 5 |
| KF-fixed d x2.0 | 0.418 | 10.04 | 5 |
| KF-cov d x0.5 | 0.359 | 10.83 | 3 |
| KF-cov d x1.0 | 0.359 | 10.83 | 3 |
| KF-cov d x2.0 | 0.361 | 10.83 | 3 |

Plots: `pareto.png`, `motion_segment.png`, `static_segment.png`, `equal_jitter_bars.png`. Full numbers: `results.json`.
