# Interim v3 outcomes

All outcome denominators include invalid executions. Forecast scores here are on-policy.

Paired 95% intervals are pointwise, not Holm-adjusted simultaneous intervals. Holm p-values apply only to the registered test families. CPU means use measured executions only; unavailable measurements are counted separately and never treated as zero.

**INTERIM · 140 / 1,536 registered cases per regime.** Nominal descriptive 95% intervals; not stopping-adjusted or confirmatory. The original sample-size power claim does not apply.

| Regime | Condition | Escape | Death | Timeout | Invalid | Mean task | Near Brier | CPU s/measured episode |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| uniform | Original memory | 130/140 | 10 | 0 | 0 | 0.9218 | 0.011111 | 0.383 |
| uniform | Original predictive seed | 111/140 | 29 | 0 | 0 | 0.8073 | 0.007599 | 0.330 |
| uniform | v2 generation 14 | 132/140 | 8 | 0 | 0 | 0.9400 | 0.008310 | 2.629 |
| uniform | Fitted predictor, frozen | 137/140 | 3 | 0 | 0 | 0.9688 | 0.007209 | 2.588 |
| uniform | Fitted predictor, online | 136/140 | 4 | 0 | 0 | 0.9623 | 0.007524 | 2.577 |
| uniform | Known-law reference | 136/140 | 4 | 0 | 0 | 0.9625 | 0.007129 | 2.454 |
| uniform | Selected evolved | 132/140 | 7 | 1 | 0 | 0.9374 | 0.007845 | 1.242 |
| uniform | Selected, frozen | 133/140 | 6 | 1 | 0 | 0.9429 | 0.007755 | 1.321 |
| uniform | Selected, uniform-law planning | 133/140 | 6 | 1 | 0 | 0.9429 | 0.007772 | 1.927 |
| uniform | Selected, frozen + uniform law | 133/140 | 6 | 1 | 0 | 0.9429 | 0.007751 | 1.966 |
| stationary | Original memory | 121/140 | 18 | 1 | 0 | 0.8730 | 0.008749 | 0.419 |
| stationary | Original predictive seed | 115/140 | 24 | 1 | 0 | 0.8292 | 0.005945 | 0.314 |
| stationary | v2 generation 14 | 132/140 | 4 | 0 | 4 | 0.9310 | 0.007198 | 3.041 |
| stationary | Fitted predictor, frozen | 131/140 | 5 | 0 | 4 | 0.9266 | 0.006016 | 2.967 |
| stationary | Fitted predictor, online | 130/140 | 5 | 0 | 5 | 0.9181 | 0.006357 | 3.054 |
| stationary | Known-law reference | 136/140 | 1 | 0 | 3 | 0.9580 | 0.006600 | 2.997 |
| stationary | Selected evolved | 132/140 | 5 | 3 | 0 | 0.9370 | 0.006506 | 1.467 |
| stationary | Selected, frozen | 127/140 | 10 | 3 | 0 | 0.9087 | 0.006610 | 1.399 |
| stationary | Selected, uniform-law planning | 127/140 | 10 | 3 | 0 | 0.9087 | 0.006465 | 2.052 |
| stationary | Selected, frozen + uniform law | 127/140 | 10 | 3 | 0 | 0.9087 | 0.006616 | 2.043 |
| switch | Original memory | 120/140 | 18 | 2 | 0 | 0.8654 | 0.008331 | 0.393 |
| switch | Original predictive seed | 116/140 | 22 | 2 | 0 | 0.8356 | 0.005844 | 0.312 |
| switch | v2 generation 14 | 131/140 | 6 | 1 | 2 | 0.9285 | 0.006807 | 2.932 |
| switch | Fitted predictor, frozen | 133/140 | 4 | 0 | 3 | 0.9396 | 0.005859 | 2.840 |
| switch | Fitted predictor, online | 133/140 | 4 | 0 | 3 | 0.9398 | 0.006166 | 2.740 |
| switch | Known-law reference | 137/140 | 0 | 0 | 3 | 0.9654 | 0.006130 | 2.858 |
| switch | Selected evolved | 135/140 | 4 | 1 | 0 | 0.9553 | 0.006619 | 1.373 |
| switch | Selected, frozen | 134/140 | 4 | 2 | 0 | 0.9504 | 0.006404 | 1.419 |
| switch | Selected, uniform-law planning | 134/140 | 4 | 2 | 0 | 0.9504 | 0.006265 | 2.118 |
| switch | Selected, frozen + uniform law | 134/140 | 4 | 2 | 0 | 0.9504 | 0.006410 | 2.114 |

**INTERIM · 140 / 1,536 registered cases per regime.** Nominal descriptive 95% intervals; not stopping-adjusted or confirmatory. The original sample-size power claim does not apply.

| Regime | Paired contrast | Escape Δ, pp [95% CI] | Wins / losses | Exact p | Holm p |
|:--|:--|--:|--:|--:|--:|
| stationary | Selected evolved − Selected, frozen | +3.57 [-2.79, +10.78] | 6 / 1 | 0.125 | 0.25 |
| switch | Selected evolved − Selected, frozen | +0.71 [-5.89, +6.73] | 2 / 1 | 1 | 1 |
| uniform | Selected evolved − Selected, frozen | -0.71 [-8.22, +6.84] | 2 / 3 | 1 | — |
| stationary | Fitted predictor, online − Fitted predictor, frozen | -0.71 [-9.85, +8.12] | 4 / 5 | 1 | 1 |
| switch | Fitted predictor, online − Fitted predictor, frozen | +0.00 [-7.11, +7.11] | 2 / 2 | 1 | 1 |
| uniform | Fitted predictor, online − Fitted predictor, frozen | -0.71 [-6.73, +5.89] | 1 / 2 | 1 | — |
| uniform | Selected evolved − Original memory | +1.43 [-8.31, +11.92] | 8 / 6 | 0.7905 | — |
| stationary | Selected evolved − Original memory | +7.86 [-0.67, +20.35] | 17 / 6 | 0.03469 | — |
| switch | Selected evolved − Original memory | +10.71 [+1.87, +21.97] | 18 / 3 | 0.00149 | — |
| uniform | Selected evolved − Original predictive seed | +15.00 [+3.86, +28.72] | 27 / 6 | 0.0003241 | — |
| stationary | Selected evolved − Original predictive seed | +12.14 [+2.27, +24.95] | 22 / 5 | 0.001514 | — |
| switch | Selected evolved − Original predictive seed | +13.57 [+3.35, +26.00] | 23 / 4 | 0.0003107 | — |
| uniform | Selected evolved − v2 generation 14 | +0.00 [-10.20, +10.20] | 7 / 7 | 1 | — |
| stationary | Selected evolved − v2 generation 14 | +0.00 [-9.29, +9.29] | 5 / 5 | 1 | — |
| switch | Selected evolved − v2 generation 14 | +2.86 [-4.57, +11.24] | 6 / 2 | 0.2891 | — |
| uniform | Selected evolved − Fitted predictor, online | -2.86 [-12.20, +5.21] | 3 / 7 | 0.3438 | — |
| stationary | Selected evolved − Fitted predictor, online | +1.43 [-7.87, +11.45] | 7 / 5 | 0.7744 | — |
| switch | Selected evolved − Fitted predictor, online | +1.43 [-6.86, +10.21] | 5 / 3 | 0.7266 | — |
| uniform | Known-law reference − Fitted predictor, frozen | -0.71 [-4.47, +4.36] | 0 / 1 | 1 | — |
| stationary | Known-law reference − Fitted predictor, frozen | +3.57 [-2.79, +10.78] | 6 / 1 | 0.125 | — |
| switch | Known-law reference − Fitted predictor, frozen | +2.86 [-3.77, +9.81] | 5 / 1 | 0.2188 | — |
| uniform | Selected evolved − Selected, uniform-law planning | -0.71 [-8.22, +6.84] | 2 / 3 | 1 | — |
| stationary | Selected evolved − Selected, uniform-law planning | +3.57 [-2.79, +10.78] | 6 / 1 | 0.125 | — |
| switch | Selected evolved − Selected, uniform-law planning | +0.71 [-5.89, +6.73] | 2 / 1 | 1 | — |
| uniform | Selected, frozen − Selected, frozen + uniform law | +0.00 [-3.08, +3.08] | 0 / 0 | 1 | — |
| stationary | Selected, frozen − Selected, frozen + uniform law | +0.00 [-3.08, +3.08] | 0 / 0 | 1 | — |
| switch | Selected, frozen − Selected, frozen + uniform law | +0.00 [-3.08, +3.08] | 0 / 0 | 1 | — |

Descriptive differences between regime-specific escape effects; conservative pointwise intervals preserve shared-case dependence.

**INTERIM · 140 / 1,536 registered cases per regime.** Nominal descriptive 95% intervals; not stopping-adjusted or confirmatory. The original sample-size power claim does not apply.

| Regime difference | Within-regime contrast | Difference of effects, pp [95% CI] |
|:--|:--|--:|
| stationary − uniform | Selected evolved − Selected, frozen | +4.29 [-11.83, +20.62] |
| switch − uniform | Selected evolved − Selected, frozen | +1.43 [-14.58, +16.46] |
| switch − stationary | Selected evolved − Selected, frozen | -2.86 [-18.28, +11.37] |
