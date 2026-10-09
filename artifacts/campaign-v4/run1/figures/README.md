# RUN1 saved-snapshot figures

These plots use the public scientific and native exports identified in [figure-manifest.json](figure-manifest.json). Source exports can advance independently; their timestamps and hashes are retained.

All task results reuse eight development layouts across six conditions. Candidate selection is biased by those same cases. Prediction scores are on-policy; no matched-learning, frozen-control, transfer or reliable-discovery result is implied.

- [Search progress](search-progress.svg): all candidate slots, missing/invalid execution, absolute task and proper prediction diagnostics.
- [Resources](resources.svg): measured local CPU, reported experiment-model tokens and registered caps; no estimated price is treated as a subscription charge.
- [Native machinery](native-machinery.svg): configured versus observed execution, bandit checkpoint counts, prompt credit. Native exponential reward sums are not mislabelled as mean rewards.
- [Usage scopes](usage-scopes.svg): separate supervising/delegated-conversation metadata; timeout usage and unreported active work remain unknown.
- [Ancestry](ancestry.svg): recorded parent/inspiration/migration relationships, retaining administrative seed copies.

```bash
.venv/bin/python scripts/run1_figures.py
```

This rendering command performs no experiment, statistical resampling or model call. SVG, PDF and PNG copies are supplied.
