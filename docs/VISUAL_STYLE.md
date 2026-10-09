# Chromatic Field — ShinkaDreamer

Adopted from [ReloadLightly/actir-backprop-neat](https://github.com/ReloadLightly/actir-backprop-neat/blob/main/docs/VISUAL_STYLE.md)
and its `scripts/visual_theme.py`, read on 2026-10-08. RUN1 explicitly extends the
style to historical and current publications. Scientific numbers, saved episodes,
programs and preregistered sources remain unchanged.

The frozen [`visual_theme.py`](../scripts/visual_theme.py) retains the original
palette and export primitives. The additive shared presentation layer is
[`chromatic_fields.py`](../scripts/chromatic_fields.py): plot roles, markers,
terrain glyphs and CSS/JavaScript tokens. [`web_presentation.py`](../scripts/web_presentation.py)
applies those tokens to fixed upstream Shinka HTML at the local HTTP boundary;
the installed upstream package is unchanged.

Use opaque warm white `#FFFEFC`, text `#161625`, secondary `#6F6A78`, cobalt
`#3534CF`, magenta `#C93683`, orange `#F26A37` and thin rules `#DCD7E0`.
DejaVu Sans at readable 10–12 pt, 16 pt titles, nominal 10-inch width; provide
SVG, PDF and PNG. Preserve individual observations, uncertainty, missing values
and failures. Captions and linked CSV/JSON carry evidence in searchable form.
The peach-to-cobalt field is exclusively a labelled numeric probability scale.
Native fitness heatmaps use a separate neutral-to-cobalt score scale.

| Meaning | Color | Redundant encoding |
|:--|:--|:--|
| Online evolved program | Cobalt | Circle, solid line |
| Frozen evolved program | Magenta | Square, dashed line |
| Fitted online / frozen comparator | Cobalt / magenta | Triangle-down / diamond, distinct line styles |
| Known-law reference | Orange | Triangle-up |
| Memory / historical control | Secondary | Diamond / triangle-down |
| Escape / death / timeout / invalid | Cobalt / magenta / orange / secondary | Explicit outcome labels |
| Native island | Four categorical colors | Explicit island labels; shape records mutation operator |
| Key / door / exit | Orange / magenta / cobalt | Diamond / square / star |

Lineage means actual recorded source ancestry, not causal necessity. Probability
maps state the target event, coordinates and horizon. Retrospective hidden-state
and next-enemy overlays are labelled; the candidate did not observe them. Saved
replay selection, frame order and frame count are retained.

The current saved-data publication command is:

```bash
.venv/bin/python scripts/visual_migration.py --group all
```

This command renders saved summaries and traces; it launches no environments,
models, statistical resampling or reproduction campaigns. Historical scientific
entrypoints are retained as provenance and are not current publication commands.
Their plot functions, when reused, run through a presentation adapter. After a
historical scientific recomputation, run this command to restore current exports.
The frozen v3 example renderer already uses Chromatic Field; its bytes remain
bound to the original assessment plan.

The [inventory](visual-migration-inventory.json) separates active presentation
paths, frozen inputs and inactive historical producers. The
[rendering manifest](../artifacts/visual-migration/rendering-manifest.json) binds
original Git/source/image identities, unchanged numerical inputs and current
outputs. Inspect exports at 960 px width and dense diagrams at full size; check
actual browser layouts at desktop and mobile widths. Do not mark the inventory
complete while any active renderer or published output remains unaccounted for.
