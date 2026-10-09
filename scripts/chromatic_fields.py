"""Shared presentation tokens for repository plots, replay glyphs and local web UI.

The preregistered visual_theme.py is intentionally unchanged. This additive
module reuses its palette, fonts and probability scale; it never reads evidence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1]/".cache/matplotlib"))

try:
    from .visual_theme import (BACKGROUND, TEXT, SECONDARY, COBALT, MAGENTA, ORANGE,
                              RULE, MUTED, FONT, MONO, FIELD_STOPS, apply_theme,
                              field_cmap, save_figure, panel_label)
except ImportError:  # direct script entrypoints
    from visual_theme import (BACKGROUND, TEXT, SECONDARY, COBALT, MAGENTA, ORANGE,
                              RULE, MUTED, FONT, MONO, FIELD_STOPS, apply_theme,
                              field_cmap, save_figure, panel_label)

PRESENTATION_VERSION = 'Chromatic Field v1 · repository presentation 2'
WALL = '#938D9F'
OUTSIDE = '#413C4C'
WASH = '#F4F3FC'
ROLES = {
    'online': {'color': COBALT, 'marker': 'o', 'linestyle': '-', 'label': 'Online updates'},
    'frozen': {'color': MAGENTA, 'marker': 's', 'linestyle': '--', 'label': 'Frozen parameters'},
    'known_law': {'color': ORANGE, 'marker': '^', 'linestyle': '-.', 'label': 'Known-law reference'},
    'memory': {'color': SECONDARY, 'marker': 'D', 'linestyle': ':', 'label': 'Original memory'},
    'historical': {'color': SECONDARY, 'marker': 'v', 'linestyle': '--', 'label': 'Historical program'},
    'fixed_risk': {'color': ORANGE, 'marker': 'P', 'linestyle': ':', 'label': 'Fixed-risk planning'},
}
OUTCOMES = {'escape': COBALT, 'death': MAGENTA, 'timeout': ORANGE, 'invalid': SECONDARY}
REGIMES = {'uniform': {'color':SECONDARY,'marker':'o'},
           'stationary': {'color':COBALT,'marker':'s'},
           'switch': {'color':MAGENTA,'marker':'^'}}
CONDITIONS = {
    'selected_online': ROLES['online'], 'selected': ROLES['online'],
    'selected_frozen': ROLES['frozen'],
    'fitted_online': {**ROLES['online'], 'marker':'v', 'linestyle':'-.'},
    'fitted_frozen': {**ROLES['frozen'], 'marker':'D', 'linestyle':':'},
    'known_law': ROLES['known_law'], 'memory': ROLES['memory'],
    'v2_gen14': ROLES['historical'],
    'selected_fixed_risk': ROLES['online'],
    'selected_frozen_fixed_risk': ROLES['frozen'],
    'no_planning': ROLES['online'], 'frozen_no_planning': ROLES['frozen'],
}

ISLAND_COLORS = (COBALT, MAGENTA, ORANGE, SECONDARY)
ISLAND_MARKERS = ('o', 's', '^', 'D')
OPERATOR_MARKERS = {'seed': 'D', 'full': 'o', 'diff': 's', 'failure': 'X'}
TERRAIN_COLORS = (MUTED, OUTSIDE, BACKGROUND, WALL, ORANGE, MAGENTA, COBALT)
TERRAIN_GLYPHS = {2: ('D', ORANGE, 'Key'), 3: ('s', MAGENTA, 'Door'), 4: ('*', COBALT, 'Exit')}


def web_tokens():
    return {'paper': BACKGROUND, 'ink': TEXT, 'muted': SECONDARY, 'cobalt': COBALT,
            'magenta': MAGENTA, 'orange': ORANGE, 'rule': RULE, 'wash': MUTED,
            'scope': WASH, 'wall': WALL, 'outside': OUTSIDE}


def css_tokens():
    return ':root{' + ';'.join(f'--{key}:{value}' for key, value in web_tokens().items()) + '}\n'


def javascript_tokens():
    return json.dumps({'colors': web_tokens(), 'islands': ISLAND_COLORS,
                       'roles': ROLES, 'operators': OPERATOR_MARKERS}, separators=(',', ':'))


def terrain_cmap():
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import numpy as np
    cmap = ListedColormap(TERRAIN_COLORS, name='chromatic-field-terrain')
    return cmap, BoundaryNorm(np.arange(-2.5, 5.5), cmap.N)


def terrain_symbols(ax, grid, *, size=22):
    """Redundant shapes distinguish categorical key, door and exit tiles."""
    import numpy as np
    for cell, (marker, color, label) in TERRAIN_GLYPHS.items():
        yy, xx = np.where(np.asarray(grid) == cell)
        if len(xx):
            ax.scatter(xx, yy, marker=marker, s=size, c=BACKGROUND,
                       edgecolor=TEXT, linewidth=.55, zorder=2)


# Used only by the additive adapters for immutable historical figure functions.
# Values and artist geometry are untouched. Colors have explicit semantic roles.
LEGACY_COLORS = {
    '#0072b2': COBALT, '#d55e00': MAGENTA, '#009e73': COBALT, '#cc6677': MAGENTA,
    '#777777': SECONDARY, '#222222': TEXT, '#666666': SECONDARY, '#dddddd': RULE,
    '#999999': SECONDARY, '#ddcc77': ORANGE, '#8f3344': ORANGE,
    '#faf8f0': BACKGROUND, '#334653': WALL, '#e8b730': ORANGE,
    '#aa88bb': MAGENTA, '#55aa88': COBALT,
    '#172c3b': TEXT, '#52616e': SECONDARY, '#007f80': COBALT,
    '#7253a8': MAGENTA, '#b2bdc5': SECONDARY, '#c5ced4': RULE,
    '#f2f6f7': MUTED, '#e8edf0': RULE, '#a8b5bd': SECONDARY,
    '#ffffff': BACKGROUND, '#000000': TEXT,
}


def restyle_legacy_figure(fig):
    """Color/type migration for retained legacy plotting geometry; no data edits."""
    from matplotlib import colors
    from matplotlib.text import Text
    from matplotlib.colors import ListedColormap
    import numpy as np

    def color(value):
        try:
            rgba = colors.to_rgba(value)
            mapped = LEGACY_COLORS.get(colors.to_hex(rgba).lower())
            return (*colors.to_rgb(mapped), rgba[3]) if mapped else value
        except (ValueError, TypeError):
            return value

    for artist in fig.findobj():
        if isinstance(artist, Text):
            artist.set_fontfamily(FONT)
            artist.set_color(color(artist.get_color()))
        for key in ('color', 'facecolor', 'edgecolor', 'markerfacecolor', 'markeredgecolor'):
            getter, setter = getattr(artist, 'get_'+key, None), getattr(artist, 'set_'+key, None)
            if not callable(getter) or not callable(setter):
                continue
            try:
                value = getter()
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    setter([color(v) for v in value])
                else:
                    setter(color(value))
            except (ValueError, TypeError, AttributeError):
                pass
        if hasattr(artist, 'get_cmap') and isinstance(artist.get_cmap(), ListedColormap):
            old = artist.get_cmap()
            artist.set_cmap(ListedColormap([color(v) for v in old.colors]))
    fig.set_facecolor(BACKGROUND)
    for ax in fig.axes:
        ax.set_facecolor(BACKGROUND)
        for spine in ax.spines.values():
            spine.set_color(RULE)
            spine.set_linewidth(.65)
        ax.tick_params(colors=SECONDARY)
        for line in ax.get_xgridlines()+ax.get_ygridlines():
            line.set_color(RULE)
    return fig
