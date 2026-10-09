"""Presentation-only adapter for unchanged upstream Shinka HTML assets.

The request handler still owns the exact campaign/database allowlist. This code
changes fixed asset colors, typography and labels; it never reads candidate data.
"""
import re
try:
    from .chromatic_fields import (BACKGROUND, TEXT, SECONDARY, COBALT, MAGENTA, ORANGE,
                                  RULE, MUTED, WALL, css_tokens, javascript_tokens,
                                  ISLAND_COLORS)
except ImportError:  # direct script entrypoints
    from chromatic_fields import (BACKGROUND, TEXT, SECONDARY, COBALT, MAGENTA, ORANGE,
                                  RULE, MUTED, WALL, css_tokens, javascript_tokens,
                                  ISLAND_COLORS)

CSS = r'''
body,button,input,select,textarea,svg text{font-family:'DejaVu Sans',Arial,sans-serif}
body{background:var(--paper);color:var(--ink)}
h1,h2,h3,h4{color:var(--ink);letter-spacing:-.015em}
button,input,select,textarea{border-color:var(--rule);border-radius:4px}
button:focus-visible,a:focus-visible,select:focus-visible{outline:3px solid var(--orange);outline-offset:3px}
pre,code{font-family:'DejaVu Sans Mono',monospace}
#tree-panel,#details-panel,.tab.active,.content,.card,.database-card,.metric-card{background:var(--paper)}
#tree-panel{box-shadow:none;border-right:1px solid var(--rule)}
.link{stroke:var(--rule)}.link.best-path{stroke:var(--cobalt)}
.node text{fill:var(--ink);font-family:'DejaVu Sans',Arial,sans-serif}
.cf-native-note{font:11px/1.5 'DejaVu Sans',Arial,sans-serif;color:var(--muted);background:var(--wash);border-left:3px solid var(--cobalt);padding:9px 11px;margin:0 0 12px}
.cf-native-note strong{color:var(--ink)}
'''

COLORS = {
    '#f8f9fa': BACKGROUND, '#ffffff': BACKGROUND, '#fff': BACKGROUND,
    '#f8f8f8': BACKGROUND, '#fafafa': BACKGROUND, '#f5f5f5': MUTED,
    '#f4f4f4': MUTED, '#f1f1f1': MUTED, '#f0f0f0': RULE,
    '#ddd': RULE, '#dddddd': RULE, '#ccc': RULE, '#cccccc': RULE,
    '#e9ecef': MUTED, '#f0f4f8': MUTED, '#f6f8fa': MUTED, '#e9f5f9': MUTED,
    '#f1f8ff': MUTED, '#d4eaff': MUTED,
    '#000': TEXT, '#000000': TEXT, '#333': TEXT, '#333333': TEXT,
    '#444': TEXT, '#444444': TEXT, '#2c3e50': TEXT, '#343a40': TEXT,
    '#555': SECONDARY, '#666': SECONDARY, '#888': SECONDARY, '#999': SECONDARY,
    '#555555': SECONDARY, '#666666': SECONDARY, '#777': SECONDARY, '#777777': SECONDARY,
    '#888888': SECONDARY, '#999999': SECONDARY, '#7f8c8d': SECONDARY, '#6a737d': SECONDARY,
    '#3498db': COBALT, '#2980b9': COBALT, '#007bff': COBALT, '#0366d6': COBALT,
    '#0000ff': COBALT, '#1f77b4': COBALT, '#aec7e8': MUTED, '#17becf': COBALT,
    '#8e44ad': MAGENTA, '#9b59b6': MAGENTA, '#9467bd': MAGENTA, '#e377c2': MAGENTA,
    '#d7bde2': MUTED, '#a31515': MAGENTA,
    '#e74c3c': MAGENTA, '#c0392b': MAGENTA, '#ff0000': MAGENTA, '#ffeef0': MUTED,
    '#f39c12': ORANGE, '#e67e22': ORANGE, '#ff8c00': ORANGE, '#ffd700': ORANGE,
    '#f1c40f': ORANGE, '#f8c291': MUTED, '#e58e26': ORANGE, '#795e26': ORANGE,
    '#8c564b': ORANGE, '#bcbd22': ORANGE,
    '#2ecc71': COBALT, '#27ae60': COBALT, '#2ca02c': COBALT, '#28a745': COBALT,
    '#155724': COBALT, '#008000': COBALT, '#098658': COBALT,
    '#d4edda': MUTED, '#e6ffed': MUTED, '#7b9ac4': SECONDARY, '#7f7f7f': SECONDARY,
}


COLORS.update({
    **dict.fromkeys(('#e0e0e0','#d8d8d8','#dee2e6'), RULE),
    **dict.fromkeys(('#e8f4fc','#eef0f2','#f5f6f7','#f5f7fa','#b3d7ff','#b3d9ff',
                    '#c1e5c5','#c3e6cb','#d0e3eb','#e7f3ff','#e9e9e9','#eaeaea',
                    '#ebf5ff','#eee','#f8d7da','#ffb3b3','#ffe0a6','#ffe6e6',
                    '#ffeaa7','#fff1b8','#fff1f0','#fff3cd','#fff5f5','#fff8e6'), MUTED),
    '#f9f9f9':BACKGROUND, '#1e1e1e':BACKGROUND, '#d4d4d4':TEXT,
    **dict.fromkeys(('#111827','#34495e'), TEXT),
    **dict.fromkeys(('#495057','#6b7280','#6c757d','#95a5a6','#607d8b'), SECONDARY),
    **dict.fromkeys(('#0066cc','#2196f3','#3b82f6','#58a6ff','#60a5fa',
                    '#10b981','#16a085','#1abc9c','#34d399','#3fb950',
                    '#009688','#00bcd4','#8bc34a'), COBALT),
    **dict.fromkeys(('#5a2d91','#6f42c1','#721c24','#7a1c1c','#dc3545',
                    '#f85149','#673ab7','#e91e63'), MAGENTA),
    **dict.fromkeys(('#b8860b','#d35400','#f59e0b','#ffc107','#795548',
                    '#ff5722','#ff9800'), ORANGE),
})
# The native fitness legend is a numeric score scale, not a probability field.
# Match its fixed CSS stops to the same neutral-to-cobalt interpolation as D3.
for index,old in enumerate(('#440154','#482878','#3e4a89','#31688e','#26838f',
                            '#1f9d8a','#6cce5a','#b6de2b','#fee825')):
    t=index/8
    rgb=[round((1-t)*int(MUTED[i:i+2],16)+t*int(COBALT[i:i+2],16)) for i in (1,3,5)]
    COLORS[old]='#'+''.join(f'{v:02X}' for v in rgb)


def native_html(source):
    """Adapt fixed upstream HTML, including native D3/Chart color definitions."""
    output = re.sub(r'#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b',
                    lambda m: COLORS.get(m[0].lower(),m[0]), source)
    rgba = {(52,152,219):COBALT,(255,140,0):ORANGE,(231,76,60):MAGENTA,
            (155,89,182):MAGENTA,(230,126,34):ORANGE,(142,68,173):MAGENTA,
            (46,204,113):COBALT,(0,0,0):TEXT}
    def alpha(m):
        rgb=tuple(int(m[i]) for i in (1,2,3));value=rgba.get(rgb)
        if not value:return m[0]
        parts=[int(value[i:i+2],16) for i in (1,3,5)]
        return f'rgba({parts[0]}, {parts[1]}, {parts[2]}, {m[4]})'
    output=re.sub(r'rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([.\d]+)\s*\)',alpha,output)
    for key,color in (('Viridis',COBALT),('Plasma',MAGENTA),('Inferno',ORANGE),('Magma',TEXT)):
        output=output.replace('d3.interpolate'+key,f'd3.interpolateRgb("{MUTED}", "{color}")')
        output=output.replace('>'+key+'</option>', '>'+{'Viridis':'Cobalt','Plasma':'Magenta','Inferno':'Orange','Magma':'Ink'}[key]+'</option>')
    palette='['+','.join('"'+c+'"' for c in ISLAND_COLORS)+']'
    output=re.sub(r'const islandColors\s*=\s*\[[\s\S]*?\];','const islandColors = '+palette+';',output)
    output=output.replace('d3.schemeCategory10',palette)
    for old,color in {'white':BACKGROUND,'black':TEXT,'blue':COBALT,'red':MAGENTA,'green':COBALT,'orange':ORANGE,'steelblue':COBALT,'lightgray':RULE,'grey':SECONDARY,'gray':SECONDARY}.items():
        output=re.sub(r'((?:color|fill|stroke|background(?:-color)?)\s*:\s*)'+old+r'\b',lambda m:m[1]+color,output)
        output=output.replace("'"+old+"'","'"+color+"'").replace('"'+old+'"','"'+color+'"')
    output=output.replace('🎏', '').replace('Total Cost', 'Total API list-price estimate').replace('API Cost', 'API list-price estimate').replace('Avg Cost/Program', 'List-price estimate / program').replace('Cost Breakdown', 'API list-price estimates (not subscription charges)').replace('Embed Cost', 'Embedding estimate').replace('Meta Cost', 'Meta estimate').replace('Novelty Cost', 'Novelty estimate')
    note="""<script>document.addEventListener('DOMContentLoaded',()=>{const host=document.getElementById('tree-panel')||document.body;const n=document.createElement('div');n.className='cf-native-note';n.innerHTML='<strong>Chromatic Field · native Shinka view.</strong> Recorded search values and lineage are unchanged. Reused development scores are selection-biased. Native API list-price estimates are not subscription charges. Tokens and CPU are the primary resource measures.';host.prepend(n);if(window.Chart){Chart.defaults.color='__SECONDARY__';Chart.defaults.font.family='DejaVu Sans, Arial, sans-serif';Chart.defaults.borderColor='__RULE__';}});</script>""".replace('__SECONDARY__',SECONDARY).replace('__RULE__',RULE)
    overlay='<style id="chromatic-field-presentation">'+css_tokens()+CSS+'</style>'
    return output.replace('</head>',overlay+'</head>').replace('</body>',note+'</body>')


def local_html(source):
    """Compile shared tokens into portable local pages; retain all data and logic."""
    return re.sub(r':root\{[^}]+\}',css_tokens().strip(),source,count=1)
