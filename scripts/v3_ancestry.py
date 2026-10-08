"""Chromatic Field view of recorded native parent links, including seed copies."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
from visual_theme import apply_theme, save_figure, COBALT, MAGENTA, ORANGE, SECONDARY, TEXT, RULE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as DrawingPath
from matplotlib.patches import PathPatch
from matplotlib.ticker import MaxNLocator


def render(directory):
    rows = json.loads((directory/'lineage.json').read_text())
    by_id = {row['id']:row for row in rows}
    children = defaultdict(list)
    roots = []
    for row in rows:
        if row['parent_id'] is None:
            roots.append(row)
        else:
            if row['parent_id'] not in by_id:
                raise ValueError('Parent missing from the recorded native lineage')
            children[row['parent_id']].append(row)
    roots.sort(key=lambda row:(row['birth_island'],row['id']))
    for group in children.values():
        group.sort(key=lambda row:(row['generation'],row['id']))
    coordinates, cursor = {}, 0.
    def place(row, depth):
        nonlocal cursor
        offspring = children[row['id']]
        if offspring:
            positions = [place(child,depth+1) for child in offspring]
            y = sum(positions)/len(positions)
        else:
            y, cursor = cursor, cursor+1
        coordinates[row['id']] = (depth,y)
        return y
    for root in roots:
        place(root,0)
        cursor += .9
    selection_path = directory/'selection/selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    highlighted = set()
    if selection:
        node = by_id[selection['native_id']]
        while node:
            highlighted.add(node['id'])
            node = by_id.get(node['parent_id'])
    apply_theme()
    fig, ax = plt.subplots(figsize=(12,max(7,cursor*.28)))
    colors = (COBALT,MAGENTA,ORANGE,SECONDARY)
    for row in rows:
        child = coordinates[row['id']]
        if row['parent_id']:
            parent = coordinates[row['parent_id']]
            middle = (parent[0]+child[0])/2
            path = DrawingPath([parent,(middle,parent[1]),(middle,child[1]),child],
                               [DrawingPath.MOVETO,DrawingPath.CURVE4,DrawingPath.CURVE4,DrawingPath.CURVE4])
            emphasis = row['id'] in highlighted
            ax.add_patch(PathPatch(path,facecolor='none',edgecolor=TEXT if emphasis else RULE,
                                   linewidth=2.1 if emphasis else 1.2,zorder=1))
        marker = 'D' if not row['parent_id'] else 's' if row['metadata'].get('patch_type') == 'diff' else 'o'
        if not row['correct']:
            marker = 'X'
        selected = bool(selection and row['id'] == selection['native_id'])
        ax.scatter(*child,s=260 if selected else 165,marker=marker,
                   c=colors[row['birth_island']],edgecolors=TEXT if selected else 'white',
                   linewidths=2.5 if selected else .7,zorder=3)
        ax.text(*child,str(row['generation']),ha='center',va='center',color='white',
                fontsize=8.5,fontweight='bold',zorder=4)
    depth = max(x for x,y in coordinates.values())
    for root in roots:
        x,y = coordinates[root['id']]
        ax.text(-.28,y,f"Island {root['birth_island']}",ha='right',va='center',
                fontsize=10,color=colors[root['birth_island']],fontweight='bold')
    slots = len({row['generation'] for row in rows})
    ax.set(xlim=(-1.35,depth+.45),ylim=(cursor-.3,-1),yticks=[],
           xlabel='Parent-link depth from native island seed')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(range(depth+1))
    ax.spines['left'].set_visible(False)
    handles = [Line2D([],[],marker=marker,linestyle='none',markersize=8,color=SECONDARY,label=label)
               for marker,label in [('D','Seed island row'),('o','Full proposal'),('s','Diff proposal')]]
    if any(not row['correct'] for row in rows):
        handles.append(Line2D([],[],marker='X',linestyle='none',color=SECONDARY,label='Failed slot'))
    ax.legend(handles=handles,loc='upper right',ncol=1,fontsize=9)
    fig.suptitle('Program ancestry across four native islands',fontsize=18,fontweight='bold',x=.12,ha='left')
    subtitle = f'{slots} candidate slots · node labels identify slots · color records island at source creation'
    fig.text(.12,.92,subtitle,color=SECONDARY,fontsize=10)
    note = 'Edges are recorded parents. Seed copies count as one experimental slot; inspiration and migration events remain in lineage.json.'
    if selection:
        note += f"\nDark path leads to selection-validation winner, slot {selection['generation']}; this is not independent discovery evidence."
    fig.text(.12,.025,note,color=SECONDARY,fontsize=9)
    fig.subplots_adjust(top=.87,bottom=.16,left=.10,right=.98)
    save_figure(fig,directory/'figures/ancestry')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=Path('artifacts/campaign-v3'))
    render(parser.parse_args().artifacts)
