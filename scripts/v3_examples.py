"""Publish deterministic recorded v3 examples after numerical analysis closes.

This module never imports an environment, invokes an agent, or reads a seed
pool. Figures use already-retained final-assessment traces exclusively. Every
published case is explicitly exposed and the entire assessment pool is retired.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
from dreamer.provenance import sha256

PAIR = ("selected", "selected_frozen")
STRATA = ("selected_only_escape", "frozen_only_escape", "both_escape", "neither_escape")
LABELS = {"selected_only_escape": "Selected escapes; frozen does not",
          "frozen_only_escape": "Frozen escapes; selected does not",
          "both_escape": "Both escape", "neither_escape": "Neither escapes"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_immutable(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Refusing to replace a published example record: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def frozen_script(plan, relative):
    expected = {**plan.get("analysis_files", {}), **plan.get("design_files", {})}.get(relative)
    if expected != sha256(ROOT / relative):
        raise ValueError("Publication helper was not frozen in the plan: " + relative)


def verify_closed_inputs(data, raw, plan_path):
    """Read numerical evidence first; access no replay before closure verifies."""
    data, raw, plan_path = Path(data), Path(raw), Path(plan_path)
    plan = read_json(plan_path)
    closed = read_json(data / "analysis-closure.json")
    if closed.get("status") != "analysis-complete" or closed.get("analysis_precedes_example_selection") is not True:
        raise ValueError("Numerical analysis has not closed before example selection")
    if closed["plan_sha256"] != sha256(plan_path):
        raise ValueError("Analysis closure plan mismatch")
    if closed["analysis_source_sha256"] != sha256(ROOT / "scripts/v3_analysis.py"):
        raise ValueError("Closed analysis source changed")
    for relative in ("scripts/v3_examples.py", "scripts/visual_theme.py", "scripts/v3_analysis.py"):
        frozen_script(plan, relative)
    required = {"episodes.jsonl.gz", "analysis.json"}
    if plan.get("matched"):
        required.update(("matched-episodes.jsonl.gz", "matched-analysis.json"))
    if not required.issubset(closed["files"]):
        raise ValueError("Closure does not bind every required numerical evidence file")
    for name, expected in closed["files"].items():
        if Path(name).name != name or sha256(data / name) != expected:
            raise ValueError("Closed numerical evidence changed: " + name)
    if read_json(raw / "frozen-plan.json") != plan:
        raise ValueError("Raw assessment uses a different plan")
    if closed.get("raw_completion_sha256") != sha256(raw / "execution-complete.json"):
        raise ValueError("Closure must bind this completed raw assessment")
    completion = read_json(raw / "execution-complete.json")
    expected_count = plan["sample_size"] * len(plan["regimes"]) * len(plan["conditions"])
    if closed.get("condition_episodes") != expected_count or closed.get("cases_per_regime") != plan["sample_size"]:
        raise ValueError("Analysis closure has incomplete denominators")
    if completion["cases"] != plan["sample_size"] or completion["condition_episodes"] != expected_count:
        raise ValueError("Final assessment is incomplete")
    analysis = read_json(data / "analysis.json")
    inputs = analysis["inputs"]
    if inputs["data_sha256"] != sha256(data / "episodes.jsonl.gz") or inputs["plan_sha256"] != sha256(plan_path):
        raise ValueError("Closed numerical analysis has different inputs")
    if inputs["analysis_sha256"] != closed["analysis_source_sha256"]:
        raise ValueError("Closed numerical analysis uses a different source")
    if inputs.get("raw_completion_sha256") != closed["raw_completion_sha256"]:
        raise ValueError("Analysis inputs are not bound to the completed raw assessment")
    if plan.get("matched"):
        matched_inputs = read_json(data / "matched-analysis.json")["inputs"]
        if (matched_inputs.get("matched_data_sha256") != sha256(data / "matched-episodes.jsonl.gz")
                or matched_inputs.get("plan_sha256") != sha256(plan_path)
                or matched_inputs.get("data_sha256") != sha256(data / "episodes.jsonl.gz")
                or matched_inputs.get("analysis_sha256") != closed["analysis_source_sha256"]
                or closed.get("matched_policy_episodes") != plan["sample_size"] * len(plan["regimes"])):
            raise ValueError("Matched analysis closure or input mismatch")
    if analysis["condition_episodes"] != expected_count or analysis["cases_per_regime"] != plan["sample_size"]:
        raise ValueError("Closed numerical analysis has incomplete denominators")
    spec = plan.get("replays", {})
    if spec.get("conditions") != list(PAIR) or spec.get("first_cases") != 32:
        raise ValueError("Freeze the selected/frozen first-32 replay rule before assessment")
    cutoff = min(32, plan["sample_size"])
    condition_sources = {c["name"]: c["program_sha256"] for c in plan["conditions"]}
    for condition in plan["conditions"]:
        path = Path(condition["program_path"])
        if not path.is_absolute():
            path = ROOT / path
        if sha256(path) != condition["program_sha256"]:
            raise ValueError("Frozen condition source changed: " + condition["name"])
    compact_hashes, retained_rows = {}, {}
    with gzip.open(data / "episodes.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            identity = (row["case"], row["regime"], row["condition"])
            if identity in compact_hashes:
                raise ValueError("Duplicate compact episode")
            compact_hashes[identity] = digest(row)
            if row["case"] < cutoff and row["condition"] in PAIR:
                retained_rows[identity] = row
    expected_ids = {(case, regime, condition) for case in range(plan["sample_size"])
                    for regime in plan["regimes"] for condition in condition_sources}
    if set(compact_hashes) != expected_ids:
        raise ValueError("Missing or undeclared compact episodes")
    case_paths = sorted((raw / "cases").glob("*.json"))
    if {p.name for p in case_paths} != set(completion["case_file_sha256"]):
        raise ValueError("Raw completion and cases differ")
    checked = set()
    for path in case_paths:
        if sha256(path) != completion["case_file_sha256"][path.name]:
            raise ValueError("Raw episode case changed: " + path.name)
        case = read_json(path)
        for row in case["conditions"]:
            identity = (row["case"], row["regime"], row["condition"])
            if row["case"] != case["case"] or identity in checked or digest(row) != compact_hashes.get(identity):
                raise ValueError("Raw and compact episode evidence disagree")
            if row["program_sha256"] != condition_sources[row["condition"]]:
                raise ValueError("Recorded source differs from plan")
            checked.add(identity)
    if checked != expected_ids:
        raise ValueError("Incomplete raw outcome denominator")
    inventory = completion.get("passive_replays", {})
    expected_replays = {}
    for row in retained_rows.values():
        replay = row.get("audit", {}).get("retained_replay")
        if not replay:
            raise ValueError("A prescribed first-32 original replay was not retained")
        expected_replays[Path(replay["path"]).name] = replay["sha256"]
    if (inventory.get("file_sha256") != expected_replays or inventory.get("new_environment_episodes") != 0):
        raise ValueError("Retained original replay inventory differs from completion")
    return plan, retained_rows, closed


def select_examples(rows, regimes, cutoff=32):
    selections = []
    for regime in regimes:
        groups = {name: [] for name in STRATA}
        for case in sorted({key[0] for key in rows if key[1] == regime and key[0] < cutoff}):
            a, b = rows[case, regime, PAIR[0]], rows[case, regime, PAIR[1]]
            escaped = (a["reason"] == "escaped", b["reason"] == "escaped")
            stratum = {(True, False): STRATA[0], (False, True): STRATA[1],
                       (True, True): STRATA[2], (False, False): STRATA[3]}[escaped]
            groups[stratum].append(case)
        selections.extend({"regime": regime, "stratum": stratum,
                           "case": min(groups[stratum]) if groups[stratum] else None,
                           "eligible_cases_in_retained_prefix": len(groups[stratum])}
                          for stratum in STRATA)
    return selections


def load_trace(raw, row):
    expected_path = f"replays/{row['case']:06d}--{row['regime']}--{row['condition']}.json.gz"
    retained = row.get("audit", {}).get("retained_replay")
    if not retained or retained["path"] != expected_path:
        raise ValueError("Missing registered original replay; world reruns are prohibited")
    path = Path(raw) / expected_path
    if sha256(path) != retained["sha256"]:
        raise ValueError("Retained original replay bytes changed")
    with gzip.open(path, "rt") as handle:
        trace = json.load(handle)
    trajectory_hash = digest([{k: frame[k] for k in ("world", "action", "next_enemies")} for frame in trace])
    if trajectory_hash != row["audit"]["trajectory_sha256"] or trajectory_hash != retained["trajectory_sha256"]:
        raise ValueError("Retained trace disagrees with the original outcome")
    if len(trace) != row["audit"]["frames"]:
        raise ValueError("Retained replay frame count changed")
    return trace


def display_index(left, right):
    common = min(len(left), len(right))
    if not common:
        return None, None
    for index in range(common):
        if left[index]["world"] != right[index]["world"]:
            raise ValueError("Paired worlds differ before any action disagreement")
        if left[index]["action"] != right[index]["action"]:
            return index, index
    return common-1, None


def public_frame(frame):
    # Explicit allowlists exclude hidden movement laws, seeds, candidate-private
    # state, predictor internals and any future schedule from published examples.
    return {"world": {key: frame["world"][key] for key in
                       ("grid", "enemies", "agent", "origin", "step", "keys", "door_open")},
            "model": {key: frame["model"][key] for key in
                       ("position", "terrain", "enemy", "default_enemy") if key in frame["model"]},
            "action": frame["action"], "next_enemies": frame["next_enemies"]}


def prepare_examples(raw, rows, selections):
    examples, private = [], {}
    for selection in selections:
        if selection["case"] is None:
            continue
        case, regime = selection["case"], selection["regime"]
        pair_rows = {condition: rows[case, regime, condition] for condition in PAIR}
        traces = {condition: load_trace(raw, row) for condition, row in pair_rows.items()}
        index, divergence = display_index(traces[PAIR[0]], traces[PAIR[1]])
        agents = {}
        for condition, row in pair_rows.items():
            trace = traces[condition]
            agents[condition] = {
                "reason": row["reason"], "steps": row["steps"], "keys": row["keys"], "door": row["door"], "error": row["error"],
                "program_sha256": row["program_sha256"],
                "recorded_trajectory_sha256": row["audit"]["trajectory_sha256"],
                "retained_replay_sha256": row["audit"]["retained_replay"]["sha256"],
                "path": [f["world"]["agent"] for f in trace],
                "display_frame": public_frame(trace[index]) if index is not None else None}
        examples.append({**selection, "display_frame_index": index,
                         "display_step": traces[PAIR[0]][index]["world"]["step"] if index is not None else None,
                         "first_action_divergence_step": traces[PAIR[0]][divergence]["world"]["step"] if divergence is not None else None,
                         "display_rule_used": "first action divergence" if divergence is not None else "last shared frame" if index is not None else "no paired frame: invalid/empty execution",
                         "agents": agents})
        private[regime, selection["stratum"], case] = traces
    return examples, private


def draw_frame(ax, frame, path, condition, subtitle, title=None):
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle, Rectangle
    from visual_theme import BACKGROUND, COBALT, MAGENTA, ORANGE, RULE, SECONDARY, TEXT, field_cmap
    color = COBALT if condition == PAIR[0] else MAGENTA
    ax.clear()
    if frame is None:
        ax.text(.5, .5, "No recorded paired frame", transform=ax.transAxes, ha="center", color=SECONDARY)
        ax.set_axis_off()
        return
    world, model = frame["world"], frame["model"]
    grid = np.asarray(world["grid"])
    ax.imshow((grid == 1).astype(float), cmap=ListedColormap([BACKGROUND, "#938D9F"]), vmin=0, vmax=1,
              extent=(-.5, 14.5, 14.5, -.5), interpolation="nearest")
    px, py = world["agent"]
    ox, oy = world["origin"]
    forecasts = {(x+ox, y+oy): p for x, y, p in model.get("enemy", [])}
    patch = np.full((15,15), np.nan)
    for y in range(max(0,py-2),min(15,py+3)):
        for x in range(max(0,px-2),min(15,px+3)):
            if grid[y,x] != 1:
                patch[y,x] = forecasts.get((x,y),model.get("default_enemy",.5))
    ax.imshow(patch,cmap=field_cmap(),vmin=0,vmax=1,alpha=1.,
              extent=(-.5,14.5,14.5,-.5),interpolation="nearest")
    if path:
        points = np.asarray(path)
        ax.plot(points[:,0],points[:,1],color=color,lw=1.5,alpha=.65,zorder=3)
    for cell, label in ((2,"K"),(3,"D"),(4,"E")):
        ys,xs=np.where(grid==cell)
        for x,y in zip(xs,ys):
            ax.text(x,y,label,ha="center",va="center",fontsize=8,color=TEXT,fontweight="bold",zorder=4)
    enemies = sorted({tuple(p) for p in world["enemies"]})
    if enemies:
        points=np.asarray(enemies)
        ax.scatter(points[:,0],points[:,1],marker="x",s=40,c=ORANGE,linewidths=1.6,zorder=5)
    ax.add_patch(Rectangle((px-2.5,py-2.5),5,5,fill=False,ec=color,lw=1.4,zorder=6))
    ax.scatter([px],[py],s=58,c=color,edgecolors=BACKGROUND,linewidths=1.1,zorder=7)
    dx,dy=frame["action"]["move"]
    if dx or dy:
        ax.annotate("",xy=(px+dx,py+dy),xytext=(px,py),arrowprops={"arrowstyle":"-|>","color":color,"lw":2},zorder=8)
    else:
        ax.add_patch(Circle((px,py),.43,fill=False,ec=color,lw=1.2,zorder=8))
    ax.set(xlim=(-.5,14.5),ylim=(14.5,-.5),xticks=[],yticks=[],aspect="equal")
    ax.set_title(title or ("Selected" if condition==PAIR[0] else "Frozen learner"),loc="left",color=color,fontsize=11)
    ax.text(0,-.045,subtitle,transform=ax.transAxes,ha="left",va="top",fontsize=9,color=SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(True);spine.set_color(RULE)


def render_figures(examples, out):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from visual_theme import apply_theme, field_cmap, save_figure, SECONDARY
    apply_theme()
    outputs=[]
    for regime in dict.fromkeys(e["regime"] for e in examples):
        subset=[e for e in examples if e["regime"]==regime]
        fig,axes=plt.subplots(len(subset),2,figsize=(10,3.45*len(subset)+1.1),squeeze=False)
        fig.subplots_adjust(top=1-.7/fig.get_figheight(),bottom=.8/fig.get_figheight(),left=.045,right=.945,hspace=.32,wspace=.12)
        fig.suptitle(f"{regime.capitalize()} dynamics · recorded assessment decisions",x=.05,ha="left",fontsize=16,fontweight="bold")
        for row,example in enumerate(subset):
            for col,condition in enumerate(PAIR):
                agent=example["agents"][condition]
                frame=agent["display_frame"]
                path=agent["path"][:example["display_frame_index"]+1] if frame is not None else []
                label="Selected" if condition==PAIR[0] else "Frozen learner"
                title=f"Case {example['case']} · {label} · {agent['reason']}"
                subtitle=f"{LABELS[example['stratum']]}\n{example['display_rule_used']}; step {example['display_step']} / {agent['steps']}"
                draw_frame(axes[row,col],frame,path,condition,subtitle,title)
        cax=fig.add_axes([.967,.19,.013,.3])
        fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=field_cmap()),cax=cax,label="Exported P(enemy next tick)")
        fig.text(.05,.018,"Retrospective world; box = observation, × = enemy, arrow = move; K/D/E = key/door/exit.\nPaths stop at decision. Cases exposed after analysis; numerical assessment remains primary.",fontsize=9,color=SECONDARY)
        target=Path(out)/f"behavior-examples-{regime}"
        save_figure(fig,target)
        outputs.extend(str(target.with_suffix('.'+extension)) for extension in ('svg','pdf','png'))
    return outputs


def render_gif(example,traces,path):
    """One bounded-size animation from original frames, never a new episode."""
    import matplotlib.pyplot as plt
    from PIL import Image
    from visual_theme import apply_theme, SECONDARY
    apply_theme()
    length=max(len(t) for t in traces.values())
    if not length:
        return None
    stride=max(1,math.ceil(length/48))
    indices=sorted(set(range(0,length,stride))|{length-1})
    frames=[]
    fig,axes=plt.subplots(1,2,figsize=(10,5),dpi=90)
    fig.subplots_adjust(top=.86,bottom=.12,left=.03,right=.98,wspace=.10)
    for index in indices:
        for ax,condition in zip(axes,PAIR):
            trace=traces[condition]
            own=min(index,len(trace)-1)
            frame=public_frame(trace[own]) if trace else None
            path_prefix=[f['world']['agent'] for f in trace[:own+1]] if trace else []
            end=example['agents'][condition]
            subtitle=f"Recorded step {own}; eventual {end['reason']} at {end['steps']}"
            draw_frame(ax,frame,path_prefix,condition,subtitle)
        fig.suptitle(f"{example['regime'].capitalize()} · case {example['case']} · {LABELS[example['stratum']]}",x=.04,ha='left',fontsize=14)
        if not fig.texts or len(fig.texts)<2:
            fig.text(.04,.025,"Original assessment frames; cases are exposed and retired. No environment rerun.",fontsize=9,color=SECONDARY)
        fig.canvas.draw()
        frames.append(Image.frombuffer('RGBA',fig.canvas.get_width_height(),fig.canvas.buffer_rgba(),'raw','RGBA',0,1).convert('RGB'))
    plt.close(fig)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    frames[0].save(path,save_all=True,append_images=frames[1:],duration=160,loop=0,optimize=True)
    return str(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='artifacts/campaign-v3/assessment')
    parser.add_argument('--raw',required=True,help='Completed assessment directory containing retained original traces')
    parser.add_argument('--plan',help='Defaults to DATA/preregistration.json')
    parser.add_argument('--gif',action='store_true',help='Animate the first available rule-selected example')
    args=parser.parse_args()
    data=Path(args.data);plan_path=Path(args.plan) if args.plan else data/'preregistration.json'
    plan,rows,closed=verify_closed_inputs(data,args.raw,plan_path)
    selections=select_examples(rows,plan['regimes'],min(32,plan['sample_size']))
    examples,private=prepare_examples(args.raw,rows,selections)
    publication={
        'study':plan['study'],'published_after_analysis_closed':True,
        'analysis_closure_sha256':sha256(data/'analysis-closure.json'),
        'example_script_sha256':sha256(__file__),'theme_sha256':sha256(ROOT/'scripts/visual_theme.py'),
        'plan_sha256':sha256(plan_path),'selection_rule':'Lowest case index among the prespecified first32 for each selected/frozen escape stratum within each regime',
        'display_rule':'First differing action; otherwise last shared frame. Missing strata/empty traces remain explicit.',
        'new_environment_episodes':0,'new_model_calls':0,'inferential_episode_count_unchanged':plan['sample_size']*len(plan['conditions'])*len(plan['regimes']),
        'path_scope':'Pre-action recorded positions; terminal destinations are not reconstructed.',
        'exposure_notice':'These displayed cases are now explicitly exposed. The entire assessment pool is retired from future tuning and fresh-test claims.',
        'hidden_laws_published':False,'selections':selections,'examples':examples}
    write_immutable(data/'behavior-examples.json',publication)
    write_immutable(data/'exposed-cases.json',{'analysis_closure_sha256':publication['analysis_closure_sha256'],
        'whole_assessment_pool_retired':True,'seed_pool_read':False,
        'cases':[{'case':e['case'],'regime':e['regime'],'stratum':e['stratum']} for e in examples],
        'status':'Published retrospectively after numerical closure; exclude all assessment cases from tuning/future fresh tests.'})
    outputs=render_figures(examples,data/'figures')
    if args.gif and examples:
        first=examples[0]
        gif=render_gif(first,private[first['regime'],first['stratum'],first['case']],data/'figures/behavior-example.gif')
        if gif:outputs.append(gif)
    print(json.dumps({'examples':len(examples),'absent_strata':sum(s['case'] is None for s in selections),
                      'new_environment_episodes':0,'figures':outputs},indent=2))


if __name__=='__main__':main()
