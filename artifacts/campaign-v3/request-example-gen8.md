# System Instructions

Evolve an executable Python agent for the Namazu unknown-enemy-dynamics v3
experiment. Jointly change world_model_step(memory, local_obs, last_action),
planner(memory, local_obs), export_model, helpers, representations, memory and
predictive updating. Substantive new programs and algorithms are permitted;
this is not a fixed algorithm catalog or parameter-only search.

The world is 15x15 with a local 5x5 observation square without occlusion. Two
keys unlock a cardinally adjacent gating door and an exit. Three anonymous
enemies move independently. Interior dynamic walls change every 25 steps; fixed
borders and a 200-step horizon are retained. Actions: move=[dx,dy], each integer
in -1,0,1, plus interact=bool. Diagonal corner cutting is blocked. Interaction
opens adjacent doors before agent movement; keys are collected after movement.
Stepping onto an enemy kills before that enemy moves. A dynamic closure of an
occupied cell is skipped until the next scheduled tick.

The three equally weighted development regimes are: uniform nine attempted
movements including stay; one privately drawn stationary categorical movement
law per episode; and an unannounced within-episode change to an independent law.
Laws use Dirichlet(1,...,1), rejecting entropy<1.2nats or maximum probability>.6.
The changing regime switches at a private uniform integer step 25..75, with total
variation distance>=.3 between laws. All enemies share the current law. Blocked
attempted moves become stays; probabilities are NOT renormalized over legal
moves. Laws, switch times, enemy identities, hidden coordinates and seeds are
never revealed to ordinary candidates.

local_obs supplies grid(enemy overlay=5), terrain(0empty,1wall,2key,3locked door,
4exit,-1outside), step,health,keys,door_open, and feedback(actual displacement,
collected,opened). Coordinates in exported models use the initial agent position
as(0,0). Memory resets to None every episode. Terrain and localization memory are
state estimation; predictive adaptation means learning dynamics from experience.

The evaluator records export_model(memory,local_obs) after action choice and
before the next transition. Export enemy=[[relative_x,relative_y,p],...] and
default_enemy=p in[0,1], predicting NEXT-step anonymous occupancy at stationary
cells in initial-agent coordinates. Export terrain=[[x,y,cell,last_seen_step],...]
for reconstruction diagnostics and position=[x,y]. It scores every local 3x3 cell
and 12 random interior audit cells; missing probabilities default to.5. Invalid
numbers, actions or malformed exports fail evaluation. Prediction quality is
reported as proper Brier losses and actionable text feedback, but it is not the
fitness objective. Prediction and risk calculations used in decisions may use
any coherent representation; the export interface does not constrain planning.

Frozen objective absolute-task-v3: task=.65*escape+.10*keys+.10*door+
.05*escape*(1-steps/200), where keys is the raw number of collected keys (0, 1, or 2).
Maximize the mean absolute task component over all equally weighted regimes.
Do not optimize advantage over a disabled version or sabotage an intervention.
Forecast improvements only matter to selection when they improve actual control.
If your representation naturally permits predictive-update or prediction-use
interventions, preserve them separately from mapping/localization. Generic flags
are not required to force your algorithm into the seed's representation.

Execution is Python 3.10 standard library, 192 MiB address space, 10 CPU seconds per
episode, 3 wall seconds per decision, 512 KiB source, 2 MiB reply. OS isolation denies
repository/private files,network,new processes and evaluator inspection. Resource
limits regulate execution, not permissible algorithms. Use only this task and
the supplied parent/inspiration code,development text feedback and native meta
recommendations. Do not read other experiment outputs or assessment data. Return
a patch or full program in ShinkaEvolve's requested format.


# Potential Recommendations
The following are potential recommendations for the next program generation:

**Maintain enemy beliefs across visibility loss.** Replace the freshly initialized diffuse hidden intensity in `_forecast` with a persistent spatial belief propagated through the learned movement kernel, removing mass from currently observed empty cells and reconciling visible occupancy without assuming enemy identities. Preserve blocked-attempt behavior and separate occupancy from collision hazard; this extends the best program’s successful forecasting while potentially reducing both blind spots and unnecessary avoidance around recently cleared areas.
You MUST respond using an edit name, description, and the exact SEARCH/REPLACE diff format shown below to indicate changes:

<NAME>
A shortened name summarizing the edit you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
A description and argumentation process of the edit you are proposing.
</DESCRIPTION>

<DIFF>
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE

</DIFF>


Example of a valid diff format:
<DIFF>
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

</DIFF>

* You may only modify text that lies below a line containing "EVOLVE-BLOCK-START" and above the next "EVOLVE-BLOCK-END". Everything outside those markers is read-only.
* Do not repeat the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the SEARCH/REPLACE blocks.  
* Every block’s SEARCH section must be copied **verbatim** from the current file, including indentation.
* You can propose multiple independent edits. SEARCH/REPLACE blocks follow one after another. DO NOT ADD ANY OTHER TEXT BETWEEN THESE BLOCKS.
* Make sure the file still runs after your changes.

# Previous Messages

[]

# User Request

Here are the performance metrics of a set of previously implemented programs:

# Prior programs

```python
"""Predictive seed. Entire representation, learner and planner may evolve."""
# EVOLVE-BLOCK-START
import heapq
import math


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _blocked(m, p):
    return m["map"].get(p, 1) in (-1, 1) or (m["map"].get(p) == 3 and m["keys"] < 2)


def _legal(m, p, d):
    return (not _blocked(m, _add(p, d)) and not
            (d[0] and d[1] and (_blocked(m, _add(p, (d[0], 0))) or _blocked(m, _add(p, (0, d[1]))))))


def _feature(terrain, enemies, p):
    # Anonymous occupancy transitions: no matching or hidden enemy identities.
    if terrain.get(p, 1) in (-1, 1, 3):
        return "blocked"
    if p in enemies:
        openings = sum(terrain.get(_add(p, d), 1) not in (-1, 1, 3)
                       for d in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        return "occupied_tight" if openings <= 2 else "occupied_open"
    neighbors = sum(max(abs(e[0]-p[0]), abs(e[1]-p[1])) == 1 for e in enemies)
    return "near_many" if neighbors > 1 else "near" if neighbors else "far"


def world_model_step(memory, local_obs, last_action):
    if memory is None:
        prior = {"occupied_tight": .65, "occupied_open": .4, "near": .08,
                 "near_many": .16, "far": .005, "blocked": .001}
        memory = {"map": {}, "seen": {}, "pos": (0, 0), "visits": {},
                  "rates": {k: [v * 12, 12.] for k, v in prior.items()},
                  "updates": 0, "previous": None}
    m = memory
    m["pos"] = _add(m["pos"], local_obs["feedback"]["displacement"])
    m["keys"], m["door_open"], m["step"] = local_obs["keys"], local_obs["door_open"], local_obs["step"]
    m["visits"][m["pos"]] = m["visits"].get(m["pos"], 0) + 1
    terrain, enemies = {}, set()
    for y, row in enumerate(local_obs["terrain"]):
        for x, cell in enumerate(row):
            p = _add(m["pos"], (x-2, y-2))
            terrain[p] = cell
            m["map"][p], m["seen"][p] = cell, m["step"]
            if local_obs["grid"][y][x] == 5:
                enemies.add(p)
    # FREEZE_PREDICTIVE_UPDATES: ordinary localization/mapping above always run.
    if m["previous"] is not None and local_obs.get("learn", True):
        prev_terrain, prev_enemies, center = m["previous"]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                p = _add(center, (dx, dy))
                # Prior 3x3 is fully surrounded by observable 5x5; next label is
                # used ONLY now, after the previous forecast was scored.
                if p in terrain and terrain[p] not in (-1, 1, 3):
                    feature = _feature(prev_terrain, prev_enemies, p)
                    if feature != "blocked":
                        m["rates"][feature][0] += p in enemies
                        m["rates"][feature][1] += 1
                        m["updates"] += 1
    m["previous"] = (terrain, enemies, m["pos"])
    m["enemies"] = enemies
    m["risk"] = {}
    for p, cell in m["map"].items():
        if max(abs(p[0]-m["pos"][0]), abs(p[1]-m["pos"][1])) <= 1:
            feature = _feature(terrain, enemies, p)
            n, d = m["rates"][feature]
            m["risk"][p] = n / d if cell not in (-1, 1, 3) else 0.
        elif p in terrain:
            n, d = m["rates"][_feature(terrain, enemies, p)]
            m["risk"][p] = n / d if cell not in (-1, 1, 3) else 0.
        else:
            m["risk"][p] = .02 if cell not in (-1, 1, 3) else 0.
    return m


def _fixed_risk(m, p):
    # Competent hand-written baseline: conservative local enemy avoidance.
    if p in m["enemies"]:
        return 1.
    return .22 if any(max(abs(e[0]-p[0]), abs(e[1]-p[1])) <= 1 for e in m["enemies"]) else 0.


def planner(memory, local_obs):
    m = memory
    start = m["pos"]
    moves = [(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1) if x or y]
    dist, first, queue = {start: 0.}, {}, [(0., start)]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != dist[p]:
            continue
        for d in moves:
            q = _add(p, d)
            if not _legal(m, p, d) or q in m["enemies"]:
                continue
            risk = m["risk"].get(q, .02) if local_obs.get("predictive_planning", True) else _fixed_risk(m, q)
            edge = (1.4 if d[0] and d[1] else 1.) + 45 * risk
            new = cost + edge + .06 * min(m["visits"].get(q, 0), 15)
            if new < dist.get(q, float("inf")):
                dist[q], first[q] = new, d if p == start else first[p]
                heapq.heappush(queue, (new, q))
    goals = [p for p in dist if p != start and m["map"].get(p) == (2 if m["keys"] < 2 else 4)]
    if not goals and m["keys"] == 2 and not m["door_open"]:
        goals = [p for p in dist if p != start and m["map"].get(p) == 3]
    if goals:
        target = min(goals, key=lambda p: (dist[p], p))
    else:
        frontier = []
        for p in dist:
            if p == start:
                continue
            gain = sum(_add(p, (dx, dy)) not in m["map"] for dx in range(-2, 3) for dy in range(-2, 3))
            if gain:
                frontier.append((dist[p] / math.sqrt(gain), p))
        if frontier:
            target = min(frontier)[1]
        elif first:
            target = min(first, key=lambda p: (m["visits"].get(p, 0), dist[p], p))
        else:
            target = start
    move = first.get(target, (0, 0))
    # If the immediate learned/fixed forecast is high, waiting is another
    # legitimate action; compare it to the chosen move plus a small delay cost.
    risk_fn = (lambda p: m["risk"].get(p, .02)) if local_obs.get("predictive_planning", True) else (lambda p: _fixed_risk(m, p))
    if move != (0, 0) and risk_fn(_add(start, move)) > risk_fn(start) + .12:
        move = (0, 0)
    return {"move": list(move), "interact": True}


def export_model(memory, local_obs):
    m = memory
    return {"enemy": [[x, y, p] for (x, y), p in m["risk"].items()],
            "default_enemy": .02,
            "terrain": [[x, y, c, m["seen"][x, y]] for (x, y), c in m["map"].items()],
            "position": list(m["pos"]),
            "learning": {"updates": m["updates"], "rates": {k: a/b for k, (a, b) in m["rates"].items()}}}
# EVOLVE-BLOCK-END

```

Performance metrics:
Combined score to maximize: 0.80
episodes: 72; escape: 0.78; death: 0.21; timeout: 0.01; invalid: 0; task: 0.80; keys: 1.88; door: 0.78; steps: 51.61; model_score: 0.99; seconds: 0.20; candidate_cpu_seconds: 0.17; evaluator_cpu_seconds: 0.04; base_audit: 0.02; base_near: 0.01; brier_audit: 0.02; brier_audit_post_switch: 0.01; brier_audit_pre_switch: 0.02; brier_destination: 0.00; brier_near: 0.01; brier_near_post_switch: 0.00; brier_near_pre_switch: 0.01; brier_threat: 0.02; coverage: 0.60; map_age: 18.96; persistence_audit: 0.02; persistence_near: 0.01; persistence_threat: 0.04; prevalence_audit: 0.02; prevalence_destination: 0.00; prevalence_near: 0.01; reconstruction: 0.59; reconstruction_stale: 0.99; reconstruction_unseen: 0.00; reconstruction_visible: 1.00; escape_steps: 54.34; regimes: {'uniform': {'episodes': 24, 'escape': 0.7916666666666666, 'task': 0.8183541666666669, 'encountered_switch': 0, 'post_switch_steps': 0, 'post_switch_visible_enemy_steps': 0}, 'stationary': {'episodes': 24, 'escape': 0.7083333333333334, 'task': 0.7408020833333331, 'encountered_switch': 0, 'post_switch_steps': 0, 'post_switch_visible_enemy_steps': 0}, 'switch': {'episodes': 24, 'escape': 0.8333333333333334, 'task': 0.8383124999999999, 'encountered_switch': 14, 'post_switch_steps': 284, 'post_switch_visible_enemy_steps': 64}}

Text feedback:
namazu-unknown-dynamics-v3; absolute-task-v1. 72 paired development condition-episodes. Absolute task fitness; prediction losses are diagnostics: {"episodes": 72, "escape": 0.7777777777777778, "death": 0.20833333333333334, "timeout": 0.013888888888888888, "invalid": 0, "task": 0.7991562499999998, "keys": 1.875, "door": 0.7777777777777778, "steps": 51.611111111111114, "model_score": 0.9883203572878558, "seconds": 0.20086731408324945, "candidate_cpu_seconds": 0.16906706944444447, "evaluator_cpu_seconds": 0.04442126886111117, "base_audit": 0.016847793326157292, "base_near": 0.008006745604592752, "brier_audit": 0.016229963907654608, "brier_audit_post_switch": 0.012563311703297145, "brier_audit_pre_switch": 0.01653338151430888, "brier_destination": 0.0036963101743201646, "brier_near": 0.006127493610904402, "brier_near_post_switch": 0.004648418378631156, "brier_near_pre_switch": 0.006249887948306968, "brier_threat": 0.024908940792286567, "coverage": 0.5986724076067456, "map_age": 18.955311657177106, "persistence_audit": 0.016712791532113518, "persistence_near": 0.008760913766295897, "persistence_threat": 0.03561876975443715, "prevalence_audit": 0.017133118048080372, "prevalence_destination": 0.004036598493003229, "prevalence_near": 0.00792369333811745, "reconstruction": 0.5918774668101902, "reconstruction_stale": 0.9858714911871678, "reconstruction_unseen": 0.0, "reconstruction_visible": 1.0, "escape_steps": 54.339285714285715, "regimes": {"uniform": {"episodes": 24, "escape": 0.7916666666666666, "task": 0.8183541666666669, "encountered_switch": 0, "post_switch_steps": 0, "post_switch_visible_enemy_steps": 0}, "stationary": {"episodes": 24, "escape": 0.7083333333333334, "task": 0.7408020833333331, "encountered_switch": 0, "post_switch_steps": 0, "post_switch_visible_enemy_steps": 0}, "switch": {"episodes": 24, "escape": 0.8333333333333334, "task": 0.8383124999999999, "encountered_switch": 14, "post_switch_steps": 284, "post_switch_visible_enemy_steps": 64}}}. Improve escape/keys/door and escape efficiency under unknown stationary and switching attempted movement laws; blocked enemy attempts stay. Predictive feedback is horizon-1 occupancy at fixed cells in starting-agent coordinates: near targets surround the OLD agent position, audit targets sample interior cells; predictions precede outcomes. Threat losses concern currently visible nearby enemies; audit prevalence is sparse. All representations, memory, updating, helpers and planning may evolve. Do not infer adaptive prediction from map accumulation, or control benefit from lower Brier.


# Current program

Here is the current program we are trying to improve (you will need to propose a modification to it below):

```python
"""Predictive seed. Entire representation, learner and planner may evolve."""
# EVOLVE-BLOCK-START
import heapq
import math


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _blocked(m, p):
    return m["map"].get(p, 1) in (-1, 1) or (m["map"].get(p) == 3 and m["keys"] < 2)


def _legal(m, p, d):
    q = _add(p, d)
    if m["map"].get(q) == 3 and not m["door_open"] and d[0] and d[1]:
        return False
    return (not _blocked(m, q) and not
            (d[0] and d[1] and (_blocked(m, _add(p, (d[0], 0))) or _blocked(m, _add(p, (0, d[1]))))))


_DIRECTIONS = tuple((x, y) for y in (-1, 0, 1) for x in (-1, 0, 1))
def _enemy_destination(terrain, p, d):
    def blocked(q):
        return terrain.get(q, 0) in (-1, 1, 3)
    q = _add(p, d)
    if blocked(q):
        return p
    if d[0] and d[1]:
        if blocked(_add(p, (d[0], 0))) or blocked(_add(p, (0, d[1]))):
            return p
    return q
def _motion_weights(m):
    total = sum(m["motion"]) + 9.
    # A uniform mixture guards against unobserved changes in the law.
    return [.85 * (v + 1.) / total + .15 / 9. for v in m["motion"]]
def _learn_motion(m, terrain, enemies):
    m["motion"] = [v * .96 for v in m["motion"]]
    if m["previous"] is None:
        return
    old_terrain, old_enemies, old_step = m["previous"]
    # Skip transitions that may include scheduled geometry changes.
    if m["step"] != old_step + 1 or m["step"] // 25 != old_step // 25:
        return
    weights = _motion_weights(m)
    evidence = [0.] * 9
    for source in sorted(old_enemies):
        neighborhood = [_add(source, d) for d in _DIRECTIONS]
        if not all(p in old_terrain and p in terrain for p in neighborhood):
            continue
        # Current terrain includes any door opened before enemy movement.
        destinations = [_enemy_destination(terrain, source, d)
                        for d in _DIRECTIONS]
        possible = set(destinations) & enemies
        if len(possible) != 1:
            continue
        # With every destination observed, this source must have reached
        # the sole occupied destination. No enemy identity is required.
        destination = next(iter(possible))
        compatible = [i for i, q in enumerate(destinations) if q == destination]
        normalizer = sum(weights[i] for i in compatible)
        for i in compatible:
            evidence[i] += weights[i] / normalizer
        m["updates"] += 1
    m["motion"] = [a + b for a, b in zip(m["motion"], evidence)]
def _forecast(m, terrain, enemies):
    weights = _motion_weights(m)
    arrivals = {}
    for source in sorted(enemies):
        distribution = {}
        for d, probability in zip(_DIRECTIONS, weights):
            q = _enemy_destination(m["map"], source, d)
            distribution[q] = distribution.get(q, 0.) + probability
        for q, probability in distribution.items():
            arrivals[q] = 1. - (1. - arrivals.get(q, 0.)) * (1. - probability)
    risk = {}
    unseen = max(0, 3 - len(enemies))
    for p, cell in m["map"].items():
        if cell in (-1, 1, 3):
            risk[p] = 0.
            continue
        distance = max(abs(p[0] - m["pos"][0]), abs(p[1] - m["pos"][1]))
        # Every predecessor of an inner-square cell is currently visible.
        background = 0. if distance <= 1 else .008 * unseen
        risk[p] = min(1., max(0., 1. - (1. - arrivals.get(p, 0.)) * (1. - background)))
    return risk


def world_model_step(memory, local_obs, last_action):
    if memory is None:
        memory = {"map": {}, "seen": {}, "pos": (0, 0), "visits": {},
                  "motion": [0.] * 9,
                  "updates": 0, "previous": None}
    m = memory
    m["pos"] = _add(m["pos"], local_obs["feedback"]["displacement"])
    m["keys"], m["door_open"], m["step"] = local_obs["keys"], local_obs["door_open"], local_obs["step"]
    m["visits"][m["pos"]] = m["visits"].get(m["pos"], 0) + 1
    terrain, enemies = {}, set()
    for y, row in enumerate(local_obs["terrain"]):
        for x, cell in enumerate(row):
            p = _add(m["pos"], (x-2, y-2))
            terrain[p] = cell
            m["map"][p], m["seen"][p] = cell, m["step"]
            if local_obs["grid"][y][x] == 5:
                enemies.add(p)
    # FREEZE_PREDICTIVE_UPDATES leaves mapping and localization intact.
    if local_obs.get("learn", True):
        _learn_motion(m, terrain, enemies)
    m["previous"] = (terrain, enemies, m["step"])
    m["enemies"] = enemies
    m["risk"] = _forecast(m, terrain, enemies)
    return m


def _fixed_risk(m, p):
    # Competent hand-written baseline: conservative local enemy avoidance.
    if p in m["enemies"]:
        return 1.
    return .22 if any(max(abs(e[0]-p[0]), abs(e[1]-p[1])) <= 1 for e in m["enemies"]) else 0.


def planner(memory, local_obs):
    m = memory
    start = m["pos"]
    moves = [(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1) if x or y]
    dist, first, queue = {start: 0.}, {}, [(0., start)]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != dist[p]:
            continue
        for d in moves:
            q = _add(p, d)
            if not _legal(m, p, d) or q in m["enemies"]:
                continue
            risk = m["risk"].get(q, .02) if local_obs.get("predictive_planning", True) else _fixed_risk(m, q)
            edge = 1. + 45 * risk
            new = cost + edge + .06 * min(m["visits"].get(q, 0), 15)
            if new < dist.get(q, float("inf")):
                dist[q], first[q] = new, d if p == start else first[p]
                heapq.heappush(queue, (new, q))
    goals = [p for p in dist if p != start and m["map"].get(p) == (2 if m["keys"] < 2 else 4)]
    if not goals and m["keys"] == 2 and not m["door_open"]:
        goals = [p for p in dist if p != start and m["map"].get(p) == 3]
    if goals:
        target = min(goals, key=lambda p: (dist[p], p))
    else:
        frontier = []
        for p in dist:
            if p == start:
                continue
            gain = sum(_add(p, (dx, dy)) not in m["map"] for dx in range(-2, 3) for dy in range(-2, 3))
            if gain:
                frontier.append((dist[p] / math.sqrt(gain), p))
        if frontier:
            target = min(frontier)[1]
        elif first:
            target = min(first, key=lambda p: (m["visits"].get(p, 0), dist[p], p))
        else:
            target = start
    move = first.get(target, (0, 0))
    # If the immediate learned/fixed forecast is high, waiting is another
    # legitimate action; compare it to the chosen move plus a small delay cost.
    risk_fn = (lambda p: m["risk"].get(p, .02)) if local_obs.get("predictive_planning", True) else (lambda p: _fixed_risk(m, p))
    if (move != (0, 0) and start not in m["enemies"]
            and risk_fn(_add(start, move)) > risk_fn(start) + .055):
        move = (0, 0)
    return {"move": list(move), "interact": True}


def export_model(memory, local_obs):
    m = memory
    return {"enemy": [[x, y, p] for (x, y), p in m["risk"].items()],
            "default_enemy": .02,
            "terrain": [[x, y, c, m["seen"][x, y]] for (x, y), c in m["map"].items()],
            "position": list(m["pos"]),
            "learning": {"updates": m["updates"],
                         "attempted_motion": [[d[0], d[1], p]
                                              for d, p in zip(_DIRECTIONS, _motion_weights(m))]}}
# EVOLVE-BLOCK-END
```

Here are the performance metrics of the program:

Combined score to maximize: 0.89
episodes: 72; escape: 0.89; death: 0.08; timeout: 0.03; invalid: 0; task: 0.89; keys: 1.92; door: 0.89; steps: 55.11; model_score: 0.99; seconds: 0.16; candidate_cpu_seconds: 0.14; evaluator_cpu_seconds: 0.03; base_audit: 0.02; base_near: 0.01; brier_audit: 0.02; brier_audit_post_switch: 0.01; brier_audit_pre_switch: 0.02; brier_destination: 0.00; brier_near: 0.01; brier_near_post_switch: 0.00; brier_near_pre_switch: 0.01; brier_threat: 0.02; coverage: 0.65; map_age: 19.38; persistence_audit: 0.02; persistence_near: 0.01; persistence_threat: 0.03; prevalence_audit: 0.02; prevalence_destination: 0.00; prevalence_near: 0.01; reconstruction: 0.64; reconstruction_stale: 0.98; reconstruction_unseen: 0.00; reconstruction_visible: 1.00; escape_steps: 52.64; regimes: {'uniform': {'episodes': 24, 'escape': 0.9166666666666666, 'task': 0.921197916666667, 'encountered_switch': 0, 'post_switch_steps': 0, 'post_switch_visible_enemy_steps': 0}, 'stationary': {'episodes': 24, 'escape': 0.8333333333333334, 'task': 0.8433333333333333, 'encountered_switch': 0, 'post_switch_steps': 0, 'post_switch_visible_enemy_steps': 0}, 'switch': {'episodes': 24, 'escape': 0.9166666666666666, 'task': 0.9087083333333332, 'encountered_switch': 15, 'post_switch_steps': 234, 'post_switch_visible_enemy_steps': 50}}

Here is additional text feedback about the current program:

namazu-unknown-dynamics-v3; absolute-task-v1. 72 paired development condition-episodes. Absolute task fitness; prediction losses are diagnostics: {"episodes": 72, "escape": 0.8888888888888888, "death": 0.08333333333333333, "timeout": 0.027777777777777776, "invalid": 0, "task": 0.891079861111111, "keys": 1.9166666666666667, "door": 0.8888888888888888, "steps": 55.111111111111114, "model_score": 0.9886695121409562, "seconds": 0.15741675806889865, "candidate_cpu_seconds": 0.1353055277777778, "evaluator_cpu_seconds": 0.03365772966666663, "base_audit": 0.017335483870967845, "base_near": 0.007496774193548394, "brier_audit": 0.016264233071974606, "brier_audit_post_switch": 0.01361971083596312, "brier_audit_pre_switch": 0.016429958354038526, "brier_destination": 0.001371405219698934, "brier_near": 0.005497247382353348, "brier_near_post_switch": 0.0036512048683085847, "brier_near_pre_switch": 0.005612934031599861, "brier_threat": 0.01945858841496707, "coverage": 0.6466943884408602, "map_age": 19.3829117007112, "persistence_audit": 0.016617103494623777, "persistence_near": 0.008792562724014337, "persistence_threat": 0.03112300525324611, "prevalence_audit": 0.017641129032258066, "prevalence_destination": 0.0015120967741935483, "prevalence_near": 0.00739247311827957, "reconstruction": 0.6381468413978495, "reconstruction_stale": 0.9838344520792787, "reconstruction_unseen": 0.0, "reconstruction_visible": 1.0, "escape_steps": 52.640625, "regimes": {"uniform": {"episodes": 24, "escape": 0.9166666666666666, "task": 0.921197916666667, "encountered_switch": 0, "post_switch_steps": 0, "post_switch_visible_enemy_steps": 0}, "stationary": {"episodes": 24, "escape": 0.8333333333333334, "task": 0.8433333333333333, "encountered_switch": 0, "post_switch_steps": 0, "post_switch_visible_enemy_steps": 0}, "switch": {"episodes": 24, "escape": 0.9166666666666666, "task": 0.9087083333333332, "encountered_switch": 15, "post_switch_steps": 234, "post_switch_visible_enemy_steps": 50}}}. Improve escape/keys/door and escape efficiency under unknown stationary and switching attempted movement laws; blocked enemy attempts stay. Predictive feedback is horizon-1 occupancy at fixed cells in starting-agent coordinates: near targets surround the OLD agent position, audit targets sample interior cells; predictions precede outcomes. Threat losses concern currently visible nearby enemies; audit prevalence is sparse. All representations, memory, updating, helpers and planning may evolve. Do not infer adaptive prediction from map accumulation, or control benefit from lower Brier.


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
