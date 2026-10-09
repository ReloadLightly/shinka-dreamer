# System Instructions

Evolve a complete executable Python world-model learner and planner for the
Namazu persistence/information maze experiment (wave v4 RUN 1). Jointly change
world_model_step(memory, local_obs, last_action), planner(memory, local_obs),
export_model, helpers, representations, inference, learning and planning.
Substantive new programs are permitted, not an algorithm catalog or weight-only
search. Start from and improve the supplied parent using native inspirations.

Physical task: 15x15 dynamic maze, 5x5 terrain observation, two keys, a gating door
and exit, three anonymous enemies, 200 transitions. Walls change every25steps.
Actions {'move':[dx,dy], 'interact':bool} with integer dx,dy in[-1,1], including
waiting. No diagonal corner cutting. Interact opens a cardinally adjacent door
when both keys have been collected, before movement. Keys are collected after
movement. Entering a currently occupied enemy cell kills BEFORE enemies move;
then all enemies attempt their movement, including on the terminal tick. Blocked
enemy attempts STAY; never renormalize their law over legal moves. Enemies share
one unknown categorical law over the nine attempted directions. They have no
observable identities. Occupied dynamic-wall closures are skipped until the
next scheduled tick. Terrain and anonymous occupancy must remain distinct.

Six equally weighted development conditions: uniform/full enemy observation,
stationary/full, one-switch/full, stationary/late, repeated-25/full and
repeated-25/late. Full means radius2; late means radius1 enemy visibility, while
terrain remains visible throughout the5x5window. The ordinary observation
explicitly provides enemy_visibility_radius. Outside it, grid equals terrain,
so lack of enemy overlay is NOT evidence of an empty cell. Stationary laws are
privately drawn from Dirichlet(1,...,1), rejecting entropy<1.2nats or max>.6.
One-switch first replacement transition is privately uniform25..75. Repeated
laws persist for privately drawn integer20..30transitions (mean25). Consecutive laws differ by total
variation>=.3. Ordinary candidates receive no laws, identities, seeds, actual switch
schedule, future innovations, private files or hidden world positions.

Observation fields: grid(enemy=5), terrain(0empty,1wall,2key,3locked door,4exit,
-1outside), enemy_visibility_radius, step, health, keys(integer0..2), door_open,
feedback(displacement,collected,opened), learn, predictive_planning. Memory
resets to None per episode. Mapping/localization/occupancy inference alone are
not evidence of predictive adaptation. Generic intervention flags need not
force a new representation into a fixed family; preserve coherent hooks if
natural, but your absolute task score governs selection.

Primary fitness is the mean absolute task score over a fixed48episode panel:
.65*escape+.10*number_of_keys+.10*door_open+.05*escape*(1-steps/200).
The two keys contribute at most.20; invalid execution scores0. No reward for
merely waiting. Never maximize advantage over your own disabled counterpart.
Prediction loss is diagnostic textual feedback, not part of fitness.

export_model(memory,local_obs) runs after planning, BEFORE the transition.
Export enemy=[[x,y,p],...] and default_enemy=p in[0,1] for NEXT-transition
anonymous occupancy at FIXED cells in initial-agent coordinates (initial
position=(0,0)). Export terrain=[[x,y,cell,last_seen_step],...] and optionally
position=[x,y]. Missing cell forecasts use default_enemy, default.5. Near Brier
scores cover the OLD agent-centered3x3; audit samples12interiorcells. Destination
Brier measures enemy occupancy AFTER enemy movement at the actual destination,
not total collision probability: entry-contact risk is separate. Forecast
coordinates/horizon/conditioning may differ from internal planning risks; be
precise about how prediction informs choices. Malformed actions/exports fail.

Development evidence motivating this wave (not assessment results): historical
v3 task performance improved rapidly, but many learned-vs-prior action changes
occurred where both actions had zero immediate estimated risk. A bounded new
selected-planner branch diagnostic held the learned state fixed, changed only
one first action, then used the same observation-only continuation for12steps.
For40 valid learned-vs-prior changed-action/future pairs, there was no collision
or escape difference;39returns were identical and one lost a key under the
learned recommendation. Only6disagreementstates had valid consequences and
5others were missing because a memory-check validator failed. This small,
selected, short-horizon negative result does not exclude useful adaptation.
Look for consequential ranking errors, uncertainty, stale evidence, anonymous
matching ambiguity and observations arriving too late to learn. A better generic
occupancy loss need not improve control. Track/update useful dynamics only where
its learning/inference cost and persistence make it relevant. All algorithms
remain open, including active information acquisition and uncertainty-aware
planning. No particular hand-designed adaptation architecture is required.

Runtime: Python3.10 standard library only, 192MiB address space,10CPU seconds per
episode,3wall seconds per reply,512KiB source,2MiB reply. Isolation denies network,
process creation, repository/private files and evaluator inspection. These
limits regulate execution rather than permissible algorithm families. Use the
provided task, parent/inspiration code, development feedback and native meta
recommendations. Return the patch/full code requested by the native operator.


# Potential Recommendations
The following are potential recommendations for the next program generation:

**Add fast and slow occupancy estimators with evidence-driven switching.** Extend the successful feature-grouped learner with recent, decayed counts alongside its cumulative counts, blending toward the recent estimator when scored occupancy errors show sustained disagreement. Update this mechanism only from observable transition labels and respect `learn=False`; the higher post-switch near loss motivates adaptation without assuming a fixed switching schedule.

Design a completely different algorithm approach to solve the same problem.
Ignore the current implementation and think of alternative algorithmic strategies that could achieve better performance.
You MUST respond using a short summary name, description and the full code:

<NAME>
A shortened name summarizing the code you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
Explain the completely different algorithmic approach you are taking and why it should perform better than the current implementation.
</DESCRIPTION>

<CODE>
```{language}
# The completely new algorithm implementation here.
```
</CODE>

* Keep the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the code.
* Your algorithm should solve the same problem but use a fundamentally different approach.
* Ensure the same inputs and outputs are maintained.
* Think outside the box - consider different data structures, algorithms, or paradigms.
* Use the <NAME>, <DESCRIPTION>, and <CODE> delimiters to structure your response. It will be parsed afterwards.

NON-EVOLVING EXPERIMENT BOUNDARY: Preserve observation-only world_model_step/planner/export_model interface and whole-program search. Hidden laws, identities, seeds, future innovations and private case pools are inaccessible; do not attempt to inspect them. Evaluator, score, 15x15 maze, candidate isolation and resource limits are fixed outside candidate scope. All tools/network/model calls by candidate code are forbidden. Improve absolute task performance; never weaken a disabled/frozen counterpart to manufacture an adaptation effect. Prediction diagnostics are feedback, not selection fitness.

Generation1 was killed by a scheduler timing bug before any episodes; its score0 is an infrastructure failure, not evidence about its algorithm. The timer now starts at evaluation launch.

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
import itertools


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


_DIRS = tuple((x, y) for y in (-1, 0, 1) for x in (-1, 0, 1))


def _enemy_destination(terrain, p, d):
    # Attempt probabilities are never renormalized over legal moves.
    q = _add(p, d)
    blocked = lambda z: terrain.get(z, 0) in (-1, 1, 3)
    if blocked(q):
        return p
    if d[0] and d[1]:
        if blocked(_add(p, (d[0], 0))) or blocked(_add(p, (0, d[1]))):
            return p
    return q


def _motion_evidence(terrain, old_enemies, visible, enemies, law):
    alternatives = []
    for p in sorted(old_enemies):
        # Learn only where the previous terrain resolves every attempted move.
        if not all(_add(p, d) in terrain for d in _DIRS):
            continue
        destinations = [_enemy_destination(terrain, p, d) for d in _DIRS]
        if not any(q in visible for q in destinations):
            continue
        choices = [(i, q) for i, q in enumerate(destinations)
                   if q not in visible or q in enemies]
        if not choices:
            return None
        alternatives.append(choices)
    if not alternatives:
        return None
    total = 0.
    evidence = [0.] * 9
    # At most three sources and nine attempted directions per source.
    # Multiple sources may explain the same anonymous occupied cell.
    for assignment in itertools.product(*alternatives):
        explained = {q for _, q in assignment if q in visible}
        weight = .04 ** len(enemies - explained)
        for i, q in assignment:
            weight *= law[i]
        total += weight
        for i, q in assignment:
            evidence[i] += weight
    if total <= 1e-30:
        return None
    return [v / total for v in evidence]


def _occupancy_forecast(terrain, belief, law):
    empty = {}
    for p, occupancy in belief.items():
        if occupancy <= 1e-8:
            continue
        transitions = {}
        for d, probability in zip(_DIRS, law):
            q = _enemy_destination(terrain, p, d)
            transitions[q] = transitions.get(q, 0.) + probability
        for q, probability in transitions.items():
            empty[q] = empty.get(q, 1.) * (1. - occupancy * probability)
    return {p: max(0., min(1., 1. - probability))
            for p, probability in empty.items()}


def world_model_step(memory, local_obs, last_action):
    if memory is None:
        prior = {"occupied_tight": .65, "occupied_open": .4, "near": .08,
                 "near_many": .16, "far": .005, "blocked": .001}
        memory = {"map": {}, "seen": {}, "pos": (0, 0), "visits": {},
                  "rates": {k: [v * 12, 12.] for k, v in prior.items()},
                  "updates": 0, "previous": None,
                  "motion_counts": [0.] * 9, "forecast": {}}
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
    radius = int(local_obs.get("enemy_visibility_radius", 2))
    visible = {p for p in terrain
               if max(abs(p[0]-m["pos"][0]), abs(p[1]-m["pos"][1])) <= radius}
    enemies.intersection_update(visible)

    # FREEZE_PREDICTIVE_UPDATES: mapping and occupancy filtering still run.
    counts = m["motion_counts"]
    if local_obs.get("learn", True):
        counts = [v * .92 for v in counts]
        denominator = sum(counts) + 6.75
        prior_law = [(v + .75) / denominator for v in counts]
        if m["previous"] is not None and m["step"] % 25 != 0:
            prev_terrain, prev_enemies, center = m["previous"]
            evidence = _motion_evidence(
                prev_terrain, prev_enemies, visible, enemies, prior_law)
            if evidence is not None:
                counts = [a + b for a, b in zip(counts, evidence)]
                m["updates"] += 1
        m["motion_counts"] = counts
    denominator = sum(counts) + 6.75
    # A residual uniform component protects against sparse evidence and switches.
    law = [.8 * (v + .75) / denominator + .2 / 9 for v in counts]
    m["law"] = law

    # The planner always interacts. Condition the transition on doors that this
    # action opens now, while preserving the observed terrain in the export.
    transition_terrain = dict(m["map"])
    previous_terrain = dict(terrain)
    if m["keys"] == 2:
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = _add(m["pos"], d)
            if transition_terrain.get(p) == 3:
                transition_terrain[p] = 0
                previous_terrain[p] = 0

    # Absence is observed only inside the enemy visibility radius. Outside it,
    # propagate the previous forecast instead of erasing possible enemies.
    support = set(m["map"])
    for p, cell in terrain.items():
        if cell not in (-1, 1, 3):
            support.update(_add(p, d) for d in _DIRS)
    belief = {}
    for p in support:
        if p in visible:
            belief[p] = float(p in enemies)
        elif transition_terrain.get(p, 0) not in (-1, 1, 3):
            belief[p] = min(.95, m["forecast"].get(p, .02))
    m["belief"] = belief
    m["enemies"] = enemies
    m["previous"] = (previous_terrain, set(enemies), m["pos"])
    forecast = _occupancy_forecast(transition_terrain, belief, law)
    m["forecast"] = forecast
    m["risk"] = {
        p: (0. if transition_terrain.get(p, 0) in (-1, 1, 3)
            else forecast.get(p, .02))
        for p in support
    }
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
            # A diagonally adjacent locked door cannot be opened this tick.
            if p == start and d[0] and d[1] and m["map"].get(q) == 3:
                continue
            risk = m["risk"].get(q, .02) if local_obs.get("predictive_planning", True) else _fixed_risk(m, q)
            distance = max(abs(q[0]-start[0]), abs(q[1]-start[1]))
            # Horizon-one occupancy is most relevant to the immediate action.
            relevance = .8 ** max(0, distance - 1)
            edge = (1.4 if d[0] and d[1] else 1.) + 45 * relevance * risk
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
    delay_cost = .055 + .004 * min(m["visits"].get(start, 0), 10)
    if move != (0, 0) and risk_fn(_add(start, move)) > risk_fn(start) + delay_cost:
        move = (0, 0)
    return {"move": list(move), "interact": True}


def export_model(memory, local_obs):
    m = memory
    return {"enemy": [[x, y, p] for (x, y), p in m["risk"].items()],
            "default_enemy": .02,
            "terrain": [[x, y, c, m["seen"][x, y]] for (x, y), c in m["map"].items()],
            "position": list(m["pos"]),
            "learning": {"updates": m["updates"],
                         "attempted_directions": [[dx, dy, p]
                             for (dx, dy), p in zip(_DIRS, m["law"])],
                         "effective_evidence": sum(m["motion_counts"])}}
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: 0.81
episodes: 48; escape: 0.81; death: 0.19; timeout: 0.00; invalid: 0; task: 0.81; keys: 1.73; door: 0.81; steps: 52.52; model_score: 0.99; seconds: 0.27; candidate_cpu_seconds: 0.22; evaluator_cpu_seconds: 0.05; base_audit: 0.02; base_near: 0.01; brier_audit: 0.02; brier_audit_post_switch: 0.02; brier_audit_pre_switch: 0.02; brier_destination: 0.00; brier_near: 0.01; brier_near_post_switch: 0.01; brier_near_pre_switch: 0.01; brier_threat: 0.03; coverage: 0.62; map_age: 14.81; persistence_audit: 0.02; persistence_near: 0.01; persistence_threat: 0.05; prevalence_audit: 0.02; prevalence_destination: 0.00; prevalence_near: 0.01; reconstruction: 0.61; reconstruction_stale: 0.98; reconstruction_unseen: 0.00; reconstruction_visible: 1.00; escape_steps: 59.03; regimes: {'uniform-full': {'episodes': 8, 'escape': 0.75, 'task': 0.7383125, 'encountered_switch': 0, 'post_switch_steps': 0, 'post_switch_visible_enemy_steps': 0}, 'stationary-full': {'episodes': 8, 'escape': 0.875, 'task': 0.8742500000000001, 'encountered_switch': 0, 'post_switch_steps': 0, 'post_switch_visible_enemy_steps': 0}, 'switch-full': {'episodes': 8, 'escape': 1.0, 'task': 0.98440625, 'encountered_switch': 4, 'post_switch_steps': 95, 'post_switch_visible_enemy_steps': 18}, 'stationary-late': {'episodes': 8, 'escape': 0.625, 'task': 0.6300625, 'encountered_switch': 0, 'post_switch_steps': 0, 'post_switch_visible_enemy_steps': 0}, 'repeat25-full': {'episodes': 8, 'escape': 1.0, 'task': 0.9841875000000002, 'encountered_switch': 8, 'post_switch_steps': 303, 'post_switch_visible_enemy_steps': 81}, 'repeat25-late': {'episodes': 8, 'escape': 0.625, 'task': 0.65434375, 'encountered_switch': 6, 'post_switch_steps': 156, 'post_switch_visible_enemy_steps': 14}}

Text feedback:
namazu-persistence-information-v4-run1; absolute-task-v1. 48 paired development condition-episodes. Absolute task fitness; prediction losses are diagnostics: {"episodes": 48, "escape": 0.8125, "death": 0.1875, "timeout": 0.0, "invalid": 0, "task": 0.8109270833333335, "keys": 1.7291666666666667, "door": 0.8125, "steps": 52.520833333333336, "model_score": 0.9866762743756426, "seconds": 0.26527566587719775, "candidate_cpu_seconds": 0.21827787500000004, "evaluator_cpu_seconds": 0.04729401027083332, "base_audit": 0.017282189607298822, "base_near": 0.008058336638899894, "brier_audit": 0.0170203631930432, "brier_audit_post_switch": 0.016883899994604122, "brier_audit_pre_switch": 0.017058797667845047, "brier_destination": 0.0036789282191833825, "brier_near": 0.0061720643093338195, "brier_near_post_switch": 0.008406691899935623, "brier_near_pre_switch": 0.005542687753566965, "brier_threat": 0.027495162662379214, "coverage": 0.6217109612587598, "map_age": 14.806890684814972, "persistence_audit": 0.01794623826523878, "persistence_near": 0.009872625501344264, "persistence_threat": 0.0454176804541768, "prevalence_audit": 0.01758561417426947, "prevalence_destination": 0.0035700119000396666, "prevalence_near": 0.00797743399885407, "reconstruction": 0.6115298162104985, "reconstruction_stale": 0.9796444385698235, "reconstruction_unseen": 0.0, "reconstruction_visible": 1.0, "escape_steps": 59.02564102564103, "regimes": {"uniform-full": {"episodes": 8, "escape": 0.75, "task": 0.7383125, "encountered_switch": 0, "post_switch_steps": 0, "post_switch_visible_enemy_steps": 0}, "stationary-full": {"episodes": 8, "escape": 0.875, "task": 0.8742500000000001, "encountered_switch": 0, "post_switch_steps": 0, "post_switch_visible_enemy_steps": 0}, "switch-full": {"episodes": 8, "escape": 1.0, "task": 0.98440625, "encountered_switch": 4, "post_switch_steps": 95, "post_switch_visible_enemy_steps": 18}, "stationary-late": {"episodes": 8, "escape": 0.625, "task": 0.6300625, "encountered_switch": 0, "post_switch_steps": 0, "post_switch_visible_enemy_steps": 0}, "repeat25-full": {"episodes": 8, "escape": 1.0, "task": 0.9841875000000002, "encountered_switch": 8, "post_switch_steps": 303, "post_switch_visible_enemy_steps": 81}, "repeat25-late": {"episodes": 8, "escape": 0.625, "task": 0.65434375, "encountered_switch": 6, "post_switch_steps": 156, "post_switch_visible_enemy_steps": 14}}}. Improve escape/keys/door and escape efficiency under unknown stationary and switching attempted movement laws; blocked enemy attempts stay. Predictive feedback is horizon-1 occupancy at fixed cells in starting-agent coordinates: near targets surround the OLD agent position, audit targets sample interior cells; predictions precede outcomes. Threat losses condition on evaluator-known enemy proximity; audit prevalence is sparse. Sensor radius is public; outside it enemy absence is unobserved. Repeated law dwell times are privately drawn from 20..30 transitions; prediction adaptation may be too slow or decision-irrelevant. Check informative exposure before consequential choices. All representations, memory, updating, helpers and planning may evolve. Do not infer adaptive prediction from map accumulation, or control benefit from lower Brier.


# Current program

Here is the current program we are trying to improve (you will need to propose a new program with the same inputs and outputs as the original program, but with improved internal implementation):

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

Here are the performance metrics of the program:

Combined score to maximize: 0.86
episodes: 48; escape: 0.85; death: 0.15; timeout: 0.00; invalid: 0; task: 0.86; keys: 1.90; door: 0.85; steps: 53.40; model_score: 0.99; seconds: 0.23; candidate_cpu_seconds: 0.18; evaluator_cpu_seconds: 0.05; base_audit: 0.02; base_near: 0.01; brier_audit: 0.02; brier_audit_post_switch: 0.02; brier_audit_pre_switch: 0.02; brier_destination: 0.00; brier_near: 0.01; brier_near_post_switch: 0.01; brier_near_pre_switch: 0.01; brier_threat: 0.03; coverage: 0.59; map_age: 14.01; persistence_audit: 0.02; persistence_near: 0.01; persistence_threat: 0.04; prevalence_audit: 0.02; prevalence_destination: 0.00; prevalence_near: 0.01; reconstruction: 0.58; reconstruction_stale: 0.98; reconstruction_unseen: 0.00; reconstruction_visible: 1.00; escape_steps: 55.29; regimes: {'uniform-full': {'episodes': 8, 'escape': 0.875, 'task': 0.8886875, 'encountered_switch': 0, 'post_switch_steps': 0, 'post_switch_visible_enemy_steps': 0}, 'stationary-full': {'episodes': 8, 'escape': 0.875, 'task': 0.8759687500000002, 'encountered_switch': 0, 'post_switch_steps': 0, 'post_switch_visible_enemy_steps': 0}, 'switch-full': {'episodes': 8, 'escape': 1.0, 'task': 0.98615625, 'encountered_switch': 2, 'post_switch_steps': 65, 'post_switch_visible_enemy_steps': 21}, 'stationary-late': {'episodes': 8, 'escape': 0.875, 'task': 0.8747187500000001, 'encountered_switch': 0, 'post_switch_steps': 0, 'post_switch_visible_enemy_steps': 0}, 'repeat25-full': {'episodes': 8, 'escape': 0.75, 'task': 0.7643125, 'encountered_switch': 8, 'post_switch_steps': 222, 'post_switch_visible_enemy_steps': 55}, 'repeat25-late': {'episodes': 8, 'escape': 0.75, 'task': 0.7768125000000001, 'encountered_switch': 8, 'post_switch_steps': 225, 'post_switch_visible_enemy_steps': 27}}

Here is additional text feedback about the current program:

namazu-persistence-information-v4-run1; absolute-task-v1. 48 paired development condition-episodes. Absolute task fitness; prediction losses are diagnostics: {"episodes": 48, "escape": 0.8541666666666666, "death": 0.14583333333333334, "timeout": 0.0, "invalid": 0, "task": 0.8611093750000004, "keys": 1.8958333333333333, "door": 0.8541666666666666, "steps": 53.395833333333336, "model_score": 0.9887326187532066, "seconds": 0.22787502025069747, "candidate_cpu_seconds": 0.1758874791666667, "evaluator_cpu_seconds": 0.05244417900000001, "base_audit": 0.017536168552477718, "base_near": 0.007391806476785009, "brier_audit": 0.017286005929940396, "brier_audit_post_switch": 0.016507599169363635, "brier_audit_pre_switch": 0.017480322975974114, "brier_destination": 0.0026213767456624, "brier_near": 0.005742449637441384, "brier_near_post_switch": 0.00869704098703402, "brier_near_pre_switch": 0.005004882221063309, "brier_threat": 0.02885108368038207, "coverage": 0.589315905839511, "map_age": 14.006124137931035, "persistence_audit": 0.01776804525946172, "persistence_near": 0.00836693111371223, "persistence_threat": 0.0420479302832244, "prevalence_audit": 0.017850175575497465, "prevalence_destination": 0.0027311744049941474, "prevalence_near": 0.0072831317466510595, "reconstruction": 0.5811548966055404, "reconstruction_stale": 0.9825524815793132, "reconstruction_unseen": 0.0, "reconstruction_visible": 1.0, "escape_steps": 55.292682926829265, "regimes": {"uniform-full": {"episodes": 8, "escape": 0.875, "task": 0.8886875, "encountered_switch": 0, "post_switch_steps": 0, "post_switch_visible_enemy_steps": 0}, "stationary-full": {"episodes": 8, "escape": 0.875, "task": 0.8759687500000002, "encountered_switch": 0, "post_switch_steps": 0, "post_switch_visible_enemy_steps": 0}, "switch-full": {"episodes": 8, "escape": 1.0, "task": 0.98615625, "encountered_switch": 2, "post_switch_steps": 65, "post_switch_visible_enemy_steps": 21}, "stationary-late": {"episodes": 8, "escape": 0.875, "task": 0.8747187500000001, "encountered_switch": 0, "post_switch_steps": 0, "post_switch_visible_enemy_steps": 0}, "repeat25-full": {"episodes": 8, "escape": 0.75, "task": 0.7643125, "encountered_switch": 8, "post_switch_steps": 222, "post_switch_visible_enemy_steps": 55}, "repeat25-late": {"episodes": 8, "escape": 0.75, "task": 0.7768125000000001, "encountered_switch": 8, "post_switch_steps": 225, "post_switch_visible_enemy_steps": 27}}}. Improve escape/keys/door and escape efficiency under unknown stationary and switching attempted movement laws; blocked enemy attempts stay. Predictive feedback is horizon-1 occupancy at fixed cells in starting-agent coordinates: near targets surround the OLD agent position, audit targets sample interior cells; predictions precede outcomes. Threat losses condition on evaluator-known enemy proximity; audit prevalence is sparse. Sensor radius is public; outside it enemy absence is unobserved. Repeated law dwell times are privately drawn from 20..30 transitions; prediction adaptation may be too slow or decision-irrelevant. Check informative exposure before consequential choices. All representations, memory, updating, helpers and planning may evolve. Do not infer adaptive prediction from map accumulation, or control benefit from lower Brier.


# Task

Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.
