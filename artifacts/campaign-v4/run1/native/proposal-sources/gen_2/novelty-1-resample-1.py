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