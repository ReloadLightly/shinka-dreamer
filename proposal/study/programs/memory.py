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

# Immutable v1 memory/pathfinding control: bind the original interventions here.
# No imports from the evolving program; evaluator supplies persistence forecasts.
_original_world_model_step = world_model_step
_original_planner = planner

def world_model_step(memory, local_obs, last_action):
    return _original_world_model_step(memory, dict(local_obs, learn=False), last_action)

def planner(memory, local_obs):
    return _original_planner(memory, dict(local_obs, predictive_planning=False))


# Original-task adapter appended to the immutable v1 memory source above.
# The ancestor already binds learning and predictive planning off.
_study_memory_updater = world_model_step
_study_memory_planner = planner
_study_memory_legal = _legal

def _legal(m, p, d):
    # Interaction opens only a cardinally adjacent closed door before movement.
    if (d[0] and d[1] and not m["door_open"]
            and m["map"].get(_add(p, d)) == 3):
        return False
    return _study_memory_legal(m, p, d)

def world_model_step(memory, local_obs, last_action):
    m = _study_memory_updater(memory, local_obs, last_action)
    m["believed_map"] = {p: cell for p, cell in m["map"].items()
                         if cell in (0, 1, 2, 3, 4)}
    for p in m["enemies"]:
        m["believed_map"][p] = 5
    return m

def planner(memory, local_obs):
    # A key underfoot needs interaction without walking away from it.
    if memory["map"].get(memory["pos"]) == 2:
        return {"move": [0, 0], "interact": True}
    return _study_memory_planner(memory, local_obs)
