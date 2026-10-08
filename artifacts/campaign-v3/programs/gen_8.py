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


def _planning_hazards(m, horizon):
    # Marginal distributions require no persistent enemy identities: each
    # currently observed occupied cell starts an independent prediction.
    weights = _motion_weights(m)
    distributions = [{p: 1.} for p in sorted(m["enemies"])]
    transitions = {}
    hazards = []
    unseen = max(0, 3 - len(distributions))

    def transition(p):
        if p not in transitions:
            row = {}
            for d, probability in zip(_DIRECTIONS, weights):
                q = _enemy_destination(m["map"], p, d)
                row[q] = row.get(q, 0.) + probability
            transitions[p] = row
        return transitions[p]

    for tick in range(1, horizon + 1):
        combined = {}
        following = []
        for distribution in distributions:
            prediction = {}
            for source, mass in distribution.items():
                for destination, probability in transition(source).items():
                    prediction[destination] = (
                        prediction.get(destination, 0.) + mass * probability)
            # Entering an occupied cell kills before the enemy moves.
            # Subtract the overlap rather than adding both marginals.
            for p in distribution.keys() | prediction.keys():
                before = distribution.get(p, 0.)
                after = prediction.get(p, 0.)
                overlap = before * transition(p).get(p, 0.)
                danger = min(1., max(0., before + after - overlap))
                combined[p] = 1. - (1. - combined.get(p, 0.)) * (1. - danger)
            following.append(prediction)
        distributions = following
        for p, cell in m["map"].items():
            if cell in (-1, 1, 3):
                continue
            distance = max(abs(p[0] - m["pos"][0]),
                           abs(p[1] - m["pos"][1]))
            # Hidden enemies cannot reach cells whose entire backwards
            # movement neighborhood lies in the currently observed square.
            background = 0. if distance + tick <= 2 else .008 * unseen
            combined[p] = 1. - (1. - combined.get(p, 0.)) * (1. - background)
        hazards.append(combined)
    return hazards


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
            if not _legal(m, p, d):
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
    predictive = local_obs.get("predictive_planning", True)
    risk_fn = ((lambda p: m["risk"].get(p, .02)) if predictive
               else (lambda p: _fixed_risk(m, p)))
    # Reverse distances supply a terminal value for a small temporal search.
    # Current enemy positions are temporary hazards, not permanent walls.
    remaining = {target: 0.}
    queue = [(0., target)]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != remaining[p]:
            continue
        for d in moves:
            q = _add(p, d)
            if _blocked(m, q) or not _legal(m, q, (-d[0], -d[1])):
                continue
            edge = (1. + 45 * risk_fn(p)
                    + .06 * min(m["visits"].get(p, 0), 15))
            new = cost + edge
            if new < remaining.get(q, float("inf")):
                remaining[q] = new
                heapq.heappush(queue, (new, q))

    horizon = 3
    hazards = (_planning_hazards(m, horizon) if predictive else
               [{p: _fixed_risk(m, p) for p in m["map"]}
                for _ in range(horizon)])
    memo = {}
    choices = {}
    actions = [(0, 0)] + moves

    def value(p, tick):
        if tick == horizon:
            return remaining.get(p, 1000.)
        # Reaching the current objective ends this short planning problem.
        if tick > 0 and p == target and target != start:
            return 0.
        key = (p, tick)
        if key in memo:
            return memo[key]
        best = float("inf")
        best_move = (0, 0)
        for d in actions:
            q = _add(p, d)
            if d != (0, 0) and not _legal(m, p, d):
                continue
            if tick == 0 and q in m["enemies"]:
                continue
            danger = hazards[tick].get(q, .02)
            cost = (1. + 45 * danger
                    + .06 * min(m["visits"].get(q, 0), 15)
                    + value(q, tick + 1))
            if cost < best:
                best, best_move = cost, d
        memo[key], choices[key] = best, best_move
        return best

    value(start, 0)
    move = choices.get((start, 0), (0, 0))
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