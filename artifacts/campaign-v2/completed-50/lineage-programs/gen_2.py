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
    if _blocked(m, q):
        return False
    if d[0] and d[1]:
        # Interaction only opens cardinally adjacent doors.
        if m["map"].get(q) == 3 and not m["door_open"]:
            return False
        for side in (_add(p, (d[0], 0)), _add(p, (0, d[1]))):
            if _blocked(m, side):
                return False
            if m["map"].get(side) == 3 and not m["door_open"]:
                return False
    return True


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


def _motion_fields(terrain, enemies):
    """Anonymous occupancy forecasts under three spatial transition kernels."""
    fields = [{}, {}, {}]
    cardinal = ((1, 0), (-1, 0), (0, 1), (0, -1))
    diagonal = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    def passable(p):
        # Unknown cells at the observation boundary remain possible destinations.
        return terrain.get(p, 0) not in (-1, 1, 3)
    for e in sorted(enemies):
        destinations = [[e]]
        for directions in (cardinal, diagonal):
            reachable = []
            for dx, dy in directions:
                q = _add(e, (dx, dy))
                if not passable(q):
                    continue
                if dx and dy and (
                    not passable(_add(e, (dx, 0))) or
                    not passable(_add(e, (0, dy)))
                ):
                    continue
                reachable.append(q)
            destinations.append(reachable or [e])
        for field, cells in zip(fields, destinations):
            probability = 1.0 / len(cells)
            for q in cells:
                # Union probability preserves the anonymous occupancy interface.
                field[q] = 1.0 - (1.0 - field.get(q, 0.0)) * (1.0 - probability)
    return fields
def _forecast(m, fields, terrain, enemies, p):
    if terrain.get(p, 0) in (-1, 1, 3):
        return 0.0
    a, b = m["rates"][_feature(terrain, enemies, p)]
    spatial = sum(w * field.get(p, 0.0)
                  for w, field in zip(m["motion"], fields))
    return .8 * spatial + .2 * a / b
def world_model_step(memory, local_obs, last_action):
    if memory is None:
        prior = {"occupied_tight": .65, "occupied_open": .4, "near": .08,
                 "near_many": .16, "far": .005, "blocked": .001}
        memory = {"map": {}, "seen": {}, "pos": (0, 0), "visits": {},
                  "rates": {k: [v * 12, 12.] for k, v in prior.items()},
                  "updates": 0, "previous": None,
                  "motion": [.55, .35, .10], "motion_updates": 0,
                  "waits": 0}
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
        fields = _motion_fields(prev_terrain, prev_enemies)
        gradient = [0.0, 0.0, 0.0]
        labels = []
        motion_labels = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                p = _add(center, (dx, dy))
                # Score the previous forecast before incorporating its labels.
                if p in terrain and terrain[p] not in (-1, 1, 3):
                    feature = _feature(prev_terrain, prev_enemies, p)
                    if feature == "blocked":
                        continue
                    label = float(p in enemies)
                    labels.append((feature, label))
                    values = [field.get(p, 0.0) for field in fields]
                    if any(values):
                        prediction = _forecast(
                            m, fields, prev_terrain, prev_enemies, p)
                        for i, value in enumerate(values):
                            gradient[i] += 1.6 * (prediction - label) * value
                        motion_labels += 1
        if motion_labels:
            # Mirror descent on Brier loss learns a stochastic motion mixture.
            scale = 1.0 / math.sqrt(max(1, len(prev_enemies)))
            weights = [
                max(1e-6, w * math.exp(max(-8.0, min(8.0, -scale * g))))
                for w, g in zip(m["motion"], gradient)
            ]
            total = sum(weights)
            m["motion"] = [w / total for w in weights]
            m["motion_updates"] += motion_labels
        for feature, label in labels:
            m["rates"][feature][0] += label
            m["rates"][feature][1] += 1
            m["updates"] += 1
    m["previous"] = (terrain, enemies, m["pos"])
    m["enemies"] = enemies
    m["risk"] = {}
    fields = _motion_fields(terrain, enemies)
    for p, cell in m["map"].items():
        if p in terrain:
            prediction = _forecast(m, fields, terrain, enemies, p)
            # Edge cells can receive enemies from beyond the observed square.
            if cell not in (-1, 1, 3) and max(
                abs(p[0] - m["pos"][0]), abs(p[1] - m["pos"][1])
            ) == 2:
                prediction = 1.0 - (1.0 - prediction) * .99
            m["risk"][p] = prediction
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
            # Immediate hazards are actionable; distant forecasts will be revised.
            risk_weight = 85.0 if p == start else 35.0
            edge = (1.4 if d[0] and d[1] else 1.) + risk_weight * risk
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
    destination_risk = risk_fn(_add(start, move))
    waiting_risk = risk_fn(start)
    if (move != (0, 0) and destination_risk > .035
            and destination_risk > waiting_risk + .025
            and start not in m["enemies"] and m["waits"] < 3):
        move = (0, 0)
        m["waits"] += 1
    else:
        m["waits"] = 0
    return {"move": list(move), "interact": True}


def export_model(memory, local_obs):
    m = memory
    return {"enemy": [[x, y, p] for (x, y), p in m["risk"].items()],
            "default_enemy": .02,
            "terrain": [[x, y, c, m["seen"][x, y]] for (x, y), c in m["map"].items()],
            "position": list(m["pos"]),
            "learning": {
                "updates": m["updates"],
                "rates": {k: a/b for k, (a, b) in m["rates"].items()},
                "motion_updates": m["motion_updates"],
                "motion_weights": list(m["motion"])
            }}
# EVOLVE-BLOCK-END