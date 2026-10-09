"""Predictive seed. Entire representation, learner and planner may evolve."""
# EVOLVE-BLOCK-START
import heapq
import math

DIRECTIONS = tuple((x, y) for y in (-1, 0, 1) for x in (-1, 0, 1))
MOVES = tuple(d for d in DIRECTIONS if d != (0, 0))
UNIFORM = (1.0 / 9.0,) * 9


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _radius(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _solid(cells, p, opened, unknown_blocked=False):
    cell = cells.get(p, 1 if unknown_blocked else 0)
    return cell in (-1, 1) or (cell == 3 and not opened)


def _legal(cells, p, d, opened, unknown_blocked=False):
    q = _add(p, d)
    if _solid(cells, q, opened, unknown_blocked):
        return False
    if d[0] and d[1]:
        if _solid(cells, _add(p, (d[0], 0)), opened, unknown_blocked):
            return False
        if _solid(cells, _add(p, (0, d[1])), opened, unknown_blocked):
            return False
    return True


def _attempt(cells, p, d, opened):
    # Illegal attempted directions retain their probability at the origin.
    return _add(p, d) if _legal(cells, p, d, opened) else p


def _new_memory():
    return {
        "atlas": {"cells": {}, "seen": {}, "visits": {}},
        "state": {
            "pos": (0, 0), "keys": 0, "open": False, "step": 0,
            "enemies": set(), "visible": set(), "local": {}
        },
        "dynamics": {
            "fast": [0.0] * 9, "slow": [0.0] * 9,
            "evidence": 0.0, "updates": 0,
            "cloud": {}, "predicted": {}, "law": list(UNIFORM)
        },
        "navigation": {"target": None, "waits": 0},
        "snapshot": None,
        "forecast": {}
    }


def _law(dynamics):
    fast = [v + 0.8 for v in dynamics["fast"]]
    slow = [v + 1.0 for v in dynamics["slow"]]
    sf, ss = sum(fast), sum(slow)
    confidence = dynamics["evidence"] / (dynamics["evidence"] + 5.0)
    uniform_weight = 0.10 + 0.25 * (1.0 - confidence)
    return [
        (1.0 - uniform_weight) * (0.8 * fast[i] / sf + 0.2 * slow[i] / ss)
        + uniform_weight / 9.0
        for i in range(9)
    ]


def _observe(m, obs):
    state, atlas = m["state"], m["atlas"]
    displacement = obs.get("feedback", {}).get("displacement", (0, 0))
    state["pos"] = _add(state["pos"], displacement)
    state["keys"] = int(obs["keys"])
    state["open"] = bool(obs["door_open"])
    state["step"] = int(obs["step"])
    pos = state["pos"]
    radius = int(obs.get("enemy_visibility_radius", 2))
    local, visible, enemies = {}, set(), set()
    for y, row in enumerate(obs["terrain"]):
        for x, cell in enumerate(row):
            p = _add(pos, (x - 2, y - 2))
            local[p] = cell
            atlas["cells"][p] = cell
            atlas["seen"][p] = state["step"]
            if max(abs(x - 2), abs(y - 2)) <= radius:
                visible.add(p)
                if obs["grid"][y][x] == 5:
                    enemies.add(p)
    state["local"] = local
    state["visible"] = visible
    state["enemies"] = enemies
    atlas["visits"][pos] = atlas["visits"].get(pos, 0) + 1


def _association_statistics(snapshot, state, law):
    """Marginalize small anonymous matchings, including censored departures."""
    cells = dict(snapshot["terrain"])
    for p, cell in state["local"].items():
        if p not in cells:
            cells[p] = cell
    opened = snapshot["open"] or state["open"]
    all_old = sorted(snapshot["enemies"])
    old = [
        p for p in all_old
        if all(_add(p, d) in cells for d in DIRECTIONS)
    ]
    if not old:
        return None

    new = state["enemies"]
    visible = state["visible"]
    options = []
    for origin in old:
        groups = {}
        for i, direction in enumerate(DIRECTIONS):
            q = _attempt(cells, origin, direction, opened)
            if q not in visible:
                groups.setdefault(None, []).append(i)
            elif q in new:
                groups.setdefault(q, []).append(i)
            # Observable empty destinations contradict this attempted outcome.
        events = []
        for destination, indices in groups.items():
            probability = sum(law[i] for i in indices)
            events.append(
                (destination, indices, probability, len(indices) / 9.0)
            )
        if not events:
            return None
        options.append(events)

    omitted = set(all_old) - set(old)
    nuisance_count = 3 - len(old)
    birth_weights = {}
    for q in new:
        unobserved = sum(
            _add(q, (-d[0], -d[1])) not in snapshot["visible"]
            for d in DIRECTIONS
        ) / 9.0
        omitted_near = any(_radius(q, p) <= 1 for p in omitted)
        birth_weights[q] = (
            0.025 + 0.22 * unobserved + 0.15 * omitted_near
        )

    totals = [0.0] * 9
    normalizer = [0.0, 0.0]

    def visit(index, used, chosen, probability, uniform_probability):
        if index == len(options):
            unmatched = new - used
            if len(unmatched) > nuisance_count:
                return
            background = 1.0
            for q in unmatched:
                background *= birth_weights[q]
            weight = probability * background
            normalizer[0] += weight
            normalizer[1] += uniform_probability * background
            for destination, indices, event_probability, _ in chosen:
                if destination is None:
                    continue
                # Broad stay masks contain less directional information.
                information = 1.0 - len(indices) / 9.0
                if event_probability > 0.0 and information > 0.0:
                    for i in indices:
                        totals[i] += (
                            weight * information * law[i] / event_probability
                        )
            return

        for event in options[index]:
            destination, indices, probability_event, uniform_event = event
            if destination is not None and destination in used:
                continue
            next_used = used if destination is None else used | {destination}
            visit(
                index + 1, next_used, chosen + [event],
                probability * probability_event,
                uniform_probability * uniform_event
            )

    visit(0, set(), [], 1.0, 1.0)
    if normalizer[0] <= 1e-16:
        return None
    statistics = [v / normalizer[0] for v in totals]
    return statistics, normalizer[0], normalizer[1]


def _learn(m, enabled):
    dynamics, state = m["dynamics"], m["state"]
    if not enabled:
        dynamics["law"] = _law(dynamics)
        return

    for i in range(9):
        dynamics["fast"][i] *= 0.90
        dynamics["slow"][i] *= 0.975
    dynamics["evidence"] *= 0.90
    snapshot = m["snapshot"]
    if (
        snapshot is not None
        and state["step"] == snapshot["step"] + 1
        and state["step"] % 25 != 0
    ):
        result = _association_statistics(snapshot, state, _law(dynamics))
        if result is not None:
            statistics, likelihood, uniform_likelihood = result
            exposure = sum(statistics)
            if (
                exposure > 0.5
                and dynamics["evidence"] > 4.0
                and likelihood < 0.45 * uniform_likelihood
            ):
                # Unexpected informative observations shorten effective memory.
                dynamics["fast"] = [v * 0.25 for v in dynamics["fast"]]
                dynamics["evidence"] *= 0.4
            for i, count in enumerate(statistics):
                dynamics["fast"][i] += count
                dynamics["slow"][i] += count
            dynamics["evidence"] += exposure
            if exposure > 0.01:
                dynamics["updates"] += 1
    dynamics["law"] = _law(dynamics)


def _background(m):
    """A board-size prior derived only from observed terrain and boundaries."""
    cells = m["atlas"]["cells"]
    inside = [p for p, cell in cells.items() if cell != -1]
    if not inside:
        return {}
    xs, ys = [p[0] for p in inside], [p[1] for p in inside]
    lx, hx = max(xs) - 14, min(xs)
    ly, hy = max(ys) - 14, min(ys)

    def interior(p):
        return p in cells and cells[p] != -1

    for (x, y), cell in cells.items():
        if cell != -1:
            continue
        if interior((x + 1, y)):
            lx, hx = max(lx, x + 1), min(hx, x + 1)
        if interior((x - 1, y)):
            lx, hx = max(lx, x - 15), min(hx, x - 15)
        if interior((x, y + 1)):
            ly, hy = max(ly, y + 1), min(hy, y + 1)
        if interior((x, y - 1)):
            ly, hy = max(ly, y - 15), min(hy, y - 15)
    if lx > hx or ly > hy:
        return {}

    opened = m["state"]["open"]
    background = {}
    for x in range(lx, hx + 15):
        wx = max(0, min(hx, x) - max(lx, x - 14) + 1)
        for y in range(ly, hy + 15):
            p = (x, y)
            if _solid(cells, p, opened):
                continue
            wy = max(0, min(hy, y) - max(ly, y - 14) + 1)
            weight = wx * wy * (1.0 if p in cells else 0.72)
            if weight:
                background[p] = weight
    return background


def _filter_occupancy(m):
    state, dynamics = m["state"], m["dynamics"]
    enemies, visible = state["enemies"], state["visible"]
    missing = max(0, 3 - len(enemies))
    cloud = {p: 1.0 for p in enemies}
    if missing:
        background = {
            p: value for p, value in _background(m).items()
            if p not in visible
        }
        prior = {
            p: value for p, value in dynamics["predicted"].items()
            if p in background and value > 1e-12
        }
        prior_total, background_total = sum(prior.values()), sum(background.values())
        if background_total:
            for p, value in background.items():
                broad = value / background_total
                posterior = prior.get(p, 0.0) / prior_total if prior_total else broad
                cloud[p] = missing * (0.97 * posterior + 0.03 * broad)
    dynamics["cloud"] = cloud


def _spread(cloud, law, cells, opened, cache):
    result = {}
    for p, mass in cloud.items():
        if mass <= 1e-12:
            continue
        kernel = cache.get(p)
        if kernel is None:
            kernel = {}
            for direction, probability in zip(DIRECTIONS, law):
                q = _attempt(cells, p, direction, opened)
                kernel[q] = kernel.get(q, 0.0) + probability
            cache[p] = kernel
        for q, probability in kernel.items():
            result[q] = result.get(q, 0.0) + mass * probability
    return result


def _occupancy(components):
    survival = {}
    for cloud, count in components:
        if count <= 0:
            continue
        for p, mass in cloud.items():
            individual_probability = min(1.0, max(0.0, mass / count))
            absence = (1.0 - individual_probability) ** count
            survival[p] = survival.get(p, 1.0) * absence
    return {p: min(1.0, max(0.0, 1.0 - value)) for p, value in survival.items()}


def _layers(m, law, opened, horizon=3):
    enemies = m["state"]["enemies"]
    cloud = m["dynamics"]["cloud"]
    components = [({p: 1.0}, 1) for p in sorted(enemies)]
    hidden = {p: mass for p, mass in cloud.items() if p not in enemies}
    missing = max(0, 3 - len(enemies))
    if missing and hidden:
        components.append((hidden, missing))

    occupancy = [_occupancy(components)]
    first_intensity = {}
    cache = {}
    cells = m["atlas"]["cells"]
    for time in range(1, horizon + 1):
        components = [
            (_spread(component, law, cells, opened, cache), count)
            for component, count in components
        ]
        occupancy.append(_occupancy(components))
        if time == 1:
            for component, _ in components:
                for p, mass in component.items():
                    first_intensity[p] = first_intensity.get(p, 0.0) + mass
    return occupancy, first_intensity


def _can_open(m, p):
    return m["state"]["keys"] >= 2 and any(
        m["atlas"]["cells"].get(_add(p, d)) == 3
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1))
    )


def _base_edge(m, q, direction):
    atlas, state = m["atlas"], m["state"]
    age = max(0, state["step"] // 25 - atlas["seen"].get(q, 0) // 25)
    return (
        (1.4 if direction[0] and direction[1] else 1.0)
        + 0.025 * min(atlas["visits"].get(q, 0), 16)
        + 0.06 * min(age, 3)
    )


def _choose_target(m, forecasts):
    atlas, state = m["atlas"], m["state"]
    cells, start = atlas["cells"], state["pos"]
    virtual_open = state["open"] or state["keys"] >= 2
    distance = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distance[p]:
            continue
        for direction in MOVES:
            if not _legal(cells, p, direction, virtual_open, True):
                continue
            q = _add(p, direction)
            local_risk = forecasts[1].get(q, 0.0) if _radius(q, start) <= 3 else 0.0
            edge = _base_edge(m, q, direction) + 32.0 * local_risk
            candidate = cost + edge
            if candidate < distance.get(q, float("inf")):
                distance[q] = candidate
                heapq.heappush(queue, (candidate, q))

    desired = 2 if state["keys"] < 2 else 4
    goals = [p for p in distance if p != start and cells.get(p) == desired]
    if not goals and state["keys"] >= 2 and not state["open"]:
        goals = [p for p in distance if p != start and cells.get(p) == 3]
    if goals:
        return min(goals, key=lambda p: (distance[p], p))

    frontiers = []
    previous_target = m["navigation"]["target"]
    for p, cost in distance.items():
        if p == start:
            continue
        gain = sum(
            _add(p, (dx, dy)) not in cells
            for dx in range(-2, 3) for dy in range(-2, 3)
        )
        if gain:
            score = cost / math.sqrt(gain)
            if p == previous_target:
                score *= 0.92
            frontiers.append((score, p))
    if frontiers:
        return min(frontiers)[1]

    alternatives = [p for p in distance if p != start]
    if alternatives:
        return min(
            alternatives,
            key=lambda p: (
                atlas["visits"].get(p, 0),
                distance[p] + 0.02 * atlas["seen"].get(p, 0),
                p
            )
        )
    return start


def _potential(m, target):
    cells = m["atlas"]["cells"]
    opened = m["state"]["open"] or m["state"]["keys"] >= 2
    distance = {target: 0.0}
    queue = [(0.0, target)]
    while queue:
        cost, q = heapq.heappop(queue)
        if cost != distance[q]:
            continue
        for direction in MOVES:
            p = _add(q, (-direction[0], -direction[1]))
            if not _legal(cells, p, direction, opened, True):
                continue
            if _solid(cells, p, opened, True):
                continue
            candidate = cost + _base_edge(m, q, direction)
            if candidate < distance.get(p, float("inf")):
                distance[p] = candidate
                heapq.heappush(queue, (candidate, p))
    return distance


def _control(m, forecasts, target, opened):
    state, atlas = m["state"], m["atlas"]
    start, cells = state["pos"], atlas["cells"]
    potential = _potential(m, target)
    horizon = len(forecasts) - 1
    layer = {(start, opened): (0.0, (0, 0))}
    candidates = []
    waits = m["navigation"]["waits"]

    for depth in range(1, horizon + 1):
        following = {}
        for (p, door_open), (cost, first) in layer.items():
            effective_open = door_open or _can_open(m, p)
            for direction in DIRECTIONS:
                if not _legal(cells, p, direction, effective_open, True):
                    continue
                q = _add(p, direction)
                if depth == 1 and q in state["enemies"]:
                    continue
                if q not in potential:
                    continue

                after = forecasts[depth].get(q, 0.0)
                if depth == 1:
                    # Every immediate destination is inside the declared sensor.
                    contact = 0.0
                else:
                    contact = forecasts[depth - 1].get(q, 0.0)
                large, small = max(contact, after), min(contact, after)
                risk = min(0.999999, large + 0.5 * small * (1.0 - large))
                danger = -math.log(max(1e-6, 1.0 - risk))

                if direction == (0, 0):
                    movement_cost = 0.70 + (0.12 * min(waits, 8) if depth == 1 else 0.0)
                else:
                    movement_cost = 0.53 if direction[0] and direction[1] else 0.38
                next_cost = (
                    cost + movement_cost
                    + 62.0 * (0.80 ** (depth - 1)) * danger
                    + 0.025 * min(atlas["visits"].get(q, 0), 16)
                )
                first_action = direction if depth == 1 else first

                if q == target and q != start:
                    candidates.append((
                        next_cost - 0.35 * (horizon - depth),
                        first_action
                    ))
                    continue

                key = (q, effective_open)
                record = following.get(key)
                if record is None or next_cost < record[0]:
                    following[key] = (next_cost, first_action)
        layer = following
        if not layer:
            break
        if depth == horizon:
            for (p, _), (cost, first) in layer.items():
                candidates.append((cost + 1.10 * potential[p], first))

    if candidates:
        return min(candidates, key=lambda item: (item[0], item[1]))[1]

    legal = []
    for direction in DIRECTIONS:
        if _legal(cells, start, direction, opened, True):
            q = _add(start, direction)
            if q not in state["enemies"]:
                legal.append((
                    forecasts[1].get(q, 0.0),
                    potential.get(q, 1000.0),
                    direction
                ))
    return min(legal)[2] if legal else (0, 0)


def world_model_step(memory, local_obs, last_action):
    m = _new_memory() if memory is None else memory
    _observe(m, local_obs)
    _learn(m, bool(local_obs.get("learn", True)))
    _filter_occupancy(m)
    state = m["state"]
    m["snapshot"] = {
        "terrain": dict(m["atlas"]["cells"]),
        "enemies": set(state["enemies"]),
        "visible": set(state["visible"]),
        "pos": state["pos"],
        "step": state["step"],
        "open": state["open"]
    }
    m["forecast"] = {}
    return m


def planner(memory, local_obs):
    m = memory
    opened = m["state"]["open"] or _can_open(m, m["state"]["pos"])
    learned, predicted = _layers(m, m["dynamics"]["law"], opened)
    m["forecast"] = learned[1]
    m["dynamics"]["predicted"] = predicted

    if local_obs.get("predictive_planning", True):
        planning_forecasts = learned
    else:
        # Same inference and controller, with the movement law replaced by prior.
        planning_forecasts, _ = _layers(m, UNIFORM, opened)

    target = _choose_target(m, planning_forecasts)
    direction = _control(m, planning_forecasts, target, opened)
    m["navigation"]["target"] = target
    if direction == (0, 0):
        m["navigation"]["waits"] += 1
    else:
        m["navigation"]["waits"] = 0
    return {"move": [int(direction[0]), int(direction[1])], "interact": True}


def export_model(memory, local_obs):
    m = memory
    if not m["forecast"]:
        opened = m["state"]["open"] or _can_open(m, m["state"]["pos"])
        forecasts, predicted = _layers(m, m["dynamics"]["law"], opened)
        m["forecast"] = forecasts[1]
        m["dynamics"]["predicted"] = predicted
    atlas, dynamics = m["atlas"], m["dynamics"]
    return {
        "enemy": [
            [x, y, float(min(1.0, max(0.0, probability)))]
            for (x, y), probability in sorted(m["forecast"].items())
        ],
        "default_enemy": 0.0,
        "terrain": [
            [x, y, int(cell), int(atlas["seen"][(x, y)])]
            for (x, y), cell in sorted(atlas["cells"].items())
        ],
        "position": list(m["state"]["pos"]),
        "learning": {
            "updates": dynamics["updates"],
            "effective_evidence": dynamics["evidence"],
            "attempt_law": list(dynamics["law"]),
            "forecast_horizon": 1
        }
    }
# EVOLVE-BLOCK-END
