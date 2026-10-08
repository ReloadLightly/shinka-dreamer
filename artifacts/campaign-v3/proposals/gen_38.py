"""Predictive seed. Entire representation, learner and planner may evolve."""
# EVOLVE-BLOCK-START
import heapq
import itertools
import math
DIRECTIONS = tuple((x, y) for y in (-1, 0, 1) for x in (-1, 0, 1))
MOVES = tuple(d for d in DIRECTIONS if d != (0, 0))
UNIFORM = (1.0 / 9.0,) * 9
EXPERT_PRIOR = (.30, .45, .25)
def _add(p, d):
    return p[0] + d[0], p[1] + d[1]
def _distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
def _inside(m, p):
    left, right, top, bottom = m["bounds"]
    return ((left is None or left <= p[0] <= right) and
            (top is None or top <= p[1] <= bottom))
def _infer_bounds(m, terrain):
    x, y = m["pos"]
    bounds = m["bounds"]
    for delta in (-2, -1, 1, 2):
        if terrain.get((x + delta, y)) == -1:
            if delta < 0:
                edge = x + delta + 1
                if bounds[0] is None or edge > bounds[0]:
                    bounds[0], bounds[1] = edge, edge + 14
            else:
                edge = x + delta - 1
                if bounds[1] is None or edge < bounds[1]:
                    bounds[0], bounds[1] = edge - 14, edge
        if terrain.get((x, y + delta)) == -1:
            if delta < 0:
                edge = y + delta + 1
                if bounds[2] is None or edge > bounds[2]:
                    bounds[2], bounds[3] = edge, edge + 14
            else:
                edge = y + delta - 1
                if bounds[3] is None or edge < bounds[3]:
                    bounds[2], bounds[3] = edge - 14, edge
def _terrain_destination(terrain, p, d, door_open):
    def blocked(q):
        cell = terrain.get(q, 0)
        return cell in (-1, 1) or (cell == 3 and not door_open)
    q = _add(p, d)
    if blocked(q):
        return p
    if d[0] and d[1]:
        if blocked((p[0] + d[0], p[1])) or blocked((p[0], p[1] + d[1])):
            return p
    return q
def _mean(alpha):
    total = sum(alpha)
    return tuple(a / total for a in alpha)
def _observation_masks(previous, terrain, enemies, door_open):
    old_terrain, old_enemies, old_center, old_step = previous
    safe = {
        _add(old_center, (dx, dy))
        for dx in (-1, 0, 1) for dy in (-1, 0, 1)
        if _add(old_center, (dx, dy)) in terrain
    }
    occupied = sorted(safe.intersection(enemies))
    bits = {p: 1 << i for i, p in enumerate(occupied)}
    full_mask = (1 << len(occupied)) - 1
    rows = []
    for origin in sorted(old_enemies):
        row = []
        for d in DIRECTIONS:
            q = _terrain_destination(old_terrain, origin, d, door_open)
            if q not in safe:
                row.append(0)
            elif q in bits:
                row.append(bits[q])
            else:
                row.append(-1)
        rows.append(row)
    return rows, full_mask
def _infer_anonymous(law, masks, full_mask):
    """Exact occupancy likelihood, permitting anonymous overlaps."""
    groups = []
    for row in masks:
        probabilities = {}
        for i, mask in enumerate(row):
            if mask >= 0:
                probabilities[mask] = probabilities.get(mask, 0.0) + law[i]
        if not probabilities:
            return 0.0, None
        groups.append(tuple(probabilities.items()))
    likelihood = 0.0
    marginals = [{} for _ in groups]
    for choice in itertools.product(*groups):
        union = 0
        probability = 1.0
        for mask, mass in choice:
            union |= mask
            probability *= mass
        if union != full_mask:
            continue
        likelihood += probability
        for j, (mask, mass) in enumerate(choice):
            marginals[j][mask] = marginals[j].get(mask, 0.0) + probability
    if likelihood <= 1e-15:
        return 0.0, None
    posteriors = []
    for j, row in enumerate(masks):
        masses = dict(groups[j])
        posterior = []
        for i, mask in enumerate(row):
            if mask < 0:
                posterior.append(0.0)
            else:
                numerator = law[i] * marginals[j].get(mask, 0.0)
                posterior.append(numerator / (likelihood * masses[mask]))
        posteriors.append(posterior)
    return likelihood, posteriors
def _learn(m, terrain, enemies, enabled):
    if not enabled:
        return
    laws = (UNIFORM, _mean(m["slow"]), _mean(m["fast"]))
    results = None
    previous = m["previous"]
    if previous is not None:
        old_step = previous[3]
        # Scheduled terrain changes are not evidence about movement laws.
        if m["step"] == old_step + 1 and m["step"] // 25 == old_step // 25:
            masks, target = _observation_masks(
                previous, terrain, enemies, m["door_open"]
            )
            if masks:
                results = [_infer_anonymous(law, masks, target) for law in laws]
                if any(result[1] is None for result in results):
                    results = None
    for name, retention, prior in (("slow", .985, .65), ("fast", .86, .35)):
        m[name] = [prior + retention * (a - prior) for a in m[name]]
    if results is None:
        return
    for i, (likelihood, posterior) in enumerate(results):
        m["log_weights"][i] = (
            .95 * m["log_weights"][i]
            + .05 * math.log(EXPERT_PRIOR[i])
            + .70 * math.log(max(likelihood, 1e-15))
        )
    normalizer = max(m["log_weights"])
    m["log_weights"] = [v - normalizer for v in m["log_weights"]]
    for expert_index, name in ((1, "slow"), (2, "fast")):
        law = laws[expert_index]
        for posterior in results[expert_index][1]:
            # Censored observations with no directional information must not
            # artificially increase concentration.
            information = min(1.0, sum(abs(a - b)
                                       for a, b in zip(posterior, law)))
            if information < 1e-8:
                continue
            for i, probability in enumerate(posterior):
                m[name][i] += information * probability
            if expert_index == 1:
                m["updates"] += 1
                m["effective_updates"] += information
def _combined_law(m):
    weights = [math.exp(v - max(m["log_weights"])) for v in m["log_weights"]]
    total = sum(weights)
    weights = [v / total for v in weights]
    laws = (UNIFORM, _mean(m["slow"]), _mean(m["fast"]))
    law = [
        .12 / 9.0 + .88 * sum(weights[j] * laws[j][i] for j in range(3))
        for i in range(9)
    ]
    m["expert_weights"] = weights
    return tuple(law)
def _solid(m, p, opened):
    if not _inside(m, p):
        return True
    left, right, top, bottom = m["bounds"]
    if ((left is not None and p[0] in (left, right)) or
            (top is not None and p[1] in (top, bottom))):
        return True
    cell = m["map"].get(p, 0)
    return cell in (-1, 1) or (
        cell == 3 and not m["door_open"] and p not in opened
    )
def _forecast(m, law, horizon=8):
    """Independent visible enemies and a diffuse unseen-enemy intensity."""
    opened = set()
    if m["keys"] >= 2:
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p = _add(m["pos"], d)
            if m["map"].get(p) == 3:
                opened.add(p)
    kernels = {}
    stay_probability = {}
    def kernel(p):
        if p in kernels:
            return kernels[p]
        result = {}
        for d, probability in zip(DIRECTIONS, law):
            q = _add(p, d)
            if (_solid(m, q, opened) or
                (d[0] and d[1] and
                 (_solid(m, (p[0] + d[0], p[1]), opened) or
                  _solid(m, (p[0], p[1] + d[1]), opened)))):
                q = p
            result[q] = result.get(q, 0.0) + probability
        kernels[p] = tuple(result.items())
        stay_probability[p] = result.get(p, 0.0)
        return kernels[p]
    def spread(distribution):
        result = {}
        for p, mass in distribution.items():
            for q, probability in kernel(p):
                result[q] = result.get(q, 0.0) + mass * probability
        return result
    free_visible = sum(not _solid(m, p, opened) for p in m["visible"])
    hidden_count = max(0, 3 - len(m["enemies"]))
    density = hidden_count / float(max(45, 130 - free_visible))
    candidates = set(m["map"])
    x, y = m["pos"]
    candidates.update((x + dx, y + dy)
                      for dx in range(-6, 7) for dy in range(-6, 7))
    hidden = {
        p: density for p in candidates
        if p not in m["visible"] and not _solid(m, p, opened) and density > 0
    }
    visible_distributions = [{p: 1.0} for p in sorted(m["enemies"])]
    forecasts = [{}]
    hazards = [{}]
    for time in range(1, horizon + 1):
        next_hidden = spread(hidden)
        no_occupancy = {}
        no_collision = {}
        next_visible_distributions = []
        for previous in visible_distributions:
            following = spread(previous)
            next_visible_distributions.append(following)
            for p in set(previous).union(following):
                before = previous.get(p, 0.0)
                after = following.get(p, 0.0)
                kernel(p)
                collision = before + after - before * stay_probability[p]
                no_occupancy[p] = no_occupancy.get(p, 1.0) * (1.0 - after)
                no_collision[p] = no_collision.get(p, 1.0) * (
                    1.0 - min(1.0, max(0.0, collision))
                )
        visible_distributions = next_visible_distributions
        cells = set(no_occupancy).union(hidden, next_hidden, m["map"])
        occupancy = {}
        collision_risk = {}
        for p in cells:
            if _solid(m, p, opened):
                occupancy[p] = 0.0
                collision_risk[p] = 0.0
                continue
            kernel(p)
            before = hidden.get(p, 0.0)
            after = next_hidden.get(p, 0.0)
            hidden_union = max(0.0, before + after -
                               before * stay_probability[p])
            occupancy[p] = min(1.0, max(
                0.0, 1.0 - no_occupancy.get(p, 1.0) * math.exp(-after)
            ))
            collision_risk[p] = min(1.0, max(
                0.0, 1.0 - no_collision.get(p, 1.0) * math.exp(-hidden_union)
            ))
        forecasts.append(occupancy)
        hazards.append(collision_risk)
        hidden = next_hidden
    return forecasts, hazards, min(.05, density)
def world_model_step(memory, local_obs, last_action):
    if memory is None:
        memory = {
            "map": {}, "seen": {}, "pos": (0, 0), "visits": {},
            "bounds": [None, None, None, None],
            "slow": [.65] * 9, "fast": [.35] * 9,
            "log_weights": [math.log(v) for v in EXPERT_PRIOR],
            "updates": 0, "effective_updates": 0.0,
            "previous": None, "target": None, "target_kind": None
        }
    m = memory
    displacement = local_obs.get("feedback", {}).get("displacement", (0, 0))
    m["pos"] = _add(m["pos"], displacement)
    m["keys"] = local_obs["keys"]
    m["door_open"] = local_obs["door_open"]
    m["step"] = local_obs["step"]
    m["visits"][m["pos"]] = m["visits"].get(m["pos"], 0) + 1
    terrain, enemies = {}, set()
    for y, row in enumerate(local_obs["terrain"]):
        for x, cell in enumerate(row):
            p = _add(m["pos"], (x - 2, y - 2))
            terrain[p] = cell
            m["map"][p] = cell
            m["seen"][p] = m["step"]
            if local_obs["grid"][y][x] == 5:
                enemies.add(p)
    _infer_bounds(m, terrain)
    # Predictive intervention leaves mapping and localization intact.
    _learn(m, terrain, enemies, local_obs.get("learn", True))
    m["previous"] = (terrain, enemies, m["pos"], m["step"])
    m["visible"], m["enemies"] = set(terrain), enemies
    m["law"] = _combined_law(m)
    forecasts, hazards, default = _forecast(m, m["law"])
    m["forecasts"] = forecasts
    m["risk"], m["hazards"], m["default_enemy"] = forecasts[1], hazards, default
    return m
def _blocked(m, p):
    if not _inside(m, p):
        return True
    cell = m["map"].get(p, 1)
    return cell in (-1, 1) or (
        cell == 3 and m["keys"] < 2 and not m["door_open"]
    )
def _legal(m, p, d):
    q = _add(p, d)
    if _blocked(m, q):
        return False
    if d[0] and d[1]:
        if (_blocked(m, (p[0] + d[0], p[1])) or
                _blocked(m, (p[0], p[1] + d[1]))):
            return False
        # Interaction opens only cardinally adjacent doors.
        if m["map"].get(q) == 3 and not m["door_open"]:
            return False
    return True
def _edge_cost(m, destination, risk):
    return (1.0 + .025 * min(12, m["visits"].get(destination, 0))
            + 16.0 * risk.get(destination, 0.0)
            + (5.0 if destination in m["enemies"] else 0.0))
def _distances(m, start, risk, reverse=False):
    distances = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distances[p]:
            continue
        for d in MOVES:
            q = _add(p, d)
            if reverse:
                if not _legal(m, q, (-d[0], -d[1])) or _blocked(m, q):
                    continue
                edge = _edge_cost(m, p, risk)
            else:
                if not _legal(m, p, d):
                    continue
                edge = _edge_cost(m, q, risk)
            new_cost = cost + edge
            if new_cost < distances.get(q, float("inf")):
                distances[q] = new_cost
                heapq.heappush(queue, (new_cost, q))
    return distances
def _gain(m, p):
    return sum(
        _inside(m, q) and q not in m["map"]
        for q in (_add(p, (dx, dy))
                  for dx in range(-2, 3) for dy in range(-2, 3))
    )
def _choose_target(m, distances):
    start = m["pos"]
    wanted = 2 if m["keys"] < 2 else 4
    goals = [p for p in distances
             if p != start and m["map"].get(p) == wanted]
    if goals:
        return min(goals, key=lambda p: (distances[p], p)), "goal"
    if m["keys"] >= 2 and not m["door_open"]:
        doors = [p for p in distances
                 if p != start and m["map"].get(p) == 3]
        if doors:
            return min(doors, key=lambda p: (distances[p], p)), "door"
    frontier = {}
    for p, cost in distances.items():
        if p == start:
            continue
        gain = _gain(m, p)
        if gain:
            frontier[p] = (cost + .3) / math.sqrt(gain)
    if frontier:
        best = min(frontier, key=lambda p: (frontier[p], p))
        old = m.get("target")
        if (m.get("target_kind") == "frontier" and old in frontier and
                frontier[old] <= 1.18 * frontier[best]):
            best = old
        return best, "frontier"
    alternatives = [p for p in distances if p != start]
    if alternatives:
        return min(alternatives, key=lambda p: (
            m["visits"].get(p, 0), distances[p], p
        )), "patrol"
    return start, "wait"
def planner(memory, local_obs):
    m = memory
    start = m["pos"]
    if local_obs.get("predictive_planning", True):
        forecasts, hazards = m["forecasts"], m["hazards"]
    else:
        # Replace only predictive dynamics, retaining the same controller.
        forecasts, hazards, _ = _forecast(m, UNIFORM)
    risk = forecasts[1]
    distances = _distances(m, start, risk)
    target, kind = _choose_target(m, distances)
    m["target"], m["target_kind"] = target, kind
    remaining = max(1, 200 - m["step"])
    horizon = min(len(hazards) - 1, remaining)
    route_length = max(1.0, distances.get(target, 1.0))
    risk_weight = 100.0 * (
        .55 + .45 * min(1.0, remaining / (route_length + 8.0))
    )

    # Cache the observed navigation graph once for all temporal layers.
    graph, reverse = {}, {}
    for p in distances:
        edges = []
        for d in DIRECTIONS:
            q = _add(p, d)
            if q in distances and _legal(m, p, d):
                edges.append((q, d))
                if q != p:
                    reverse.setdefault(q, []).append(p)
        graph[p] = edges

    # Beyond the explicit temporal horizon, use a continuation route under
    # the forecast at that time. Current visible enemies are not permanent
    # obstacles or permanent surcharges in this future route.
    terminal = {target: 0.0}
    queue = [(0.0, target)]
    tail_risk = forecasts[horizon]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != terminal[p]:
            continue
        occupancy = min(1.0, max(0.0, tail_risk.get(p, 0.0)))
        edge = (
            1.0 + .025 * min(12, m["visits"].get(p, 0))
            + 16.0 * -math.log(max(1e-7, 1.0 - occupancy))
        )
        for q in reverse.get(p, ()):
            candidate = cost + edge
            if candidate < terminal.get(q, float("inf")):
                terminal[q] = candidate
                heapq.heappush(queue, (candidate, q))

    # Survival costs use the same scale at every time. A crossing therefore
    # cannot become attractive solely by moving it to a discounted layer.
    stage_cache, memo = {}, {}
    def stage(q, waiting, time):
        key = (q, waiting, time)
        if key not in stage_cache:
            hazard = min(1.0, max(0.0, hazards[time].get(q, 0.0)))
            stage_cache[key] = (
                1.0 + (.16 if waiting else 0.0)
                + .02 * min(10, m["visits"].get(q, 0))
                + risk_weight * -math.log(max(1e-7, 1.0 - hazard))
            )
        return stage_cache[key]

    def value(p, time):
        if p == target and target != start:
            return 0.0
        if time > horizon:
            return terminal.get(p, 250.0)
        key = (p, time)
        if key not in memo:
            memo[key] = min(
                (
                    stage(q, d == (0, 0), time) + value(q, time + 1)
                    for q, d in graph.get(p, ())
                ),
                default=250.0,
            )
        return memo[key]

    choices = []
    for q, d in graph.get(start, ()):
        # Agent movement precedes enemy movement.
        if q in m["enemies"]:
            continue
        score = stage(q, d == (0, 0), 1) + value(q, 2)
        choices.append((
            score, hazards[1].get(q, 0.0),
            terminal.get(q, 250.0), d
        ))
    move = min(choices)[3] if choices else (0, 0)
    return {"move": [move[0], move[1]], "interact": True}
def export_model(memory, local_obs):
    m = memory
    return {
        "enemy": [[x, y, float(probability)]
                  for (x, y), probability in m["risk"].items()],
        "default_enemy": float(m["default_enemy"]),
        "terrain": [[x, y, cell, m["seen"][(x, y)]]
                    for (x, y), cell in m["map"].items()],
        "position": list(m["pos"]),
        "learning": {
            "updates": m["updates"],
            "effective_updates": m["effective_updates"],
            "attempted_movement_law": list(m["law"]),
            "directions": [list(d) for d in DIRECTIONS],
            "expert_weights": list(m["expert_weights"]),
            "slow_law": list(_mean(m["slow"])),
            "fast_law": list(_mean(m["fast"]))
        }
    }
# EVOLVE-BLOCK-END