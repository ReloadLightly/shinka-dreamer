"""Predictive seed. Entire representation, learner and planner may evolve."""
# EVOLVE-BLOCK-START
import heapq
import math
_CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
_MOVES = tuple((x, y) for y in (-1, 0, 1)
               for x in (-1, 0, 1) if x or y)
_ACTIONS = ((0, 0),) + _MOVES
_PRIOR = [math.log(.55 / .0875), math.log(.025 / .0875),
          0., 0., 0., .10, 0., 0., 0., 0.]
def _add(a, b):
    return a[0] + b[0], a[1] + b[1]
def _distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
def _possible(m, p):
    # Bounds follow only from observed terrain and the supplied board size.
    left_lo, left_hi, top_lo, top_hi = m["bounds"]
    return (left_lo + 1 <= p[0] <= left_hi + 13 and
            top_lo + 1 <= p[1] <= top_hi + 13)
def _blocked(m, p):
    cell = m["map"].get(p, 1)
    return (cell in (-1, 1) or
            (cell == 3 and not m["door_open"] and m["keys"] < 2))
def _legal(m, p, d):
    q = _add(p, d)
    if _blocked(m, q):
        return False
    if d[0] and d[1]:
        if m["map"].get(q) == 3 and not m["door_open"]:
            return False
        for side in (_add(p, (d[0], 0)), _add(p, (0, d[1]))):
            if _blocked(m, side):
                return False
            if m["map"].get(side) == 3 and not m["door_open"]:
                return False
    return True
def _enemy_open(m, p):
    if not _possible(m, p):
        return False
    cell = m["map"].get(p, 0)
    if cell in (-1, 1):
        return False
    if cell == 3 and not m["door_open"]:
        # Every returned action interacts before movement.
        return (m["keys"] >= 2 and
                abs(p[0] - m["pos"][0]) +
                abs(p[1] - m["pos"][1]) == 1)
    return True
def _transition_rows(m, e, center, velocity, degrees):
    def degree(p):
        if p not in degrees:
            degrees[p] = sum(_enemy_open(m, _add(p, d))
                             for d in _CARDINAL)
        return degrees[p]
    rx, ry = center[0] - e[0], center[1] - e[1]
    length = max(1., math.hypot(rx, ry))
    near = float(max(abs(rx), abs(ry)) <= 2)
    tight = float(degree(e) <= 2)
    vx, vy = velocity
    speed = min(1., vx * vx + vy * vy)
    choices = []
    for dx, dy in _ACTIONS:
        q = e[0] + dx, e[1] + dy
        if not _enemy_open(m, q):
            continue
        if dx and dy:
            if (not _enemy_open(m, (e[0] + dx, e[1])) or
                    not _enemy_open(m, (e[0], e[1] + dy))):
                continue
        stay = float(dx == 0 and dy == 0)
        diagonal = float(bool(dx and dy))
        norm = math.sqrt(2.) if diagonal else 1.
        approach = (dx * rx + dy * ry) / (length * norm)
        alignment = (dx * vx + dy * vy) / norm
        features = (
            stay,
            diagonal,
            approach,
            approach * near,
            stay * near,
            stay * tight,
            (1. - stay) * (degree(q) - 4.) / 4.,
            alignment,
            stay * speed,
            approach * tight,
        )
        score = sum(w * f for w, f in zip(m["theta"], features))
        choices.append((q, score, features))
    if not choices:
        return []
    maximum = max(row[1] for row in choices)
    weights = [math.exp(row[1] - maximum) for row in choices]
    total = sum(weights)
    return [(q, weight / total, features)
            for (q, _, features), weight in zip(choices, weights)]
def _predict(m, belief, center, keep_rows=False):
    survival, saved, degrees = {}, [], {}
    for e, mass in belief.items():
        if mass <= 1e-8 or not _enemy_open(m, e):
            continue
        velocity = m["flow"].get(e, (0., 0.))
        rows = _transition_rows(m, e, center, velocity, degrees)
        for q, probability, _ in rows:
            contribution = min(1., mass * probability)
            survival[q] = survival.get(q, 1.) * (1. - contribution)
        if keep_rows and e in m["enemies"]:
            saved.append((e, rows))
    risk = {p: 1. - value for p, value in survival.items()}
    for p in m["map"]:
        risk.setdefault(p, 0.)
    return risk, saved
def _learn(m, previous, terrain, enemies):
    if previous is None:
        return
    labels = {}
    center = previous["center"]
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            p = _add(center, (dx, dy))
            if (p not in terrain or
                    terrain[p] in (-1, 1, 3) or
                    previous["terrain"].get(p, 1) in (-1, 1, 3)):
                continue
            label = float(p in enemies)
            prediction = previous["risk"].get(p, 0.)
            labels[p] = label
            m["squared_error"] += (prediction - label) ** 2
            m["persistence_error"] += (
                float(p in previous["enemies"]) - label) ** 2
            m["updates"] += 1
    gradient = [0.] * len(m["theta"])
    informative = False
    # Differentiate anonymous union occupancy through each spatial softmax.
    # No enemy matching or hidden identities are needed.
    for _, rows in previous["rows"]:
        mean = [0.] * len(gradient)
        for _, probability, features in rows:
            for i, feature in enumerate(features):
                mean[i] += probability * feature
        for p, probability, features in rows:
            if p not in labels:
                continue
            prediction = previous["risk"].get(p, 0.)
            other_survival = ((1. - prediction) /
                              max(1e-8, 1. - probability))
            factor = (2. * (prediction - labels[p]) *
                      other_survival * probability)
            if abs(factor) > 1e-10:
                informative = True
            for i, feature in enumerate(features):
                gradient[i] += factor * (feature - mean[i])
    if informative:
        for i, value in enumerate(gradient):
            value += .003 * (m["theta"][i] - _PRIOR[i])
            value = max(-3., min(3., value))
            m["accumulator"][i] += value * value
            step = .20 * value / math.sqrt(.20 + m["accumulator"][i])
            m["theta"][i] = max(-6., min(6., m["theta"][i] - step))
        m["parameter_updates"] += 1
def _infer_flow(previous, enemies):
    # Conditional expected incoming displacement, not persistent identities.
    flow = {}
    if previous is None:
        return flow
    for q in enemies:
        total, vx, vy = 0., 0., 0.
        for e, rows in previous["rows"]:
            for destination, probability, _ in rows:
                if destination == q:
                    total += probability
                    vx += probability * (q[0] - e[0])
                    vy += probability * (q[1] - e[1])
        if total > 1e-8:
            flow[q] = vx / total, vy / total
    return flow
def _record_forecast(m, center):
    risk, rows = _predict(m, m["belief"], center, True)
    m["risk"] = risk
    m["previous"] = {
        "risk": risk,
        "rows": rows,
        "terrain": m["terrain"],
        "enemies": set(m["enemies"]),
        "center": m["pos"],
    }
def world_model_step(memory, local_obs, last_action):
    if memory is None:
        memory = {
            "map": {}, "seen": {}, "pos": (0, 0), "visits": {},
            "bounds": [-13, -1, -13, -1],
            "theta": list(_PRIOR),
            "accumulator": [0.] * len(_PRIOR),
            "updates": 0, "parameter_updates": 0,
            "squared_error": 0., "persistence_error": 0.,
            "previous": None, "risk": {}, "flow": {},
            "waits": 0, "target": None,
        }
    m = memory
    previous = m["previous"]
    old_risk = m["risk"]
    m["pos"] = _add(m["pos"], local_obs["feedback"]["displacement"])
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
            if cell != -1:
                margin = 0 if cell == 1 else 1
                b = m["bounds"]
                b[0] = max(b[0], p[0] - 14 + margin)
                b[1] = min(b[1], p[0] - margin)
                b[2] = max(b[2], p[1] - 14 + margin)
                b[3] = min(b[3], p[1] - margin)
    m["terrain"], m["enemies"] = terrain, enemies
    # FREEZE_PREDICTIVE_UPDATES: observations and state inference always run.
    if local_obs.get("learn", True):
        _learn(m, previous, terrain, enemies)
    m["flow"] = _infer_flow(previous, enemies)
    # Track occupancy in explored space and an unseen observation-boundary ring.
    support = set(m["map"])
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            support.add(_add(m["pos"], (dx, dy)))
    belief, hidden_mass = {}, 0.
    for p in support:
        if not _enemy_open(m, p):
            continue
        if p in terrain:
            value = float(p in enemies)
        else:
            value = .96 * old_risk.get(p, .02) + .04 * .02
            hidden_mass += value
        belief[p] = value
    # Unobserved probability may not represent more than the remaining enemies.
    available = max(0., 3. - len(enemies))
    if hidden_mass > available:
        scale = available / max(hidden_mass, 1e-8)
        for p in belief:
            if p not in terrain:
                belief[p] *= scale
    m["belief"] = belief
    _record_forecast(m, m["pos"])
    return m
def _fixed_risk(m, p):
    if p in m["enemies"]:
        return 1.
    return .22 if any(_distance(e, p) <= 1
                      for e in m["enemies"]) else 0.
def _paths(m, start, risk, reverse=False):
    distance = {start: 0.}
    queue = [(0., start)]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distance[p]:
            continue
        for d in _MOVES:
            q = _add(p, d)
            if q in m["enemies"]:
                continue
            if reverse:
                legal = _legal(m, q, (-d[0], -d[1]))
                destination = p
            else:
                legal = _legal(m, p, d)
                destination = q
            if not legal or _blocked(m, q):
                continue
            # Both diagonal and cardinal moves consume one environment step.
            weight = 12. if _distance(destination, m["pos"]) <= 2 else 4.
            edge = (1. + weight * risk(destination) +
                    .035 * min(m["visits"].get(destination, 0), 12))
            new = cost + edge
            if new < distance.get(q, float("inf")):
                distance[q] = new
                heapq.heappush(queue, (new, q))
    return distance
def _choose_target(m, distance):
    start = m["pos"]
    desired = 2 if m["keys"] < 2 else 4
    goals = [p for p in distance
             if p != start and m["map"].get(p) == desired]
    if not goals and m["keys"] >= 2 and not m["door_open"]:
        goals = [p for p in distance
                 if p != start and m["map"].get(p) == 3]
    if goals:
        return min(goals, key=lambda p: (distance[p], p))
    frontiers = {}
    for p in distance:
        if p == start:
            continue
        gain = sum(
            _add(p, (dx, dy)) not in m["map"] and
            _possible(m, _add(p, (dx, dy)))
            for dx in range(-2, 3) for dy in range(-2, 3)
        )
        if gain:
            frontiers[p] = distance[p] / math.sqrt(gain)
    if frontiers:
        best = min(frontiers, key=lambda p: (frontiers[p], p))
        old = m["target"]
        if old in frontiers and frontiers[old] <= 1.12 * frontiers[best]:
            return old
        return best
    # Refresh stale views if remembered connectivity no longer reaches a goal.
    candidates = []
    for p in distance:
        if p == start:
            continue
        age_gain = sum(
            min(2., max(0., m["step"] -
                        m["seen"].get(_add(p, (dx, dy)), 0)) / 25.)
            for dx in range(-2, 3) for dy in range(-2, 3)
            if _possible(m, _add(p, (dx, dy)))
        )
        value = ((distance[p] + .3 * m["visits"].get(p, 0)) /
                 math.sqrt(1. + age_gain))
        candidates.append((value, p))
    return min(candidates)[1] if candidates else start
def planner(memory, local_obs):
    m = memory
    start = m["pos"]
    predictive = local_obs.get("predictive_planning", True)
    if predictive:
        risk = lambda p: m["risk"].get(p, .02)
    else:
        risk = lambda p: _fixed_risk(m, p)
    distance = _paths(m, start, risk)
    target = _choose_target(m, distance)
    m["target"] = target
    remaining = _paths(m, target, risk, reverse=True)
    if predictive:
        second, _ = _predict(m, m["risk"], start)
        future_risk = lambda p: second.get(p, .02)
    else:
        future_risk = lambda p: _fixed_risk(m, p)
    best = None
    for move in _ACTIONS:
        q = _add(start, move)
        if q in m["enemies"] or not _legal(m, start, move):
            continue
        if q not in remaining:
            continue
        immediate = risk(q)
        continuation = float("inf")
        if q == target:
            continuation = 0.
        else:
            for next_move in _ACTIONS:
                r = _add(q, next_move)
                if not _legal(m, q, next_move) or r not in remaining:
                    continue
                # At the next decision, entering a then-occupied cell also kills.
                before_move = risk(r) if r != q else 0.
                hazard = min(1., future_risk(r) + .65 * before_move)
                value = 1. + remaining[r] + 75. * hazard
                if r == q:
                    value += .2
                continuation = min(continuation, value)
        if not math.isfinite(continuation):
            continuation = remaining[q] + 2.
        score = (1. + 180. * immediate + continuation +
                 .06 * min(m["visits"].get(q, 0), 20))
        if move == (0, 0):
            score += .45 + min(18., .30 * m["waits"] ** 2)
        candidate = (score, immediate, remaining[q], move)
        if best is None or candidate < best:
            best = candidate
    move = best[3] if best is not None else (0, 0)
    m["waits"] = m["waits"] + 1 if move == (0, 0) else 0
    # Export and subsequently train the forecast conditioned on this action.
    # The forecast is still made before observing the next transition.
    _record_forecast(m, _add(start, move))
    return {"move": list(move), "interact": True}
def export_model(memory, local_obs):
    m = memory
    count = max(1, m["updates"])
    return {
        "enemy": [[x, y, max(0., min(1., probability))]
                  for (x, y), probability in m["risk"].items()],
        "default_enemy": .02,
        "terrain": [[x, y, cell, m["seen"][(x, y)]]
                    for (x, y), cell in m["map"].items()],
        "position": list(m["pos"]),
        "learning": {
            "updates": m["updates"],
            "parameter_updates": m["parameter_updates"],
            "transition_weights": list(m["theta"]),
            "preupdate_brier": m["squared_error"] / count,
            "persistence_brier": m["persistence_error"] / count,
            "represented_enemy_mass": sum(m["belief"].values()),
        },
    }
# EVOLVE-BLOCK-END