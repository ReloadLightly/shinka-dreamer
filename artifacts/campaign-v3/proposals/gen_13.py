"""Predictive seed. Entire representation, learner and planner may evolve."""
# EVOLVE-BLOCK-START
import heapq
import math
MOVES = tuple((x, y) for y in (-1, 0, 1) for x in (-1, 0, 1))
STEPS = tuple(d for d in MOVES if d != (0, 0))
def _add(p, d):
    return p[0] + d[0], p[1] + d[1]
def _distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
def _wall(terrain, p, opened=False, unknown=False):
    c = terrain.get(p)
    return unknown if c is None else c in (-1, 1) or (c == 3 and not opened)
def _destination(terrain, p, d, opened):
    q = _add(p, d)
    if _wall(terrain, q, opened):
        return p
    if d[0] and d[1]:
        if (_wall(terrain, _add(p, (d[0], 0)), opened) or
                _wall(terrain, _add(p, (0, d[1])), opened)):
            return p
    return q
def _legal(m, p, d):
    q = _add(p, d)
    terrain = m["map"]
    unlocked = m["keys"] >= 2 or m["door_open"]
    if _wall(terrain, q, unlocked, True):
        return False
    if terrain.get(q) == 3 and not m["door_open"]:
        if d[0] and d[1]:
            return False
    if d[0] and d[1]:
        # A closed door cannot serve as a diagonal corner before interaction.
        for side in ((d[0], 0), (0, d[1])):
            s = _add(p, side)
            open_here = m["door_open"] or (
                m["keys"] >= 2 and
                abs(s[0] - m["pos"][0]) + abs(s[1] - m["pos"][1]) == 1)
            if _wall(terrain, s, open_here, True):
                return False
    return True
def _law(m):
    total = sum(m["counts"]) + 9.0
    return [(c + 1.0) / total for c in m["counts"]]
def _learn(m, terrain, enemies, obs):
    previous = m["previous"]
    if previous is None or not obs.get("learn", True):
        return
    old_terrain, old_enemies, old_center, old_step, opened = previous
    if m["step"] != old_step + 1:
        return
    # Avoid attributing unobserved wall changes to the movement law.
    if m["step"] % 25 == 0 or old_step % 25 == 0:
        return
    m["counts"] = [c * .965 for c in m["counts"]]
    if not old_enemies:
        return
    terrain_union = dict(old_terrain)
    terrain_union.update(terrain)
    for p in set(old_terrain).intersection(terrain):
        if old_terrain[p] != terrain[p] and (
                old_terrain[p] == 1 or terrain[p] == 1):
            return
    law = _law(m)
    sources = sorted(old_enemies)
    options = []
    informative = []
    for e in sources:
        groups = {}
        allowed = 0
        for k, d in enumerate(MOVES):
            q = _destination(terrain_union, e, d, opened)
            if q in terrain and q not in enemies:
                continue
            key = q if q in terrain else None
            groups.setdefault(key, []).append(k)
            allowed += 1
        if not groups:
            return
        options.append([(q, ks, sum(law[k] for k in ks))
                        for q, ks in groups.items()])
        known = all(_add(e, d) in terrain_union for d in MOVES)
        informative.append(known and allowed < 9)
    # These observations cannot have arrived from beyond the previous window.
    required = {e for e in enemies if _distance(e, old_center) <= 1}
    expected = [0.0] * 9
    normalizer = [0.0]
    def enumerate_assignments(i, covered, weight, choices):
        if i == len(sources):
            if not required.issubset(covered):
                return
            normalizer[0] += weight
            for j, (_, ks, probability) in enumerate(choices):
                if informative[j]:
                    for k in ks:
                        expected[k] += weight * law[k] / probability
            return
        for option in options[i]:
            q, ks, probability = option
            enumerate_assignments(
                i + 1, covered if q is None else covered | {q},
                weight * probability, choices + [option])
    enumerate_assignments(0, set(), 1.0, [])
    if normalizer[0] > 1e-15:
        for k in range(9):
            m["counts"][k] += expected[k] / normalizer[0]
        m["updates"] += sum(informative)
        total = sum(m["counts"])
        if total > 45:
            m["counts"] = [c * 45 / total for c in m["counts"]]
def _forecasts(m, law):
    opened = m["door_open"] or (
        m["keys"] >= 2 and any(
            m["map"].get(_add(m["pos"], d)) == 3
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1))))
    tracks = [{e: 1.0} for e in sorted(m["enemies"])]
    forecasts = []
    background = max(0, 3 - len(tracks)) / 140.0
    for horizon in (1, 2):
        new_tracks = []
        for track in tracks:
            new = {}
            for p, mass in track.items():
                for d, probability in zip(MOVES, law):
                    q = _destination(m["map"], p, d, opened)
                    new[q] = new.get(q, 0.0) + mass * probability
            new_tracks.append(new)
        tracks = new_tracks
        cells = set(m["map"])
        for track in tracks:
            cells.update(track)
        risk = {}
        for p in cells:
            if _wall(m["map"], p, opened):
                risk[p] = 0.0
                continue
            distance = _distance(p, m["pos"])
            if distance <= 2 - horizon:
                base = 0.0
            elif p in m["local"]:
                base = background * min(1.0, .18 * horizon)
            else:
                base = background
            empty = 1.0 - base
            for track in tracks:
                empty *= 1.0 - min(1.0, track.get(p, 0.0))
            risk[p] = min(1.0, max(0.0, 1.0 - empty))
        forecasts.append(risk)
    return forecasts, background, opened
def world_model_step(memory, local_obs, last_action):
    if memory is None:
        memory = {"map": {}, "seen": {}, "pos": (0, 0), "visits": {},
                  "counts": [0.0] * 9, "updates": 0, "previous": None,
                  "target": None}
    m = memory
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
    _learn(m, terrain, enemies, local_obs)
    m["local"], m["enemies"] = terrain, enemies
    m["forecast"], m["background"], opened = _forecasts(m, _law(m))
    m["previous"] = (terrain, enemies, m["pos"], m["step"], opened)
    return m
def _paths(m, start, risk, reverse=False):
    dist = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != dist[p]:
            continue
        for d in STEPS:
            q = _add(p, d)
            a, b = (q, p) if reverse else (p, q)
            direction = (b[0] - a[0], b[1] - a[1])
            if _wall(m["map"], q, m["keys"] >= 2 or m["door_open"], True):
                continue
            if not _legal(m, a, direction):
                continue
            edge = 1.0 + 4.0 * risk.get(b, m["background"])
            edge += .025 * min(12, m["visits"].get(b, 0))
            new_cost = cost + edge
            if new_cost < dist.get(q, float("inf")):
                dist[q] = new_cost
                heapq.heappush(queue, (new_cost, q))
    return dist
def _choose_target(m, dist):
    start = m["pos"]
    wanted = 2 if m["keys"] < 2 else 4
    goals = [p for p in dist if m["map"].get(p) == wanted and p != start]
    if not goals and m["keys"] >= 2 and not m["door_open"]:
        goals = [p for p in dist if m["map"].get(p) == 3 and p != start]
    if goals:
        return min(goals, key=lambda p: (dist[p], p))
    frontier = []
    refresh_views = {}
    tick = (m["step"] // 25) * 25
    for p in dist:
        if p == start:
            continue
        gain = 0.0
        refresh = 0.0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                q = _add(p, (dx, dy))
                if q not in m["map"]:
                    gain += 1.0
                elif m["map"][q] != -1 and m["seen"][q] < tick:
                    refresh += 1.0
        refresh_views[p] = refresh
        if gain:
            score = (dist[p] + .08 * m["visits"].get(p, 0)) / gain ** .55
            if p == m["target"]:
                score *= .92
            frontier.append((score, p))
    if frontier:
        return min(frontier)[1]
    # Once exploration is exhausted, revisit stale views after wall ticks.
    alternatives = []
    for p in dist:
        if p == start:
            continue
        age = min(50, m["step"] - m["seen"].get(p, 0))
        score = (dist[p] + .7 * m["visits"].get(p, 0) - .08 * age
                 - .10 * refresh_views.get(p, 0.0))
        alternatives.append((score, p))
    return min(alternatives)[1] if alternatives else start
def _route_steps(m, target):
    # Reverse breadth-first search measures progress independently of
    # changing enemy forecasts and accumulated visitation penalties.
    steps = {target: 0}
    queue = [target]
    for p in queue:
        for dx, dy in STEPS:
            q = _add(p, (dx, dy))
            if q in steps:
                continue
            if _wall(m["map"], q, m["keys"] >= 2 or m["door_open"], True):
                continue
            if _legal(m, q, (-dx, -dy)):
                steps[q] = steps[p] + 1
                queue.append(q)
    return steps
def _recovery_target(m, target, dist, risk):
    start, step = m["pos"], m["step"]
    state = m.setdefault("route_recovery", {
        "signature": None, "best": {}, "advance": step,
        "history": [], "waypoint": None, "objective": None,
        "until": 0, "cooldown": 0,
    })
    signature = (m["keys"], m["door_open"], len(m["map"]))
    if signature != state["signature"]:
        state["signature"] = signature
        state["best"] = {}
        state["advance"] = step
        state["waypoint"] = None
    geometry = _route_steps(m, target)
    route = geometry.get(start, 1000)
    if route < state["best"].get(target, 1000):
        state["best"][target] = route
        state["advance"] = step
    state["history"].append((start, target))
    state["history"] = state["history"][-12:]
    waypoint = state["waypoint"]
    if waypoint is not None:
        if (step < state["until"] and waypoint != start
                and waypoint in dist and target == state["objective"]):
            return waypoint, True
        state["waypoint"] = None
        state["cooldown"] = step + 8
    repeated = sum(p == start and goal == target
                   for p, goal in state["history"])
    if (step - state["advance"] < 8 or repeated < 3
            or step < state["cooldown"] or 200 - step < route + 12):
        return target, False
    tick = (step // 25) * 25
    candidates = []
    for p, travel in dist.items():
        if (travel > 6.0 or _distance(start, p) < 2
                or geometry.get(p, 1000) > route + 3
                or risk.get(p, m["background"]) > .025):
            continue
        revisits = sum(old == p for old, _ in state["history"])
        if revisits > 1:
            continue
        gain = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                cell = _add(p, (dx, dy))
                if cell not in m["map"]:
                    gain += 2
                elif (m["map"][cell] != -1
                      and m["seen"].get(cell, step) < tick):
                    gain += 1
        score = travel + .65 * geometry[p] + .8 * revisits - .15 * gain
        candidates.append((score, p))
    if not candidates:
        state["cooldown"] = step + 4
        return target, False
    waypoint = min(candidates)[1]
    state["waypoint"] = waypoint
    state["objective"] = target
    state["until"] = step + 6
    return waypoint, True
def _hazard(p):
    return -math.log(max(1e-6, 1.0 - min(.999999, max(0.0, p))))
def planner(memory, local_obs):
    m = memory
    start = m["pos"]
    if local_obs.get("predictive_planning", True):
        risk, risk2 = m["forecast"]
    else:
        # This intervention removes learned dynamics from control, while
        # retaining localization, terrain and an ordinary uniform predictor.
        fixed, _, _ = _forecasts(m, [1.0 / 9] * 9)
        risk, risk2 = fixed
    dist = _paths(m, start, risk)
    target = _choose_target(m, dist)
    m["target"] = target
    target, recovering = _recovery_target(m, target, dist, risk)
    potential = _paths(m, target, risk, True)
    pressure = max(20.0, min(85.0, (200 - m["step"]) * 2.5))
    safe_detour = recovering and any(
        _legal(m, start, d)
        and _add(start, d) not in m["enemies"]
        and _add(start, d) in potential
        and risk.get(_add(start, d), m["background"])
            <= min(.02, risk.get(start, m["background"]) + .003)
        for d in STEPS
    )
    choices = []
    for d in MOVES:
        q = _add(start, d)
        if not _legal(m, start, d) or q in m["enemies"]:
            continue
        if q not in potential:
            continue
        immediate = risk.get(q, m["background"])
        future = float("inf")
        if q == target:
            future = 0.0
        else:
            for e in MOVES:
                r = _add(q, e)
                if not _legal(m, q, e) or r not in potential:
                    continue
                before = 0.0 if r == q else risk.get(r, m["background"])
                after = risk2.get(r, m["background"])
                danger = 1.0 - (1.0 - before) * (1.0 - after)
                value = 1.0 + potential[r] + .42 * pressure * _hazard(danger)
                future = min(future, value)
        if not math.isfinite(future):
            future = potential[q] + 4.0
        score = 1.0 + .65 * potential[q] + .35 * future
        score += pressure * _hazard(immediate)
        score += .04 * min(20, m["visits"].get(q, 0))
        if d == (0, 0):
            score += .12 + (1.25 if safe_detour else 0.0)
        choices.append((score, immediate, d))
    move = min(choices)[2] if choices else (0, 0)
    return {"move": list(move), "interact": True}
def export_model(memory, local_obs):
    m = memory
    return {"enemy": [[x, y, p] for (x, y), p in sorted(m["forecast"][0].items())],
            "default_enemy": m["background"],
            "terrain": [[x, y, c, m["seen"][(x, y)]]
                        for (x, y), c in sorted(m["map"].items())],
            "position": list(m["pos"]),
            "learning": {"updates": m["updates"],
                         "attempt_probabilities": _law(m)}}
# EVOLVE-BLOCK-END