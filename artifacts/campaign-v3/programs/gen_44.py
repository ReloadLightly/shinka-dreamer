"""Predictive seed. Entire representation, learner and planner may evolve."""
# EVOLVE-BLOCK-START
import heapq
import math

MOVES = tuple((x, y) for y in (-1, 0, 1) for x in (-1, 0, 1))
STEPS = tuple(d for d in MOVES if d != (0, 0))
CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _wall(terrain, p, opened=False, unknown=False):
    c = terrain.get(p)
    return unknown if c is None else (
        c in (-1, 1) or (c == 3 and not opened)
    )


def _destination(terrain, p, d, opened):
    q = _add(p, d)
    if _wall(terrain, q, opened):
        return p
    if d[0] and d[1]:
        if (
            _wall(terrain, _add(p, (d[0], 0)), opened)
            or _wall(terrain, _add(p, (0, d[1])), opened)
        ):
            return p
    return q


def _door_after(m, p, opened, keys=None, interact=True):
    if opened:
        return True
    if keys is None:
        keys = m["keys"]
    return bool(
        interact and keys >= 2
        and any(m["map"].get(_add(p, d)) == 3 for d in CARDINAL)
    )


def _legal(m, p, d, opened=None):
    if opened is None:
        opened = _door_after(m, p, m["door_open"])
    terrain = m["map"]
    if _wall(terrain, _add(p, d), opened, True):
        return False
    if d[0] and d[1]:
        for side in ((d[0], 0), (0, d[1])):
            if _wall(terrain, _add(p, side), opened, True):
                return False
    return True


def _law(m):
    total = sum(m["counts"]) + 9.0
    return [(c + 1.0) / total for c in m["counts"]]


def _learn(m, terrain, enemies, obs):
    previous = m["previous"]
    if previous is None or not obs.get("learn", True):
        return
    old_terrain, old_enemies, old_center, old_step, old_opened = previous
    if m["step"] != old_step + 1:
        return

    # Interaction precedes enemy movement, so use its confirmed outcome.
    opened = bool(m["door_open"])
    if old_opened and not opened:
        return
    if m["step"] % 25 == 0 or old_step % 25 == 0:
        return

    m["counts"] = [c * .965 for c in m["counts"]]
    if not old_enemies:
        return

    terrain_union = dict(old_terrain)
    terrain_union.update(terrain)
    for p in set(old_terrain).intersection(terrain):
        if old_terrain[p] != terrain[p] and (
            old_terrain[p] == 1 or terrain[p] == 1
        ):
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
        options.append([
            (q, ks, sum(law[k] for k in ks))
            for q, ks in groups.items()
        ])
        known = all(_add(e, d) in terrain_union for d in MOVES)
        informative.append(known and allowed < 9)

    # No previously unseen enemy can enter this inner observation square.
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
                i + 1,
                covered if q is None else covered | {q},
                weight * probability,
                choices + [option],
            )

    enumerate_assignments(0, set(), 1.0, [])
    if normalizer[0] > 1e-15:
        for k in range(9):
            m["counts"][k] += expected[k] / normalizer[0]
        m["updates"] += sum(informative)
        total = sum(m["counts"])
        if total > 45:
            m["counts"] = [c * 45 / total for c in m["counts"]]


def _forecasts(m, law, opened):
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
        memory = {
            "map": {}, "seen": {}, "pos": (0, 0), "visits": {},
            "counts": [0.0] * 9, "updates": 0, "previous": None,
            "target": None, "recent": [], "progress": None,
            "stationary": 0,
        }
    m = memory
    displacement = tuple(local_obs["feedback"]["displacement"])
    m["pos"] = _add(m["pos"], displacement)
    m["keys"] = local_obs["keys"]
    m["door_open"] = local_obs["door_open"]
    m["step"] = local_obs["step"]
    m["visits"][m["pos"]] = m["visits"].get(m["pos"], 0) + 1
    m["stationary"] = (
        m["stationary"] + 1
        if m["previous"] is not None and displacement == (0, 0)
        else 0
    )
    m["recent"].append(m["pos"])
    m["recent"] = m["recent"][-12:]

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
    opened = _door_after(m, m["pos"], m["door_open"], interact=True)
    m["forecast"], m["background"], _ = _forecasts(m, _law(m), opened)
    m["previous"] = (
        terrain, enemies, m["pos"], m["step"], bool(m["door_open"])
    )
    return m


def _paths(m, start, risk, reverse=False, keys=None, opened=None,
           geometric=False):
    if keys is None:
        keys = m["keys"]
    if opened is None:
        opened = m["door_open"]
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
            if _wall(m["map"], q, keys >= 2 or opened, True):
                continue
            edge_open = _door_after(m, a, opened, keys, True)
            if not _legal(m, a, direction, edge_open):
                continue
            edge = 1.0
            if not geometric:
                edge += 4.0 * risk.get(b, m["background"])
                edge += .025 * min(12, m["visits"].get(b, 0))
            new_cost = cost + edge
            if new_cost < dist.get(q, float("inf")):
                dist[q] = new_cost
                heapq.heappush(queue, (new_cost, q))
    return dist


def _choose_target(m, dist, observed_from=None):
    start = m["pos"]
    wanted = 2 if m["keys"] < 2 else 4
    goals = [
        p for p in dist
        if m["map"].get(p) == wanted and p != start
    ]
    if not goals and m["keys"] >= 2 and not m["door_open"]:
        goals = [
            p for p in dist
            if m["map"].get(p) == 3 and p != start
        ]
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
                # A hypothetical completed observation has already revealed
                # these cells, without inventing their terrain values.
                already_revealed = (
                    observed_from is not None
                    and _distance(q, observed_from) <= 2
                )
                if q not in m["map"]:
                    if not already_revealed:
                        gain += 1.0
                elif (
                    m["map"][q] != -1 and m["seen"][q] < tick
                    and not already_revealed
                ):
                    refresh += 1.0
        refresh_views[p] = refresh
        if gain:
            score = (
                dist[p] + .08 * m["visits"].get(p, 0)
            ) / gain ** .55
            if p == m["target"]:
                score *= .92
            frontier.append((score, p))
    if frontier:
        return min(frontier)[1]

    alternatives = []
    for p in dist:
        if p == start:
            continue
        age = min(50, m["step"] - m["seen"].get(p, 0))
        score = (
            dist[p] + .7 * m["visits"].get(p, 0) - .08 * age
            - .10 * refresh_views.get(p, 0.0)
        )
        alternatives.append((score, p))
    return min(alternatives)[1] if alternatives else start


def _hazard(p):
    return -math.log(max(1e-6, 1.0 - min(.999999, max(0.0, p))))


def _planning_transition(m, law, opened):
    cache = {}

    def transition(p):
        if p not in cache:
            result = {}
            for d, probability in zip(MOVES, law):
                q = _destination(m["map"], p, d, opened)
                result[q] = result.get(q, 0.0) + probability
            cache[p] = result
        return cache[p]

    return transition


def _project_tracks(tracks, transition):
    projected = []
    for track in tracks:
        result = {}
        for p, mass in track.items():
            for q, probability in transition(p).items():
                result[q] = result.get(q, 0.0) + mass * probability
        projected.append(result)
    return projected


def _collision_cost(tracks, projected, q, transition):
    stay = transition(q).get(q, 0.0)
    result = 0.0
    for before, after in zip(tracks, projected):
        danger = after.get(q, 0.0) + before.get(q, 0.0) * (1.0 - stay)
        if danger >= 1.0 - 1e-12:
            return float("inf")
        result += _hazard(danger)
    return result


def _surviving_tracks(tracks, projected, q, transition):
    outgoing = transition(q)
    result = []
    for before, after in zip(tracks, projected):
        at_q = before.get(q, 0.0)
        filtered = {}
        for p, mass in after.items():
            if p != q:
                surviving = mass - at_q * outgoing.get(p, 0.0)
                if surviving > 1e-15:
                    filtered[p] = surviving
        total = sum(filtered.values())
        if total <= 1e-12:
            return None
        result.append({
            p: mass / total for p, mass in filtered.items()
        })
    return result


def _background_cost(m, p, q, depth):
    def probability(cell, horizon):
        if horizon <= 0:
            return 0.0
        if _distance(cell, m["pos"]) <= 2 - horizon:
            return 0.0
        if cell in m["local"]:
            return m["background"] * min(1.0, .18 * horizon)
        return m["background"]

    after = probability(q, depth)
    before = 0.0 if p == q else probability(q, depth - 1)
    return _hazard(before) + _hazard(after)


def _stagnation(m, target, geometric):
    route = geometric.get(m["pos"], 250.0)
    signature = (target, m["keys"], m["door_open"])
    state = m["progress"]
    if state is None or state["signature"] != signature:
        state = {
            "signature": signature,
            "best": route,
            "step": m["step"],
        }
        m["progress"] = state
    elif route < state["best"]:
        state["best"] = route
        state["step"] = m["step"]
    repeats = m["recent"].count(m["pos"])
    return min(1.0, max(
        0.0,
        (m["stationary"] - 6) / 12.0,
        (m["step"] - state["step"] - 12) / 20.0,
        (repeats - 5) / 7.0,
    ))


def planner(memory, local_obs):
    m = memory
    start = m["pos"]
    opened0 = _door_after(m, start, m["door_open"], interact=True)
    if local_obs.get("predictive_planning", True):
        learned = _law(m)
        # A modest uncertainty floor is applied only to control.
        law = [.90 * p + .10 / 9.0 for p in learned]
    else:
        law = [1.0 / 9.0] * 9
    fixed, _, _ = _forecasts(m, law, opened0)
    risk = fixed[0]

    dist = _paths(m, start, risk)
    target = _choose_target(m, dist)
    m["target"] = target
    target_cell = m["map"].get(target)
    primary = _paths(m, target, risk, reverse=True)
    geometric = _paths(
        m, target, {}, reverse=True, geometric=True
    )
    stagnation = _stagnation(m, target, geometric)
    remaining = max(1, 200 - m["step"])
    pressure = max(22.0, min(95.0, remaining * 2.5))
    pressure *= 1.0 - .15 * stagnation

    key_cells = sorted(
        p for p, cell in m["map"].items() if cell == 2
    )
    key_bits = {p: 1 << i for i, p in enumerate(key_cells)}
    transitions = {
        state: _planning_transition(m, law, state)
        for state in (False, True)
    }
    primary_cache = {(m["keys"], bool(m["door_open"])): primary}
    continuation_cache = {}
    recent_counts = {}
    for p in m["recent"]:
        recent_counts[p] = recent_counts.get(p, 0) + 1

    completion_reward = (
        1.0 if target_cell == 2
        else .65 if target_cell == 3
        else .30
    )

    def collect(q, keys, mask):
        bit = key_bits.get(q, 0)
        if bit and not (mask & bit):
            return min(2, keys + 1), mask | bit
        return keys, mask

    def completed(q, opened, done):
        return bool(
            done
            or (q == target and target != start)
            or (
                target_cell == 3
                and opened and not m["door_open"]
            )
        )

    def escaped(q, keys, opened):
        return bool(
            m["map"].get(q) == 4 and keys >= 2 and opened
        )

    def continuation(keys, mask, opened):
        cache_key = (keys, mask, bool(opened))
        if cache_key in continuation_cache:
            return continuation_cache[cache_key]

        # This shadow state changes only hypothetical inventory and
        # navigation. It never writes predicted terrain into real memory.
        shadow = dict(m)
        shadow["map"] = dict(m["map"])
        for p, bit in key_bits.items():
            if mask & bit:
                shadow["map"][p] = 0
        shadow["pos"] = target
        shadow["keys"] = keys
        shadow["door_open"] = bool(opened)
        shadow["target"] = None
        reachable = _paths(
            shadow, target, risk, keys=keys, opened=opened
        )
        next_target = _choose_target(
            shadow, reachable, observed_from=target
        )
        if next_target == target:
            result = ({}, 0.0, False)
        else:
            field = _paths(
                shadow, next_target, risk, reverse=True,
                keys=keys, opened=opened,
            )
            result = (field, field.get(target, 0.0), True)
        continuation_cache[cache_key] = result
        return result

    def potential(p, keys, mask, opened, done):
        if not done:
            cache_key = (keys, bool(opened))
            if cache_key not in primary_cache:
                primary_cache[cache_key] = _paths(
                    m, target, risk, reverse=True,
                    keys=keys, opened=opened,
                )
            return primary_cache[cache_key].get(p, 1e6)

        if target_cell == 4 and keys >= 2 and opened:
            return 0.0

        field, baseline, useful = continuation(keys, mask, opened)
        if useful:
            onward = field.get(p, baseline + 8.0) - baseline
            onward = max(-4.0, min(8.0, onward))
        else:
            onward = 0.0
        # A branch retains this completed phase forever. Returning to the
        # same key, door, or frontier therefore cannot collect more reward.
        return onward - completion_reward

    def escape_tail(q, keys, mask, opened, done):
        if target_cell == 4:
            return 0.0
        return min(0.0, potential(q, keys, mask, opened, done))

    def stage(p, q, depth, hazard):
        weight = (0.0, 1.0, .60, .35)[depth]
        result = 1.0 + pressure * weight * hazard
        result += (.035 if depth == 1 else .02) * min(
            20, m["visits"].get(q, 0)
        )
        result += .07 * stagnation * recent_counts.get(q, 0)
        if p == q:
            result += .12 + .35 * stagnation
        return result

    tracks0 = [{e: 1.0} for e in sorted(m["enemies"])]
    transition0 = transitions[opened0]
    projected0 = _project_tracks(tracks0, transition0)
    choices = []
    emergency = []

    for d in MOVES:
        q = _add(start, d)
        if not _legal(m, start, d, opened0) or q in m["enemies"]:
            continue

        h1 = _collision_cost(tracks0, projected0, q, transition0)
        h1 += _background_cost(m, start, q, 1)
        emergency.append((h1, primary.get(q, 1e6), d))
        if not math.isfinite(h1):
            continue

        keys1, mask1 = collect(q, m["keys"], 0)
        done1 = completed(q, opened0, False)
        first_cost = stage(start, q, 1, h1)

        if escaped(q, keys1, opened0):
            score = first_cost + escape_tail(
                q, keys1, mask1, opened0, done1
            )
            choices.append((score, h1, d))
            continue

        tracks1 = _surviving_tracks(
            tracks0, projected0, q, transition0
        )
        if tracks1 is None:
            continue
        opened1 = _door_after(m, q, opened0, keys1, True)
        transition1 = transitions[opened1]
        projected1 = _project_tracks(tracks1, transition1)
        seconds = []

        for e in MOVES:
            r = _add(q, e)
            if not _legal(m, q, e, opened1):
                continue
            h2 = _collision_cost(
                tracks1, projected1, r, transition1
            )
            h2 += _background_cost(m, q, r, 2)
            if not math.isfinite(h2):
                continue
            keys2, mask2 = collect(r, keys1, mask1)
            done2 = completed(r, opened1, done1)
            terminal2 = escaped(r, keys2, opened1)
            cost2 = stage(q, r, 2, h2)
            estimate = cost2 + potential(
                r, keys2, mask2, opened1, done2
            )
            seconds.append((
                estimate, h2, r, keys2, mask2, done2,
                terminal2, cost2,
            ))

        # Keep promising progress choices plus the safest continuation.
        # Every immediate action receives a separate continuation beam.
        seconds.sort()
        beam = seconds[:4]
        if seconds:
            safest = min(seconds, key=lambda item: (item[1], item[0]))
            if safest not in beam:
                beam.append(safest)

        future = float("inf")
        for (
            estimate, h2, r, keys2, mask2, done2,
            terminal2, cost2,
        ) in beam:
            if terminal2:
                value = cost2 + escape_tail(
                    r, keys2, mask2, opened1, done2
                )
                future = min(future, value)
                continue

            tracks2 = _surviving_tracks(
                tracks1, projected1, r, transition1
            )
            if tracks2 is None:
                continue
            opened2 = _door_after(m, r, opened1, keys2, True)
            transition2 = transitions[opened2]
            projected2 = _project_tracks(tracks2, transition2)
            third = float("inf")

            for f in MOVES:
                s = _add(r, f)
                if not _legal(m, r, f, opened2):
                    continue
                h3 = _collision_cost(
                    tracks2, projected2, s, transition2
                )
                h3 += _background_cost(m, r, s, 3)
                if not math.isfinite(h3):
                    continue
                keys3, mask3 = collect(s, keys2, mask2)
                done3 = completed(s, opened2, done2)
                if escaped(s, keys3, opened2):
                    tail = escape_tail(
                        s, keys3, mask3, opened2, done3
                    )
                else:
                    tail = potential(
                        s, keys3, mask3, opened2, done3
                    )
                third = min(
                    third, stage(r, s, 3, h3) + tail
                )

            future = min(future, cost2 + third)

        score = first_cost + future
        if math.isfinite(score):
            choices.append((score, h1, d))

    if choices:
        move = min(choices)[2]
    else:
        move = min(emergency)[2] if emergency else (0, 0)
    return {"move": list(move), "interact": True}


def export_model(memory, local_obs):
    m = memory
    return {
        "enemy": [
            [x, y, float(min(1.0, max(0.0, p)))]
            for (x, y), p in sorted(m["forecast"][0].items())
        ],
        "default_enemy": float(m["background"]),
        "terrain": [
            [x, y, c, m["seen"][(x, y)]]
            for (x, y), c in sorted(m["map"].items())
        ],
        "position": list(m["pos"]),
        "learning": {
            "updates": m["updates"],
            "attempt_probabilities": _law(m),
        },
    }
# EVOLVE-BLOCK-END