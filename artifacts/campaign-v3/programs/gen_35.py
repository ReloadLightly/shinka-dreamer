"""Predictive seed. Entire representation, learner and planner may evolve."""
# EVOLVE-BLOCK-START
import heapq
import math
from functools import lru_cache

MOVES = tuple((x, y) for y in (-1, 0, 1) for x in (-1, 0, 1))
WALK = tuple(d for d in MOVES if d != (0, 0))
CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
UNIFORM = [1.0 / 9.0] * 9
HORIZON = 4
NEVER = HORIZON + 1


def add(p, d):
    return p[0] + d[0], p[1] + d[1]


def distance(p, q):
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def blocked(terrain, p, opened=False, unknown=False):
    cell = terrain.get(p)
    if cell is None:
        return unknown
    return cell in (-1, 1) or (cell == 3 and not opened)


def destination(terrain, p, d, opened):
    q = add(p, d)
    if blocked(terrain, q, opened):
        return p
    if d[0] and d[1]:
        if (blocked(terrain, add(p, (d[0], 0)), opened)
                or blocked(terrain, add(p, (0, d[1])), opened)):
            return p
    return q


def interaction_state(m, p, opened, keys, interact):
    if opened:
        return True
    return bool(
        interact and keys >= 2
        and any(m["map"].get(add(p, d)) == 3 for d in CARDINAL)
    )


def legal(m, p, q, opened):
    dx, dy = q[0] - p[0], q[1] - p[1]
    if abs(dx) > 1 or abs(dy) > 1:
        return False
    if blocked(m["map"], q, opened, True):
        return False
    if dx and dy:
        if (blocked(m["map"], (p[0] + dx, p[1]), opened, True)
                or blocked(m["map"], (p[0], p[1] + dy), opened, True)):
            return False
    return True


def count_law(counts):
    total = sum(counts) + 6.3
    return [(c + .7) / total for c in counts]


def component_laws(m):
    return [UNIFORM] + [count_law(c) for c in m["counts"]]


def learned_law(m):
    laws = component_laws(m)
    return [
        sum(w * law[k] for w, law in zip(m["weights"], laws))
        for k in range(9)
    ]


def transition_terms(m):
    previous = m["previous"]
    if previous is None:
        return None
    old_map, sources, old_center, step, was_open, old_seen, keys = previous
    if m["step"] != step + 1 or not sources:
        return None
    if m["step"] % 25 == 0 or step % 25 == 0:
        return None

    action = m.get("last_action") or {}
    expected_open = was_open or (
        bool(action.get("interact", False)) and keys >= 2
        and any(old_map.get(add(old_center, d)) == 3 for d in CARDINAL)
    )
    opened = m["door_open"]
    if opened != bool(expected_open):
        return None
    opening_transition = opened and not was_open

    terrain = dict(old_map)
    for p, cell in m["local"].items():
        old = old_map.get(p)
        if old is not None and old != cell:
            # Opening precedes both agent movement and enemy movement.
            # A door becoming empty is explained by this action, not by
            # a change in the unknown attempted-movement law.
            explained = opening_transition and old == 3 and cell == 0
            if not explained and (old in (1, 3) or cell in (1, 3)):
                return None
        terrain[p] = cell

    epoch = 25 * (step // 25)
    options = []
    for source in sources:
        for d in MOVES:
            p = add(source, d)
            if p not in terrain:
                return None
            if p not in m["local"] and old_seen.get(p, -1) < epoch:
                return None
        groups = {}
        for k, d in enumerate(MOVES):
            q = destination(terrain, source, d, opened)
            if q in m["local"]:
                if q not in m["enemies"]:
                    continue
                label = q
            else:
                label = None
            groups[label] = groups.get(label, 0) | (1 << k)
        if not groups:
            return None
        options.append(list(groups.items()))

    required = {
        p for p in m["enemies"] if distance(p, old_center) <= 1
    }
    terms = []

    def visit(i, covered, masks):
        if i == len(options):
            if required.issubset(covered):
                terms.append(tuple(masks))
            return
        for label, mask in options[i]:
            visit(
                i + 1,
                covered if label is None else covered | {label},
                masks + [mask],
            )

    visit(0, set(), [])
    return terms or None


def evidence_and_counts(law, terms):
    masks = {mask for term in terms for mask in term}
    probabilities = {
        mask: sum(law[k] for k in range(9) if mask & (1 << k))
        for mask in masks
    }
    evidence = 0.0
    expected = [0.0] * 9
    for term in terms:
        weight = 1.0
        for mask in term:
            weight *= probabilities[mask]
        evidence += weight
        for mask in term:
            if mask == 511:
                continue
            denominator = probabilities[mask]
            if denominator > 1e-15:
                for k in range(9):
                    if mask & (1 << k):
                        expected[k] += weight * law[k] / denominator
    if evidence > 1e-15:
        expected = [value / evidence for value in expected]
    return evidence, expected


def cap_counts(counts, cap):
    total = sum(counts)
    if total > cap:
        scale = cap / total
        for k in range(9):
            counts[k] *= scale


def rebuild_fast(m):
    counts = [0.0] * 9
    for _, terms in m["recent_terms"]:
        evidence, expected = evidence_and_counts(count_law(counts), terms)
        for k in range(9):
            counts[k] *= .78
            if evidence > 1e-15:
                counts[k] += expected[k]
    cap_counts(counts, 10.0)
    m["counts"][1] = counts
    uniform = max(.15, min(.35, m["weights"][0]))
    fast = max(.40, min(.55, m["weights"][2]))
    m["weights"] = [uniform, 1.0 - uniform - fast, fast]
    m["surprise"] = [0.0, 0.0]
    m["surprise_signs"] = []
    m["last_reset"] = m["step"]
    m["recoveries"] += 1


def update_dynamics(m, obs):
    if not obs.get("learn", True):
        return

    for counts, decay in zip(m["counts"], (.992, .88)):
        for k in range(9):
            counts[k] *= decay
    m["weights"] = [
        .975 * w + .025 * prior
        for w, prior in zip(m["weights"], (.30, .45, .25))
    ]

    terms = transition_terms(m)
    if not terms:
        return
    if all(all(mask == 511 for mask in term) for term in terms):
        return

    results = [
        evidence_and_counts(law, terms) for law in component_laws(m)
    ]
    if results[0][0] >= 1.0 - 1e-10:
        return

    count = max(1, len(terms[0]))
    contamination = .004 * (1.0 / 9.0) ** count
    likelihood = [
        max(1e-30, .996 * result[0] + contamination)
        for result in results
    ]
    advantages = [
        math.log(likelihood[0] / likelihood[1]) / count,
        math.log(likelihood[2] / likelihood[1]) / count,
    ]
    for i, advantage in enumerate(advantages):
        m["surprise"][i] = max(
            0.0,
            .94 * m["surprise"][i]
            + max(-1.5, min(1.5, advantage)) - .035,
        )
    m["surprise_signs"].append(
        tuple(advantage > .035 for advantage in advantages)
    )
    m["surprise_signs"] = m["surprise_signs"][-5:]
    m["recent_terms"].append((m["step"], terms))
    m["recent_terms"] = [
        item for item in m["recent_terms"][-5:]
        if m["step"] - item[0] <= 16
    ]

    weights = [
        w * probability ** .8
        for w, probability in zip(m["weights"], likelihood)
    ]
    total = sum(weights)
    if total > 1e-25:
        m["weights"] = [.94 * w / total + .02 for w in weights]

    for counts, result, cap in zip(
            m["counts"], results[1:], (80.0, 16.0)):
        if result[0] <= 1e-15:
            continue
        for k in range(9):
            counts[k] += result[1][k]
        cap_counts(counts, cap)

    sustained = any(
        m["surprise"][i] > 1.15
        and sum(signs[i] for signs in m["surprise_signs"]) >= 3
        for i in range(2)
    )
    if (sustained and sum(m["counts"][0]) >= 10.0
            and len(m["recent_terms"]) >= 3
            and m["step"] - m["last_reset"] >= 10):
        rebuild_fast(m)

    m["updates"] += 1
    if m["previous"] is not None:
        if m["door_open"] and not m["previous"][4]:
            m["opening_updates"] += 1


def make_transition(m, law, opened):
    terrain = m["map"]
    left, right, top, bottom = m["enemy_bounds"]
    cache = {}

    def transition(p):
        if p not in cache:
            result = {}
            for d, probability in zip(MOVES, law):
                q = destination(terrain, p, d, opened)
                if not (left <= q[0] <= right and top <= q[1] <= bottom):
                    q = p
                result[q] = result.get(q, 0.0) + probability
            cache[p] = result
        return cache[p]

    return transition


def propagate(distribution, transition):
    result = {}
    for p, mass in distribution.items():
        if mass < 1e-10:
            continue
        for q, probability in transition(p).items():
            result[q] = result.get(q, 0.0) + mass * probability
    return result


def occupancy(tracks, fog, p):
    empty = math.exp(-max(0.0, fog.get(p, 0.0)))
    for track in tracks:
        empty *= 1.0 - min(1.0, max(0.0, track.get(p, 0.0)))
    return min(1.0, max(0.0, 1.0 - empty))


def reconcile_fog(m):
    terrain = m["map"]
    physical = [p for p, cell in terrain.items() if cell != -1]
    left = max(-13, max(p[0] for p in physical) - 13)
    right = min(13, min(p[0] for p in physical) + 13)
    top = max(-13, max(p[1] for p in physical) - 13)
    bottom = min(13, min(p[1] for p in physical) + 13)

    for p, cell in terrain.items():
        if cell != -1:
            continue
        for dx, dy in CARDINAL:
            q = add(p, (dx, dy))
            if q not in terrain or terrain[q] == -1:
                continue
            if dx == 1:
                left = max(left, q[0] + 1)
            elif dx == -1:
                right = min(right, q[0] - 1)
            elif dy == 1:
                top = max(top, q[1] + 1)
            else:
                bottom = min(bottom, q[1] - 1)

    m["enemy_bounds"] = (left, right, top, bottom)
    support = {}
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            p = (x, y)
            if p in m["local"] or blocked(terrain, p, m["door_open"]):
                continue
            support[p] = 1.0 if p in terrain else .68

    missing = max(0, 3 - len(m["enemies"]))
    if not missing or not support:
        return {}

    prior_total = sum(support.values())
    residual = {
        p: max(0.0, m["density"].get(p, 0.0))
        for p in support
    }
    residual_total = sum(residual.values())
    if residual_total <= 1e-12:
        return {
            p: missing * weight / prior_total
            for p, weight in support.items()
        }
    return {
        p: missing * (
            .995 * residual[p] / residual_total
            + .005 * weight / prior_total
        )
        for p, weight in support.items()
    }


def first_forecast(m, law, opened):
    transition = make_transition(m, law, opened)
    tracks = [dict(transition(p)) for p in sorted(m["enemies"])]
    fog = propagate(m["fog"], transition)
    return tracks, fog, transition


def set_prediction(m, opened):
    if m.get("forecast_opened") == opened:
        return
    forecasts = []
    cells = set(m["map"])
    density = {}
    for weight, law in zip(m["weights"], component_laws(m)):
        tracks, fog, _ = first_forecast(m, law, opened)
        forecasts.append((list(law), tracks, fog))
        cells.update(fog)
        for p, mass in fog.items():
            density[p] = density.get(p, 0.0) + weight * mass
        for track in tracks:
            cells.update(track)
            for p, mass in track.items():
                density[p] = density.get(p, 0.0) + weight * mass

    m["forecast_components"] = forecasts
    m["forecast_opened"] = opened
    m["prediction"] = {
        p: 0.0 if blocked(m["map"], p, opened) else min(
            1.0, max(0.0, sum(
                weight * occupancy(tracks, fog, p)
                for weight, (_, tracks, fog)
                in zip(m["weights"], forecasts)
            ))
        )
        for p in cells
    }
    m["default"] = 0.0
    m["density"] = density


def world_model_step(memory, local_obs, last_action):
    if memory is None:
        memory = {
            "map": {}, "seen": {}, "pos": (0, 0), "visits": {},
            "counts": [[0.0] * 9, [0.0] * 9],
            "weights": [.30, .45, .25], "updates": 0,
            "previous": None, "target": None, "density": {},
            "fog": {}, "stall": 0, "last_distance": None,
            "last_objective": None,
            "surprise": [0.0, 0.0], "surprise_signs": [],
            "recent_terms": [], "last_reset": -100,
            "recoveries": 0, "opening_updates": 0,
            "chosen_action": None,
        }
    m = memory
    m["last_action"] = (
        last_action if last_action is not None else m.get("chosen_action")
    )
    m["pos"] = add(m["pos"], local_obs["feedback"]["displacement"])
    m["keys"] = int(local_obs["keys"])
    m["door_open"] = bool(local_obs["door_open"])
    m["step"] = int(local_obs["step"])
    m["visits"][m["pos"]] = m["visits"].get(m["pos"], 0) + 1

    local, enemies = {}, set()
    for y, row in enumerate(local_obs["terrain"]):
        for x, cell in enumerate(row):
            p = add(m["pos"], (x - 2, y - 2))
            local[p] = int(cell)
            m["map"][p], m["seen"][p] = int(cell), m["step"]
            if local_obs["grid"][y][x] == 5:
                enemies.add(p)
    m["local"], m["enemies"] = local, enemies

    # Mapping/localization remain active under predictive interventions.
    update_dynamics(m, local_obs)
    m["law"] = learned_law(m)
    m["fog"] = reconcile_fog(m)
    m["forecast_opened"] = None
    m["chosen_action"] = None
    set_prediction(m, m["door_open"])

    # Preserve actual preceding state, not an assumed future interaction.
    m["previous"] = (
        dict(m["map"]), tuple(sorted(enemies)), m["pos"], m["step"],
        m["door_open"], dict(m["seen"]), m["keys"],
    )
    return m


def graph(m):
    # Nodes contain physical position and the actual persistent door state.
    states = (
        (True,) if m["door_open"]
        else ((False, True) if m["keys"] >= 2 else (False,))
    )
    edges = {
        (p, opened): []
        for opened in states
        for p in m["map"]
        if not blocked(m["map"], p, opened, True)
    }
    for node in edges:
        p, opened = node
        outcomes = [opened]
        if not opened and interaction_state(m, p, False, m["keys"], True):
            outcomes.append(True)
        for next_open in outcomes:
            # Staying while opening changes state and is a real graph edge.
            moves = WALK if next_open == opened else MOVES
            for d in moves:
                q = add(p, d)
                following = (q, next_open)
                if following in edges and legal(m, p, q, next_open):
                    edges[node].append(following)
    return edges


def reverse_graph(edges):
    reverse = {node: [] for node in edges}
    for node, neighbors in edges.items():
        for following in neighbors:
            reverse.setdefault(following, []).append(node)
    return reverse


def breadth_distances(edges, sources):
    dist = {node: 0 for node in sources}
    queue = list(dist)
    for node in queue:
        for following in edges.get(node, ()):
            if following not in dist:
                dist[following] = dist[node] + 1
                queue.append(following)
    return dist


def paths(edges, sources, m, risk, reverse=False):
    adjacency = reverse_graph(edges) if reverse else edges
    dist = {node: 0.0 for node in sources}
    queue = [(0.0, node) for node in dist]
    heapq.heapify(queue)
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != dist[node]:
            continue
        for following in adjacency.get(node, ()):
            cell = node if reverse else following
            edge = 1.0 + 4.0 * risk.get(cell, m["default"])
            edge += .02 * min(15, m["visits"].get(cell[0], 0))
            new = cost + edge
            if new < dist.get(following, float("inf")):
                dist[following] = new
                heapq.heappush(queue, (new, following))
    return dist


def position_costs(dist):
    result = {}
    for (p, _), cost in dist.items():
        result[p] = min(result.get(p, float("inf")), cost)
    return result


def bottleneck_view(m, state_dist, objectives):
    """Use optimistic connectivity only to choose an observation target."""
    step = m["step"]
    epoch = 25 * (step // 25)
    start = m["pos"]
    objectives = set(objectives)
    dist = position_costs(state_dist)
    active = m.get("bottleneck_observation")

    if active is not None:
        goal, blockers, view, selected, deadline, old_epoch = active
        pending = any(m["seen"].get(p, -1) < selected for p in blockers)
        if (
            goal in objectives and view in dist and view != start
            and pending and step < deadline and epoch == old_epoch
        ):
            return view
        m["bottleneck_observation"] = None
        m["bottleneck_cooldown"] = step + 3

    if (
        not objectives or epoch == 0
        or step < m.get("bottleneck_cooldown", -1)
    ):
        return None

    reachable = objectives.intersection(dist)
    if reachable and (
        m["stall"] < 12 or min(dist[p] for p in reachable) <= 4.0
    ):
        return None

    left, right, top, bottom = m["enemy_bounds"]
    classification = {}

    def classify(p, opened):
        key = (p, opened)
        if key in classification:
            return classification[key]
        cell = m["map"].get(p, -1)
        if (
            not (left <= p[0] <= right and top <= p[1] <= bottom)
            or cell == -1
            or (cell == 3 and not opened)
        ):
            result = -1
        elif cell == 1:
            permanent = any(
                m["map"].get(add(p, d)) == -1 for d in CARDINAL
            )
            result = (
                1 if not permanent and m["seen"].get(p, step) < epoch
                else -1
            )
        else:
            result = 0
        classification[key] = result
        return result

    source = (start, m["door_open"])
    costs = {source: 0.0}
    parents = {}
    queue = [(0.0, source)]
    finish = None
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != costs[node]:
            continue
        p, opened = node
        if p in objectives and p != start:
            finish = node
            break
        outcomes = [opened]
        if not opened and interaction_state(m, p, False, m["keys"], True):
            outcomes.append(True)
        for next_open in outcomes:
            moves = WALK if next_open == opened else MOVES
            for d in moves:
                q = add(p, d)
                required = [q]
                if d[0] and d[1]:
                    required.extend((
                        (p[0] + d[0], p[1]),
                        (p[0], p[1] + d[1]),
                    ))
                kinds = [classify(cell, next_open) for cell in required]
                if any(kind < 0 for kind in kinds):
                    continue
                blockers = tuple(
                    cell for cell, kind in zip(required, kinds) if kind == 1
                )
                following = (q, next_open)
                candidate = cost + 1.0 + 5.0 * len(blockers)
                if candidate < costs.get(following, float("inf")):
                    costs[following] = candidate
                    parents[following] = (node, blockers)
                    heapq.heappush(queue, (candidate, following))

    if finish is None:
        m["bottleneck_cooldown"] = step + 5
        return None

    crossings = []
    node = finish
    while node != source:
        node, blockers = parents[node]
        crossings.append(blockers)
    ordered = []
    for blockers in reversed(crossings):
        for p in blockers:
            if p not in ordered:
                ordered.append(p)
    if not ordered:
        m["bottleneck_cooldown"] = step + 5
        return None

    # The first uncertain crossing has priority. Looking farther along a
    # hypothetical route is useful only when the same view reveals both.
    first = ordered[0]
    options = []
    for view, travel in dist.items():
        if view == start or view in m["enemies"] or distance(view, first) > 2:
            continue
        revealed = tuple(p for p in ordered if distance(view, p) <= 2)
        score = (
            travel + .06 * min(20, m["visits"].get(view, 0))
            + .08 * distance(view, finish[0])
            - .40 * min(3, len(revealed))
        )
        options.append((score, travel, view, revealed))
    if not options:
        m["bottleneck_cooldown"] = step + 5
        return None

    _, travel, view, revealed = min(options)
    duration = min(20, max(6, int(math.ceil(travel)) + 5))
    m["bottleneck_observation"] = (
        finish[0], revealed, view, step, step + duration, epoch,
    )
    m["bottleneck_routes"] = m.get("bottleneck_routes", 0) + 1
    return view


def choose_target(m, edges, state_dist, risk):
    dist = position_costs(state_dist)
    start = m["pos"]
    desired = 2 if m["keys"] < 2 else 4
    goals = [
        p for p in dist if p != start and m["map"].get(p) == desired
    ]
    # Preserve ordinary objective priority. An unreachable remembered exit
    # may instead be approached through its gating door.
    observation_goals = list(goals)
    if not observation_goals:
        observation_goals = [
            p for p, cell in m["map"].items()
            if p != start and cell == desired
        ]
        if m["keys"] >= 2 and not m["door_open"]:
            observation_goals.extend(
                p for p, cell in m["map"].items()
                if p != start and cell == 3
            )
    view = bottleneck_view(m, state_dist, observation_goals)
    if view is not None:
        return view
    if goals:
        if desired == 2 and m["keys"] == 0 and len(goals) > 1:
            doors = [p for p, cell in m["map"].items() if cell in (3, 4)]
            tours = []
            for first in goals:
                sources = [node for node in state_dist if node[0] == first]
                onward = position_costs(paths(edges, sources, m, risk))
                for second in goals:
                    if second == first or second not in onward:
                        continue
                    tail = min(
                        (distance(second, p) for p in doors), default=0
                    )
                    tours.append((
                        dist[first] + onward[second] + .6 * tail, first,
                    ))
            if tours:
                return min(tours)[1]
        return min(goals, key=lambda p: (dist[p], p))

    if m["keys"] >= 2 and not m["door_open"]:
        doors = [
            p for p in dist if p != start and m["map"].get(p) == 3
        ]
        if doors:
            return min(doors, key=lambda p: (dist[p], p))

    frontier, fallback = [], []
    tick = 25 * (m["step"] // 25)
    for p, travel in dist.items():
        if p == start:
            continue
        unseen = stale = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                q = add(p, (dx, dy))
                if q not in m["map"]:
                    unseen += 1
                elif m["map"][q] != -1 and m["seen"][q] < tick:
                    stale += 1
        visits = m["visits"].get(p, 0)
        if unseen:
            score = (travel + .15 * visits) / unseen ** .55
            if p == m["target"]:
                score *= .94
            frontier.append((score, p))
        fallback.append((travel + .9 * visits - .20 * stale, p))
    if frontier:
        return min(frontier)[1]
    return min(fallback)[1] if fallback else start


def hazard(probability):
    return -math.log(max(
        1e-10, 1.0 - min(1.0, max(0.0, probability))
    ))


def mixture_hazard(probabilities, weights):
    total = sum(weights)
    if total <= 1e-15:
        return hazard(1.0)
    weights = [w / total for w in weights]
    mean = sum(w * p for w, p in zip(weights, probabilities))
    cutoff = max(.08, .20 * max(weights))
    plausible = [
        (w, p) for w, p in zip(weights, probabilities) if w >= cutoff
    ]
    mass = sum(w for w, _ in plausible)
    center = sum(w * p for w, p in plausible) / mass
    variance = sum(
        w * (p - center) ** 2 for w, p in plausible
    ) / mass
    upper = max(mean, max(p for _, p in plausible))
    conservative = min(upper, mean + .45 * math.sqrt(variance))
    return hazard(conservative)


def prepare_second(expert, second_kernel):
    tracks, fog, _ = expert
    return (
        tracks,
        [propagate(track, second_kernel) for track in tracks],
        fog,
        propagate(fog, second_kernel),
        second_kernel,
    )


def second_probability(prepared, q, r):
    tracks, after, fog, fog_after, transition = prepared
    qr = transition(q).get(r, 0.0)
    rr = transition(r).get(r, 0.0)
    total = 0.0
    for old, new in zip(tracks, after):
        at_q = old.get(q, 0.0)
        surviving = 1.0 - at_q
        if surviving <= 1e-12:
            return 1.0
        danger = max(0.0, new.get(r, 0.0) - at_q * qr) / surviving
        if r != q:
            # Entering an occupied cell kills before enemy movement.
            danger += old.get(r, 0.0) * (1.0 - rr) / surviving
        total += hazard(danger)

    intensity = max(
        0.0, fog_after.get(r, 0.0) - fog.get(q, 0.0) * qr
    )
    if r != q:
        intensity += fog.get(r, 0.0) * (1.0 - rr)
    total += max(0.0, intensity)
    return min(1.0, max(0.0, -math.expm1(-total)))


def pool_states(states, weights):
    tracks = [{} for _ in states[0][0]]
    fog = {}
    for weight, (source_tracks, source_fog) in zip(weights, states):
        for pooled, track in zip(tracks, source_tracks):
            for p, mass in track.items():
                pooled[p] = pooled.get(p, 0.0) + weight * mass
        for p, mass in source_fog.items():
            fog[p] = fog.get(p, 0.0) + weight * mass
    return tracks, fog


def route_forecasts(m, roots, prepared, kernels, weights):
    # Opening time is sufficient history for this irreversible transition.
    # Histories share their identical prefixes. Expert identities are kept
    # through horizon two; later marginal pooling is a route approximation.
    cells = list(m["map"])
    state_cache = {}
    field_cache = {}
    mixed_cache = {}

    def mixed_kernel(opened):
        if opened not in mixed_cache:
            cache = {}

            def transition(p):
                if p not in cache:
                    result = {}
                    for weight, kernel in zip(weights, kernels(opened)):
                        for q, probability in kernel(p).items():
                            result[q] = result.get(q, 0.0) + weight * probability
                    cache[p] = result
                return cache[p]

            mixed_cache[opened] = transition
        return mixed_cache[opened]

    def state(t, onset):
        onset = min(onset, t + 1)
        key = (t, onset)
        if key in state_cache:
            return state_cache[key]
        if t == 1:
            result = [
                (tracks, fog) for tracks, fog, _ in roots(onset <= 1)
            ]
        elif t == 2:
            result = [
                (item[1], item[3])
                for item in prepared(onset <= 1, onset <= 2)
            ]
        else:
            previous = state(t - 1, onset)
            if t == 3:
                previous = [pool_states(previous, weights)]
            transition = mixed_kernel(onset <= t)
            result = [
                (
                    [propagate(track, transition) for track in tracks],
                    propagate(fog, transition),
                )
                for tracks, fog in previous
            ]
        state_cache[key] = result
        return result

    def field(t, onset):
        key = (t, min(onset, t + 1))
        if key not in field_cache:
            states = state(t, onset)
            active_weights = weights if t <= 2 else [1.0]
            field_cache[key] = {
                p: sum(
                    weight * occupancy(tracks, fog, p)
                    for weight, (tracks, fog) in zip(active_weights, states)
                )
                for p in cells
            }
        return field_cache[key]

    if m["door_open"]:
        onsets = (0,)
    elif m["keys"] >= 2:
        onsets = tuple(range(1, NEVER + 1))
    else:
        onsets = (NEVER,)
    profiles = {
        onset: [field(t, onset) for t in range(1, HORIZON + 1)]
        for onset in onsets
    }

    left, right, top, bottom = m["enemy_bounds"]
    area = 0.0
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            p = (x, y)
            if p not in m["map"]:
                area += .68
            elif not blocked(m["map"], p, m["door_open"]):
                area += 1.0
    background = min(.06, max(.012, 3.0 / max(60.0, area)))
    return profiles, background


def arrival_route_risk(edges, source, profiles, background):
    arrival = {source: 0}
    onset = {source: 0 if source[1] else NEVER}
    queue = [source]
    for node in queue:
        for following in edges.get(node, ()):
            if following not in arrival:
                step = arrival[node] + 1
                arrival[following] = step
                onset[following] = (
                    min(NEVER, step)
                    if not node[1] and following[1]
                    else onset[node]
                )
                queue.append(following)

    risk = {}
    for node, steps in arrival.items():
        fields = profiles[onset[node]]
        horizon = max(1, steps)
        if horizon <= HORIZON:
            value = fields[horizon - 1].get(node[0], background)
        else:
            retained = math.exp(-(horizon - HORIZON) / 5.0)
            value = (
                retained * fields[-1].get(node[0], background)
                + (1.0 - retained) * background
            )
        risk[node] = value
    return risk


def timed_route_values(m, edges, target, profiles, background):
    targets = [node for node in edges if node[0] == target]
    tail = paths(
        edges, targets, m, {node: background for node in edges}, True
    )

    @lru_cache(maxsize=None)
    def value(t, node, onset):
        if node[0] == target:
            return 0.0
        if node not in tail:
            return float("inf")
        if t >= HORIZON:
            return tail[node]

        best = float("inf")
        for following in [node] + edges.get(node, []):
            next_onset = (
                t + 1 if not node[1] and following[1] else onset
            )
            future = value(t + 1, following, next_onset)
            cost = 1.0 + future
            cost += 4.0 * profiles[next_onset][t].get(
                following[0], background
            )
            cost += .02 * min(15, m["visits"].get(following[0], 0))
            best = min(best, cost)
        return best

    return value, tail, targets


def planner(memory, local_obs):
    m = memory
    start = m["pos"]
    source = (start, m["door_open"])
    initial_onset = 0 if m["door_open"] else NEVER
    predictive = local_obs.get("predictive_planning", True)
    nearby = (
        any(distance(start, p) <= 3 for p in m["enemies"])
        or any(
            distance(start, p) <= 2 and probability > .006
            for p, probability in m["prediction"].items()
        )
    )
    if predictive and nearby:
        laws = component_laws(m)
        weights = list(m["weights"])
    else:
        laws = [m["law"] if predictive else UNIFORM]
        weights = [1.0]

    kernel_cache = {}
    root_cache = {}
    prepared_cache = {}

    def kernels(opened):
        if opened not in kernel_cache:
            kernel_cache[opened] = [
                make_transition(m, law, opened) for law in laws
            ]
        return kernel_cache[opened]

    def roots(opened):
        if opened not in root_cache:
            if (predictive and nearby
                    and m["forecast_opened"] == opened):
                root_cache[opened] = [
                    (tracks, fog, kernel)
                    for (_, tracks, fog), kernel
                    in zip(m["forecast_components"], kernels(opened))
                ]
            else:
                root_cache[opened] = [
                    (
                        [dict(kernel(p)) for p in sorted(m["enemies"])],
                        propagate(m["fog"], kernel),
                        kernel,
                    )
                    for kernel in kernels(opened)
                ]
        return root_cache[opened]

    def prepared(first_open, second_open):
        key = (first_open, second_open)
        if key not in prepared_cache:
            prepared_cache[key] = [
                prepare_second(expert, kernel)
                for expert, kernel
                in zip(roots(first_open), kernels(second_open))
            ]
        return prepared_cache[key]

    edges = graph(m)
    profiles, background = route_forecasts(
        m, roots, prepared, kernels, weights
    )
    route_risk = arrival_route_risk(edges, source, profiles, background)
    dist = paths(edges, [source], m, route_risk)
    target = choose_target(m, edges, dist, route_risk)
    route_value, tail, targets = timed_route_values(
        m, edges, target, profiles, background
    )
    geometry = breadth_distances(reverse_graph(edges), targets)

    objective = (target, m["keys"], m["door_open"])
    progress = geometry.get(source, 225)
    if objective == m["last_objective"]:
        if progress < m["last_distance"]:
            m["stall"] = max(0, m["stall"] - 3)
        else:
            m["stall"] += 1
    else:
        m["stall"] = 0
    m["last_objective"] = objective
    m["last_distance"] = progress
    m["target"] = target

    remaining = max(1, 200 - m["step"])
    route = route_value(0, source, initial_onset)
    if not math.isfinite(route):
        route = 25.0
    slack = remaining - route
    pressure = min(72.0, max(15.0, 1.5 * slack))
    pressure /= 1.0 + .055 * max(0, m["stall"] - 5)
    pressure = max(12.0, pressure)

    can_open = (
        not m["door_open"]
        and interaction_state(m, start, False, m["keys"], True)
    )
    interactions = (True, False) if can_open else (True,)
    choices = []
    emergency = []

    for interact in interactions:
        first_open = interaction_state(
            m, start, m["door_open"], m["keys"], interact
        )
        onset1 = 1 if not m["door_open"] and first_open else initial_onset
        experts = roots(first_open)

        for d in MOVES:
            q = add(start, d)
            node1 = (q, first_open)
            if not legal(m, start, q, first_open) or q in m["enemies"]:
                continue

            first_probabilities = [
                occupancy(tracks, fog, q) for tracks, fog, _ in experts
            ]
            first_hazard = mixture_hazard(first_probabilities, weights)
            first_value = route_value(1, node1, onset1)
            emergency.append((
                first_hazard, tail.get(node1, 1e6),
                not interact, d, first_open,
            ))
            if not math.isfinite(first_value):
                continue

            keys1 = min(2, m["keys"] + int(m["map"].get(q) == 2))
            terminal = (
                m["map"].get(q) == 4 and keys1 >= 2 and first_open
            )
            # Collecting a goal or reaching a frontier causes a new
            # objective next turn; stop the old route at that boundary.
            achieved = q == target
            continuation = 0.0

            if not terminal and not achieved:
                survivor_weights = [
                    weight * max(0.0, 1.0 - probability)
                    for weight, probability
                    in zip(weights, first_probabilities)
                ]
                continuation = float("inf")
                next_states = [first_open]
                if (not first_open
                        and interaction_state(m, q, False, keys1, True)):
                    next_states.append(True)

                if sum(survivor_weights) > 1e-12:
                    for second_open in next_states:
                        onset2 = (
                            2 if not first_open and second_open else onset1
                        )
                        states = prepared(first_open, second_open)
                        for e in MOVES:
                            r = add(q, e)
                            node2 = (r, second_open)
                            if not legal(m, q, r, second_open):
                                continue
                            if node2 not in tail:
                                continue
                            second_probabilities = [
                                second_probability(state, q, r)
                                for state in states
                            ]
                            h2 = mixture_hazard(
                                second_probabilities, survivor_weights
                            )
                            future = route_value(2, node2, onset2)
                            terminal2 = (
                                m["map"].get(r) == 4
                                and keys1 >= 2 and second_open
                            )
                            value = 1.0 + .85 * pressure * h2
                            if not terminal2:
                                value += future
                            value += .025 * min(16, m["visits"].get(r, 0))
                            if r == q:
                                value += .15
                            continuation = min(continuation, value)

                if not math.isfinite(continuation):
                    continuation = first_value + 10.0

            score = 1.0 + pressure * first_hazard
            if not terminal:
                score += .60 * first_value + .40 * continuation
            score += .04 * min(20, m["visits"].get(q, 0))
            if q == start:
                score += .20 + .025 * min(12, m["stall"])
            choices.append((
                score, first_hazard, not interact, d, first_open,
            ))

    if choices:
        _, _, no_interact, move, opened = min(choices)
        interact = not no_interact
    elif emergency:
        _, _, no_interact, move, opened = min(emergency)
        interact = not no_interact
    else:
        move = (0, 0)
        interact = True
        opened = interaction_state(
            m, start, m["door_open"], m["keys"], interact
        )

    action = {"move": list(move), "interact": bool(interact)}
    m["chosen_action"] = action
    # Evaluation occurs after action choice. Commit the forecast and
    # next-step density for exactly the interaction that was selected.
    set_prediction(m, opened)
    return action


def export_model(memory, local_obs):
    m = memory
    return {
        "enemy": [
            [p[0], p[1], probability]
            for p, probability in sorted(m["prediction"].items())
        ],
        "default_enemy": m["default"],
        "terrain": [
            [p[0], p[1], cell, m["seen"][p]]
            for p, cell in sorted(m["map"].items())
        ],
        "position": list(m["pos"]),
        "navigation": {
            "bottleneck_routes": m.get("bottleneck_routes", 0),
            "observing_bottleneck": (
                m.get("bottleneck_observation") is not None
            ),
        },
        "learning": {
            "updates": m["updates"],
            "attempt_probabilities": list(m["law"]),
            "mixture_weights": list(m["weights"]),
            "effective_evidence": [sum(c) for c in m["counts"]],
            "component_attempt_probabilities": [
                list(law) for law, _, _ in m["forecast_components"]
            ],
            "surprise_evidence": list(m["surprise"]),
            "fast_recoveries": m["recoveries"],
            "opening_transition_updates": m["opening_updates"],
            "forecast_door_open": bool(m["forecast_opened"]),
            "route_forecast_horizons": HORIZON,
            "near_expert_disagreement": max(
                (
                    math.sqrt(max(0.0, sum(
                        weight * (
                            occupancy(tracks, fog, p) - m["prediction"][p]
                        ) ** 2
                        for weight, (_, tracks, fog) in zip(
                            m["weights"], m["forecast_components"]
                        )
                    )))
                    for p in m["local"]
                    if distance(p, m["pos"]) <= 1
                    and not blocked(m["map"], p, m["forecast_opened"])
                ),
                default=0.0,
            ),
        },
    }
# EVOLVE-BLOCK-END