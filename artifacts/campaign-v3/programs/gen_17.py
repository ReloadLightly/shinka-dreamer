"""Predictive seed. Entire representation, learner and planner may evolve."""
# EVOLVE-BLOCK-START
import heapq
import math

MOVES = tuple((x, y) for y in (-1, 0, 1) for x in (-1, 0, 1))
WALK = tuple(d for d in MOVES if d != (0, 0))
CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
UNIFORM = [1.0 / 9.0] * 9


def add(p, d):
    return p[0] + d[0], p[1] + d[1]


def distance(p, q):
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def blocked(terrain, p, opened=False, unknown=False):
    c = terrain.get(p)
    if c is None:
        return unknown
    return c in (-1, 1) or (c == 3 and not opened)


def destination(terrain, p, d, opened):
    q = add(p, d)
    if blocked(terrain, q, opened):
        return p
    if d[0] and d[1]:
        if (blocked(terrain, add(p, (d[0], 0)), opened) or
                blocked(terrain, add(p, (0, d[1])), opened)):
            return p
    return q


def opening_now(m):
    return m["door_open"] or (
        m["keys"] >= 2 and any(
            m["map"].get(add(m["pos"], d)) == 3 for d in CARDINAL
        )
    )


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
    old_map, sources, old_center, step, opened, old_seen = previous
    if m["step"] != step + 1 or not sources:
        return None
    if m["step"] % 25 == 0 or step % 25 == 0:
        return None

    terrain = dict(old_map)
    for p, c in m["local"].items():
        old = old_map.get(p)
        if old is not None and old != c and (
                old in (1, 3) or c in (1, 3)):
            return None
        terrain[p] = c

    epoch = 25 * (step // 25)
    options = []
    for source in sources:
        # Only geometry observed during this wall epoch is reliable.
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
        expected = [v / evidence for v in expected]
    return evidence, expected


def cap_counts(counts, cap):
    total = sum(counts)
    if total > cap:
        scale = cap / total
        for k in range(9):
            counts[k] *= scale


def rebuild_fast(m):
    # Replay only recent observed constraints, starting from a fresh prior.
    # The persistent expert is left intact.
    counts = [0.0] * 9
    for _, terms in m["recent_terms"]:
        law = count_law(counts)
        evidence, expected = evidence_and_counts(law, terms)
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

    laws = component_laws(m)
    results = [evidence_and_counts(law, terms) for law in laws]
    if results[0][0] >= 1.0 - 1e-10:
        return

    count = max(1, len(terms[0]))
    contamination = .004 * (1.0 / 9.0) ** count
    likelihood = [
        max(1e-30, .996 * result[0] + contamination)
        for result in results
    ]

    # Compare pre-update predictions, normalized by visible source count.
    # Two one-sided evidence accumulators avoid rewarding whichever
    # alternative happened to win a single observation.
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


def make_transition(m, law):
    terrain = m["map"]
    opened = opening_now(m)
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


def first_forecast(m, law):
    transition = make_transition(m, law)
    tracks = [dict(transition(p)) for p in sorted(m["enemies"])]
    fog = propagate(m["fog"], transition)
    return tracks, fog, transition


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
            "recoveries": 0,
        }
    m = memory
    m["pos"] = add(m["pos"], local_obs["feedback"]["displacement"])
    m["keys"] = int(local_obs["keys"])
    m["door_open"] = bool(local_obs["door_open"])
    m["step"] = int(local_obs["step"])
    m["visits"][m["pos"]] = m["visits"].get(m["pos"], 0) + 1

    local, enemies = {}, set()
    for y, row in enumerate(local_obs["terrain"]):
        for x, c in enumerate(row):
            p = add(m["pos"], (x - 2, y - 2))
            local[p] = int(c)
            m["map"][p], m["seen"][p] = int(c), m["step"]
            if local_obs["grid"][y][x] == 5:
                enemies.add(p)
    m["local"], m["enemies"] = local, enemies

    # Mapping and localization always run, independently of this intervention.
    update_dynamics(m, local_obs)
    m["law"] = learned_law(m)
    m["fog"] = reconcile_fog(m)

    forecasts = []
    cells = set(m["map"])
    density = {}
    for weight, law in zip(m["weights"], component_laws(m)):
        tracks, next_fog, _ = first_forecast(m, law)
        forecasts.append((list(law), tracks, next_fog))
        cells.update(next_fog)
        for p, value in next_fog.items():
            density[p] = density.get(p, 0.0) + weight * value
        for track in tracks:
            cells.update(track)
            for p, value in track.items():
                density[p] = density.get(p, 0.0) + weight * value

    m["forecast_components"] = forecasts
    opened = opening_now(m)
    m["prediction"] = {
        p: 0.0 if blocked(m["map"], p, opened)
        else min(1.0, max(0.0, sum(
            weight * occupancy(tracks, fog, p)
            for weight, (_, tracks, fog)
            in zip(m["weights"], forecasts)
        )))
        for p in cells
    }
    m["default"] = 0.0
    m["density"] = density
    m["previous"] = (
        dict(m["map"]), tuple(sorted(enemies)), m["pos"],
        m["step"], opened, dict(m["seen"]),
    )
    return m


def legal(m, p, q, immediate=False):
    dx, dy = q[0] - p[0], q[1] - p[1]
    opened = opening_now(m) if immediate else (
        m["door_open"] or m["keys"] >= 2)
    if blocked(m["map"], q, opened, True):
        return False
    if dx and dy:
        for side in ((p[0] + dx, p[1]), (p[0], p[1] + dy)):
            if blocked(m["map"], side, opened, True):
                return False
        if m["map"].get(q) == 3 and not m["door_open"]:
            return False
    return True


def graph(m):
    opened = m["door_open"] or m["keys"] >= 2
    return {
        p: [add(p, d) for d in WALK if legal(m, p, add(p, d))]
        for p in m["map"] if not blocked(m["map"], p, opened, True)
    }


def reverse_graph(edges):
    result = {p: [] for p in edges}
    for p, neighbors in edges.items():
        for q in neighbors:
            result.setdefault(q, []).append(p)
    return result


def paths(edges, source, m, risk, reverse=False):
    adjacency = reverse_graph(edges) if reverse else edges
    dist = {source: 0.0}
    queue = [(0.0, source)]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != dist[p]:
            continue
        for q in adjacency.get(p, ()):
            cell = p if reverse else q
            edge = 1.0 + 4.0 * risk.get(cell, m["default"])
            edge += .02 * min(15, m["visits"].get(cell, 0))
            new = cost + edge
            if new < dist.get(q, float("inf")):
                dist[q] = new
                heapq.heappush(queue, (new, q))
    return dist


def geometric_distances(edges, target):
    reverse = reverse_graph(edges)
    dist = {target: 0}
    queue = [target]
    index = 0
    while index < len(queue):
        p = queue[index]
        index += 1
        for q in reverse.get(p, ()):
            if q not in dist:
                dist[q] = dist[p] + 1
                queue.append(q)
    return dist


def choose_target(m, edges, dist, risk):
    start = m["pos"]
    desired = 2 if m["keys"] < 2 else 4
    goals = [p for p in dist if p != start and m["map"].get(p) == desired]
    if goals:
        if desired == 2 and m["keys"] == 0 and len(goals) > 1:
            doors = [p for p, c in m["map"].items() if c in (3, 4)]
            tours = []
            for first in goals:
                onward = paths(edges, first, m, risk)
                for second in goals:
                    if second == first or second not in onward:
                        continue
                    tail = min((distance(second, p) for p in doors), default=0)
                    tours.append((dist[first] + onward[second] + .6 * tail,
                                  first))
            if tours:
                return min(tours)[1]
        return min(goals, key=lambda p: (dist[p], p))

    if m["keys"] >= 2 and not m["door_open"]:
        doors = [p for p in dist if p != start and m["map"].get(p) == 3]
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
    return -math.log(max(1e-10, 1.0 - min(1.0, max(0.0, probability))))


def mixture_hazard(probabilities, weights):
    total = sum(weights)
    if total <= 1e-15:
        return hazard(1.0)
    weights = [w / total for w in weights]
    mean = sum(w * p for w, p in zip(weights, probabilities))
    cutoff = max(.08, .20 * max(weights))
    plausible = [
        (w, p) for w, p in zip(weights, probabilities)
        if w >= cutoff
    ]
    mass = sum(w for w, _ in plausible)
    center = sum(w * p for w, p in plausible) / mass
    variance = sum(w * (p - center) ** 2 for w, p in plausible) / mass
    upper = max(mean, max(p for _, p in plausible))
    conservative = min(upper, mean + .45 * math.sqrt(variance))
    return hazard(conservative)


def prepare_second(expert):
    tracks, fog, transition = expert
    return (
        tracks,
        [propagate(track, transition) for track in tracks],
        fog,
        propagate(fog, transition),
        transition,
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
        # Propagation is linear: remove paths through the cell excluded
        # by first-step survival before renormalizing the track.
        danger = max(0.0, new.get(r, 0.0) - at_q * qr) / surviving
        if r != q:
            danger += old.get(r, 0.0) * (1.0 - rr) / surviving
        total += hazard(danger)

    intensity = max(
        0.0, fog_after.get(r, 0.0) - fog.get(q, 0.0) * qr
    )
    if r != q:
        intensity += fog.get(r, 0.0) * (1.0 - rr)
    total += max(0.0, intensity)
    return min(1.0, max(0.0, -math.expm1(-total)))


def planner(memory, local_obs):
    m = memory
    start = m["pos"]
    predictive = local_obs.get("predictive_planning", True)
    nearby = (
        any(distance(start, p) <= 3 for p in m["enemies"])
        or any(
            distance(start, p) <= 2 and probability > .006
            for p, probability in m["prediction"].items()
        )
    )
    if predictive and nearby:
        weights = list(m["weights"])
        experts = [
            (tracks, fog, make_transition(m, law))
            for law, tracks, fog in m["forecast_components"]
        ]
    else:
        law = m["law"] if predictive else UNIFORM
        experts = [first_forecast(m, law)]
        weights = [1.0]

    route_risk = {
        p: sum(
            weight * occupancy(tracks, fog, p)
            for weight, (tracks, fog, _) in zip(weights, experts)
        )
        for p in m["map"]
    }
    edges = graph(m)
    dist = paths(edges, start, m, route_risk)
    target = choose_target(m, edges, dist, route_risk)
    potential = paths(edges, target, m, route_risk, True)
    geometry = geometric_distances(edges, target)

    objective = (target, m["keys"], m["door_open"])
    progress = geometry.get(start, 225)
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
    route = potential.get(start, 25.0)
    slack = remaining - route
    pressure = min(72.0, max(15.0, 1.5 * slack))
    pressure /= 1.0 + .055 * max(0, m["stall"] - 5)
    pressure = max(12.0, pressure)

    # Each expert needs only one unconditional second propagation.
    prepared = [prepare_second(expert) for expert in experts]
    choices = []

    for d in MOVES:
        q = add(start, d)
        if not legal(m, start, q, True) or q in m["enemies"]:
            continue
        if q not in potential:
            continue

        first_probabilities = [
            occupancy(tracks, fog, q) for tracks, fog, _ in experts
        ]
        first_hazard = mixture_hazard(first_probabilities, weights)
        terminal = (
            m["map"].get(q) == 4
            and m["keys"] >= 2
            and opening_now(m)
        )

        continuation = 0.0
        if not terminal:
            survivor_weights = [
                weight * max(0.0, 1.0 - probability)
                for weight, probability in zip(weights, first_probabilities)
            ]
            continuation = float("inf")
            if sum(survivor_weights) > 1e-12:
                for e in MOVES:
                    r = add(q, e)
                    if r not in potential or not legal(m, q, r):
                        continue
                    second_probabilities = [
                        second_probability(state, q, r)
                        for state in prepared
                    ]
                    h2 = mixture_hazard(
                        second_probabilities, survivor_weights
                    )
                    value = 1.0 + potential[r] + .85 * pressure * h2
                    value += .025 * min(16, m["visits"].get(r, 0))
                    if r == q:
                        value += .15
                    continuation = min(continuation, value)

            if not math.isfinite(continuation):
                continuation = potential[q] + 10.0

        score = 1.0 + pressure * first_hazard
        if not terminal:
            # Keys, doors and exploration targets are intermediate states.
            # Check whether their surroundings permit surviving another turn.
            score += .60 * potential[q] + .40 * continuation
        score += .04 * min(20, m["visits"].get(q, 0))
        if q == start:
            score += .20 + .025 * min(12, m["stall"])
        choices.append((score, first_hazard, d))

    if choices:
        move = min(choices)[2]
    else:
        emergency = []
        for d in MOVES:
            q = add(start, d)
            if legal(m, start, q, True) and q not in m["enemies"]:
                emergency.append((
                    mixture_hazard(
                        [occupancy(tracks, fog, q)
                         for tracks, fog, _ in experts],
                        weights,
                    ),
                    potential.get(q, 1e6), d,
                ))
        move = min(emergency)[2] if emergency else (0, 0)

    return {"move": list(move), "interact": True}


def export_model(memory, local_obs):
    m = memory
    return {
        "enemy": [[p[0], p[1], probability]
                  for p, probability in sorted(m["prediction"].items())],
        "default_enemy": m["default"],
        "terrain": [[p[0], p[1], cell, m["seen"][p]]
                    for p, cell in sorted(m["map"].items())],
        "position": list(m["pos"]),
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
                    and not blocked(m["map"], p, opening_now(m))
                ),
                default=0.0,
            ),
        },
    }
# EVOLVE-BLOCK-END