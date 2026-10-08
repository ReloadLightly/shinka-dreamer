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
def component_laws(m):
    laws = [UNIFORM]
    for counts in m["counts"]:
        total = sum(counts) + 6.3
        laws.append([(c + .7) / total for c in counts])
    return laws
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
    old_map, sources, old_center, step, opened = previous
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
    options = []
    for source in sources:
        # Do not interpret unknown geometry as an observed movement attempt.
        if any(add(source, d) not in terrain for d in MOVES):
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
            # A completely censored source supplies no directional evidence.
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
def update_dynamics(m, obs):
    if not obs.get("learn", True):
        return
    # Slow evidence persists through short gaps; the fast learner recovers
    # quickly after changes without requiring a declared switch event.
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
    results = [evidence_and_counts(law, terms)
               for law in component_laws(m)]
    contamination = .004 * (1.0 / 9.0) ** len(terms[0])
    # Score experts before incorporating this observation. The alternative
    # is a fixed mixture, avoiding a hindsight choice of the best expert.
    likelihoods = [.996 * result[0] + contamination for result in results]
    informative = max(1.0, sum(results[0][1]))
    alternative = .5 * likelihoods[0] + .5 * likelihoods[2]
    surprise = math.log(
        max(1e-30, alternative) / max(1e-30, likelihoods[1])
    ) / informative
    surprise = max(-1.25, min(1.25, surprise))
    m["recovery_score"] = max(
        0.0, .94 * m["recovery_score"] + surprise - .04
    )
    if surprise > .08:
        m["recovery_support"] = min(8, m["recovery_support"] + 1)
    else:
        m["recovery_support"] = max(0, m["recovery_support"] - 1)
    m["recent_terms"].append(terms)
    m["recent_terms"] = m["recent_terms"][-4:]
    weights = [
        w * (.996 * result[0] + contamination) ** .8
        for w, result in zip(m["weights"], results)
    ]
    total = sum(weights)
    if total > 1e-25:
        # A probability floor permits recovery from a previously poor model.
        m["weights"] = [.94 * w / total + .02 for w in weights]
    for counts, result, cap in zip(
            m["counts"], results[1:], (80.0, 16.0)):
        if result[0] <= 1e-15:
            continue
        for k in range(9):
            counts[k] += result[1][k]
        total = sum(counts)
        if total > cap:
            for k in range(9):
                counts[k] *= cap / total
    m["updates"] += 1
    if (m["recovery_score"] > 1.2
            and m["recovery_support"] >= 3
            and sum(m["counts"][0]) >= 8.0
            and m["updates"] - m["last_recovery"] >= 8):
        # Reinterpret recent anonymous transitions from a fresh prior.
        # Each transition contributes once; the slow expert is retained.
        fresh = [0.0] * 9
        for recent in m["recent_terms"]:
            fresh = [.88 * value for value in fresh]
            denominator = sum(fresh) + 6.3
            fresh_law = [(value + .7) / denominator for value in fresh]
            evidence, expected = evidence_and_counts(fresh_law, recent)
            if evidence > 1e-15:
                fresh = [value + increment
                         for value, increment in zip(fresh, expected)]
        total = sum(fresh)
        if total > 16.0:
            fresh = [value * 16.0 / total for value in fresh]
        m["counts"][1] = fresh
        # Give recovery some control influence while preserving every floor.
        transfer = .35 * max(0.0, m["weights"][1] - .02)
        m["weights"][1] -= transfer
        m["weights"][2] += transfer
        m["recoveries"] += 1
        m["last_recovery"] = m["updates"]
        m["recovery_score"] = 0.0
        m["recovery_support"] = 0
def make_transition(m, law):
    terrain = m["map"]
    opened = opening_now(m)
    cache = {}
    def transition(p):
        if p not in cache:
            result = {}
            for d, probability in zip(MOVES, law):
                q = destination(terrain, p, d, opened)
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
            "recovery_score": 0.0, "recovery_support": 0,
            "recent_terms": [], "last_recovery": -100,
            "recoveries": 0,
            "previous": None, "target": None, "density": {},
            "fog": {}, "stall": 0, "last_distance": None,
            "last_objective": None,
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
    update_dynamics(m, local_obs)
    m["law"] = learned_law(m)
    free = [p for p in m["map"]
            if not blocked(m["map"], p, m["door_open"])]
    outside = [p for p in free if p not in local]
    known = sum(c != -1 for c in m["map"].values())
    reservoir = .68 * max(0, 225 - known)
    base = 3.0 / max(60.0, len(free) + reservoir)
    fog = {
        p: .94 * m["density"].get(p, base) + .06 * base
        for p in outside
    }
    missing = max(0, 3 - len(enemies))
    denominator = sum(fog.values()) + reservoir * base
    if denominator > 1e-12:
        factor = missing / denominator
        fog = {p: max(0.0, value * factor)
               for p, value in fog.items()}
    else:
        fog = {}
    m["fog"] = fog
    tracks, next_fog, _ = first_forecast(m, m["law"])
    cells = set(m["map"]) | set(next_fog)
    for track in tracks:
        cells.update(track)
    opened = opening_now(m)
    m["prediction"] = {
        p: 0.0 if blocked(m["map"], p, opened)
        else occupancy(tracks, next_fog, p)
        for p in cells
    }
    m["default"] = min(.06, missing / max(80.0, len(free) + reservoir))
    density = dict(next_fog)
    for track in tracks:
        for p, value in track.items():
            density[p] = density.get(p, 0.0) + value
    m["density"] = density
    m["previous"] = (
        dict(m["map"]), tuple(sorted(enemies)), m["pos"],
        m["step"], opened,
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
def paths(edges, source, m, risk, reverse=False):
    adjacency = edges
    if reverse:
        adjacency = {p: [] for p in edges}
        for p, neighbors in edges.items():
            for q in neighbors:
                adjacency.setdefault(q, []).append(p)
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
def conditioned_future(tracks, fog, q, transition):
    # First-step survival means each enemy is absent from q.
    # Conditioning each independent track preserves this information.
    before, after = [], []
    for track in tracks:
        remaining = 1.0 - track.get(q, 0.0)
        if remaining <= 1e-10:
            return None
        filtered = {
            p: mass / remaining for p, mass in track.items()
            if p != q and mass > 1e-10
        }
        before.append(filtered)
        after.append(propagate(filtered, transition))
    fog_before = {p: mass for p, mass in fog.items() if p != q}
    fog_after = propagate(fog_before, transition)
    return before, after, fog_before, fog_after
def second_hazard(state, q, r, transition):
    before, after, fog_before, fog_after = state
    self_probability = transition(r).get(r, 0.0)
    total = 0.0
    for old, new in zip(before, after):
        danger = new.get(r, 0.0)
        if r != q:
            # Union of collision before and after the enemy move.
            danger += old.get(r, 0.0) * (1.0 - self_probability)
        total += hazard(danger)
    intensity = fog_after.get(r, 0.0)
    if r != q:
        intensity += fog_before.get(r, 0.0) * (1.0 - self_probability)
    return total + max(0.0, intensity)
def planner(memory, local_obs):
    m = memory
    start = m["pos"]
    if local_obs.get("predictive_planning", True):
        law = [.93 * p + .07 / 9.0 for p in m["law"]]
    else:
        law = list(UNIFORM)
    tracks, fog, transition = first_forecast(m, law)
    route_risk = {
        p: occupancy(tracks, fog, p) for p in m["map"]
    }
    edges = graph(m)
    dist = paths(edges, start, m, route_risk)
    target = choose_target(m, edges, dist, route_risk)
    potential = paths(edges, target, m, route_risk, True)
    # Measure geometric progress separately from fluctuating risk estimates.
    objective = (target, m["keys"], m["door_open"])
    progress = distance(start, target)
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
    choices = []
    for d in MOVES:
        q = add(start, d)
        if not legal(m, start, q, True) or q in m["enemies"]:
            continue
        if q not in potential:
            continue
        first_hazard = max(0.0, fog.get(q, 0.0))
        for track in tracks:
            first_hazard += hazard(track.get(q, 0.0))
        if q == target and q != start:
            continuation = 0.0
        else:
            state = conditioned_future(tracks, fog, q, transition)
            continuation = float("inf")
            if state is not None:
                for e in MOVES:
                    r = add(q, e)
                    if r not in potential or not legal(m, q, r):
                        continue
                    h2 = second_hazard(state, q, r, transition)
                    value = 1.0 + potential[r] + .85 * pressure * h2
                    value += .025 * min(16, m["visits"].get(r, 0))
                    if r == q:
                        value += .15
                    continuation = min(continuation, value)
            if not math.isfinite(continuation):
                continuation = potential[q] + 10.0
        score = 1.0 + pressure * first_hazard
        if q != target or q == start:
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
                    occupancy(tracks, fog, q),
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
            "fast_recoveries": m["recoveries"],
            "recovery_score": m["recovery_score"],
            "recovery_support": m["recovery_support"],
        },
    }
# EVOLVE-BLOCK-END