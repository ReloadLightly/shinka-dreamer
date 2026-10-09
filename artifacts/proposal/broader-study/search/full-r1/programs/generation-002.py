"""Namazu's original map-memory and greedy/random-walk seed.

The evaluator owns the world and seeds Python's random module independently.
"""

# Repairs from docs/namazu-proposal.md:
# - Apply observed actual displacement before mapping, keeping initial-relative
#   coordinates even when an attempted move is blocked.
# - Read observable timestep, key inventory, collection feedback and door state.
# - Store anonymous occupied-cell sightings, never invented enemy identities.
# - Permit interaction on the current target tile, so a diagonally reached key
#   can be collected on the following turn rather than becoming a deadlock.
# The live-world evaluation wrapper is deliberately outside candidate code.

# EVOLVE-BLOCK-START
import heapq
import math

_MOVES = tuple(
    (dx, dy)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
)
_CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _inside(memory, p):
    if p in memory["outside"]:
        return False
    xmin, xmax, ymin, ymax = memory["possible_bounds"]
    return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax


def _enemy_kernel(memory, positions):
    """Anonymous random-walk transitions, including blocked attempts."""
    terrain = memory["terrain"]
    doors_passable = memory["door_open"] or memory["keys"] >= 2
    result = {}

    def free(p):
        if not _inside(memory, p):
            return False
        cell = terrain.get(p, 0)
        return cell != 1 and (cell != 3 or doors_passable)

    for p in positions:
        x, y = p
        counts = {}
        for dx, dy in _MOVES:
            q = (x + dx, y + dy)
            if (
                not free(q)
                or (
                    dx and dy
                    and (
                        not free((x + dx, y))
                        or not free((x, y + dy))
                    )
                )
            ):
                q = p
            counts[q] = counts.get(q, 0) + 1
        result[p] = tuple((q, n / 9.0) for q, n in counts.items())
    return result


def _advance(memory, mass, kernel=None):
    if kernel is None:
        kernel = _enemy_kernel(memory, mass)
    following = {}
    for p, amount in mass.items():
        for q, probability in kernel[p]:
            following[q] = following.get(q, 0.0) + amount * probability
    return {
        p: amount
        for p, amount in following.items()
        if amount >= 0.00015 and _inside(memory, p)
    }


def world_model_step(memory: dict, local_obs: dict, last_action: dict) -> dict:
    if not memory or "terrain" not in memory:
        memory = {
            "agent_pos": (0, 0),
            "terrain": {},
            "seen": {},
            "outside": set(),
            "visits": {},
            "collected": set(),
            "enemy_mass": {},
            "possible_bounds": (-14, 14, -14, 14),
            "step": local_obs.get("step", 0),
            "keys": 0,
            "door_open": False,
            "believed_map": {},
        }

    previous_step = memory["step"]
    feedback = local_obs.get("feedback", {})
    dx, dy = feedback.get("displacement", (0, 0))
    ax, ay = memory["agent_pos"]
    position = (ax + dx, ay + dy)
    ax, ay = position
    step = int(local_obs.get("step", previous_step))

    memory["agent_pos"] = position
    memory["step"] = step
    memory["keys"] = int(local_obs.get("keys", 0))
    memory["door_open"] = bool(local_obs.get("door_open", False))
    memory["visits"][position] = memory["visits"].get(position, 0) + 1

    if feedback.get("collected", False):
        memory["collected"].add(position)

    terrain = memory["terrain"]
    seen = memory["seen"]
    grid = local_obs["grid"]
    base_grid = local_obs.get("terrain")
    usable_base = (
        isinstance(base_grid, (list, tuple))
        and len(base_grid) == 5
        and all(
            isinstance(row, (list, tuple)) and len(row) == 5
            for row in base_grid
        )
    )

    visible = {}
    occupied = set()
    for oy in range(-2, 3):
        for ox in range(-2, 3):
            p = (ax + ox, ay + oy)
            cell = grid[oy + 2][ox + 2]
            if cell == -1:
                memory["outside"].add(p)
                continue

            memory["outside"].discard(p)
            visible[p] = cell
            if cell == 5:
                occupied.add(p)

            base = base_grid[oy + 2][ox + 2] if usable_base else cell
            if base in (0, 1, 2, 3, 4):
                terrain[p] = base
                seen[p] = step
            elif cell == 5:
                # Occupancy establishes traversability, but not a landmark.
                if terrain.get(p) not in (0, 2, 3, 4):
                    terrain[p] = 0
                seen[p] = step
            else:
                terrain.setdefault(p, 0)
                seen[p] = step

    for p in memory["collected"]:
        if p not in visible or visible[p] != 2:
            terrain[p] = 0
    if memory["door_open"]:
        for p in tuple(terrain):
            if terrain[p] == 3:
                terrain[p] = 0

    # Any two in-bounds coordinates differ by at most fourteen.
    if terrain:
        xs = [p[0] for p in terrain]
        ys = [p[1] for p in terrain]
        memory["possible_bounds"] = (
            max(xs) - 14, min(xs) + 14,
            max(ys) - 14, min(ys) + 14,
        )

    mass = memory["enemy_mass"]
    elapsed = max(0, step - previous_step)
    if elapsed > 6:
        mass = {}
    else:
        for _ in range(elapsed):
            mass = _advance(memory, mass)

    # Negative observations remove occupancy hypotheses immediately.
    hidden = {
        p: amount
        for p, amount in mass.items()
        if p not in visible
        and _inside(memory, p)
        and terrain.get(p) != 1
    }
    total = sum(hidden.values())
    capacity = max(0, 3 - len(occupied))
    if total > capacity:
        scale = capacity / total
        hidden = {p: amount * scale for p, amount in hidden.items()}
    hidden.update({p: 1.0 for p in occupied})

    memory["enemy_mass"] = hidden
    memory["occupied"] = occupied
    memory["visible"] = set(visible)

    # Reconstruction is separate from navigation and future risk forecasts.
    memory["believed_map"] = visible
    return memory


def _graph(memory, speculative):
    terrain = memory["terrain"]
    phase = memory["step"] // 25
    keys = memory["keys"]
    opened = memory["door_open"]

    def stale(p):
        return memory["seen"].get(p, -25) // 25 < phase

    def allowed(p):
        if p not in terrain or not _inside(memory, p):
            return False
        cell = terrain[p]
        if cell == 1:
            return speculative and stale(p)
        return cell != 3 or opened or keys >= 2

    nodes = {p for p in terrain if allowed(p)}
    nodes.add(memory["agent_pos"])
    graph = {}

    for p in nodes:
        x, y = p
        edges = []
        for dx, dy in _MOVES:
            if not (dx or dy):
                continue
            q = (x + dx, y + dy)
            if q not in nodes:
                continue
            if dx and dy:
                side1, side2 = (x + dx, y), (x, y + dy)
                if not allowed(side1) or not allowed(side2):
                    continue
                # A closed door cannot be opened from a diagonal position.
                if terrain.get(q) == 3 and not opened:
                    continue

            cost = 1.0
            if speculative:
                if stale(q):
                    cost += 3.5 if terrain.get(q) == 1 else 0.30
                if dx and dy:
                    for side in ((x + dx, y), (x, y + dy)):
                        if terrain.get(side) == 1:
                            cost += 1.5
            edges.append((q, cost, (dx, dy)))
        graph[p] = edges

    return graph


def _distances(graph, source):
    distance = {source: 0.0}
    queue = [(0.0, source)]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distance.get(p):
            continue
        for q, weight, _ in graph.get(p, ()):
            candidate = cost + weight
            if candidate < distance.get(q, float("inf")):
                distance[q] = candidate
                heapq.heappush(queue, (candidate, q))
    return distance


def _reverse_distances(graph, goal):
    reverse = {p: [] for p in graph}
    for p, edges in graph.items():
        for q, cost, _ in edges:
            reverse.setdefault(q, []).append((p, cost, (0, 0)))
    return _distances(reverse, goal)


def _information(memory, p):
    x, y = p
    terrain = memory["terrain"]
    phase = memory["step"] // 25
    unknown = 0
    refresh = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            q = (x + dx, y + dy)
            if not _inside(memory, q):
                continue
            if q not in terrain:
                unknown += 1
            elif memory["seen"].get(q, -25) // 25 < phase:
                refresh += 1
    return unknown, refresh


def _choose_goal(memory, graph, distance):
    start = memory["agent_pos"]
    terrain = memory["terrain"]

    exits = [
        p for p in distance
        if terrain.get(p) == 4
    ]
    doors = [
        p for p, cell in terrain.items()
        if cell == 3
    ]

    if memory["keys"] >= 2:
        if exits:
            return min(exits, key=distance.get)
        reachable_doors = [p for p in doors if p in distance]
        if reachable_doors:
            return min(reachable_doors, key=distance.get)

    keys = [
        p for p in distance
        if terrain.get(p) == 2 and p not in memory["collected"]
    ]
    if memory["keys"] < 2 and keys:
        anchors = list(exits)
        if not anchors:
            for door in doors:
                for dx, dy in _CARDINAL:
                    p = (door[0] + dx, door[1] + dy)
                    if p in distance:
                        anchors.append(p)

        from_key = {p: _distances(graph, p) for p in keys}

        def tail(p):
            available = [
                from_key[p][a] for a in anchors if a in from_key[p]
            ]
            return min(available) if available else 0.0

        def itinerary(first):
            if memory["keys"] == 0 and len(keys) >= 2:
                continuations = [
                    from_key[first][second] + 0.55 * tail(second)
                    for second in keys
                    if second != first and second in from_key[first]
                ]
                if continuations:
                    return distance[first] + min(continuations)
            return distance[first] + 0.20 * tail(first)

        return min(keys, key=itinerary)

    previous_goal = memory.get("goal")
    candidates = []
    for p, travel in distance.items():
        if p == start:
            continue
        unknown, refresh = _information(memory, p)
        gain = unknown + 0.30 * refresh
        if gain <= 0:
            continue
        visits = memory["visits"].get(p, 0)
        score = gain / (2.5 + travel + 0.8 * visits)
        if p == previous_goal:
            score *= 1.12
        candidates.append((score, -travel, p))

    if candidates:
        return max(candidates)[2]

    alternatives = [p for p in distance if p != start]
    return min(
        alternatives,
        key=lambda p: (
            memory["visits"].get(p, 0)
            + 0.12 * distance[p]
            + 0.025 * memory["seen"].get(p, 0)
        ),
        default=start,
    )


def _risk_forecast(memory, horizon):
    """Upper bounds on contact before or after each enemy movement."""
    mass = dict(memory["enemy_mass"])
    risks = []
    for _ in range(horizon):
        kernel = _enemy_kernel(memory, mass)
        following = _advance(memory, mass, kernel)
        risk = {}
        for p in set(mass) | set(following):
            stay_probability = 0.0
            if p in kernel:
                for q, probability in kernel[p]:
                    if q == p:
                        stay_probability = probability
                        break
            # Subtract the overlap between pre- and post-move contact.
            contact = (
                mass.get(p, 0.0)
                + following.get(p, 0.0)
                - mass.get(p, 0.0) * stay_probability
            )
            risk[p] = min(1.0, max(0.0, contact))
        risks.append(risk)
        mass = following
    return risks


def planner(memory: dict, local_obs: dict) -> dict:
    start = memory["agent_pos"]
    graph = _graph(memory, speculative=True)
    distance = _distances(graph, start)
    goal = _choose_goal(memory, graph, distance)
    memory["goal"] = goal

    potential = _reverse_distances(graph, goal)
    legal = _graph(memory, speculative=False)

    horizon = min(5, max(1, 200 - memory["step"]))
    risks = _risk_forecast(memory, horizon)

    # Restrict the time-expanded controller to its reachable local region.
    layers = [{start}]
    for _ in range(horizon):
        reachable = set(layers[-1])
        for p in layers[-1]:
            reachable.update(q for q, _, _ in legal.get(p, ()))
        layers.append(reachable)

    default_distance = potential.get(start, 40.0) + 12.0

    def remaining(p):
        return potential.get(p, default_distance)

    values = {p: remaining(p) for p in layers[horizon]}
    chosen = (0, 0)
    occupied = memory["occupied"]
    visits = memory["visits"]

    # Survival has more value when there is ample time for a safer route.
    turns_left = max(1, 200 - memory["step"])
    risk_price = 110.0 if turns_left > remaining(start) + 15 else 65.0

    for t in range(horizon - 1, -1, -1):
        current_values = {}
        for p in layers[t]:
            options = [(p, 1.0, (0, 0))]
            options.extend(legal.get(p, ()))
            best_value = float("inf")
            best_move = (0, 0)

            for q, _, move in options:
                if q not in values:
                    continue
                if t == 0 and q in occupied:
                    continue

                probability = risks[t].get(q, 0.0)
                if probability >= 0.999999:
                    risk_cost = 10000.0
                else:
                    risk_cost = -risk_price * math.log1p(-probability)

                value = (
                    1.0
                    + risk_cost
                    + 0.10 * remaining(q)
                    + 0.018 * min(visits.get(q, 0), 15)
                    + values[q]
                )
                if move == (0, 0) and p != goal:
                    value += 0.035

                if value < best_value:
                    best_value = value
                    best_move = move

            current_values[p] = best_value
            if t == 0 and p == start:
                chosen = best_move
        values = current_values

    return {"move": [chosen[0], chosen[1]], "interact": True}
# EVOLVE-BLOCK-END