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

_MOVES = tuple((dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
_CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
_INF = float("inf")


def _inside(memory, p):
    xmin, xmax, ymin, ymax = memory["possible_bounds"]
    return (
        xmin <= p[0] <= xmax
        and ymin <= p[1] <= ymax
        and p not in memory["outside"]
    )


def _enemy_kernel(memory, positions, cache=None, opened=None):
    """Cache anonymous attempted-move transitions for one fixed map."""
    if cache is None:
        cache = {}
    if opened is None:
        opened = memory["door_open"]
    terrain = memory["terrain"]

    def free(p):
        if not _inside(memory, p):
            return False
        cell = terrain.get(p, 0)
        return cell != 1 and (cell != 3 or opened)

    for p in positions:
        if p in cache:
            continue
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
        cache[p] = tuple((q, count / 9.0) for q, count in counts.items())
    return cache


def _advance(memory, mass, kernel=None):
    if not mass:
        return {}
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
    if not memory or "origin_ranges" not in memory:
        memory = {
            "agent_pos": (0, 0),
            "terrain": {},
            "seen": {},
            "outside": set(),
            "visits": {},
            "collected": set(),
            "enemy_mass": {},
            "origin_ranges": [-14, 0, -14, 0],
            "possible_bounds": (-14, 14, -14, 14),
            "step": int(local_obs.get("step", 0)),
            "keys": 0,
            "door_open": False,
            "believed_map": {},
        }

    previous_step = memory["step"]
    feedback = local_obs.get("feedback", {})
    dx, dy = feedback.get("displacement", (0, 0))
    px, py = memory["agent_pos"]
    ax, ay = px + dx, py + dy
    position = (ax, ay)
    step = int(local_obs.get("step", previous_step))

    memory["agent_pos"] = position
    memory["step"] = step
    memory["keys"] = int(local_obs.get("keys", memory["keys"]))
    memory["door_open"] = bool(
        local_obs.get(
            "door_open",
            memory["door_open"] or feedback.get("opened", False),
        )
    )
    memory["visits"][position] = memory["visits"].get(position, 0) + 1
    if feedback.get("collected", False):
        memory["collected"].add(position)

    # Advance the previous posterior before applying negative observations.
    # Opening precedes enemy movement, so the new door state applies here.
    mass = memory["enemy_mass"]
    elapsed = max(0, step - previous_step)
    if elapsed > 6:
        mass = {}
    else:
        kernel = {}
        for _ in range(elapsed):
            _enemy_kernel(memory, mass, kernel)
            mass = _advance(memory, mass, kernel)

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
    left_lo, left_hi, top_lo, top_hi = memory["origin_ranges"]

    for oy in range(-2, 3):
        for ox in range(-2, 3):
            p = (ax + ox, ay + oy)
            x, y = p
            cell = grid[oy + 2][ox + 2]
            if cell == -1:
                memory["outside"].add(p)
                # The center row/column intersects the maze interior,
                # making these observations unambiguous edge evidence.
                if oy == 0:
                    if ox < 0:
                        left_lo = max(left_lo, x + 1)
                    elif ox > 0:
                        left_hi = min(left_hi, x - 15)
                if ox == 0:
                    if oy < 0:
                        top_lo = max(top_lo, y + 1)
                    elif oy > 0:
                        top_hi = min(top_hi, y - 15)
                continue

            memory["outside"].discard(p)
            visible[p] = cell
            left_lo = max(left_lo, x - 14)
            left_hi = min(left_hi, x)
            top_lo = max(top_lo, y - 14)
            top_hi = min(top_hi, y)

            if cell == 5:
                occupied.add(p)

            base = base_grid[oy + 2][ox + 2] if usable_base else cell
            if base in (0, 1, 2, 3, 4):
                terrain[p] = base
            elif cell == 5:
                if terrain.get(p) not in (0, 2, 3, 4):
                    terrain[p] = 0
            else:
                terrain.setdefault(p, 0)
            seen[p] = step

    memory["origin_ranges"] = [left_lo, left_hi, top_lo, top_hi]
    memory["possible_bounds"] = (
        left_lo, left_hi + 14, top_lo, top_hi + 14
    )

    # Clear obsolete landmarks without overwriting newly observed walls.
    for p in memory["collected"]:
        if terrain.get(p) == 2:
            terrain[p] = 0
    if memory["door_open"]:
        for p in tuple(terrain):
            if terrain[p] == 3:
                terrain[p] = 0

    phase = step // 25
    hidden = {
        p: amount
        for p, amount in mass.items()
        if p not in visible
        and _inside(memory, p)
        and (
            terrain.get(p) != 1
            or seen.get(p, -25) // 25 < phase
        )
    }
    capacity = max(0, 3 - len(occupied))
    total = sum(hidden.values())
    if total > capacity:
        scale = capacity / total
        hidden = {p: amount * scale for p, amount in hidden.items()}
    hidden.update({p: 1.0 for p in occupied})

    memory["enemy_mass"] = hidden
    memory["occupied"] = occupied
    memory["visible"] = set(visible)
    # Only observed current facts are exported to the evaluator.
    memory["believed_map"] = visible
    return memory


def _graph(memory, speculative, inventory=None):
    terrain = memory["terrain"]
    phase = memory["step"] // 25
    keys = memory["keys"] if inventory is None else inventory
    opened = memory["door_open"]
    stale = {
        p for p in terrain
        if memory["seen"].get(p, -25) // 25 < phase
    }
    nodes = {
        p for p, cell in terrain.items()
        if _inside(memory, p)
        and (cell != 1 or (speculative and p in stale))
        and (cell != 3 or opened or keys >= 2)
    }
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
            sides = ()
            if dx and dy:
                sides = ((x + dx, y), (x, y + dy))
                if any(side not in nodes for side in sides):
                    continue
                # Interaction can open a cardinally adjacent side door,
                # but cannot open a diagonally located destination door.
                if terrain.get(q) == 3 and not opened:
                    continue

            cost = 1.0
            if speculative:
                if q in stale:
                    cost += 3.5 if terrain.get(q) == 1 else 0.30
                for side in sides:
                    if terrain.get(side) == 1:
                        cost += 1.5
            edges.append((q, cost, (dx, dy)))
        graph[p] = edges
    return graph


def _distances(graph, sources):
    if isinstance(sources, tuple) and len(sources) == 2:
        sources = (sources,)
    distance = {p: 0.0 for p in sources}
    queue = [(0.0, p) for p in distance]
    heapq.heapify(queue)
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distance.get(p):
            continue
        for q, weight, _ in graph.get(p, ()):
            candidate = cost + weight
            if candidate < distance.get(q, _INF):
                distance[q] = candidate
                heapq.heappush(queue, (candidate, q))
    return distance


def _reverse(graph):
    reverse = {p: [] for p in graph}
    for p, edges in graph.items():
        for q, cost, _ in edges:
            reverse.setdefault(q, []).append((p, cost, (0, 0)))
    return reverse


def _information(memory, p):
    x, y = p
    terrain = memory["terrain"]
    seen = memory["seen"]
    phase = memory["step"] // 25
    unknown = refresh = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            q = (x + dx, y + dy)
            if not _inside(memory, q):
                continue
            if q not in terrain:
                unknown += 1
            elif seen.get(q, -25) // 25 < phase:
                refresh += 1
    return unknown + 0.30 * refresh


def _choose_goal(memory, graph, distance):
    start = memory["agent_pos"]
    terrain = memory["terrain"]
    exits = [p for p, cell in terrain.items() if cell == 4]
    doors = [p for p, cell in terrain.items() if cell == 3]

    if memory["keys"] >= 2:
        for targets in (exits, doors):
            reachable = [p for p in targets if p in distance]
            if reachable:
                return min(reachable, key=lambda p: (distance[p], p))

    keys = sorted(
        p for p in distance
        if terrain.get(p) == 2 and p not in memory["collected"]
    )
    if memory["keys"] < 2 and keys:
        if len(keys) == 1:
            return keys[0]

        # Only the route after the final key can use closed doors.
        unlocked = _graph(memory, True, inventory=2)
        reverse = _reverse(unlocked)
        exit_tail = _distances(reverse, [p for p in exits if p in unlocked])
        door_tail = _distances(reverse, [p for p in doors if p in unlocked])

        def tail(p):
            if p in exit_tail:
                return 0, exit_tail[p]
            if p in door_tail:
                return 1, door_tail[p]
            return 2, 0.0

        if memory["keys"] == 1:
            def final_key_cost(p):
                rank, continuation = tail(p)
                return rank, distance[p] + continuation, p
            return min(keys, key=final_key_cost)

        from_key = {p: _distances(graph, p) for p in keys}

        def itinerary(first):
            options = []
            for second in keys:
                if second == first or second not in from_key[first]:
                    continue
                rank, continuation = tail(second)
                options.append((
                    rank,
                    distance[first] + from_key[first][second] + continuation,
                ))
            return min(options, default=(3, distance[first])), first

        return min(keys, key=itinerary)

    previous_goal = memory.get("goal")
    best = None
    for p, travel in distance.items():
        if p == start:
            continue
        gain = _information(memory, p)
        if gain <= 0:
            continue
        visits = memory["visits"].get(p, 0)
        score = gain / (2.5 + travel + 0.8 * visits)
        if p == previous_goal:
            score *= 1.12
        candidate = (score, -travel, p)
        if best is None or candidate > best:
            best = candidate

    if best is not None:
        return best[2]

    return min(
        (p for p in distance if p != start),
        key=lambda p: (
            memory["visits"].get(p, 0)
            + 0.12 * distance[p]
            + 0.025 * memory["seen"].get(p, 0),
            p,
        ),
        default=start,
    )


def _risk_forecast(memory, horizon):
    mass = dict(memory["enemy_mass"])
    risks = []
    kernel = {}

    # Our always-enabled interaction opens the adjacent gate before
    # this turn's enemy movement. Keys alone never open distant gates.
    opened = memory["door_open"]
    if not opened and memory["keys"] >= 2:
        x, y = memory["agent_pos"]
        opened = any(
            memory["terrain"].get((x + dx, y + dy)) == 3
            for dx, dy in _CARDINAL
        )

    for _ in range(horizon):
        _enemy_kernel(memory, mass, kernel, opened)
        following = _advance(memory, mass, kernel)
        risk = {}
        for p in mass.keys() | following.keys():
            stay = 0.0
            if p in mass:
                stay = next(
                    (probability for q, probability in kernel[p] if q == p),
                    0.0,
                )
            contact = (
                mass.get(p, 0.0) * (1.0 - stay)
                + following.get(p, 0.0)
            )
            risk[p] = min(1.0, max(0.0, contact))
        risks.append(risk)
        mass = following
    return risks


def planner(memory: dict, local_obs: dict) -> dict:
    start = memory["agent_pos"]
    graph = _graph(memory, True)
    distance = _distances(graph, start)
    goal = _choose_goal(memory, graph, distance)
    memory["goal"] = goal

    potential = _distances(_reverse(graph), goal)
    legal = _graph(memory, False)
    visits = memory["visits"]
    occupied = memory["occupied"]
    default_distance = potential.get(start, 40.0) + 12.0
    remaining = {
        p: potential.get(p, default_distance)
        for p in legal
    }

    options = {
        p: [(p, 1.0, (0, 0))] + edges
        for p, edges in legal.items()
    }
    first_options = [
        edge for edge in options.get(start, [(start, 1.0, (0, 0))])
        if edge[0] not in occupied
    ]
    if not first_options:
        first_options = [(start, 1.0, (0, 0))]
    options[start] = options.get(start, [(start, 1.0, (0, 0))])

    # Without tracked occupancy, shortest-path descent avoids the
    # expense of a time-expanded search with identically zero risk.
    if not memory["enemy_mass"]:
        def route_cost(edge):
            q, _, move = edge
            return (
                potential.get(q, default_distance)
                + 0.018 * min(visits.get(q, 0), 15)
                + (0.035 if move == (0, 0) and q != goal else 0.0),
                move,
            )
        chosen = min(first_options, key=route_cost)[2]
        return {"move": [chosen[0], chosen[1]], "interact": True}

    horizon = min(5, max(1, 200 - memory["step"]))
    risks = _risk_forecast(memory, horizon)
    turns_left = max(1, 200 - memory["step"])
    risk_price = (
        110.0
        if turns_left > remaining.get(start, default_distance) + 15
        else 65.0
    )

    # Compute each logarithmic collision penalty once per cell and turn.
    risk_costs = [
        {
            p: (
                10000.0 if probability >= 0.999999
                else -risk_price * math.log1p(-probability)
            )
            for p, probability in risk.items()
        }
        for risk in risks
    ]

    layers = [{start}]
    for t in range(horizon):
        reachable = set()
        for p in layers[-1]:
            edges = first_options if t == 0 else options.get(p, ())
            reachable.update(q for q, _, _ in edges)
        layers.append(reachable)

    base_cost = {
        p: (
            1.0
            + 0.10 * remaining[p]
            + 0.018 * min(visits.get(p, 0), 15)
        )
        for p in legal
    }
    values = {p: remaining.get(p, default_distance) for p in layers[-1]}
    chosen = (0, 0)

    for t in range(horizon - 1, -1, -1):
        current_values = {}
        collision = risk_costs[t]
        for p in layers[t]:
            edges = first_options if t == 0 else options.get(p, ())
            best_value = _INF
            best_move = (0, 0)
            for q, _, move in edges:
                if q not in values:
                    continue
                value = (
                    base_cost.get(q, 1.0 + 0.10 * default_distance)
                    + collision.get(q, 0.0)
                    + values[q]
                )
                if move == (0, 0) and p != goal:
                    value += 0.035
                if value < best_value:
                    best_value = value
                    best_move = move
            current_values[p] = best_value
            if t == 0:
                chosen = best_move
        values = current_values

    return {"move": [chosen[0], chosen[1]], "interact": True}
# EVOLVE-BLOCK-END