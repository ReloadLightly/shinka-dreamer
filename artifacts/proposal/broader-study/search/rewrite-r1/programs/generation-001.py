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

_DIRECTIONS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),            (1, 0),
    (-1, 1),  (0, 1),  (1, 1),
)
_ATTEMPTS = ((0, 0),) + _DIRECTIONS
_CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _cell(matrix, row, column):
    if matrix is None:
        return None
    try:
        return matrix[row][column]
    except (IndexError, KeyError, TypeError):
        return None


def _limits(memory):
    xmin, xmax, ymin, ymax = memory["extent"]
    borders = memory["borders"]
    # Every observed coordinate belongs to the same 15 by 15 world.
    left = borders.get("left")
    right = borders.get("right")
    top = borders.get("top")
    bottom = borders.get("bottom")
    return (
        xmax - 14 if left is None else left,
        xmin + 14 if right is None else right,
        ymax - 14 if top is None else top,
        ymin + 14 if bottom is None else bottom,
    )


def _inside(memory, position):
    left, right, top, bottom = memory["limits"]
    x, y = position
    return left <= x <= right and top <= y <= bottom


def _enemy_blocked(memory, position, opened):
    if not _inside(memory, position):
        return True
    cell = memory["terrain"].get(position)
    return cell == 1 or (cell == 3 and not opened)


def _enemy_destination(memory, position, displacement, opened):
    x, y = position
    dx, dy = displacement
    destination = (x + dx, y + dy)
    if _enemy_blocked(memory, destination, opened):
        return position
    if dx and dy:
        if (
            _enemy_blocked(memory, (x + dx, y), opened)
            or _enemy_blocked(memory, (x, y + dy), opened)
        ):
            return position
    return destination


def _spread(memory, occupancy, opened=None):
    """Propagate anonymous occupancy mass through uniform attempted moves."""
    if opened is None:
        opened = memory["door_open"]
    result = {}
    for position, mass in occupancy.items():
        if mass < 0.00001:
            continue
        share = mass / 9.0
        for displacement in _ATTEMPTS:
            destination = _enemy_destination(
                memory, position, displacement, opened
            )
            result[destination] = result.get(destination, 0.0) + share
    return {
        position: min(1.0, mass)
        for position, mass in result.items()
        if mass >= 0.00001
    }


def _infer_borders(memory, observation):
    """A cropped central row/column identifies a world edge exactly."""
    ax, ay = memory["agent_pos"]
    borders = memory["borders"]
    row = [observation.get((ax + dx, ay), -1) for dx in range(-2, 3)]
    column = [
        observation.get((ax, ay + dy), -1) for dy in range(-2, 3)
    ]
    for values, coordinate, low_name, high_name in (
        (row, ax, "left", "right"),
        (column, ay, "top", "bottom"),
    ):
        valid = [i for i, value in enumerate(values) if value != -1]
        if not valid:
            continue
        first, last = valid[0], valid[-1]
        if first > 0 and all(value == -1 for value in values[:first]):
            borders[low_name] = coordinate + first - 2
        if last < 4 and all(value == -1 for value in values[last + 1:]):
            borders[high_name] = coordinate + last - 2


def world_model_step(memory: dict, local_obs: dict, last_action: dict) -> dict:
    """Integrate observations without confusing terrain with occupancy."""
    if memory is None:
        memory = {}

    defaults = {
        "believed_map": {},
        "agent_pos": (0, 0),
        "terrain": {},
        "seen": {},
        "occupancy": {},
        "enemy_sightings": {},
        "keys_collected": set(),
        "key_sites": set(),
        "door_sites": set(),
        "exit_sites": set(),
        "visits": {},
        "extent": (0, 0, 0, 0),
        "borders": {},
        "keys": 0,
        "door_open": False,
        "step": 0,
    }
    for name, value in defaults.items():
        memory.setdefault(name, value)

    previous_step = memory["step"]
    feedback = local_obs.get("feedback") or {}
    dx, dy = feedback.get("displacement", (0, 0))
    ax, ay = memory["agent_pos"]
    ax, ay = ax + dx, ay + dy
    memory["agent_pos"] = (ax, ay)
    memory["step"] = int(local_obs.get("step", previous_step + 1))
    memory["keys"] = int(local_obs.get("keys", memory["keys"]))
    memory["door_open"] = bool(
        local_obs.get("door_open", memory["door_open"])
    )

    if feedback.get("collected"):
        memory["keys_collected"].add((ax, ay))
        memory["key_sites"].discard((ax, ay))

    visits = memory["visits"]
    visits[(ax, ay)] = visits.get((ax, ay), 0) + 1

    grid = local_obs.get("grid")
    terrain_view = local_obs.get("terrain")
    terrain = memory["terrain"]
    seen = memory["seen"]
    current = {}
    enemies = set()
    xmin, xmax, ymin, ymax = memory["extent"]
    step = memory["step"]

    for row in range(5):
        for column in range(5):
            position = (ax + column - 2, ay + row - 2)
            visible = _cell(grid, row, column)
            ground = _cell(terrain_view, row, column)
            if visible is None:
                visible = ground
            if visible is None or visible == -1:
                continue

            current[position] = visible
            x, y = position
            xmin, xmax = min(xmin, x), max(xmax, x)
            ymin, ymax = min(ymin, y), max(ymax, y)

            if visible == 5:
                enemies.add(position)

            # Terrain is independent of an enemy overlay. Without a terrain
            # channel, retain the last substrate beneath an occupied cell.
            if ground not in (0, 1, 2, 3, 4):
                ground = visible if visible in (0, 1, 2, 3, 4) else None

            if ground is not None:
                terrain[position] = ground
                seen[position] = step
                if ground == 2 and position not in memory["keys_collected"]:
                    memory["key_sites"].add(position)
                else:
                    memory["key_sites"].discard(position)
                if ground == 3:
                    memory["door_sites"].add(position)
                if ground == 4:
                    memory["exit_sites"].add(position)
            elif position not in terrain:
                # Useful internally, but never exported as a terrain claim.
                terrain[position] = 0
                seen[position] = step

    memory["extent"] = (xmin, xmax, ymin, ymax)
    _infer_borders(memory, current)
    memory["limits"] = _limits(memory)

    occupancy = memory["occupancy"]
    elapsed = max(0, step - previous_step)
    if elapsed > 8:
        occupancy = {}
    else:
        for _ in range(elapsed):
            occupancy = _spread(memory, occupancy)

    # Condition on observable occupied and empty cells. No enemy identities
    # or correspondences are introduced.
    for position in current:
        occupancy.pop(position, None)
    hidden_budget = max(0.0, 3.0 - len(enemies))
    hidden_mass = sum(occupancy.values())
    if hidden_mass > hidden_budget and hidden_mass > 0:
        scale = hidden_budget / hidden_mass
        occupancy = {
            position: mass * scale
            for position, mass in occupancy.items()
            if mass * scale >= 0.00001
        }
    for position in enemies:
        occupancy[position] = 1.0

    memory["occupancy"] = occupancy
    memory["enemy_sightings"] = {
        position: step for position in enemies
    }
    memory["current_view"] = current

    # Reconstruction certificates: an unobserved wall cannot be occupied,
    # and its state is certified until the next global wall-toggle boundary.
    # Other unobserved terrain remains available in the private route map.
    epoch = step // 25
    reported = {
        position: 1
        for position, ground in terrain.items()
        if ground == 1
        and seen.get(position, -25) // 25 == epoch
        and _inside(memory, position)
    }
    reported.update(current)
    memory["believed_map"] = reported
    return memory


def _will_open(memory):
    if memory["door_open"]:
        return True
    if memory["keys"] < 2:
        return False
    x, y = memory["agent_pos"]
    terrain = memory["terrain"]
    return any(
        terrain.get((x + dx, y + dy)) == 3
        for dx, dy in _CARDINAL
    )


def _node_cost(memory, position, danger):
    if not _inside(memory, position):
        return None
    terrain = memory["terrain"]
    cell = terrain.get(position)
    epoch = memory["step"] // 25
    fresh_epoch = memory["seen"].get(position, -25) // 25 == epoch

    if cell == 1 and fresh_epoch:
        return None
    if cell == 3 and not memory["door_open"] and memory["keys"] < 2:
        return None

    if cell is None:
        base = 2.6
    elif cell == 1:
        # A remembered wall may have disappeared at a toggle boundary.
        base = 4.5
    elif not fresh_epoch and cell == 0:
        base = 1.3
    else:
        base = 1.0

    return (
        base
        + 14.0 * danger.get(position, 0.0)
        + 3.0 * memory["occupancy"].get(position, 0.0)
        + 0.025 * min(10, memory["visits"].get(position, 0))
    )


def _graph(memory, danger):
    xmin, xmax, ymin, ymax = memory["extent"]
    left, right, top, bottom = memory["limits"]
    xmin, xmax = max(left, xmin - 2), min(right, xmax + 2)
    ymin, ymax = max(top, ymin - 2), min(bottom, ymax + 2)

    costs = {}
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            position = (x, y)
            cost = _node_cost(memory, position, danger)
            if cost is not None:
                costs[position] = cost

    start = memory["agent_pos"]
    costs.setdefault(start, 1.0)
    forward = {position: [] for position in costs}
    reverse = {position: [] for position in costs}
    terrain = memory["terrain"]
    enemies = memory["enemy_sightings"]

    for position in costs:
        x, y = position
        for dx, dy in _DIRECTIONS:
            destination = (x + dx, y + dy)
            if destination not in costs:
                continue
            if position == start and destination in enemies:
                continue
            if dx and dy:
                if (x + dx, y) not in costs or (x, y + dy) not in costs:
                    continue
                # A closed destination door cannot be opened from a
                # diagonally adjacent source.
                if (
                    terrain.get(destination) == 3
                    and not memory["door_open"]
                ):
                    continue
            weight = costs[destination]
            forward[position].append((destination, weight))
            reverse[destination].append((position, weight))
    return forward, reverse


def _distances(graph, start):
    distance = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        cost, position = heapq.heappop(queue)
        if cost != distance.get(position):
            continue
        for destination, weight in graph.get(position, ()):
            candidate = cost + weight
            if candidate < distance.get(destination, float("inf")):
                distance[destination] = candidate
                heapq.heappush(queue, (candidate, destination))
    return distance


def _information_gain(memory, position):
    terrain = memory["terrain"]
    seen = memory["seen"]
    step = memory["step"]
    x, y = position
    gain = 0.0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            candidate = (x + dx, y + dy)
            if not _inside(memory, candidate):
                continue
            if candidate not in terrain:
                gain += 1.0
            elif seen.get(candidate, -25) // 25 != step // 25:
                # Refreshing old terrain matters, but discovering new
                # territory is usually more valuable before finding goals.
                gain += 0.12
    return gain


def _choose_goal(memory, distance):
    start = memory["agent_pos"]
    if memory["keys"] < 2:
        targets = [
            position for position in memory["key_sites"]
            if position not in memory["keys_collected"]
            and position in distance
        ]
    else:
        targets = [
            position for position in memory["exit_sites"]
            if position in distance
        ]
        if not targets and not memory["door_open"]:
            targets = [
                position for position in memory["door_sites"]
                if position in distance
            ]

    if targets:
        return min(
            targets,
            key=lambda position: (distance[position], position),
        )

    previous = memory.get("navigation_goal")
    best, best_value = None, -float("inf")
    for position, cost in distance.items():
        if position == start:
            continue
        gain = _information_gain(memory, position)
        if gain <= 0:
            continue
        value = gain / (cost + 3.0)
        value /= 1.0 + 0.06 * min(
            12, memory["visits"].get(position, 0)
        )
        if position == previous:
            value *= 1.12
        if value > best_value:
            best_value, best = value, position

    if best is not None:
        return best

    # If everything has been surveyed, seek the oldest accessible view.
    candidates = [position for position in distance if position != start]
    if candidates:
        return max(
            candidates,
            key=lambda position: (
                (
                    memory["step"]
                    - memory["seen"].get(position, -25)
                    + 1.0
                )
                / (distance[position] + 4.0)
                / (1.0 + memory["visits"].get(position, 0)),
                position,
            ),
        )
    return start


def planner(memory: dict, local_obs: dict) -> dict:
    """Choose a useful route, then reassess its first move for survival."""
    start = memory.get("agent_pos", (0, 0))
    if "terrain" not in memory or "limits" not in memory:
        return {"move": [0, 0], "interact": True}

    opened = _will_open(memory)
    danger = _spread(memory, memory["occupancy"], opened)
    forward, reverse = _graph(memory, danger)
    distance = _distances(forward, start)
    goal = _choose_goal(memory, distance)
    memory["navigation_goal"] = goal
    remaining = _distances(reverse, goal)

    # Every enemy able to reach a one-step destination is within the
    # current 5x5 view. Use its actual uniform attempted-move distribution
    # for the immediate decision, rather than an old sighting radius.
    enemies = tuple(memory["enemy_sightings"])
    immediate_risk = {}
    for enemy in enemies:
        outcomes = {}
        for displacement in _ATTEMPTS:
            destination = _enemy_destination(
                memory, enemy, displacement, opened
            )
            outcomes[destination] = outcomes.get(destination, 0) + 1
        for destination, count in outcomes.items():
            probability = count / 9.0
            previous = immediate_risk.get(destination, 0.0)
            immediate_risk[destination] = (
                1.0 - (1.0 - previous) * (1.0 - probability)
            )

    candidates = [start]
    candidates.extend(destination for destination, _ in forward.get(start, ()))
    best_position = start
    best_score = float("inf")

    for destination in candidates:
        if destination in memory["enemy_sightings"]:
            continue
        risk = immediate_risk.get(destination, 0.0)
        route = remaining.get(destination)
        if route is None:
            continue

        score = 1.0 + route + 45.0 * risk
        score += 0.025 * min(
            12, memory["visits"].get(destination, 0)
        )
        if destination == start and goal != start:
            score += 0.2

        # Stable deterministic tie-breaking without random wandering.
        if score < best_score:
            best_score = score
            best_position = destination

    if best_score == float("inf"):
        best_position = min(
            candidates,
            key=lambda position: (
                immediate_risk.get(position, 0.0),
                memory["visits"].get(position, 0),
                -_information_gain(memory, position),
                position,
            ),
        )

    dx = best_position[0] - start[0]
    dy = best_position[1] - start[1]
    # Interaction has no stated cost: it collects destination keys,
    # including diagonal arrivals, and opens cardinally adjacent doors.
    return {"move": [dx, dy], "interact": True}
# EVOLVE-BLOCK-END
