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
_TRAVEL = tuple(move for move in _MOVES if move != (0, 0))


def _add(pos, move):
    return pos[0] + move[0], pos[1] + move[1]


def _distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _new_memory():
    return {
        "believed_map": {},
        "agent_pos": (0, 0),
        "step": 0,
        "keys": 0,
        "door_open": False,
        "keys_collected": set(),
        "terrain": {},
        "seen": {},
        "outside": set(),
        "visible": {},
        "occupancy": {},
        "visits": {},
        "changed_walls": set(),
        "door_changed_step": -1,
        "observed_step": None,
        "target": None,
    }


def _terrain_sample(terrain, row, column):
    if isinstance(terrain, (list, tuple)) and row < len(terrain):
        values = terrain[row]
        if isinstance(values, (list, tuple)) and column < len(values):
            value = values[column]
            if value in (0, 1, 2, 3, 4):
                return value
    return None


def _enemy_open(memory, pos, doors_open):
    if pos in memory["outside"]:
        return False
    # Every in-bounds position is within fourteen cells of the origin.
    if abs(pos[0]) > 14 or abs(pos[1]) > 14:
        return False
    tile = memory["terrain"].get(pos)
    if tile == 1:
        return False
    if tile == 3 and not doors_open:
        return False
    return True


def _spread_occupancy(memory, occupancy, doors_open):
    """Propagate anonymous occupancy mass through uniform attempted moves."""
    result = {}
    for pos, probability in occupancy.items():
        if probability < 0.0002:
            continue
        share = probability / 9.0
        for dx, dy in _MOVES:
            destination = (pos[0] + dx, pos[1] + dy)
            valid = _enemy_open(memory, destination, doors_open)
            if valid and dx and dy:
                valid = (
                    _enemy_open(memory, (pos[0] + dx, pos[1]), doors_open)
                    and _enemy_open(memory, (pos[0], pos[1] + dy), doors_open)
                )
            if not valid:
                destination = pos
            result[destination] = result.get(destination, 0.0) + share
    return {
        pos: min(1.0, probability)
        for pos, probability in result.items()
        if probability >= 0.0002
    }


def _publish_map(memory):
    """Publish present evidence separately from planning hypotheses."""
    phase = memory["step"] // 25
    published = {}
    visible = memory["visible"]
    for pos, tile in memory["terrain"].items():
        if pos in visible:
            continue

        stamp = memory["seen"].get(pos, -1)
        # Generic terrain may have changed at the latest wall update.
        if tile not in (2, 3, 4) and stamp // 25 != phase:
            continue
        if tile == 3 and stamp < memory["door_changed_step"]:
            continue
        if memory["occupancy"].get(pos, 0.0) > 0.12:
            continue
        published[pos] = tile

    # Exact current sightings override inferred underlying terrain.
    published.update(visible)
    memory["believed_map"] = published


def world_model_step(memory: dict, local_obs: dict,
                     last_action: dict) -> dict:
    if memory is None or "terrain" not in memory:
        memory = _new_memory()

    step = int(local_obs.get("step", 0))
    previous_step = memory["observed_step"]
    if previous_step == step:
        return memory

    feedback = local_obs.get("feedback") or {}
    displacement = feedback.get("displacement", (0, 0))
    ax, ay = memory["agent_pos"]
    ax += int(displacement[0])
    ay += int(displacement[1])
    memory["agent_pos"] = (ax, ay)

    old_door_open = memory["door_open"]
    occupancy = memory["occupancy"]
    if previous_step is not None:
        elapsed = max(0, step - previous_step)
        if elapsed > 12:
            occupancy = {}
        else:
            # Interaction happens before enemy movement.
            transition_door_open = bool(
                local_obs.get("door_open", old_door_open)
            )
            for _ in range(elapsed):
                occupancy = _spread_occupancy(
                    memory, occupancy, transition_door_open
                )

    memory["step"] = step
    memory["keys"] = int(local_obs.get("keys", memory["keys"]))
    memory["door_open"] = bool(
        local_obs.get("door_open", old_door_open)
    )
    if memory["door_open"] != old_door_open:
        memory["door_changed_step"] = step
    if feedback.get("collected"):
        memory["keys_collected"].add((ax, ay))

    grid = local_obs["grid"]
    terrain_view = local_obs.get("terrain")
    visible = {}
    enemies = set()

    for row, values in enumerate(grid):
        for column, cell in enumerate(values):
            pos = (ax + column - 2, ay + row - 2)
            if cell == -1:
                memory["outside"].add(pos)
                occupancy.pop(pos, None)
                continue

            memory["outside"].discard(pos)
            visible[pos] = cell
            underlying = _terrain_sample(terrain_view, row, column)
            if underlying is None:
                underlying = (
                    memory["terrain"].get(pos, 0)
                    if cell == 5 else cell
                )
                # Occupancy proves that an old wall is currently passable.
                if cell == 5 and underlying == 1:
                    underlying = 0

            old = memory["terrain"].get(pos)
            if old is not None and (old == 1) != (underlying == 1):
                memory["changed_walls"].add(pos)
            memory["terrain"][pos] = underlying
            memory["seen"][pos] = step

            if cell == 5:
                enemies.add(pos)
            occupancy.pop(pos, None)

    # Unseen probability cannot account for more than the remaining enemies.
    unseen_mass = sum(occupancy.values())
    remaining = max(0, 3 - len(enemies))
    if unseen_mass > remaining and unseen_mass > 0.0:
        scale = remaining / unseen_mass
        occupancy = {
            pos: probability * scale
            for pos, probability in occupancy.items()
            if probability * scale >= 0.0002
        }
    for pos in enemies:
        occupancy[pos] = 1.0

    memory["occupancy"] = occupancy
    memory["visible"] = visible
    memory["observed_step"] = step
    memory["visits"][(ax, ay)] = (
        memory["visits"].get((ax, ay), 0) + 1
    )
    _publish_map(memory)
    return memory


def _passable(memory, pos, immediate=False):
    if pos in memory["outside"]:
        return False
    terrain = memory["terrain"]
    if pos not in terrain:
        return False
    tile = terrain[pos]
    if tile == 1:
        # Old walls are hypotheses for routing, never legal immediate moves.
        return (
            not immediate
            and memory["seen"].get(pos, -1) // 25
            < memory["step"] // 25
        )
    if tile == 3:
        return memory["door_open"] or memory["keys"] >= 2
    return True


def _edge_allowed(memory, source, destination, immediate=False):
    if not _passable(memory, destination, immediate):
        return False
    dx = destination[0] - source[0]
    dy = destination[1] - source[1]
    if dx and dy:
        terrain = memory["terrain"]
        # A closed door must first be approached cardinally.
        if not memory["door_open"]:
            if terrain.get(source) == 3 or terrain.get(destination) == 3:
                return False
        for side in ((source[0] + dx, source[1]),
                     (source[0], source[1] + dy)):
            if not _passable(memory, side, immediate):
                return False
            if terrain.get(side) == 3 and not memory["door_open"]:
                return False
    return True


def _route_cost(memory, destination, forecast):
    tile = memory["terrain"][destination]
    stale = (
        memory["seen"].get(destination, -1) // 25
        < memory["step"] // 25
    )
    cost = 1.0
    if tile == 1:
        cost += 7.0
    elif stale:
        cost += 0.35
    if stale and destination in memory["changed_walls"]:
        cost += 1.5
    if tile == 3 and not memory["door_open"]:
        cost += 0.5
    cost += 6.0 * memory["occupancy"].get(destination, 0.0)
    cost += 10.0 * forecast.get(destination, 0.0)
    return cost


def _search(memory, start, forecast, reverse=False):
    """Dijkstra search over the planning layer; supports reverse potentials."""
    distances = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        distance, pos = heapq.heappop(queue)
        if distance != distances.get(pos):
            continue
        for move in _TRAVEL:
            neighbor = _add(pos, move)
            if not _passable(memory, neighbor):
                continue
            source, destination = (
                (neighbor, pos) if reverse else (pos, neighbor)
            )
            if not _edge_allowed(memory, source, destination):
                continue
            candidate = distance + _route_cost(
                memory, destination, forecast
            )
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distances


def _information_gain(memory, pos):
    gain = 0.0
    phase = memory["step"] // 25
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cell = (pos[0] + dx, pos[1] + dy)
            if cell in memory["outside"]:
                continue
            if abs(cell[0]) > 14 or abs(cell[1]) > 14:
                continue
            if cell not in memory["terrain"]:
                gain += 1.0
            elif memory["seen"].get(cell, -1) // 25 < phase:
                gain += 0.20
    return gain


def _choose_target(memory, distances):
    terrain = memory["terrain"]
    start = memory["agent_pos"]
    keys = memory["keys"]

    if keys < 2:
        objectives = [
            pos for pos in distances
            if terrain.get(pos) == 2
            and pos not in memory["keys_collected"]
        ]
    else:
        objectives = [
            pos for pos in distances if terrain.get(pos) == 4
        ]
        if not objectives and not memory["door_open"]:
            objectives = [
                pos for pos in distances if terrain.get(pos) == 3
            ]

    if objectives:
        def objective_score(pos):
            score = distances[pos]
            if keys == 0:
                others = [other for other in objectives if other != pos]
                if others:
                    score += 0.25 * min(
                        _distance(pos, other) for other in others
                    )
            if pos == memory.get("target"):
                score -= 0.4
            return score, pos

        return min(objectives, key=objective_score)

    best = None
    best_score = -math.inf
    for pos, distance in distances.items():
        if pos == start or terrain.get(pos) == 1:
            continue
        gain = _information_gain(memory, pos)
        if gain <= 0.0:
            continue
        score = gain / ((distance + 2.0) ** 0.85)
        score /= 1.0 + 0.22 * memory["visits"].get(pos, 0)
        if pos == memory.get("target"):
            score *= 1.12
        if score > best_score:
            best_score, best = score, pos
    return best


def planner(memory: dict, local_obs: dict) -> dict:
    start = memory.get("agent_pos", (0, 0))
    terrain = memory.get("terrain", {})
    if not terrain:
        return {"move": [0, 0], "interact": True}

    # All actions interact: this collects keys at their destination and opens
    # a cardinally adjacent door as soon as the inventory permits it.
    doors_open = memory["door_open"]
    if memory["keys"] >= 2 and not doors_open:
        doors_open = any(
            terrain.get(_add(start, move)) == 3
            for move in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )

    forecast = _spread_occupancy(
        memory, memory["occupancy"], doors_open
    )
    distances = _search(memory, start, forecast)
    target = _choose_target(memory, distances)
    memory["target"] = target

    potential = (
        _search(memory, target, forecast, reverse=True)
        if target is not None else {}
    )

    candidates = []
    for move in _MOVES:
        destination = _add(start, move)
        if move != (0, 0):
            if not _edge_allowed(memory, start, destination, True):
                continue

        occupied = memory["occupancy"].get(destination, 0.0)
        # Known current contact is fatal before enemies get a chance to move.
        if occupied >= 0.999:
            continue
        after = forecast.get(destination, 0.0)
        risk = occupied + (1.0 - occupied) * after

        if target is not None:
            remaining = potential.get(destination, math.inf)
            if not math.isfinite(remaining):
                continue
            cost = remaining
            if move != (0, 0):
                cost += _route_cost(memory, destination, forecast)
            else:
                cost += 1.35
            # Small revisit cost breaks equal-length loops without defeating
            # required backtracking through narrow passages.
            cost += 0.035 * min(
                memory["visits"].get(destination, 0), 20
            )
        else:
            cost = 0.25 * min(
                memory["visits"].get(destination, 0), 20
            )
            cost -= 0.2 * _information_gain(memory, destination)
            if move == (0, 0):
                cost += 0.3

        # Immediate survival dominates modest route savings.
        cost += 180.0 * risk
        candidates.append((risk, cost, move))

    if not candidates:
        return {"move": [0, 0], "interact": True}

    lowest_risk = min(item[0] for item in candidates)
    # Prefer a safe detour or wait over entering an enemy's reachable cell.
    admissible = [
        item for item in candidates
        if item[0] <= lowest_risk + 0.045
    ]
    _, _, move = min(
        admissible,
        key=lambda item: (
            item[1],
            item[0],
            item[2] == (0, 0),
            item[2],
        ),
    )
    return {"move": [move[0], move[1]], "interact": True}
# EVOLVE-BLOCK-END
