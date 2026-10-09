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
# Both functions, their representations and any added helpers may evolve.
_MOVES = tuple(
    (dx, dy)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
)


def _destination(memory, pos, move, doors=False):
    """Resolve an attempted move, including diagonal corner constraints."""
    terrain = memory.get("terrain_map", {})
    outside = memory.get("outside", set())
    x, y = pos
    dx, dy = move

    def passable(p):
        if p in outside:
            return False
        cell = terrain.get(p, 0)
        if cell == 1:
            return False
        if cell == 3 and not memory.get("door_open", False):
            return (
                doors
                and memory.get("keys", 0) >= 2
                and abs(p[0] - x) + abs(p[1] - y) == 1
            )
        return True

    target = (x + dx, y + dy)
    if not passable(target):
        return pos
    if dx and dy:
        if not passable((x + dx, y)) or not passable((x, y + dy)):
            return pos
    return target


def _spread_occupancy(memory, belief):
    """Propagate anonymous occupancy mass through uniform attempted moves."""
    result = {}
    for pos, mass in belief.items():
        if mass <= 0:
            continue
        for move in _MOVES:
            target = _destination(memory, pos, move)
            result[target] = result.get(target, 0.0) + mass / 9.0
    return {p: mass for p, mass in result.items() if mass >= 0.0001}


def world_model_step(memory: dict, local_obs: dict, last_action: dict) -> dict:
    """Integrate the local 5x5 observation into persistent relative map memory."""
    if memory is None:
        memory = {
            "believed_map": {},
            "agent_pos": (0, 0),
            "enemy_sightings": {},
            "keys_collected": set(),
            "keys": 0,
            "door_open": False,
            "step": 0,
        }

    feedback = local_obs["feedback"]
    dx, dy = feedback["displacement"]
    ax, ay = memory["agent_pos"]
    ax, ay = ax + dx, ay + dy
    memory["agent_pos"] = (ax, ay)
    memory["step"] = local_obs["step"]
    memory["keys"] = local_obs["keys"]
    memory["door_open"] = local_obs["door_open"]
    if feedback["collected"]:
        memory["keys_collected"].add((ax, ay))

    terrain = memory.setdefault("terrain_map", {})
    seen = memory.setdefault("terrain_seen", {})
    outside = memory.setdefault("outside", set())
    landmarks = memory.setdefault("landmarks", set())
    visits = memory.setdefault("visits", {})
    visits[(ax, ay)] = visits.get((ax, ay), 0) + 1

    step = memory["step"]
    belief = memory.get("enemy_belief", {})
    elapsed = step - memory.get("belief_step", step)
    if elapsed > 3:
        belief = {}
    else:
        for _ in range(max(0, elapsed)):
            belief = _spread_occupancy(memory, belief)

    grid = local_obs["grid"]
    terrain_grid = local_obs.get("terrain")
    terrain_available = (
        isinstance(terrain_grid, (list, tuple))
        and len(terrain_grid) == 5
        and all(
            isinstance(row, (list, tuple)) and len(row) == 5
            for row in terrain_grid
        )
    )
    visible = set()
    occupied = set()
    for oy in range(-2, 3):
        for ox in range(-2, 3):
            cell = grid[oy + 2][ox + 2]
            pos = (ax + ox, ay + oy)
            if cell == -1:
                outside.add(pos)
                belief.pop(pos, None)
                continue
            outside.discard(pos)
            visible.add(pos)
            base = (
                terrain_grid[oy + 2][ox + 2]
                if terrain_available else cell
            )
            if base not in (0, 1, 2, 3, 4):
                base = terrain.get(pos, 0)
            terrain[pos] = base
            seen[pos] = step
            if base in (2, 3, 4):
                landmarks.add(pos)
            if cell == 5:
                occupied.add(pos)

    for pos in memory["keys_collected"]:
        terrain[pos] = 0
    if memory["door_open"]:
        for pos, cell in list(terrain.items()):
            if cell == 3:
                terrain[pos] = 0

    # Visible empty cells rule out occupancy; sightings never acquire identities.
    hidden = {
        p: mass for p, mass in belief.items()
        if p not in visible and p not in outside and terrain.get(p) != 1
    }
    total = sum(hidden.values())
    remaining = max(0, 3 - len(occupied))
    if total > remaining:
        scale = remaining / total
        hidden = {p: mass * scale for p, mass in hidden.items()}
    hidden.update({p: 1.0 for p in occupied})
    memory["enemy_belief"] = hidden
    memory["belief_step"] = step
    memory["enemy_sightings"] = {p: step for p in occupied}

    # Keep old terrain for navigation, but report only sufficiently fresh cells.
    phase = step // 25
    reported = {
        p: cell for p, cell in terrain.items()
        if seen.get(p, -25) // 25 == phase or p in landmarks
    }
    for pos in occupied:
        reported[pos] = 5
    memory["believed_map"] = reported
    return memory


def planner(memory: dict, local_obs: dict) -> dict:
    """Route to objectives or informative frontiers, avoiding contact risk."""
    import heapq

    terrain = memory.get("terrain_map", {})
    start = memory.get("agent_pos", (0, 0))
    visits = memory.get("visits", {})
    outside = memory.get("outside", set())
    occupied = set(memory.get("enemy_sightings", {}))
    belief = memory.get("enemy_belief", {})
    following = _spread_occupancy(memory, belief)
    hazard = {
        p: min(1.0, belief.get(p, 0.0) + following.get(p, 0.0))
        for p in set(belief) | set(following)
    }

    # Legal paths respect walls, door interaction, and both diagonal side cells.
    distance = {start: 0.0}
    first_move = {start: (0, 0)}
    queue = [(0.0, start)]
    while queue:
        cost, pos = heapq.heappop(queue)
        if cost != distance.get(pos):
            continue
        for move in _MOVES:
            if move == (0, 0):
                continue
            target = _destination(memory, pos, move, doors=True)
            if target == pos or target not in terrain:
                continue
            if pos == start and target in occupied:
                continue
            stale = (
                memory.get("terrain_seen", {}).get(target, -25) // 25
                < memory.get("step", 0) // 25
            )
            edge = (
                1.0 + 12.0 * hazard.get(target, 0.0)
                + 0.04 * min(visits.get(target, 0), 10)
                + 0.25 * stale
            )
            candidate = cost + edge
            if candidate < distance.get(target, float("inf")):
                distance[target] = candidate
                first_move[target] = (
                    move if pos == start else first_move[pos]
                )
                heapq.heappush(queue, (candidate, target))

    if memory.get("keys", 0) < 2:
        targets = [
            p for p, cell in terrain.items()
            if cell == 2
            and p not in memory.get("keys_collected", set())
            and p in distance
        ]
    else:
        targets = [
            p for p, cell in terrain.items()
            if cell == 4 and p in distance
        ]
        if not targets and not memory.get("door_open", False):
            targets = [
                p for p, cell in terrain.items()
                if cell == 3 and p in distance
            ]

    def information(pos):
        x, y = pos
        return sum(
            (x + dx, y + dy) not in terrain
            and (x + dx, y + dy) not in outside
            for dx in range(-2, 3)
            for dy in range(-2, 3)
        )

    if targets:
        goal = min(targets, key=lambda p: distance[p])
    else:
        frontiers = [
            p for p in distance if p != start and information(p) > 0
        ]
        if frontiers:
            goal = min(
                frontiers,
                key=lambda p: (
                    distance[p] / (1.0 + 0.3 * information(p))
                    + 0.35 * visits.get(p, 0)
                ),
            )
        else:
            # Revisiting remote cells refreshes topology after wall changes.
            candidates = [p for p in distance if p != start]
            goal = min(
                candidates,
                key=lambda p: (
                    visits.get(p, 0)
                    + 0.15 * distance[p]
                    + 0.03 * memory.get("terrain_seen", {}).get(p, 0)
                ),
                default=start,
            )

    move = first_move[goal]
    options = []
    for alternative in _MOVES:
        target = _destination(memory, start, alternative, doors=True)
        if alternative != (0, 0) and target == start:
            continue
        if target not in terrain or target in occupied:
            continue
        options.append((hazard.get(target, 0.0), alternative, target))

    if options:
        safest = min(item[0] for item in options)
        destination = _destination(memory, start, move, doors=True)
        if hazard.get(destination, 0.0) > safest + 0.055:
            safe_options = [
                item for item in options if item[0] <= safest + 0.02
            ]
            _, move, _ = min(
                safe_options,
                key=lambda item: (
                    max(
                        abs(item[2][0] - goal[0]),
                        abs(item[2][1] - goal[1]),
                    )
                    + 20.0 * item[0]
                    + 0.12 * visits.get(item[2], 0)
                    - 0.15 * information(item[2])
                ),
            )

    return {"move": list(move), "interact": True}
# EVOLVE-BLOCK-END