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
def world_model_step(memory: dict, local_obs: dict, last_action: dict) -> dict:
    """Integrate the local 5x5 observation into persistent relative map memory."""
    if memory is None:
        memory = {
            "believed_map": {},
            "agent_pos": (0, 0),
            "enemy_sightings": {},
            "terrain_map": {},
            "outside": set(),
            "visits": {},
            "seen_at": {},
            "unstable": set(),
            "map_epoch": 0,
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

    believed = memory["believed_map"]
    terrain = memory.setdefault("terrain_map", {})
    outside = memory.setdefault("outside", set())
    visits = memory.setdefault("visits", {})
    seen_at = memory.setdefault("seen_at", {})
    unstable = memory.setdefault("unstable", set())
    visits[(ax, ay)] = visits.get((ax, ay), 0) + 1

    # Occupancy is ephemeral; preserve the underlying terrain separately.
    for pos, cell in list(believed.items()):
        if cell == 5:
            if pos in terrain:
                believed[pos] = terrain[pos]
            else:
                del believed[pos]
    memory["enemy_sightings"] = {}

    epoch = memory["step"] // 25
    if epoch != memory.get("map_epoch", 0):
        for pos in unstable:
            believed.pop(pos, None)
    memory["map_epoch"] = epoch

    grid = local_obs["grid"]
    observed_terrain = local_obs.get("terrain")
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cell = grid[dy + 2][dx + 2]
            pos = (ax + dx, ay + dy)
            if cell == -1:
                outside.add(pos)
                continue
            outside.discard(pos)

            underlying = cell if cell != 5 else None
            if (
                isinstance(observed_terrain, (list, tuple))
                and len(observed_terrain) > dy + 2
                and isinstance(observed_terrain[dy + 2], (list, tuple))
                and len(observed_terrain[dy + 2]) > dx + 2
            ):
                value = observed_terrain[dy + 2][dx + 2]
                if isinstance(value, int) and value in (0, 1, 2, 3, 4):
                    underlying = value
            if underlying is not None:
                previous = terrain.get(pos)
                if previous in (0, 1) and underlying in (0, 1):
                    if previous != underlying:
                        unstable.add(pos)
                terrain[pos] = underlying

            believed[pos] = cell
            seen_at[pos] = memory["step"]
            if cell == 5:
                memory["enemy_sightings"][pos] = memory["step"]

    return memory


def planner(memory: dict, local_obs: dict) -> dict:
    """Replan safe routes to objectives and informative exploration positions."""
    import heapq
    import math

    believed = memory.get("believed_map", {})
    terrain = memory.get("terrain_map", {})
    start = memory.get("agent_pos", (0, 0))
    outside = memory.get("outside", set())
    visits = memory.get("visits", {})
    keys = memory.get("keys", 0)
    opened = memory.get("door_open", False)
    enemies = set(memory.get("enemy_sightings", {}))
    moves = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    wait = {"move": [0, 0], "interact": True}

    def cell_at(pos):
        cell = terrain.get(pos, believed.get(pos))
        return 0 if cell == 5 else cell

    def passable(pos, enemy=False):
        cell = cell_at(pos)
        if pos in outside or cell is None or cell == 1:
            return False
        if cell == 3 and not opened:
            return not enemy and keys >= 2
        return True

    def legal(source, destination, enemy=False):
        if not passable(destination, enemy):
            return False
        dx = destination[0] - source[0]
        dy = destination[1] - source[1]
        if dx and dy:
            if not passable((source[0] + dx, source[1]), enemy):
                return False
            if not passable((source[0], source[1] + dy), enemy):
                return False
            # A closed door must first be approached cardinally.
            if cell_at(destination) == 3 and not opened:
                return False
        return True

    # Marginal next-cell probabilities, without inventing enemy identities.
    survival = {}
    for enemy in enemies:
        distribution = {}
        for dx, dy in moves:
            destination = (enemy[0] + dx, enemy[1] + dy)
            if not legal(enemy, destination, True):
                destination = enemy
            distribution[destination] = distribution.get(destination, 0) + 1
        for destination, count in distribution.items():
            survival[destination] = survival.get(destination, 1.0) * (
                1.0 - count / 9.0
            )

    immediate = {}
    for dx, dy in moves:
        destination = (start[0] + dx, start[1] + dy)
        if destination not in enemies and legal(start, destination):
            immediate[(dx, dy)] = 1.0 - survival.get(destination, 1.0)
    if not immediate:
        return wait

    minimum_risk = min(immediate.values())
    permitted = {
        move for move, risk in immediate.items()
        if risk <= minimum_risk + 1e-9
    }
    if permitted == {(0, 0)}:
        return wait
    if (
        cell_at(start) == 2
        and start not in memory.get("keys_collected", set())
        and (0, 0) in permitted
    ):
        return wait

    # First edges respect current danger; later edges are replanned next turn.
    distances = {start: 0.0}
    first_moves = {}
    queue = [(0.0, start)]
    while queue:
        distance, position = heapq.heappop(queue)
        if distance != distances[position]:
            continue
        for dx, dy in moves:
            if dx == 0 and dy == 0:
                continue
            move = (dx, dy)
            if position == start and move not in permitted:
                continue
            destination = (position[0] + dx, position[1] + dy)
            if destination in enemies or not legal(position, destination):
                continue
            cost = distance + 1.0 + 0.015 * visits.get(destination, 0)
            if cost < distances.get(destination, float("inf")):
                distances[destination] = cost
                first_moves[destination] = (
                    move if position == start else first_moves[position]
                )
                heapq.heappush(queue, (cost, destination))

    collected = memory.get("keys_collected", set())
    reachable_keys = [
        pos for pos in distances
        if cell_at(pos) == 2 and pos not in collected
    ]
    reachable_exits = [
        pos for pos in distances if cell_at(pos) == 4
    ]
    reachable_doors = [
        pos for pos in distances if cell_at(pos) == 3
    ]

    target = None
    if keys < 2 and reachable_keys:
        target = min(reachable_keys, key=lambda pos: distances[pos])
    elif keys >= 2 and reachable_exits:
        target = min(reachable_exits, key=lambda pos: distances[pos])
    elif keys >= 2 and not opened and reachable_doors:
        target = min(reachable_doors, key=lambda pos: distances[pos])

    if target is None:
        frontiers = []
        for pos, distance in distances.items():
            if pos == start:
                continue
            information = sum(
                (pos[0] + dx, pos[1] + dy) not in terrain
                and (pos[0] + dx, pos[1] + dy) not in believed
                and (pos[0] + dx, pos[1] + dy) not in outside
                for dx in range(-2, 3)
                for dy in range(-2, 3)
            )
            if information:
                score = (
                    distance + 0.3 * visits.get(pos, 0)
                ) / math.sqrt(information)
                frontiers.append((score, pos))
        if frontiers:
            target = min(frontiers)[1]
        else:
            # Revisit old observations when the known region has no frontier.
            alternatives = [pos for pos in distances if pos != start]
            if alternatives:
                step = memory.get("step", 0)
                seen = memory.get("seen_at", {})
                target = min(
                    alternatives,
                    key=lambda pos: (
                        distances[pos] + 0.6 * visits.get(pos, 0)
                        - 0.03 * (step - seen.get(pos, 0))
                    ),
                )

    if target is None or target == start:
        return wait
    dx, dy = first_moves[target]
    # Interaction handles destination keys and adjacent doors before movement.
    return {"move": [dx, dy], "interact": True}
# EVOLVE-BLOCK-END

# Intervention: retain only the current 5x5 spatial observation in map fields.
# Localization, inventory, visits and collected-key history remain intact.
_study_full_updater = world_model_step

def world_model_step(memory, local_obs, last_action):
    m = _study_full_updater(memory, local_obs, last_action)
    ax, ay = m["agent_pos"]
    visible = {(ax + dx, ay + dy) for dx in range(-2, 3)
               for dy in range(-2, 3)}
    for field in ("believed_map", "terrain_map", "seen_at", "enemy_sightings"):
        m[field] = {p: value for p, value in m[field].items() if p in visible}
    for field in ("outside", "unstable"):
        m[field].intersection_update(visible)
    return m
