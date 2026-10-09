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
    visits = memory.setdefault("visits", {})
    visits[(ax, ay)] = visits.get((ax, ay), 0) + 1
    void = memory.setdefault("void", set())
    bounds = memory.setdefault("bounds", {})
    step = memory["step"]
    if feedback["opened"]:
        memory["opened_step"] = step
    if feedback["collected"]:
        terrain[(ax, ay)] = 0

    grid = local_obs["grid"]
    terrain_grid = local_obs.get("terrain")
    has_terrain = (
        isinstance(terrain_grid, (list, tuple))
        and len(terrain_grid) == 5
        and all(isinstance(row, (list, tuple)) and len(row) == 5
                for row in terrain_grid)
    )
    visible = {}
    enemies = {}
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cell = grid[dy + 2][dx + 2]
            pos = (ax + dx, ay + dy)
            if cell == -1:
                void.add(pos)
                if dy == 0 and dx < 0:
                    bounds["min_x"] = max(bounds.get("min_x", pos[0] + 1),
                                          pos[0] + 1)
                elif dy == 0 and dx > 0:
                    bounds["max_x"] = min(bounds.get("max_x", pos[0] - 1),
                                          pos[0] - 1)
                if dx == 0 and dy < 0:
                    bounds["min_y"] = max(bounds.get("min_y", pos[1] + 1),
                                          pos[1] + 1)
                elif dx == 0 and dy > 0:
                    bounds["max_y"] = min(bounds.get("max_y", pos[1] - 1),
                                          pos[1] - 1)
                continue
            visible[pos] = cell
            base = terrain_grid[dy + 2][dx + 2] if has_terrain else cell
            if base in (-1, 5):
                base = terrain.get(pos, 0)
            terrain[pos] = base
            seen[pos] = step
            if cell == 5:
                enemies[pos] = step

    # One observed boundary determines the other boundary of this 15x15 maze.
    for axis in ("x", "y"):
        low, high = "min_" + axis, "max_" + axis
        if low in bounds:
            bounds[high] = bounds[low] + 14
        elif high in bounds:
            bounds[low] = bounds[high] - 14

    # Old terrain remains useful for navigation, but changed walls are uncertain.
    epoch = step // 25
    believed = {}
    for pos, cell in terrain.items():
        fresh = seen.get(pos, -25) // 25 == epoch
        stable = cell in (2, 4)
        if cell == 3:
            stable = (
                not memory["door_open"]
                or seen.get(pos, -1) >= memory.get("opened_step", step)
            )
        if fresh or stable:
            believed[pos] = cell
    believed.update(visible)
    memory["believed_map"] = believed
    memory["enemy_sightings"] = enemies
    return memory


def _maze_edges(pos, terrain, keys, door_open):
    """Known legal displacements, including interaction before movement."""
    x, y = pos

    def passable(p):
        cell = terrain.get(p, -1)
        return cell not in (-1, 1) and (
            cell != 3 or door_open or keys >= 2
        )

    for dx, dy in (
        (0, -1), (1, 0), (0, 1), (-1, 0),
        (1, -1), (1, 1), (-1, 1), (-1, -1),
    ):
        dest = (x + dx, y + dy)
        if not passable(dest):
            continue
        if dx and dy:
            if not passable((x + dx, y)) or not passable((x, y + dy)):
                continue
            # A diagonal destination door cannot be opened from this position.
            if terrain.get(dest) == 3 and not door_open:
                continue
        yield dest, (dx, dy)


def planner(memory: dict, local_obs: dict) -> dict:
    """Follow useful legal paths while minimizing immediate collision risk."""
    import heapq

    terrain = memory.get("terrain_map", memory["believed_map"])
    agent = memory["agent_pos"]
    keys = memory["keys"]
    door_open = memory["door_open"]
    step = memory["step"]
    seen = memory.get("terrain_seen", {})
    visits = memory.get("visits", {})
    enemies = set(memory.get("enemy_sightings", {}))
    wait = {"move": [0, 0], "interact": True}

    # Any enemy able to hit a next destination lies within the current 5x5 view.
    # Each legal attempted displacement has probability 1/9. Blocked attempts
    # stay at the enemy's current cell, which is forbidden before enemy movement.
    enemy_reach = [
        {dest for dest, move in _maze_edges(e, terrain, 0, door_open)}
        for e in enemies
    ]

    def risk(dest):
        if dest in enemies:
            return 1.0
        survival = 1.0
        for reach in enemy_reach:
            if dest in reach:
                survival *= 8.0 / 9.0
        return 1.0 - survival

    moves = list(_maze_edges(agent, terrain, keys, door_open))
    if not moves:
        return wait
    risks = {dest: risk(dest) for dest, move in moves}
    minimum = min(risks.values())
    wait_risk = risk(agent)
    if wait_risk < minimum - 1e-12:
        return wait
    if terrain.get(agent) == 2 and keys < 2 and wait_risk <= minimum:
        return wait
    safe = {dest for dest, value in risks.items()
            if value <= minimum + 1e-12}

    # Dijkstra retains the first action of each route. Only that action uses
    # current occupancy; later enemy positions are unknown.
    distance = {agent: 0.0}
    first = {}
    queue = [(0.0, agent)]
    while queue:
        cost, pos = heapq.heappop(queue)
        if cost > distance[pos] + 1e-12:
            continue
        for dest, move in _maze_edges(pos, terrain, keys, door_open):
            if pos == agent and dest not in safe:
                continue
            stale = seen.get(dest, -25) // 25 != step // 25
            edge = 1.0 + 0.04 * min(visits.get(dest, 0), 6)
            if stale:
                edge += 0.35
            new_cost = cost + edge
            if new_cost + 1e-12 < distance.get(dest, float("inf")):
                distance[dest] = new_cost
                first[dest] = move if pos == agent else first[pos]
                heapq.heappush(queue, (new_cost, dest))

    if keys < 2:
        targets = [
            pos for pos, cell in terrain.items()
            if cell == 2 and pos not in memory["keys_collected"]
        ]
    elif not door_open:
        targets = [pos for pos, cell in terrain.items() if cell == 3]
    else:
        targets = [pos for pos, cell in terrain.items() if cell == 4]
    reachable = [pos for pos in targets if pos in first]
    if reachable:
        goal = min(reachable, key=lambda pos: distance[pos])
        return {"move": list(first[goal]), "interact": True}

    # Choose reachable observation positions by information gained per effort.
    bounds = memory.get("bounds", {})
    void = memory.get("void", set())

    def inside(pos):
        x, y = pos
        return (
            pos not in void
            and bounds.get("min_x", x) <= x <= bounds.get("max_x", x)
            and bounds.get("min_y", y) <= y <= bounds.get("max_y", y)
        )

    best_goal = None
    best_score = -float("inf")
    for pos in first:
        x, y = pos
        gain = 0.0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                cell_pos = (x + dx, y + dy)
                if not inside(cell_pos):
                    continue
                if cell_pos not in terrain:
                    gain += 1.0
                elif seen.get(cell_pos, -25) // 25 != step // 25:
                    gain += 0.3
        score = gain / (distance[pos] + 2.5)
        score -= 0.08 * visits.get(pos, 0)
        if score > best_score:
            best_score = score
            best_goal = pos
    if best_goal is not None:
        return {"move": list(first[best_goal]), "interact": True}
    return wait
# EVOLVE-BLOCK-END