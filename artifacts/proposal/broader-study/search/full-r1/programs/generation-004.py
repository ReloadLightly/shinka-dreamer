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

    memory.setdefault("terrain_map", {})
    memory.setdefault("seen_at", {})
    memory.setdefault("outside", set())
    memory.setdefault("visits", {})
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

    step = memory["step"]
    terrain_map = memory["terrain_map"]
    seen_at = memory["seen_at"]
    believed = memory["believed_map"]
    visits = memory["visits"]
    visits[(ax, ay)] = visits.get((ax, ay), 0) + 1

    # Retain spatial memory for navigation, but stop reporting stale occupants
    # and terrain whose last observation preceded a wall-change boundary.
    for pos, cell in list(believed.items()):
        if cell == 5 or (
            cell in (0, 1) and seen_at.get(pos, -25) // 25 != step // 25
        ):
            del believed[pos]

    memory["enemy_sightings"] = {}
    grid = local_obs["grid"]
    terrain = local_obs.get("terrain")
    terrain_is_grid = (
        isinstance(terrain, (list, tuple))
        and len(terrain) == 5
        and all(isinstance(row, (list, tuple)) and len(row) == 5
                for row in terrain)
    )
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cell = grid[dy + 2][dx + 2]
            pos = (ax + dx, ay + dy)
            if cell == -1:
                memory["outside"].add(pos)
                believed.pop(pos, None)
                terrain_map.pop(pos, None)
                continue
            believed[pos] = cell
            seen_at[pos] = step
            ground = terrain[dy + 2][dx + 2] if terrain_is_grid else cell
            if ground not in (0, 1, 2, 3, 4):
                ground = terrain_map.get(pos, 0)
                if ground == 1:
                    ground = 0
            terrain_map[pos] = ground
            if cell == 5:
                memory["enemy_sightings"][pos] = step

    for pos in memory["keys_collected"]:
        if terrain_map.get(pos) == 2:
            terrain_map[pos] = 0
        if believed.get(pos) == 2:
            believed[pos] = 0
    return memory


def planner(memory: dict, local_obs: dict) -> dict:
    """Navigate remembered terrain while avoiding immediate contact risks."""
    import heapq

    terrain = memory["terrain_map"]
    start = memory["agent_pos"]
    step = memory["step"]
    keys = memory["keys"]
    door_open = memory["door_open"]
    seen = memory["seen_at"]
    visits = memory["visits"]
    enemies = set(memory["enemy_sightings"])
    outside = memory["outside"]
    moves = [(dx, dy) for dx in (-1, 0, 1)
             for dy in (-1, 0, 1) if dx or dy]

    # Interaction occurs before movement, so these doors open this turn.
    opening = {
        p for p, cell in terrain.items()
        if cell == 3 and keys >= 2
        and abs(p[0] - start[0]) + abs(p[1] - start[1]) == 1
    }

    def passable(p, future=False, enemy=False):
        cell = terrain.get(p)
        if cell is None or p in outside:
            return False
        if cell == 1:
            return future and seen.get(p, step) // 25 < step // 25
        if cell == 3 and not door_open:
            return p in opening or (future and not enemy and keys >= 2)
        return True

    def legal(a, b, future=False, enemy=False):
        if not passable(b, future, enemy):
            return False
        dx, dy = b[0] - a[0], b[1] - a[1]
        if dx and dy:
            if not (passable((a[0] + dx, a[1]), future, enemy)
                    and passable((a[0], a[1] + dy), future, enemy)):
                return False
            # A future closed door must be approached cardinally to open it.
            if (terrain.get(b) == 3 and not door_open and b not in opening):
                return False
        return True

    # Anonymous occupancy is sufficient: sum each visible enemy's probability
    # of arriving at a destination, including attempted moves that are blocked.
    risk = {}
    for enemy in enemies:
        for dx, dy in moves + [(0, 0)]:
            destination = (enemy[0] + dx, enemy[1] + dy)
            if not legal(enemy, destination, enemy=True):
                destination = enemy
            risk[destination] = risk.get(destination, 0.0) + 1.0 / 9.0

    immediate = {}
    for move in moves + [(0, 0)]:
        destination = (start[0] + move[0], start[1] + move[1])
        if legal(start, destination) and destination not in enemies:
            immediate[move] = risk.get(destination, 0.0)

    if not immediate:
        return {"move": [0, 0], "interact": True}

    minimum_risk = min(immediate.values())
    # Waiting is a useful safe action while a roaming enemy blocks a passage.
    allowed = {move for move, value in immediate.items()
               if value <= minimum_risk + 1e-9}

    distance = {start: 0.0}
    first_move = {}
    queue = [(0.0, start)]
    while queue:
        cost, pos = heapq.heappop(queue)
        if cost != distance.get(pos):
            continue
        for move in moves:
            if pos == start and move not in allowed:
                continue
            nxt = (pos[0] + move[0], pos[1] + move[1])
            if nxt in enemies or not legal(pos, nxt, future=True):
                continue
            extra = 1.0 + 0.06 * min(visits.get(nxt, 0), 15)
            extra += 8.0 * risk.get(nxt, 0.0)
            if terrain.get(nxt) == 1:
                extra += 5.0
            elif seen.get(nxt, step) // 25 < step // 25:
                extra += 0.35
            new_cost = cost + extra
            if new_cost < distance.get(nxt, float("inf")):
                distance[nxt] = new_cost
                first_move[nxt] = move if pos == start else first_move[pos]
                heapq.heappush(queue, (new_cost, nxt))

    if keys < 2:
        goals = [p for p, cell in terrain.items()
                 if cell == 2 and p not in memory["keys_collected"]]
    elif not door_open:
        goals = [p for p, cell in terrain.items() if cell == 3]
    else:
        goals = [p for p, cell in terrain.items() if cell == 4]

    reachable = [p for p in goals if p in distance]
    if reachable:
        target = min(reachable, key=lambda p: distance[p])
        if target == start:
            if (0, 0) in allowed:
                return {"move": [0, 0], "interact": True}
        else:
            return {"move": list(first_move[target]), "interact": True}

    # Choose a reachable vantage point by travel cost and new 5x5 coverage.
    # Revisiting old terrain around a wall-change boundary also has value.
    best_target = None
    best_score = float("inf")
    for pos, cost in distance.items():
        if pos == start:
            continue
        gain = 0.0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                q = (pos[0] + dx, pos[1] + dy)
                if q in outside:
                    continue
                if q not in terrain:
                    gain += 1.0
                elif seen.get(q, step) // 25 < step // 25:
                    gain += 0.12
        if gain <= 0:
            continue
        score = cost - 0.9 * gain + 0.4 * visits.get(pos, 0)
        if score < best_score:
            best_score, best_target = score, pos
    if best_target is not None:
        return {"move": list(first_move[best_target]), "interact": True}

    # If a known objective is temporarily unsafe, wait or retreat locally.
    if goals:
        target = min(goals, key=lambda p:
                     max(abs(p[0] - start[0]), abs(p[1] - start[1])))
        def fallback_score(move):
            p = (start[0] + move[0], start[1] + move[1])
            return (max(abs(p[0] - target[0]), abs(p[1] - target[1]))
                    + 0.03 * visits.get(p, 0))
    else:
        def fallback_score(move):
            p = (start[0] + move[0], start[1] + move[1])
            return visits.get(p, 0) + (0.25 if move == (0, 0) else 0)
    move = min(sorted(allowed), key=fallback_score)
    return {"move": list(move), "interact": True}
# EVOLVE-BLOCK-END