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

    grid = local_obs["grid"]
    terrain = local_obs.get("terrain")
    ground = memory.setdefault("terrain_map", {})
    observed = memory.setdefault("observed_positions", set())
    seen_at = memory.setdefault("terrain_seen_at", {})
    dynamic = memory.setdefault("dynamic_cells", set())
    visits = memory.setdefault("visits", {})
    visits[(ax, ay)] = visits.get((ax, ay), 0) + 1
    memory["enemy_sightings"] = {}
    visible = {}
    step = memory["step"]

    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cell = grid[dy + 2][dx + 2]
            pos = (ax + dx, ay + dy)
            observed.add(pos)
            if cell == -1:
                continue
            visible[pos] = cell
            base = cell
            if isinstance(terrain, (list, tuple)):
                row = terrain[dy + 2]
                if isinstance(row, (list, tuple)):
                    base = row[dx + 2]
            if base in (-1, 5):
                base = ground.get(pos, 0)
            if pos in memory["keys_collected"] and base == 2:
                base = 0
            old = ground.get(pos)
            if old in (0, 1) and base in (0, 1) and old != base:
                dynamic.add(pos)
            ground[pos] = base
            seen_at[pos] = step
            if cell == 5:
                memory["enemy_sightings"][pos] = step

    # Retain terrain for navigation, but do not report a known toggling cell
    # as certain after another wall-update boundary has passed.
    memory["believed_map"] = {
        pos: cell
        for pos, cell in ground.items()
        if pos not in dynamic or seen_at[pos] // 25 == step // 25
    }
    memory["believed_map"].update(visible)
    return memory


def planner(memory: dict, local_obs: dict) -> dict:
    """Search remembered terrain, balancing progress against immediate danger."""
    import heapq

    ground = memory.get("terrain_map", memory["believed_map"])
    start = memory["agent_pos"]
    keys = memory.get("keys", 0)
    door_open = memory.get("door_open", False)
    visits = memory.get("visits", {})
    enemies = set(memory.get("enemy_sightings", {}))
    moves = (
        (0, -1), (-1, 0), (1, 0), (0, 1),
        (-1, -1), (1, -1), (-1, 1), (1, 1),
    )

    def passable(pos, unknown=False):
        cell = ground.get(pos)
        if cell is None:
            return unknown
        return cell not in (-1, 1) and (
            cell != 3 or door_open or keys >= 2
        )

    def legal(origin, move, unknown=False):
        dx, dy = move
        x, y = origin
        dest = (x + dx, y + dy)
        if not passable(dest, unknown):
            return False
        if dx and dy:
            sides = ((x + dx, y), (x, y + dy))
            if not all(passable(p, unknown) for p in sides):
                return False
            # A closed door needs a cardinal approach to open it.
            if not door_open and any(
                ground.get(p) == 3 for p in (dest,) + sides
            ):
                return False
        return True

    # A visible enemy contributes 1/9 for each attempted displacement.
    # Blocked attempts accumulate probability at its current position.
    danger = {}
    for enemy in enemies:
        for move in moves + ((0, 0),):
            dest = (enemy[0] + move[0], enemy[1] + move[1])
            if not legal(enemy, move, unknown=True):
                dest = enemy
            danger[dest] = danger.get(dest, 0.0) + 1.0 / 9.0
    for enemy in enemies:
        danger[enemy] = 1.0  # Contact before enemy movement is also fatal.

    if ground.get(start) == 2 and danger.get(start, 0.0) == 0:
        return {"move": [0, 0], "interact": True}

    distance = {start: 0.0}
    first_move = {}
    queue = [(0.0, start)]
    while queue:
        cost, pos = heapq.heappop(queue)
        if cost > distance[pos]:
            continue
        for move in moves:
            if not legal(pos, move):
                continue
            dest = (pos[0] + move[0], pos[1] + move[1])
            if pos == start and dest in enemies:
                continue
            risk = danger.get(dest, 0.0)
            edge = 1.0 + 0.06 * min(visits.get(dest, 0), 15)
            edge += risk * (2000.0 if pos == start else 12.0)
            candidate = cost + edge
            if candidate < distance.get(dest, float("inf")):
                distance[dest] = candidate
                first_move[dest] = move if pos == start else first_move[pos]
                heapq.heappush(queue, (candidate, dest))

    reachable = [pos for pos in distance if pos != start]
    targets = []
    if keys < 2:
        targets = [
            pos for pos in reachable
            if ground.get(pos) == 2
            and pos not in memory.get("keys_collected", set())
        ]
    elif not door_open:
        targets = [pos for pos in reachable if ground.get(pos) == 3]
        # Opening is performed before movement, so an adjacent door can
        # also be opened while choosing a safer displacement.
        if any(
            ground.get((start[0] + dx, start[1] + dy)) == 3
            for dx, dy in moves[:4]
        ):
            targets = [pos for pos in reachable if ground.get(pos) == 4] or targets
    else:
        targets = [pos for pos in reachable if ground.get(pos) == 4]

    if targets:
        goal = min(targets, key=lambda pos: (distance[pos], pos))
    elif reachable:
        observed = memory.get("observed_positions", set(ground))
        frontier = []
        for pos in reachable:
            gain = sum(
                (pos[0] + dx, pos[1] + dy) not in observed
                for dy in range(-2, 3)
                for dx in range(-2, 3)
            )
            if gain:
                score = distance[pos] - 1.1 * gain + 2.0 * visits.get(pos, 0)
                frontier.append((score, pos))
        if frontier:
            goal = min(frontier)[1]
        else:
            # Revisit neglected regions when no new frontier is reachable;
            # later wall toggles may expose another route.
            goal = min(
                reachable,
                key=lambda pos: (
                    distance[pos] + 3.0 * visits.get(pos, 0), pos
                ),
            )
    else:
        goal = None

    move = first_move[goal] if goal is not None else (0, 0)
    destination = (start[0] + move[0], start[1] + move[1])
    stay_risk = danger.get(start, 0.0)
    if danger.get(destination, 0.0) > stay_risk + 1e-9:
        move = (0, 0)

    # If the chosen action is dangerous, use the least exposed legal action.
    destination = (start[0] + move[0], start[1] + move[1])
    if danger.get(destination, 0.0) > 0:
        options = [(0, 0)] + [
            candidate for candidate in moves if legal(start, candidate)
        ]
        move = min(
            options,
            key=lambda candidate: (
                danger.get(
                    (start[0] + candidate[0], start[1] + candidate[1]), 0.0
                ),
                candidate != move,
                visits.get(
                    (start[0] + candidate[0], start[1] + candidate[1]), 0
                ),
            ),
        )
    return {"move": list(move), "interact": True}
# EVOLVE-BLOCK-END