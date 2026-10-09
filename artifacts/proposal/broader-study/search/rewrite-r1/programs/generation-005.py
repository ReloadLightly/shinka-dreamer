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
import heapq

_MOVES = (
    (0, -1), (1, 0), (0, 1), (-1, 0),
    (1, -1), (1, 1), (-1, 1), (-1, -1),
)


def _inside(memory, pos):
    x, y = pos
    xmin, xmax, ymin, ymax = memory["bounds"]
    return (
        -14 <= x <= 14 and -14 <= y <= 14
        and (xmin is None or x >= xmin)
        and (xmax is None or x <= xmax)
        and (ymin is None or y >= ymin)
        and (ymax is None or y <= ymax)
        and pos not in memory["outside"]
    )


def _terrain_value(terrain, row, col):
    """Accept a numeric local terrain matrix when one is provided."""
    if isinstance(terrain, (list, tuple)) and row < len(terrain):
        line = terrain[row]
        if isinstance(line, (list, tuple)) and col < len(line):
            value = line[col]
            if isinstance(value, int) and 0 <= value <= 4:
                return value
    return None


def world_model_step(memory: dict, local_obs: dict, last_action: dict) -> dict:
    """Keep navigation memory separate from current-state reconstruction."""
    if memory is None or "terrain_map" not in memory:
        memory = {
            "believed_map": {},
            "terrain_map": {},
            "last_seen": {},
            "agent_pos": (0, 0),
            "enemy_sightings": {},
            "visible_enemies": set(),
            "keys_collected": set(),
            "visits": {},
            "outside": set(),
            "bounds": [None, None, None, None],
            "dynamic_cells": set(),
            "keys": 0,
            "door_open": False,
            "step": 0,
            "exploration_target": None,
        }

    feedback = local_obs.get("feedback", {})
    dx, dy = feedback.get("displacement", (0, 0))
    ax, ay = memory["agent_pos"]
    ax, ay = ax + dx, ay + dy
    position = (ax, ay)
    memory["agent_pos"] = position

    step = int(local_obs.get("step", memory["step"]))
    memory["step"] = step
    memory["keys"] = local_obs.get("keys", memory["keys"])
    memory["door_open"] = bool(
        local_obs.get("door_open", memory["door_open"])
    )
    memory["visits"][position] = memory["visits"].get(position, 0) + 1
    if feedback.get("collected"):
        memory["keys_collected"].add(position)

    grid = local_obs["grid"]
    terrain = local_obs.get("terrain")
    terrain_map = memory["terrain_map"]
    last_seen = memory["last_seen"]
    sightings = memory["enemy_sightings"]
    observed = {}
    enemies = set()

    for oy in range(-2, 3):
        for ox in range(-2, 3):
            pos = (ax + ox, ay + oy)
            cell = grid[oy + 2][ox + 2]
            if cell == -1:
                memory["outside"].add(pos)
                continue

            observed[pos] = cell
            previous = terrain_map.get(pos)
            base = _terrain_value(terrain, oy + 2, ox + 2)
            if cell != 5:
                base = cell
            elif base is None:
                # Occupancy does not erase a remembered key, gate or exit.
                base = previous if previous in (2, 3, 4) else 0

            if (
                previous in (0, 1)
                and base in (0, 1)
                and previous != base
            ):
                memory["dynamic_cells"].add(pos)

            terrain_map[pos] = base
            last_seen[pos] = step
            if cell == 5:
                enemies.add(pos)
                sightings[pos] = step
            else:
                sightings.pop(pos, None)

    # Center-row/column out-of-bounds observations establish exact borders.
    bounds = memory["bounds"]
    left = [
        ax + ox for ox in (-2, -1)
        if grid[2][ox + 2] == -1
    ]
    right = [
        ax + ox for ox in (1, 2)
        if grid[2][ox + 2] == -1
    ]
    above = [
        ay + oy for oy in (-2, -1)
        if grid[oy + 2][2] == -1
    ]
    below = [
        ay + oy for oy in (1, 2)
        if grid[oy + 2][2] == -1
    ]
    if left:
        bounds[0] = max(left) + 1
        bounds[1] = bounds[0] + 14
    elif right:
        bounds[1] = min(right) - 1
        bounds[0] = bounds[1] - 14
    if above:
        bounds[2] = max(above) + 1
        bounds[3] = bounds[2] + 14
    elif below:
        bounds[3] = min(below) - 1
        bounds[2] = bounds[3] - 14

    for pos, seen in list(sightings.items()):
        if step - seen > 6:
            del sightings[pos]

    memory["visible_enemies"] = enemies
    memory["observed"] = observed

    # Old free cells may now contain anonymous enemies. Old walls may toggle.
    # Report exact observations plus walls known to persist in this phase.
    phase = step // 25
    believed = {
        pos: 1
        for pos, cell in terrain_map.items()
        if cell == 1
        and last_seen.get(pos, -25) // 25 == phase
        and _inside(memory, pos)
    }
    believed.update(observed)
    memory["believed_map"] = believed
    return memory


def _passable(memory, pos, optimistic, door_open):
    if not _inside(memory, pos):
        return False
    cell = memory["terrain_map"].get(pos)
    if cell == 1:
        return (
            optimistic
            and memory["last_seen"].get(pos, 0) // 25
            != memory["step"] // 25
        )
    if cell == 3:
        return door_open
    return cell is not None or optimistic


def _edge(memory, source, destination, optimistic, door_open,
          allow_opening=False):
    dx = destination[0] - source[0]
    dy = destination[1] - source[1]

    can_open = allow_opening and memory["keys"] >= 2
    effective_open = door_open
    if can_open and not door_open:
        effective_open = any(
            memory["terrain_map"].get(
                (source[0] + mx, source[1] + my)
            ) == 3
            for mx, my in _MOVES[:4]
        )

    if not _passable(memory, destination, optimistic, effective_open):
        return False
    if dx and dy:
        if not _passable(
            memory, (source[0] + dx, source[1]),
            optimistic, effective_open
        ):
            return False
        if not _passable(
            memory, (source[0], source[1] + dy),
            optimistic, effective_open
        ):
            return False
    return True


def _contact_risk(memory, destination, door_open):
    """Immediate pre-move contact plus uniform enemy attempted moves."""
    survival = 1.0
    for enemy in memory["visible_enemies"]:
        if destination == enemy:
            return 1.0
        dx = destination[0] - enemy[0]
        dy = destination[1] - enemy[1]
        if (
            abs(dx) <= 1 and abs(dy) <= 1
            and _edge(
                memory, enemy, destination, False, door_open, False
            )
        ):
            survival *= 8.0 / 9.0
    return 1.0 - survival


def _information(memory, pos):
    gain = 0.0
    step = memory["step"]
    phase = step // 25
    terrain_map = memory["terrain_map"]
    last_seen = memory["last_seen"]
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            q = (pos[0] + dx, pos[1] + dy)
            if not _inside(memory, q):
                continue
            if q not in terrain_map:
                gain += 1.0
            elif (
                terrain_map[q] in (0, 1)
                and last_seen[q] // 25 != phase
            ):
                gain += 0.65 if q in memory["dynamic_cells"] else 0.18
    return gain


def planner(memory: dict, local_obs: dict) -> dict:
    """Pursue objectives along safe routes, otherwise reveal useful terrain."""
    start = memory["agent_pos"]
    terrain_map = memory["terrain_map"]
    step = memory["step"]
    keys = memory["keys"]
    door_open = memory["door_open"]

    # Interaction is useful for both destination keys and adjacent gates.
    action = {"move": [0, 0], "interact": True}

    opening_now = door_open or (
        keys >= 2 and any(
            terrain_map.get((start[0] + dx, start[1] + dy)) == 3
            for dx, dy in _MOVES[:4]
        )
    )

    candidates = []
    wait_risk = _contact_risk(memory, start, opening_now)
    for dx, dy in _MOVES:
        pos = (start[0] + dx, start[1] + dy)
        if _edge(memory, start, pos, False, door_open, True):
            candidates.append(
                (_contact_risk(memory, pos, opening_now), (dx, dy), pos)
            )

    if not candidates:
        return action

    minimum_risk = min([wait_risk] + [item[0] for item in candidates])
    allowed = [
        item for item in candidates
        if item[0] <= minimum_risk + 1e-12
    ]
    if not allowed:
        return action

    # Bounded optimistic search can connect discovered regions through unknown
    # terrain, but the actual first displacement always uses the current view.
    xs = [pos[0] for pos in terrain_map]
    ys = [pos[1] for pos in terrain_map]
    xmin, xmax, ymin, ymax = memory["bounds"]
    low_x = max(-14, min(xs) - 3) if xmin is None else xmin
    high_x = min(14, max(xs) + 3) if xmax is None else xmax
    low_y = max(-14, min(ys) - 3) if ymin is None else ymin
    high_y = min(14, max(ys) + 3) if ymax is None else ymax

    # Anonymous evidence affects route preference, never exported occupancy.
    danger = {}
    for enemy, seen in memory["enemy_sightings"].items():
        age = step - seen
        weight = 1.7 if age == 0 else 0.45 * (0.65 ** age)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                pos = (enemy[0] + dx, enemy[1] + dy)
                danger[pos] = danger.get(pos, 0.0) + weight

    visits = memory["visits"]

    def enter_cost(pos):
        cell = terrain_map.get(pos)
        if cell is None:
            cost = 1.8
        elif cell == 1:
            cost = 4.0
        elif memory["last_seen"].get(pos, step) // 25 != step // 25:
            cost = 1.2
        else:
            cost = 1.0
        return cost + danger.get(pos, 0.0) + 0.04 * min(
            visits.get(pos, 0), 6
        )

    distance = {start: 0.0}
    first_move = {}
    heap = []
    for risk, move, pos in allowed:
        cost = enter_cost(pos)
        if cost < distance.get(pos, float("inf")):
            distance[pos] = cost
            first_move[pos] = move
            heapq.heappush(heap, (cost, pos))

    while heap:
        cost, pos = heapq.heappop(heap)
        if cost != distance.get(pos):
            continue
        for dx, dy in _MOVES:
            q = (pos[0] + dx, pos[1] + dy)
            if not (low_x <= q[0] <= high_x and low_y <= q[1] <= high_y):
                continue
            if q in memory["visible_enemies"]:
                continue
            if not _edge(memory, pos, q, True, door_open, True):
                continue
            new_cost = cost + enter_cost(q)
            if new_cost < distance.get(q, float("inf")):
                distance[q] = new_cost
                first_move[q] = first_move[pos]
                heapq.heappush(heap, (new_cost, q))

    if (
        terrain_map.get(start) == 2
        and start not in memory["keys_collected"]
        and wait_risk <= minimum_risk + 1e-12
    ):
        return action

    targets = []
    if keys < 2:
        targets = [
            pos for pos, cell in terrain_map.items()
            if cell == 2 and pos not in memory["keys_collected"]
        ]
    elif not door_open:
        targets = [
            pos for pos, cell in terrain_map.items() if cell == 3
        ]
        if not targets:
            targets = [
                pos for pos, cell in terrain_map.items() if cell == 4
            ]
    else:
        targets = [
            pos for pos, cell in terrain_map.items() if cell == 4
        ]

    reachable_targets = [
        pos for pos in targets if pos in first_move
    ]
    if reachable_targets:
        target = min(reachable_targets, key=lambda pos: distance[pos])
        action["move"] = list(first_move[target])
        memory["exploration_target"] = None
        return action

    best_target = None
    best_score = -1.0
    old_target = memory.get("exploration_target")
    for pos, cost in distance.items():
        if pos not in first_move or pos not in terrain_map:
            continue
        cell = terrain_map[pos]
        if cell == 1 or (cell == 3 and keys < 2 and not door_open):
            continue

        gain = _information(memory, pos)
        if gain <= 0:
            continue
        score = gain / (cost + 3.0 + 0.3 * visits.get(pos, 0))
        if pos == old_target:
            score *= 1.08
        if score > best_score:
            best_score = score
            best_target = pos

    if best_target is not None:
        memory["exploration_target"] = best_target
        action["move"] = list(first_move[best_target])
        return action

    # Revisit old areas when exploration is exhausted; updated wall phases can
    # reveal routes that were previously closed.
    if first_move:
        target = max(
            first_move,
            key=lambda pos: (
                (step - memory["last_seen"].get(pos, step))
                / (distance[pos] + 4.0)
                - 0.025 * visits.get(pos, 0)
            ),
        )
        if step - memory["last_seen"].get(target, step) >= 8:
            action["move"] = list(first_move[target])
            return action

    # If there is no useful route, waiting avoids unnecessary exposure.
    if wait_risk <= minimum_risk + 1e-12:
        return action

    _, move, _ = min(
        allowed,
        key=lambda item: (
            danger.get(item[2], 0.0),
            visits.get(item[2], 0),
        ),
    )
    action["move"] = list(move)
    return action
# EVOLVE-BLOCK-END
