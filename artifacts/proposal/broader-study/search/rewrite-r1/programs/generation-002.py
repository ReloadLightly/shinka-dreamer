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

_MOVES = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (0, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)
_CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _possible(memory, p):
    if p in memory["outside"]:
        return False
    left, right, top, bottom = memory["extent"]
    return left <= p[0] <= right and top <= p[1] <= bottom


def _passable(memory, p, opened=False, permit_door=False,
              unknown=False):
    if not _possible(memory, p):
        return False
    cell = memory["terrain"].get(p)
    if cell is None:
        return unknown
    if cell == 1:
        return False
    if cell == 3:
        return opened or permit_door
    return True


def _edge(memory, p, q, opened=False, permit_door=False,
          unknown=False):
    if not _passable(memory, q, opened, permit_door, unknown):
        return False
    dx, dy = q[0] - p[0], q[1] - p[1]
    if dx and dy:
        # A closed door must first be approached cardinally.
        terrain = memory["terrain"]
        if not opened and (terrain.get(p) == 3 or terrain.get(q) == 3):
            return False
        if not _passable(memory, (p[0] + dx, p[1]),
                         opened, False, unknown):
            return False
        if not _passable(memory, (p[0], p[1] + dy),
                         opened, False, unknown):
            return False
    return True


def _spread(memory, field, opened=None):
    """Propagate anonymous occupancy mass through attempted moves."""
    if opened is None:
        opened = memory["door_open"]
    result = {}
    for p, mass in field.items():
        if mass < 0.001:
            continue
        part = mass / 9.0
        for move in _MOVES:
            q = _add(p, move)
            if not _edge(memory, p, q, opened, False, True):
                q = p
            result[q] = result.get(q, 0.0) + part
    return {p: mass for p, mass in result.items() if mass >= 0.001}


def _terrain_cell(observation, row, col, fallback):
    terrain = observation.get("terrain")
    if terrain is not None:
        try:
            value = terrain[row][col]
            if value in (0, 1, 2, 3, 4):
                return value
        except (IndexError, KeyError, TypeError):
            pass
    return fallback


def world_model_step(memory: dict, local_obs: dict,
                     last_action: dict) -> dict:
    if memory is None or "terrain" not in memory:
        memory = {
            "believed_map": {},
            "terrain": {},
            "seen": {},
            "outside": set(),
            "origins": [(x, y) for x in range(-14, 1)
                        for y in range(-14, 1)],
            "extent": (-14, 14, -14, 14),
            "agent_pos": (0, 0),
            "keys_collected": set(),
            "keys": 0,
            "door_open": False,
            "step": 0,
            "occupancy": {},
            "enemies": set(),
            "dynamic": set(),
            "visits": {},
            "goal": None,
        }

    previous_step = memory["step"]
    step = int(local_obs.get("step", previous_step + 1))
    feedback = local_obs.get("feedback", {})
    displacement = feedback.get("displacement", (0, 0))
    ax, ay = memory["agent_pos"]
    position = (ax + displacement[0], ay + displacement[1])

    # Predict between observations, then condition on the new observation.
    occupancy = memory["occupancy"]
    elapsed = max(0, step - previous_step)
    for _ in range(min(elapsed, 6)):
        occupancy = _spread(memory, occupancy)
    if elapsed > 6:
        occupancy = {}

    memory["agent_pos"] = position
    memory["step"] = step
    memory["keys"] = local_obs.get("keys", memory["keys"])
    memory["door_open"] = bool(
        local_obs.get("door_open", memory["door_open"])
    )
    memory["visits"][position] = memory["visits"].get(position, 0) + 1

    if feedback.get("collected"):
        memory["keys_collected"].add(position)

    grid = local_obs["grid"]
    visible = {}
    constraints = []
    terrain = memory["terrain"]
    seen = memory["seen"]
    enemies = set()

    for row in range(5):
        for col in range(5):
            p = (position[0] + col - 2, position[1] + row - 2)
            cell = grid[row][col]
            valid = cell != -1
            constraints.append((p, valid))
            if not valid:
                memory["outside"].add(p)
                continue

            visible[p] = cell
            old = terrain.get(p)
            fallback = cell if cell != 5 else (
                old if old is not None else 0
            )
            base = _terrain_cell(local_obs, row, col, fallback)
            if base not in (0, 1, 2, 3, 4):
                base = old if old is not None else 0
            if p in memory["keys_collected"] and base == 2:
                base = 0

            if old in (0, 1) and base in (0, 1) and old != base:
                memory["dynamic"].add(p)
            terrain[p] = base
            seen[p] = step

            if cell == 5:
                enemies.add(p)

    origins = []
    for left, top in memory["origins"]:
        consistent = True
        for (x, y), valid in constraints:
            inside = left <= x < left + 15 and top <= y < top + 15
            if inside != valid:
                consistent = False
                break
        if consistent:
            origins.append((left, top))
    if origins:
        memory["origins"] = origins
        memory["extent"] = (
            min(p[0] for p in origins),
            max(p[0] for p in origins) + 14,
            min(p[1] for p in origins),
            max(p[1] for p in origins) + 14,
        )

    # Negative sightings erase old occupancy; positive sightings have no IDs.
    occupancy = {
        p: mass for p, mass in occupancy.items()
        if p not in visible and _possible(memory, p)
        and terrain.get(p) != 1
    }
    remaining = max(0, 3 - len(enemies))
    total = sum(occupancy.values())
    if total > remaining and total > 0:
        scale = remaining / total
        occupancy = {p: mass * scale for p, mass in occupancy.items()}
    for p in enemies:
        occupancy[p] = 1.0
    memory["occupancy"] = occupancy
    memory["enemies"] = enemies
    memory["visible"] = visible

    # Export present-state evidence, never stale enemy markers.
    believed = {}
    epoch = step // 25
    for p, cell in terrain.items():
        if not _possible(memory, p):
            continue
        if p in visible:
            believed[p] = visible[p]
            continue
        if p in memory["dynamic"] and seen[p] // 25 != epoch:
            continue
        if occupancy.get(p, 0.0) > 0.20:
            continue
        believed[p] = cell
    memory["believed_map"] = believed
    return memory


def _penalties(memory):
    mass = memory["occupancy"]
    nearby = {}
    for p, value in mass.items():
        nearby[p] = nearby.get(p, 0.0) + 2.5 * value
        for move in _MOVES:
            if move == (0, 0):
                continue
            q = _add(p, move)
            nearby[q] = nearby.get(q, 0.0) + 0.30 * value

    penalties = {}
    epoch = memory["step"] // 25
    for p in memory["terrain"]:
        uncertainty = (
            0.6 if p in memory["dynamic"]
            and memory["seen"][p] // 25 != epoch else 0.0
        )
        penalties[p] = (
            nearby.get(p, 0.0)
            + 0.035 * min(memory["visits"].get(p, 0), 12)
            + uncertainty
        )
    return penalties


def _distances(memory, start, penalties):
    opened = memory["door_open"]
    permit_door = memory["keys"] >= 2
    distance = {start: 0.0}
    queue = [(0.0, start)]
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distance.get(p):
            continue
        for move in _MOVES:
            if move == (0, 0):
                continue
            q = _add(p, move)
            if not _edge(memory, p, q, opened, permit_door):
                continue
            weight = 1.0 + 0.5 * (
                penalties.get(p, 0.0) + penalties.get(q, 0.0)
            )
            new_cost = cost + weight
            if new_cost < distance.get(q, float("inf")):
                distance[q] = new_cost
                heapq.heappush(queue, (new_cost, q))
    return distance


def _information(memory, p):
    terrain = memory["terrain"]
    epoch = memory["step"] // 25
    gain = 0.0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            q = (p[0] + dx, p[1] + dy)
            if not _possible(memory, q):
                continue
            if q not in terrain:
                gain += 1.0
            elif memory["seen"][q] // 25 != epoch:
                gain += 0.65 if q in memory["dynamic"] else 0.035
    return gain


def _choose_goal(memory, distance):
    terrain = memory["terrain"]
    position = memory["agent_pos"]
    keys = memory["keys"]

    if keys < 2:
        targets = [
            p for p in distance if terrain.get(p) == 2
            and p not in memory["keys_collected"]
        ]
    else:
        targets = [p for p in distance if terrain.get(p) == 4]
        if not targets and not memory["door_open"]:
            targets = [p for p in distance if terrain.get(p) == 3]

    if targets:
        return min(targets, key=lambda p: (
            distance[p], memory["visits"].get(p, 0), p
        ))

    best_goal = None
    best_score = float("inf")
    old_goal = memory.get("goal")
    for p, cost in distance.items():
        if p == position:
            continue
        information = _information(memory, p)
        if information < 0.15:
            continue
        score = (
            cost - 2.7 * math.sqrt(information)
            + 0.22 * memory["visits"].get(p, 0)
            - (0.65 if p == old_goal else 0.0)
        )
        if score < best_score:
            best_score, best_goal = score, p
    return best_goal


def _collision_probabilities(memory, candidates, opened):
    survival = {p: 1.0 for p in candidates}
    future_mass = {}

    for enemy in memory["enemies"]:
        distribution = _spread(memory, {enemy: 1.0}, opened)
        for p in candidates:
            collision = 1.0 if p == enemy else distribution.get(p, 0.0)
            survival[p] *= max(0.0, 1.0 - collision)
        for p, probability in distribution.items():
            future_mass[p] = future_mass.get(p, 0.0) + probability

    two_step = _spread(memory, future_mass, opened)
    return (
        {p: 1.0 - value for p, value in survival.items()},
        two_step,
    )


def planner(memory: dict, local_obs: dict) -> dict:
    position = memory["agent_pos"]
    terrain = memory["terrain"]

    # Interaction is harmless elsewhere and collects even a diagonally
    # reached key or a key on the current tile.
    opened = memory["door_open"]
    if memory["keys"] >= 2:
        opened = opened or any(
            terrain.get(_add(position, d)) == 3 for d in _CARDINAL
        )

    actions = []
    for move in _MOVES:
        q = _add(position, move)
        if move == (0, 0) or _edge(memory, position, q, opened):
            actions.append((move, q))
    if not actions:
        return {"move": [0, 0], "interact": True}

    candidates = [q for _, q in actions]
    collision, future = _collision_probabilities(
        memory, candidates, opened
    )
    minimum_risk = min(collision.values())

    # Prefer a guaranteed safe displacement whenever one exists.
    safe_actions = [
        (move, q) for move, q in actions
        if collision[q] <= minimum_risk + 1e-9
    ]

    if (terrain.get(position) == 2
            and position not in memory["keys_collected"]
            and collision[position] <= minimum_risk + 1e-9):
        return {"move": [0, 0], "interact": True}

    penalties = _penalties(memory)
    forward = _distances(memory, position, penalties)
    goal = _choose_goal(memory, forward)
    memory["goal"] = goal
    backward = _distances(memory, goal, penalties) if goal else {}

    best = None
    best_score = float("inf")
    for move, q in safe_actions:
        if goal is not None:
            remaining = backward.get(q, float("inf"))
            travel = 1.0 + 0.5 * (
                penalties.get(position, 0.0) + penalties.get(q, 0.0)
            )
            score = remaining + travel
            if move == (0, 0):
                score += 0.30
        else:
            # With no frontier left, prefer a quiet refuge while waiting
            # for a wall change to reveal another route.
            score = (
                2.0 * penalties.get(q, 0.0)
                + 0.08 * memory["visits"].get(q, 0)
                - 0.4 * _information(memory, q)
            )
            if move == (0, 0):
                score -= 0.20

        exits = sum(
            _edge(memory, q, _add(q, d), opened)
            for d in _MOVES if d != (0, 0)
        )
        score += 3.5 * future.get(q, 0.0) + 0.20 / (1 + exits)
        score += 100.0 * collision[q]

        # Stable tie breaking discourages immediate reversals.
        if move == memory.get("previous_move"):
            score -= 0.025
        if best is None or score < best_score:
            best_score, best = score, move

    if best is None:
        best = (0, 0)
    memory["previous_move"] = best
    return {"move": [best[0], best[1]], "interact": True}
# EVOLVE-BLOCK-END
