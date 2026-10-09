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
from collections import deque
from itertools import product

_MOVES = (
    (0, 0), (0, -1), (1, 0), (0, 1), (-1, 0),
    (1, -1), (1, 1), (-1, 1), (-1, -1),
)
_CARDINAL = _MOVES[1:5]


def _inside(memory, pos):
    x, y = pos
    left, right, top, bottom = memory.get(
        "domain", (-14, 14, -14, 14)
    )
    return (
        left <= x <= right
        and top <= y <= bottom
        and pos not in memory.get("void", ())
    )


def _passable(memory, pos, opened=(), unknown=False, optimistic=False):
    if not _inside(memory, pos):
        return False
    cell = memory["terrain"].get(pos)
    if cell is None:
        return unknown
    if cell == 1:
        return (
            optimistic
            and memory["seen"].get(pos, -25) // 25
            != memory["step"] // 25
        )
    if cell == 3:
        return memory["door_open"] or pos in opened
    return True


def _opened_here(memory, pos, previous=()):
    opened = set(previous)
    if memory["keys"] >= 2 and not memory["door_open"]:
        x, y = pos
        for dx, dy in _CARDINAL:
            target = (x + dx, y + dy)
            if memory["terrain"].get(target) == 3:
                opened.add(target)
    return opened


def _resolve(
    memory, pos, move, opened=(), unknown=False, optimistic=False
):
    dx, dy = move
    if dx == 0 and dy == 0:
        return pos
    x, y = pos
    target = (x + dx, y + dy)
    if not _passable(memory, target, opened, unknown, optimistic):
        return pos
    if dx and dy:
        if not _passable(
            memory, (x + dx, y), opened, unknown, optimistic
        ):
            return pos
        if not _passable(
            memory, (x, y + dy), opened, unknown, optimistic
        ):
            return pos
    return target


def _agent_moves(memory, pos, optimistic=False, previous=()):
    opened = _opened_here(memory, pos, previous)
    result = []
    for move in _MOVES:
        target = _resolve(
            memory, pos, move, opened, False, optimistic
        )
        if move == (0, 0) or target != pos:
            result.append((move, target))
    return result


def _enemy_outcomes(memory, pos, opened=()):
    counts = {}
    for move in _MOVES:
        target = _resolve(
            memory, pos, move, opened, True, True
        )
        counts[target] = counts.get(target, 0) + 1
    return tuple((p, count / 9.0) for p, count in counts.items())


def _make_kernel(memory):
    left, right, top, bottom = memory["domain"]
    kernel = {}
    for y in range(top, bottom + 1):
        for x in range(left, right + 1):
            pos = (x, y)
            if _passable(memory, pos, (), True, True):
                kernel[pos] = _enemy_outcomes(memory, pos)
    return kernel


def _spread(memory, mass):
    result = {}
    kernel = memory["kernel"]
    for pos, amount in mass.items():
        for target, probability in kernel.get(pos, ()):
            result[target] = (
                result.get(target, 0.0) + amount * probability
            )
    return result


def _rebalance(weights, capacity):
    """Conserve residual anonymous mass with a one-cell occupancy cap."""
    result = {}
    free = dict(weights)
    remaining = min(float(capacity), float(len(free)))
    while free and remaining > 1e-12:
        total = sum(free.values())
        if total <= 0.0:
            amount = remaining / len(free)
            result.update((p, amount) for p in free)
            break

        scale = remaining / total
        saturated = [
            p for p, amount in free.items() if amount * scale > 1.0
        ]
        if not saturated:
            result.update(
                (p, amount * scale)
                for p, amount in free.items()
                if amount > 0.0
            )
            break

        for pos in saturated:
            result[pos] = 1.0
            remaining -= 1.0
            del free[pos]
    return result


def world_model_step(memory: dict, local_obs: dict, last_action: dict) -> dict:
    if memory is None:
        memory = {
            "agent_pos": (0, 0),
            "terrain": {},
            "seen": {},
            "void": set(),
            "bounds": {},
            "visits": {},
            "keys_collected": set(),
            "enemy_mass": {},
            "step": int(local_obs["step"]),
            "door_open": False,
            "keys": 0,
            "camping": False,
            "finish_now": False,
        }

    previous_step = memory["step"]
    feedback = local_obs["feedback"]
    dx, dy = feedback["displacement"]
    x, y = memory["agent_pos"]
    agent = (x + dx, y + dy)
    memory["agent_pos"] = agent
    memory["step"] = int(local_obs["step"])
    memory["keys"] = int(local_obs["keys"])
    memory["door_open"] = bool(local_obs["door_open"])
    memory["health"] = local_obs.get("health")
    memory["visits"][agent] = memory["visits"].get(agent, 0) + 1

    terrain = memory["terrain"]
    seen = memory["seen"]
    void = memory["void"]
    bounds = memory["bounds"]
    step = memory["step"]
    grid = local_obs["grid"]
    underlying = local_obs.get("terrain")
    has_terrain = (
        isinstance(underlying, (list, tuple))
        and len(underlying) == 5
        and all(
            isinstance(row, (list, tuple)) and len(row) == 5
            for row in underlying
        )
    )

    visible = {}
    occupied = set()
    x, y = agent

    for oy in range(-2, 3):
        for ox in range(-2, 3):
            pos = (x + ox, y + oy)
            cell = grid[oy + 2][ox + 2]
            if cell == -1:
                void.add(pos)
                if oy == 0:
                    if ox < 0:
                        bounds["left"] = max(
                            bounds.get("left", -14), pos[0] + 1
                        )
                    elif ox > 0:
                        bounds["right"] = min(
                            bounds.get("right", 14), pos[0] - 1
                        )
                if ox == 0:
                    if oy < 0:
                        bounds["top"] = max(
                            bounds.get("top", -14), pos[1] + 1
                        )
                    elif oy > 0:
                        bounds["bottom"] = min(
                            bounds.get("bottom", 14), pos[1] - 1
                        )
                continue

            void.discard(pos)
            visible[pos] = cell
            base = underlying[oy + 2][ox + 2] if has_terrain else cell
            if base not in (0, 1, 2, 3, 4):
                base = terrain.get(pos, 0)
                if base == 1:
                    base = 0
            terrain[pos] = base
            seen[pos] = step
            if cell == 5:
                occupied.add(pos)

    if feedback.get("collected"):
        memory["keys_collected"].add(agent)
    for pos in memory["keys_collected"]:
        terrain[pos] = 0

    if memory["door_open"]:
        for pos, cell in list(terrain.items()):
            if cell == 3:
                terrain[pos] = 0

    # A observed edge fixes the complete extent of that axis.
    if "left" in bounds:
        bounds["right"] = bounds["left"] + 14
    elif "right" in bounds:
        bounds["left"] = bounds["right"] - 14
    if "top" in bounds:
        bounds["bottom"] = bounds["top"] + 14
    elif "bottom" in bounds:
        bounds["top"] = bounds["bottom"] - 14

    # Before an edge is observed, retain every possible arena coordinate.
    known = list(terrain)
    minimum_x = min(p[0] for p in known)
    maximum_x = max(p[0] for p in known)
    minimum_y = min(p[1] for p in known)
    maximum_y = max(p[1] for p in known)
    memory["domain"] = (
        bounds.get("left", maximum_x - 14),
        bounds.get("right", minimum_x + 14),
        bounds.get("top", maximum_y - 14),
        bounds.get("bottom", minimum_y + 14),
    )

    memory["kernel"] = _make_kernel(memory)
    mass = memory.get("enemy_mass", {})
    elapsed = max(0, step - previous_step)
    if elapsed > 5:
        mass = {}
    else:
        for _ in range(elapsed):
            mass = _spread(memory, mass)

    hidden = {
        pos: mass.get(pos, 0.0)
        for pos in memory["kernel"]
        if pos not in visible
    }
    mass = _rebalance(hidden, max(0, 3 - len(occupied)))
    mass.update((pos, 1.0) for pos in occupied)
    memory["enemy_mass"] = mass
    memory["visible"] = visible
    memory["enemies"] = occupied

    # Old traversable terrain is useful for routes but cannot certify current
    # occupancy. Walls cannot contain enemies and stay valid within an epoch.
    epoch = step // 25
    reported = {
        pos: 1
        for pos, cell in terrain.items()
        if cell == 1 and seen.get(pos, -25) // 25 == epoch
    }
    reported.update(visible)
    memory["believed_map"] = reported

    if memory["camping"] and (occupied or step >= 185):
        memory["finish_now"] = True
    return memory


def _bfs(start, graph):
    distance = {start: 0}
    queue = deque([start])
    while queue:
        pos = queue.popleft()
        next_distance = distance[pos] + 1
        for target in graph.get(pos, ()):
            if target not in distance:
                distance[target] = next_distance
                queue.append(target)
    return distance


def _information(memory, pos):
    x, y = pos
    terrain = memory["terrain"]
    seen = memory["seen"]
    epoch = memory["step"] // 25
    gain = 0.0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            target = (x + dx, y + dy)
            if not _inside(memory, target):
                continue
            if target not in terrain:
                gain += 1.0
            elif seen.get(target, -25) // 25 != epoch:
                gain += 0.15 if terrain[target] == 1 else 0.04
    return gain


def _choose_goal(memory, graph, distances):
    terrain = memory["terrain"]
    keys = memory["keys"]
    if keys < 2:
        kind = "key"
        targets = [
            p for p, cell in terrain.items()
            if cell == 2 and p not in memory["keys_collected"]
        ]
    elif not memory["door_open"]:
        kind = "door"
        targets = [p for p, cell in terrain.items() if cell == 3]
    else:
        kind = "exit"
        targets = [p for p, cell in terrain.items() if cell == 4]

    reachable = [p for p in targets if p in distances]
    if reachable:
        goal = min(reachable, key=lambda p: (distances[p], p))
        memory["goal"] = goal
        memory["goal_kind"] = kind
        return goal, kind

    previous = memory.get("goal")
    if (
        memory.get("goal_kind") == "explore"
        and previous in distances
        and previous != memory["agent_pos"]
        and _information(memory, previous) >= 1.0
    ):
        return previous, "explore"

    candidates = [
        p for p in distances
        if p != memory["agent_pos"] and _information(memory, p) > 0.0
    ]
    if candidates:
        visits = memory["visits"]

        def value(pos):
            return (
                _information(memory, pos)
                / (distances[pos] + 3.0) ** 0.85
                - 0.025 * min(visits.get(pos, 0), 12)
            )

        goal = max(candidates, key=lambda p: (value(p), -distances[p], p))
        kind = "explore"
    else:
        candidates = [
            p for p in distances if p != memory["agent_pos"]
        ]
        goal = min(
            candidates,
            key=lambda p: (
                memory["visits"].get(p, 0)
                + 0.15 * distances[p]
                + 0.01 * memory["seen"].get(p, 0),
                p,
            ),
            default=memory["agent_pos"],
        )
        kind = "refresh"

    memory["goal"] = goal
    memory["goal_kind"] = kind
    return goal, kind


def _first_risk(target, enemies, distributions):
    if target in enemies:
        return 1.0
    survival = 1.0
    for distribution in distributions:
        probability = sum(
            weight for pos, weight in distribution if pos == target
        )
        survival *= 1.0 - probability
    return 1.0 - survival


def _scenarios(distributions):
    if not distributions:
        return [((), 1.0)]
    result = []
    for outcomes in product(*distributions):
        positions = tuple(item[0] for item in outcomes)
        probability = 1.0
        for _, weight in outcomes:
            probability *= weight
        result.append((positions, probability))
    return result


def _trap_risk(memory, destination, scenarios, opened):
    """Expected best next-action risk after observing the next enemy positions."""
    choices = [
        target
        for _, target in _agent_moves(
            memory, destination, previous=opened
        )
    ]
    if not choices:
        return 1.0

    opened_next = _opened_here(memory, destination, opened)
    all_bits = (1 << len(choices)) - 1
    cache = {}
    expectation = 0.0
    surviving_weight = 0.0

    for positions, weight in scenarios:
        if destination in positions:
            continue
        surviving_weight += weight
        threats = []
        union = 0

        for enemy in positions:
            if enemy not in cache:
                outcomes = dict(
                    _enemy_outcomes(memory, enemy, opened_next)
                )
                probabilities = tuple(
                    1.0 if target == enemy
                    else outcomes.get(target, 0.0)
                    for target in choices
                )
                mask = 0
                for index, probability in enumerate(probabilities):
                    if probability > 0.0:
                        mask |= 1 << index
                cache[enemy] = (mask, probabilities)

            mask, probabilities = cache[enemy]
            union |= mask
            threats.append(probabilities)

        # An uncovered destination provides a zero-risk response.
        if union != all_bits:
            continue

        best = 1.0
        for index in range(len(choices)):
            survival = 1.0
            for probabilities in threats:
                survival *= 1.0 - probabilities[index]
            best = min(best, 1.0 - survival)
            if best == 0.0:
                break
        expectation += weight * best

    if surviving_weight <= 1e-12:
        return 1.0
    return expectation / surviving_weight


def planner(memory: dict, local_obs: dict) -> dict:
    agent = memory["agent_pos"]
    terrain = memory["terrain"]
    enemies = memory["enemies"]
    opened = _opened_here(memory, agent)
    actions = _agent_moves(memory, agent)
    wait = {"move": [0, 0], "interact": True}

    # Build a wavefront graph. Previously observed walls become possible
    # passages after a toggle, but visible walls always remain obstacles.
    graph = {}
    reverse = {}
    for pos, cell in terrain.items():
        if not _inside(memory, pos):
            continue
        if cell == 3:
            usable = memory["door_open"] or memory["keys"] >= 2
        else:
            usable = _passable(memory, pos, (), False, True)
        if not usable:
            continue
        neighbors = [
            target
            for move, target in _agent_moves(memory, pos, True)
            if move != (0, 0)
        ]
        graph[pos] = neighbors
        for target in neighbors:
            reverse.setdefault(target, []).append(pos)

    distances = _bfs(agent, graph)
    goal, kind = _choose_goal(memory, graph, distances)
    remaining = _bfs(goal, reverse)

    distributions = [
        _enemy_outcomes(memory, enemy, opened)
        for enemy in sorted(enemies)
    ]
    risks = {
        target: _first_risk(target, enemies, distributions)
        for _, target in actions
    }
    minimum = min(risks.values(), default=1.0)

    if (
        terrain.get(agent) == 2
        and memory["keys"] < 2
        and risks.get(agent, 1.0) <= minimum + 1e-12
    ):
        return wait

    exit_actions = [
        (move, target)
        for move, target in actions
        if move != (0, 0)
        and terrain.get(target) == 4
        and memory["keys"] >= 2
        and memory["door_open"]
    ]

    for move, exit_pos in exit_actions:
        cardinal = abs(move[0]) + abs(move[1]) == 1
        escape_routes = 0
        if cardinal and not enemies:
            for _, alternative in actions:
                if alternative == exit_pos:
                    continue
                if any(
                    target == exit_pos
                    for _, target in _agent_moves(memory, alternative)
                ):
                    escape_routes += 1

        if (
            cardinal
            and not enemies
            and not memory["finish_now"]
            and memory["step"] < 185
            and risks.get(agent, 1.0) == 0.0
            and risks[exit_pos] == 0.0
            and escape_routes >= 2
        ):
            memory["camping"] = True
            return wait

        if risks[exit_pos] <= minimum + 1e-12:
            memory["finish_now"] = True
            return {"move": list(move), "interact": True}

    scenarios = _scenarios(distributions)

    # Preserve diffuse hidden-enemy uncertainty for the second-step choice.
    hidden = {
        p: amount
        for p, amount in memory["enemy_mass"].items()
        if p not in memory["visible"]
    }
    hidden_next = _spread(memory, hidden)
    hidden_after = _spread(memory, hidden_next)

    candidates = []
    for index, (move, target) in enumerate(actions):
        if risks[target] > minimum + 1e-12:
            continue

        trap = _trap_risk(memory, target, scenarios, opened)
        next_choices = [
            p for _, p in _agent_moves(
                memory, target, previous=opened
            )
        ]
        hidden_risk = min(
            (
                hidden_next.get(p, 0.0)
                + hidden_after.get(p, 0.0)
                for p in next_choices
            ),
            default=1.0,
        )

        route = remaining.get(target, 50 + distances.get(target, 0))
        visit_cost = 0.10 * min(memory["visits"].get(target, 0), 15)
        stale = (
            memory["seen"].get(target, -25) // 25
            != memory["step"] // 25
        )
        score = (
            route
            + 50.0 * trap
            + 4.0 * hidden_risk
            + visit_cost
            + 0.3 * stale
        )
        if move == (0, 0):
            score += 0.45
        if kind == "explore":
            score -= 0.06 * _information(memory, target)

        # Deterministic rotating ties avoid persistent directional bias.
        tie = (index - memory["step"]) % len(_MOVES)
        candidates.append((score, tie, move))

    if not candidates:
        return wait
    _, _, move = min(candidates)
    return {"move": list(move), "interact": True}
# EVOLVE-BLOCK-END