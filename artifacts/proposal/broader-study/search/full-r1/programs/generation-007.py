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

MOVES = tuple((dx, dy) for dy in (-1, 0, 1)
              for dx in (-1, 0, 1))
SIZE = 15
SUPPORT_LIMIT = 256
BEAM_WIDTH = 36
SEARCH_DEPTH = 5


def _cell(value):
    if isinstance(value, str):
        names = {
            "empty": 0, "floor": 0, ".": 0,
            "wall": 1, "#": 1,
            "key": 2, "k": 2,
            "door": 3, "d": 3,
            "exit": 4, "e": 4,
            "enemy": 5,
            "outside": -1, "out": -1, "boundary": -1,
        }
        value = names.get(value.lower(), value)
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return -1 if value in (-1, 6) else value


def _key_count(value, default=0):
    if value is None:
        return default
    if isinstance(value, dict):
        return sum(bool(v) for v in value.values())
    if isinstance(value, (tuple, list, set)):
        return len(value)
    try:
        return max(0, min(2, int(value)))
    except (TypeError, ValueError):
        return default


def _delta(feedback):
    if not isinstance(feedback, dict):
        return (0, 0)
    for name in ("actual_displacement", "displacement",
                 "actual_move", "moved", "move", "delta"):
        value = feedback.get(name)
        if isinstance(value, dict):
            if "dx" in value or "dy" in value:
                return int(value.get("dx", 0)), int(value.get("dy", 0))
            if "x" in value and "y" in value:
                return int(value["x"]), int(value["y"])
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            return int(value[0]), int(value[1])
    if "dx" in feedback or "dy" in feedback:
        return int(feedback.get("dx", 0)), int(feedback.get("dy", 0))
    return (0, 0)


def _rows(grid):
    if not isinstance(grid, (tuple, list)):
        return []
    return grid


def _inside(memory, p):
    if p in memory["outside"]:
        return False
    lo_x, hi_x, lo_y, hi_y = memory["bounds"]
    return lo_x <= p[0] <= hi_x and lo_y <= p[1] <= hi_y


def _fresh(memory, p):
    return memory["stamp"].get(p, -100) // 25 == memory["step"] // 25


def _enemy_pass(memory, p, opened):
    if not _inside(memory, p):
        return 0.0
    cell = memory["terrain"].get(p)
    if cell == 3:
        return 1.0 if opened else 0.0
    if cell == 1:
        return 0.0 if _fresh(memory, p) else 0.28
    if cell is None:
        return 0.70
    return 1.0 if _fresh(memory, p) else 0.90


def _spread(memory, mass, opened, cache):
    result = {}
    for p, amount in mass.items():
        if amount <= 1e-9 or not _inside(memory, p):
            continue
        cache_key = (p, bool(opened))
        transitions = cache.get(cache_key)
        if transitions is None:
            transitions = []
            stay = 1.0 / 9.0
            x, y = p
            for dx, dy in MOVES:
                if dx == 0 and dy == 0:
                    continue
                q = (x + dx, y + dy)
                probability = _enemy_pass(memory, q, opened)
                if dx and dy and probability:
                    probability *= _enemy_pass(
                        memory, (x + dx, y), opened)
                    probability *= _enemy_pass(
                        memory, (x, y + dy), opened)
                probability /= 9.0
                if probability:
                    transitions.append((q, probability))
                stay += 1.0 / 9.0 - probability
            transitions.append((p, stay))
            cache[cache_key] = transitions
        for q, probability in transitions:
            result[q] = result.get(q, 0.0) + amount * probability

    if len(result) > SUPPORT_LIMIT:
        total = sum(result.values())
        kept = heapq.nlargest(
            SUPPORT_LIMIT, result.items(), key=lambda item: item[1])
        retained = sum(value for _, value in kept)
        scale = total / retained if retained else 1.0
        result = {p: value * scale for p, value in kept}
    return result


def _update_bounds(memory, observed, exterior):
    terrain = memory["terrain"]
    if not terrain:
        return
    min_x = min(p[0] for p in terrain)
    max_x = max(p[0] for p in terrain)
    min_y = min(p[1] for p in terrain)
    max_y = max(p[1] for p in terrain)

    candidates = []
    for left, top in memory["rectangles"]:
        if not (left <= min_x and max_x < left + SIZE
                and top <= min_y and max_y < top + SIZE):
            continue
        if any(left <= x < left + SIZE and top <= y < top + SIZE
               for x, y in exterior):
            continue
        candidates.append((left, top))

    if candidates:
        memory["rectangles"] = candidates
        memory["bounds"] = (
            min(p[0] for p in candidates),
            max(p[0] for p in candidates) + SIZE - 1,
            min(p[1] for p in candidates),
            max(p[1] for p in candidates) + SIZE - 1,
        )


def world_model_step(memory, local_obs, last_action):
    memory = memory if isinstance(memory, dict) else {}
    first = "terrain" not in memory
    if first:
        memory.update({
            "position": (0, 0),
            "terrain": {},
            "stamp": {},
            "visits": {},
            "outside": set(),
            "collected": set(),
            "doors": set(),
            "exits": set(),
            "enemy_mass": {},
            "enemy_seen": set(),
            "rectangles": [
                (x, y) for x in range(-14, 1)
                for y in range(-14, 1)
            ],
            "bounds": (-14, 14, -14, 14),
            "step": 0,
            "keys": 0,
            "door_open": False,
            "goal": None,
        })

    observation = local_obs if isinstance(local_obs, dict) else {}
    feedback = observation.get("feedback", {})
    previous_step = memory["step"]
    step = int(observation.get(
        "step", previous_step + (0 if first else 1)))

    if not first:
        dx, dy = _delta(feedback)
        px, py = memory["position"]
        memory["position"] = (px + dx, py + dy)

    position = memory["position"]
    memory["step"] = step
    memory["keys"] = _key_count(
        observation.get("keys"), memory["keys"])
    memory["door_open"] = bool(
        observation.get("door_open", memory["door_open"]))
    if isinstance(feedback, dict) and feedback.get("opened"):
        memory["door_open"] = True

    mass = memory["enemy_mass"]
    if not first:
        cache = {}
        for _ in range(min(8, max(0, step - previous_step))):
            mass = _spread(
                memory, mass, memory["door_open"], cache)

    grid = _rows(observation.get("grid", []))
    ground = _rows(observation.get("terrain", []))
    report = {}
    observed = set()
    exterior = set()
    enemies = set()
    px, py = position
    height = len(grid)
    center_y = height // 2

    for row_index, row in enumerate(grid):
        if not isinstance(row, (tuple, list)):
            continue
        center_x = len(row) // 2
        for column_index, raw in enumerate(row):
            p = (px + column_index - center_x,
                 py + row_index - center_y)
            cell = _cell(raw)
            if cell is None:
                continue
            if cell == -1:
                exterior.add(p)
                memory["outside"].add(p)
                continue

            report[p] = cell
            observed.add(p)
            memory["outside"].discard(p)
            if cell == 5:
                enemies.add(p)

            underlying = None
            if row_index < len(ground):
                ground_row = ground[row_index]
                if (isinstance(ground_row, (tuple, list))
                        and column_index < len(ground_row)):
                    underlying = _cell(ground_row[column_index])

            if underlying is None or underlying in (-1, 5):
                underlying = (
                    cell if cell != 5
                    else memory["terrain"].get(p, 0)
                )
            memory["terrain"][p] = underlying
            memory["stamp"][p] = step
            if underlying == 3:
                memory["doors"].add(p)
            if underlying == 4:
                memory["exits"].add(p)

    if isinstance(feedback, dict) and feedback.get("collected"):
        memory["collected"].add(position)
        if memory["terrain"].get(position) == 2:
            memory["terrain"][position] = 0

    for p in memory["collected"]:
        if memory["terrain"].get(p) == 2:
            memory["terrain"][p] = 0

    _update_bounds(memory, observed, exterior)

    # Condition an anonymous occupancy field on this exact observation.
    unseen = {
        p: amount for p, amount in mass.items()
        if p not in observed and _inside(memory, p)
        and _enemy_pass(memory, p, memory["door_open"]) > 0
    }
    missing = max(0, 3 - len(enemies))
    total = sum(unseen.values())
    if missing and total < 1e-8:
        seeds = []
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if max(abs(dx), abs(dy)) < 3:
                    continue
                p = (px + dx, py + dy)
                if (p not in observed and _inside(memory, p)
                        and _enemy_pass(
                            memory, p, memory["door_open"]) > 0):
                    seeds.append(p)
        if seeds:
            unseen = {p: missing / len(seeds) for p in seeds}
    elif total:
        unseen = {
            p: amount * missing / total
            for p, amount in unseen.items()
        }
    else:
        unseen = {}

    unseen.update({p: 1.0 for p in enemies})
    memory["enemy_mass"] = unseen
    memory["enemy_seen"] = enemies
    memory["believed_map"] = report
    memory["visits"][position] = (
        memory["visits"].get(position, 0) + 1)
    return memory


def _agent_pass(memory, p, keys, opened, source):
    if not _inside(memory, p):
        return False
    cell = memory["terrain"].get(p)
    if cell is None:
        return False
    if cell == 1:
        return not _fresh(memory, p)
    if cell == 3 and not opened:
        return (keys >= 2
                and abs(p[0] - source[0])
                + abs(p[1] - source[1]) == 1)
    return True


def _legal(memory, source, destination, keys, opened,
           executable=False):
    dx = destination[0] - source[0]
    dy = destination[1] - source[1]
    if abs(dx) > 1 or abs(dy) > 1:
        return False
    if not dx and not dy:
        return True
    if not _agent_pass(memory, destination, keys, opened, source):
        return False
    sides = []
    if dx and dy:
        sides = [
            (source[0] + dx, source[1]),
            (source[0], source[1] + dy),
        ]
        if any(not _agent_pass(memory, p, keys, opened, source)
               for p in sides):
            return False
    if executable:
        for p in [destination] + sides:
            if (not _fresh(memory, p)
                    or memory["terrain"].get(p) == 1):
                return False
    return True


def _hazard(probability):
    return -math.log1p(-min(0.99999, max(0.0, probability)))


def _risk_field(memory, forecasts):
    current = memory["position"]
    result = {}
    for p in memory["terrain"]:
        distance = max(abs(p[0] - current[0]),
                       abs(p[1] - current[1]))
        t = min(SEARCH_DEPTH, max(1, distance))
        probability = (
            0.10 * forecasts[0].get(p, 0.0)
            + 0.60 * forecasts[t].get(p, 0.0)
            + 0.30 * forecasts[SEARCH_DEPTH].get(p, 0.0)
        )
        result[p] = min(0.98, probability)
    return result


def _edge_cost(memory, source, destination, risk):
    stale = not _fresh(memory, destination)
    cell = memory["terrain"].get(destination)
    cost = 1.0
    if stale:
        cost += 1.9 if cell == 1 else 0.45
    if (source[0] != destination[0]
            and source[1] != destination[1]):
        for p in ((destination[0], source[1]),
                  (source[0], destination[1])):
            if not _fresh(memory, p):
                cost += (
                    0.65 if memory["terrain"].get(p) == 1
                    else 0.15
                )
    remaining = max(1, 200 - memory["step"])
    risk_weight = 7.0 if remaining > 45 else 4.8
    cost += risk_weight * _hazard(risk.get(destination, 0.0))
    cost += 0.025 * min(
        12, memory["visits"].get(destination, 0))
    return cost


def _dijkstra(memory, start, keys, risk, reverse=False):
    distances = {start: 0.0}
    queue = [(0.0, start)]
    opened = memory["door_open"]
    while queue:
        distance, p = heapq.heappop(queue)
        if distance != distances.get(p):
            continue
        for dx, dy in MOVES:
            if not dx and not dy:
                continue
            q = (p[0] + dx, p[1] + dy)
            source, destination = (q, p) if reverse else (p, q)
            if q not in memory["terrain"]:
                continue
            if not _legal(
                    memory, source, destination, keys, opened):
                continue
            candidate = distance + _edge_cost(
                memory, source, destination, risk)
            if candidate < distances.get(q, float("inf")):
                distances[q] = candidate
                heapq.heappush(queue, (candidate, q))
    return distances


def _finish_distance(memory, start, risk):
    distances = _dijkstra(memory, start, 2, risk)
    exits = [
        distances[p] for p in memory["exits"]
        if p in distances
    ]
    if exits:
        return min(exits)
    doors = [
        distances[p] for p in memory["doors"]
        if p in distances
    ]
    return min(doors) + 5.0 if doors else 10.0


def _select_goal(memory, distances, risk):
    terrain = memory["terrain"]
    position = memory["position"]
    keys = memory["keys"]
    known_keys = [
        p for p, cell in terrain.items()
        if cell == 2 and p not in memory["collected"]
    ]

    if keys < 2:
        choices = []
        for first in known_keys:
            if first not in distances:
                continue
            score = distances[first]
            if keys == 0:
                continuation = _dijkstra(memory, first, 1, risk)
                alternatives = []
                for second in known_keys:
                    if second == first or second not in continuation:
                        continue
                    alternatives.append(
                        0.85 * continuation[second]
                        + 0.30 * _finish_distance(
                            memory, second, risk)
                    )
                score += min(alternatives) if alternatives else 9.0
            else:
                score += 0.40 * _finish_distance(
                    memory, first, risk)
            choices.append((score, first))
        if choices:
            return min(choices)[1]

    if keys >= 2:
        exits = [
            (distances[p], p) for p in memory["exits"]
            if p in distances
        ]
        if exits:
            return min(exits)[1]

    best = None
    best_score = -1.0
    previous_goal = memory.get("goal")
    for p, distance in distances.items():
        if terrain.get(p) == 1:
            continue
        gain = 0.0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                q = (p[0] + dx, p[1] + dy)
                if not _inside(memory, q):
                    continue
                if q not in terrain:
                    gain += 1.0
                elif not _fresh(memory, q):
                    gain += (
                        0.30 if terrain[q] == 1 else 0.18
                    )
        if (keys >= 2 and not memory["door_open"]
                and p in memory["doors"]):
            gain += 3.5
        if gain <= 0:
            continue
        visits = memory["visits"].get(p, 0)
        score = (
            gain / ((distance + 2.8) ** 0.90)
            / (1.0 + 0.32 * visits)
        )
        score *= math.exp(
            -2.0 * min(0.9, risk.get(p, 0.0)))
        if p == previous_goal and p != position:
            score *= 1.14
        if score > best_score:
            best_score, best = score, p

    if best is not None:
        return best

    alternatives = [
        (distance + 1.5 * memory["visits"].get(p, 0), p)
        for p, distance in distances.items()
        if p != position and terrain.get(p) != 1
    ]
    return min(alternatives)[1] if alternatives else position


def _opens_here(memory, position, keys, opened):
    if opened or keys < 2:
        return bool(opened)
    return any(
        abs(p[0] - position[0]) + abs(p[1] - position[1]) == 1
        for p in memory["doors"]
    )


def _contact(forecasts, time, destination):
    # Union bound includes contact before and after enemy movement.
    return min(
        0.99999,
        forecasts[time - 1].get(destination, 0.0)
        + forecasts[time].get(destination, 0.0),
    )


def planner(memory, local_obs=None):
    if not isinstance(memory, dict) or "terrain" not in memory:
        return {"move": [0, 0], "interact": True}

    position = memory["position"]
    keys = memory["keys"]
    remaining = max(1, 200 - memory["step"])
    depth = min(SEARCH_DEPTH, remaining)

    transition_cache = {}
    closed_forecasts = [memory["enemy_mass"]]
    open_forecasts = [memory["enemy_mass"]]
    for _ in range(SEARCH_DEPTH):
        closed_forecasts.append(_spread(
            memory, closed_forecasts[-1],
            memory["door_open"], transition_cache))
        open_forecasts.append(_spread(
            memory, open_forecasts[-1], True, transition_cache))

    initially_open = _opens_here(
        memory, position, keys, memory["door_open"])
    forecasts = open_forecasts if initially_open else closed_forecasts
    risk = _risk_field(memory, forecasts)
    distances = _dijkstra(memory, position, keys, risk)
    goal = _select_goal(memory, distances, risk)
    memory["goal"] = goal

    terminal_maps = {}

    def terminal(p, held):
        held = min(2, held)
        if held not in terminal_maps:
            terminal_maps[held] = _dijkstra(
                memory, goal, held, risk, reverse=True)
        value = terminal_maps[held].get(p)
        if value is None:
            value = (
                25.0 + 3.0 * max(
                    abs(p[0] - goal[0]), abs(p[1] - goal[1]))
            )
        return value

    key_positions = [
        p for p, cell in memory["terrain"].items()
        if cell == 2 and p not in memory["collected"]
    ][:2]
    key_bits = {p: 1 << i for i, p in enumerate(key_positions)}

    root_options = []
    for dx, dy in MOVES:
        destination = (position[0] + dx, position[1] + dy)
        if not _legal(
                memory, position, destination, keys,
                initially_open, executable=True):
            continue
        probability = _contact(forecasts, 1, destination)
        root_options.append(
            ((dx, dy), destination, probability))

    nonoccupied = [
        item for item in root_options
        if item[1] not in memory["enemy_seen"]
    ]
    if nonoccupied:
        root_options = nonoccupied
    if not root_options:
        return {"move": [0, 0], "interact": True}

    minimum_risk = min(item[2] for item in root_options)
    slack = 0.060 if remaining > 40 else 0.095
    allowed_first = {
        move for move, _, probability in root_options
        if probability <= minimum_risk + slack
    }

    collision_weight = 46.0 if remaining > 40 else 32.0
    # State: position, collected-key mask, door status, cost, first move.
    beam = [(position, 0, memory["door_open"], 0.0, None)]
    completed = []
    for time in range(1, depth + 1):
        candidates = {}
        for source, mask, opened, cost, first_move in beam:
            held = min(2, keys + bin(mask).count("1"))
            now_open = _opens_here(memory, source, held, opened)
            field = open_forecasts if now_open else closed_forecasts

            for dx, dy in MOVES:
                move = (dx, dy)
                if time == 1 and move not in allowed_first:
                    continue
                destination = (source[0] + dx, source[1] + dy)
                if not _legal(
                        memory, source, destination, held,
                        now_open, executable=(time == 1)):
                    continue
                probability = _contact(field, time, destination)
                new_mask = mask | key_bits.get(destination, 0)
                new_held = min(
                    2, keys + bin(new_mask).count("1"))
                action_cost = 1.08 if move == (0, 0) else 1.0
                action_cost += (
                    collision_weight * _hazard(probability))
                action_cost += 0.045 * min(
                    10, memory["visits"].get(destination, 0))
                if not _fresh(memory, destination):
                    action_cost += (
                        2.0 if memory["terrain"].get(destination) == 1
                        else 0.35
                    )
                new_cost = cost + action_cost
                chosen = move if first_move is None else first_move
                state = (destination, new_mask, now_open)
                ranking = new_cost + 1.22 * terminal(
                    destination, new_held)
                node = (
                    destination, new_mask, now_open,
                    new_cost, chosen,
                )

                # Stop evaluating a plan once its current objective is met.
                # The real controller replans immediately after execution.
                if destination == goal:
                    completed.append((new_cost, chosen))
                    continue
                previous = candidates.get(state)
                if previous is None or ranking < previous[0]:
                    candidates[state] = (ranking, node)

        if not candidates:
            beam = []
            break
        selected = heapq.nsmallest(
            BEAM_WIDTH, candidates.values(), key=lambda item: item[0])
        beam = [node for _, node in selected]

    final_choices = list(completed)
    for p, mask, opened, cost, first_move in beam:
        held = min(2, keys + bin(mask).count("1"))
        final_choices.append(
            (cost + 1.22 * terminal(p, held), first_move))

    if final_choices:
        _, move = min(final_choices, key=lambda item: item[0])
    else:
        move, _, _ = min(
            root_options,
            key=lambda item: (
                item[2],
                terminal(item[1], keys),
                item[0] == (0, 0),
            ),
        )
    if move is None:
        move = (0, 0)
    return {"move": [move[0], move[1]], "interact": True}

# EVOLVE-BLOCK-END
