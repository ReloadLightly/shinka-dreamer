"""Predictive seed. Entire representation, learner and planner may evolve."""
# EVOLVE-BLOCK-START
import heapq
import itertools
import math

DIRECTIONS = tuple((x, y) for y in (-1, 0, 1) for x in (-1, 0, 1))
MOVES = tuple(d for d in DIRECTIONS if d != (0, 0))
CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
UNIFORM = (1.0 / 9.0,) * 9
EXPERT_PRIOR = (.30, .45, .25)
HORIZON = 4


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _mean(values):
    total = sum(values)
    return tuple(v / total for v in values)


def _clip(value):
    return min(1.0, max(0.0, value))


def _inside(m, p):
    left, right, top, bottom = m["bounds"]
    return (
        (left is None or left <= p[0] <= right)
        and (top is None or top <= p[1] <= bottom)
    )


def _infer_bounds(m, terrain):
    x, y = m["pos"]
    bounds = m["bounds"]
    for delta in (-2, -1, 1, 2):
        if terrain.get((x + delta, y)) == -1:
            if delta < 0:
                edge = x + delta + 1
                if bounds[0] is None or edge > bounds[0]:
                    bounds[0], bounds[1] = edge, edge + 14
            else:
                edge = x + delta - 1
                if bounds[1] is None or edge < bounds[1]:
                    bounds[0], bounds[1] = edge - 14, edge
        if terrain.get((x, y + delta)) == -1:
            if delta < 0:
                edge = y + delta + 1
                if bounds[2] is None or edge > bounds[2]:
                    bounds[2], bounds[3] = edge, edge + 14
            else:
                edge = y + delta - 1
                if bounds[3] is None or edge < bounds[3]:
                    bounds[2], bounds[3] = edge - 14, edge


class Geometry:
    """Shared geometric transition table, independent of movement laws."""

    def __init__(self, m):
        self.m = m
        self.opened = set()
        if m["keys"] >= 2:
            self.opened = {
                _add(m["pos"], d) for d in CARDINAL
                if m["map"].get(_add(m["pos"], d)) == 3
            }

        known = [p for p, cell in m["map"].items() if cell != -1]
        xs = [p[0] for p in known]
        ys = [p[1] for p in known]
        left, right, top, bottom = m["bounds"]

        # The initial agent cell is interior. These are possible interior
        # coordinates, not a guessed hidden map.
        self.limits = (
            left + 1 if left is not None else max(-12, max(xs) - 13),
            right - 1 if right is not None else min(12, min(xs) + 13),
            top + 1 if top is not None else max(-12, max(ys) - 13),
            bottom - 1 if bottom is not None else min(12, min(ys) + 13),
        )
        self.solids = {}
        self.destinations = {}

    def solid(self, p):
        if p not in self.solids:
            left, right, top, bottom = self.limits
            cell = self.m["map"].get(p, 0)
            self.solids[p] = (
                not (left <= p[0] <= right and top <= p[1] <= bottom)
                or cell in (-1, 1)
                or (
                    cell == 3 and not self.m["door_open"]
                    and p not in self.opened
                )
            )
        return self.solids[p]

    def row(self, p):
        if p not in self.destinations:
            destinations = []
            for dx, dy in DIRECTIONS:
                q = (p[0] + dx, p[1] + dy)
                if self.solid(q) or (
                    dx and dy and (
                        self.solid((p[0] + dx, p[1]))
                        or self.solid((p[0], p[1] + dy))
                    )
                ):
                    q = p
                destinations.append(q)
            self.destinations[p] = tuple(destinations)
        return self.destinations[p]

    def hidden_cells(self):
        left, right, top, bottom = self.limits
        return [
            (x, y)
            for y in range(top, bottom + 1)
            for x in range(left, right + 1)
            if (x, y) not in self.m["visible"]
            and not self.solid((x, y))
        ]


class Kernel:
    """Blocked attempted movements remain at their source."""

    def __init__(self, geometry, law):
        self.geometry = geometry
        self.law = law
        self.rows = {}
        self.stays = {}

    def row(self, p):
        if p not in self.rows:
            grouped = {}
            for q, probability in zip(self.geometry.row(p), self.law):
                grouped[q] = grouped.get(q, 0.0) + probability
            self.rows[p] = tuple(grouped.items())
            self.stays[p] = grouped.get(p, 0.0)
        return self.rows[p]

    def spread(self, distribution):
        result = {}
        for p, mass in distribution.items():
            if mass <= 1e-14:
                continue
            for q, probability in self.row(p):
                result[q] = result.get(q, 0.0) + mass * probability
        return result


class MotionLearner:
    """Censored anonymous observations update slow and fast law experts."""

    @staticmethod
    def destination(terrain, p, d, door_open):
        def blocked(q):
            cell = terrain.get(q, 0)
            return cell in (-1, 1) or (cell == 3 and not door_open)

        q = _add(p, d)
        if blocked(q) or (
            d[0] and d[1] and (
                blocked((p[0] + d[0], p[1]))
                or blocked((p[0], p[1] + d[1]))
            )
        ):
            return p
        return q

    @staticmethod
    def masks(previous, terrain, enemies):
        old_terrain, old_enemies, center, step, opened = previous
        monitored = {
            _add(center, (dx, dy))
            for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if _add(center, (dx, dy)) in terrain
        }
        occupied = sorted(monitored.intersection(enemies))
        bits = {p: 1 << i for i, p in enumerate(occupied)}
        rows = []
        for origin in sorted(old_enemies):
            row = []
            for d in DIRECTIONS:
                q = MotionLearner.destination(
                    old_terrain, origin, d, opened
                )
                row.append(
                    0 if q not in monitored else bits.get(q, -1)
                )
            rows.append(row)
        return rows, (1 << len(occupied)) - 1

    @staticmethod
    def posterior(law, masks, target):
        groups = []
        for row in masks:
            masses = {}
            for i, mask in enumerate(row):
                if mask >= 0:
                    masses[mask] = masses.get(mask, 0.0) + law[i]
            if not masses:
                return 0.0, None
            groups.append(tuple(masses.items()))

        likelihood = 0.0
        marginals = [{} for _ in groups]
        for assignment in itertools.product(*groups):
            union, probability = 0, 1.0
            for mask, mass in assignment:
                union |= mask
                probability *= mass
            if union != target:
                continue
            likelihood += probability
            for j, (mask, mass) in enumerate(assignment):
                marginals[j][mask] = (
                    marginals[j].get(mask, 0.0) + probability
                )

        if likelihood <= 1e-15:
            return 0.0, None
        result = []
        for j, row in enumerate(masks):
            masses = dict(groups[j])
            result.append([
                0.0 if mask < 0 else (
                    law[i] * marginals[j].get(mask, 0.0)
                    / (likelihood * masses[mask])
                )
                for i, mask in enumerate(row)
            ])
        return likelihood, result

    @staticmethod
    def update(m, terrain, enemies, enabled):
        if not enabled:
            return
        model = m["motion"]
        laws = (UNIFORM, _mean(model["slow"]), _mean(model["fast"]))
        previous = m["previous"]

        for name, retention, prior in (
            ("slow", .985, .65), ("fast", .86, .35)
        ):
            model[name] = [
                prior + retention * (a - prior) for a in model[name]
            ]

        if previous is None or not previous[1]:
            return
        if (
            m["step"] != previous[3] + 1
            or m["step"] // 25 != previous[3] // 25
            or m["step"] % 25 == 1
            or m["door_open"] != previous[4]
        ):
            return

        masks, target = MotionLearner.masks(previous, terrain, enemies)
        results = [
            MotionLearner.posterior(law, masks, target) for law in laws
        ]
        if any(posteriors is None for _, posteriors in results):
            return

        information = [
            [
                min(1.0, sum(abs(a - b) for a, b in zip(row, law)))
                for row in posteriors
            ]
            for law, (_, posteriors) in zip(laws, results)
        ]
        if sum(information[0]) < 1e-8:
            return

        # Fixed sharing lets a suppressed expert recover after a switch.
        weights = [
            (.965 * model["weights"][i] + .035 * EXPERT_PRIOR[i])
            * max(results[i][0], 1e-15) ** .70
            for i in range(3)
        ]
        model["weights"] = list(_mean(weights))

        for expert, name in ((1, "slow"), (2, "fast")):
            for row, strength in zip(
                results[expert][1], information[expert]
            ):
                if strength < 1e-8:
                    continue
                for i, probability in enumerate(row):
                    model[name][i] += strength * probability
                if expert == 1:
                    model["updates"] += 1
                    model["effective_updates"] += strength

    @staticmethod
    def law(m):
        model = m["motion"]
        laws = (UNIFORM, _mean(model["slow"]), _mean(model["fast"]))
        return tuple(
            .12 / 9.0 + .88 * sum(
                model["weights"][j] * laws[j][i] for j in range(3)
            )
            for i in range(9)
        )


class EnemyFilter:
    """Anonymous first-moment filtering with exact visible occupied cells.

    Hidden intensity is an approximation, not an inferred set of identities.
    Visible sources and unresolved mass are represented separately.
    """

    @staticmethod
    def condition(predicted, candidates, count):
        if count <= 0 or not candidates:
            return {}
        retained = {
            p: max(0.0, predicted.get(p, 0.0)) for p in candidates
        }
        total = sum(retained.values())
        uniform = count / float(len(candidates))
        if total < 1e-12:
            return {p: uniform for p in candidates}

        # A small diffuse component covers geometry uncertainty and the
        # first-moment approximation without adding extra enemy mass.
        scale = .97 * count / total
        return {
            p: scale * retained[p] + .03 * uniform for p in candidates
        }

    @staticmethod
    def aggregate(hidden, enemies):
        result = dict(hidden)
        for p in enemies:
            result[p] = result.get(p, 0.0) + 1.0
        return result

    @staticmethod
    def forecast(m, kernel, hidden, horizon):
        sources = [{p: 1.0} for p in sorted(m["enemies"])]
        hazards = [{}]
        risk, next_intensity = {}, {}
        geometry = kernel.geometry

        for tick in range(1, horizon + 1):
            next_hidden = kernel.spread(hidden)
            following_sources = []
            no_collision, no_occupancy = {}, {}

            for before_distribution in sources:
                after_distribution = kernel.spread(before_distribution)
                following_sources.append(after_distribution)
                for p in set(before_distribution).union(after_distribution):
                    kernel.row(p)
                    before = before_distribution.get(p, 0.0)
                    after = after_distribution.get(p, 0.0)
                    union = _clip(
                        before + after - before * kernel.stays[p]
                    )
                    no_collision[p] = (
                        no_collision.get(p, 1.0) * (1.0 - union)
                    )
                    if tick == 1:
                        no_occupancy[p] = (
                            no_occupancy.get(p, 1.0) * (1.0 - _clip(after))
                        )

            cells = set(m["map"]).union(
                hidden, next_hidden, no_collision
            )
            field = {}
            for p in cells:
                if geometry.solid(p):
                    field[p] = 0.0
                    if tick == 1:
                        risk[p] = 0.0
                    continue
                kernel.row(p)
                before = hidden.get(p, 0.0)
                after = next_hidden.get(p, 0.0)
                hidden_union = max(
                    0.0, before + after - before * kernel.stays[p]
                )
                field[p] = _clip(
                    1.0 - no_collision.get(p, 1.0)
                    * math.exp(-hidden_union)
                )
                if tick == 1:
                    risk[p] = _clip(
                        1.0 - no_occupancy.get(p, 1.0) * math.exp(-after)
                    )

            if tick == 1:
                next_intensity = dict(next_hidden)
                for distribution in following_sources:
                    for p, mass in distribution.items():
                        next_intensity[p] = next_intensity.get(p, 0.0) + mass
            hazards.append(field)
            hidden, sources = next_hidden, following_sources

        return risk, hazards, next_intensity


class Navigation:
    """One navigation graph serves route selection and temporal control."""

    def __init__(self, m):
        self.m = m
        self.graph = {}
        self.reverse = {}
        # Interaction at the current cell precedes the chosen movement.
        self.open_now = m["door_open"] or (
            m["keys"] >= 2 and any(
                m["map"].get(_add(m["pos"], d)) == 3 for d in CARDINAL
            )
        )
        for p in m["map"]:
            if self.blocked(p):
                continue
            edges = []
            for d in MOVES:
                q = _add(p, d)
                if self.legal(p, d):
                    edges.append((q, d))
                    self.reverse.setdefault(q, []).append(p)
            self.graph[p] = edges

    def blocked(self, p):
        cell = self.m["map"].get(p, 1)
        return (
            not _inside(self.m, p)
            or cell in (-1, 1)
            or (cell == 3 and self.m["keys"] < 2 and not self.open_now)
        )

    def legal(self, p, d):
        q = _add(p, d)
        if self.blocked(q):
            return False
        if d[0] and d[1]:
            if (
                self.blocked((p[0] + d[0], p[1]))
                or self.blocked((p[0], p[1] + d[1]))
            ):
                return False
            if self.m["map"].get(q) == 3 and not self.open_now:
                return False
        return True

    def distances(self, start, risk, reverse=False, weight=16.0,
                  geometric=False):
        distances = {start: 0.0}
        queue = [(0.0, start)]
        while queue:
            cost, p = heapq.heappop(queue)
            if cost != distances[p]:
                continue
            neighbors = (
                self.reverse.get(p, ()) if reverse
                else (q for q, d in self.graph.get(p, ()))
            )
            for q in neighbors:
                destination = p if reverse else q
                edge = 1.0
                if not geometric:
                    edge += (
                        .025 * min(
                            12, self.m["visits"].get(destination, 0)
                        )
                        + weight * risk.get(destination, 0.0)
                        + (5.0 if destination in self.m["enemies"] else 0.0)
                    )
                new = cost + edge
                if new < distances.get(q, float("inf")):
                    distances[q] = new
                    heapq.heappush(queue, (new, q))
        return distances

    def information(self, p):
        m = self.m
        unseen, stale = 0, 0
        epoch = (m["step"] // 25) * 25
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                q = (p[0] + dx, p[1] + dy)
                if not _inside(m, q):
                    continue
                if q not in m["map"]:
                    unseen += 1
                elif (
                    m["map"][q] in (0, 1)
                    and m["seen"][q] < epoch
                    and q not in m["visible"]
                ):
                    stale += 1
        return unseen, stale

    def target(self, distances):
        m = self.m
        start = m["pos"]
        wanted = 2 if m["keys"] < 2 else 4
        goals = [
            p for p in distances
            if p != start and m["map"].get(p) == wanted
        ]
        if goals:
            return min(goals, key=lambda p: (distances[p], p)), "goal"
        if m["keys"] >= 2 and not m["door_open"]:
            doors = [
                p for p in distances
                if p != start and m["map"].get(p) == 3
            ]
            if doors:
                return min(doors, key=lambda p: (distances[p], p)), "door"

        frontier, stale_gain = {}, {}
        for p, cost in distances.items():
            if p == start:
                continue
            unseen, stale = self.information(p)
            stale_gain[p] = stale
            if unseen:
                frontier[p] = (cost + .3) / math.sqrt(unseen)

        if frontier:
            best = min(frontier, key=lambda p: (frontier[p], p))
            old = m["control"]["target"]
            if (
                m["control"]["kind"] == "frontier"
                and old in frontier
                and frontier[old] <= 1.15 * frontier[best]
            ):
                best = old
            return best, "frontier"

        alternatives = [p for p in distances if p != start]
        if alternatives:
            return min(alternatives, key=lambda p: (
                .8 * min(20, m["visits"].get(p, 0))
                + .18 * distances[p] - .14 * stale_gain.get(p, 0),
                p
            )), "patrol"
        return start, "wait"

    def pressure(self, target, geometric):
        m = self.m
        state = m["control"]
        route = geometric.get(m["pos"], 250.0)
        if state["progress_target"] != target:
            state["progress_target"] = target
            state["best_distance"] = route
            state["last_progress"] = m["step"]
        elif route < state["best_distance"] - .5:
            state["best_distance"] = route
            state["last_progress"] = m["step"]
        idle = m["step"] - state["last_progress"]
        repeats = state["recent"][-12:].count(m["pos"])
        return _clip(max(
            (state["stationary"] - 5) / 12.0,
            (idle - 12) / 20.0,
            (repeats - 5) / 7.0,
        ))

    def action(self, risk, hazards):
        m = self.m
        start = m["pos"]
        distances = self.distances(start, risk)
        target, kind = self.target(distances)
        m["control"]["target"], m["control"]["kind"] = target, kind

        geometric = self.distances(
            target, {}, reverse=True, geometric=True
        )
        pressure = self.pressure(target, geometric)
        potential = self.distances(
            target, risk, reverse=True, weight=16.0 + 34.0 * pressure
        )
        remaining = max(1, 200 - m["step"])
        route = max(1.0, geometric.get(start, 1.0))
        risk_weight = 100.0 * (
            .35 + .65 * min(1.0, remaining / (route + 15.0))
        )
        risk_weight *= .35 + .65 * min(1.0, remaining / 35.0)

        # Raising later penalties after repeated nonprogress reduces the
        # incentive to postpone the same crossing on every replanning step.
        future = (
            0.0, 1.0,
            .55 + .30 * pressure,
            .25 + .45 * pressure,
            .12 + .45 * pressure,
        )
        recent = m["control"]["recent"][-10:]
        stage_cache, memo = {}, {}

        def stage(q, waiting, tick):
            key = (q, waiting, tick)
            if key not in stage_cache:
                hazard = _clip(hazards[tick].get(q, 0.0))
                cost = (
                    1.0
                    + ((.16 + .45 * pressure) if waiting else 0.0)
                    + .02 * min(10, m["visits"].get(q, 0))
                    + .05 * pressure * recent.count(q)
                    + risk_weight * future[tick]
                    * -math.log(max(1e-7, 1.0 - hazard))
                )
                stage_cache[key] = cost
            return stage_cache[key]

        def value(p, tick):
            if p == target and target != start:
                return 0.0
            if tick > HORIZON:
                return potential.get(p, 250.0)
            key = (p, tick)
            if key not in memo:
                best = stage(p, True, tick) + value(p, tick + 1)
                for q, d in self.graph.get(p, ()):
                    candidate = stage(q, False, tick) + value(q, tick + 1)
                    if candidate < best:
                        best = candidate
                memo[key] = best
            return memo[key]

        choices = []
        for q, d in [(start, (0, 0))] + self.graph.get(start, []):
            if q in m["enemies"]:
                continue
            score = stage(q, d == (0, 0), 1) + value(q, 2)
            choices.append((
                score, hazards[1].get(q, 0.0),
                potential.get(q, 250.0), d
            ))
        move = min(choices)[3] if choices else (0, 0)
        return {"move": [move[0], move[1]], "interact": True}


def _new_memory():
    return {
        "map": {}, "seen": {}, "pos": (0, 0), "visits": {},
        "bounds": [None, None, None, None],
        "keys": 0, "door_open": False, "previous": None,
        "motion": {
            "slow": [.65] * 9, "fast": [.35] * 9,
            "weights": list(EXPERT_PRIOR),
            "updates": 0, "effective_updates": 0.0,
        },
        "belief": {
            "predicted": {}, "uniform_predicted": {},
            "hidden": {}, "uniform_hidden": {},
        },
        "control": {
            "target": None, "kind": None, "progress_target": None,
            "best_distance": float("inf"), "last_progress": 0,
            "stationary": 0, "recent": [],
        },
    }


def world_model_step(memory, local_obs, last_action):
    m = _new_memory() if memory is None else memory
    old_keys, old_open = m["keys"], m["door_open"]
    displacement = tuple(
        local_obs.get("feedback", {}).get("displacement", (0, 0))
    )
    m["pos"] = _add(m["pos"], displacement)
    m["keys"] = local_obs["keys"]
    m["door_open"] = local_obs["door_open"]
    m["step"] = local_obs["step"]
    m["visits"][m["pos"]] = m["visits"].get(m["pos"], 0) + 1

    control = m["control"]
    control["stationary"] = (
        control["stationary"] + 1
        if m["previous"] is not None and displacement == (0, 0) else 0
    )
    control["recent"] = (control["recent"] + [m["pos"]])[-16:]
    if m["keys"] > old_keys or (m["door_open"] and not old_open):
        control["progress_target"] = None
        control["last_progress"] = m["step"]
        control["stationary"] = 0
        control["recent"] = [m["pos"]]

    terrain, enemies = {}, set()
    for y, row in enumerate(local_obs["terrain"]):
        for x, cell in enumerate(row):
            p = _add(m["pos"], (x - 2, y - 2))
            terrain[p] = cell
            m["map"][p], m["seen"][p] = cell, m["step"]
            if local_obs["grid"][y][x] == 5:
                enemies.add(p)

    _infer_bounds(m, terrain)
    MotionLearner.update(m, terrain, enemies, local_obs.get("learn", True))
    m["visible"], m["enemies"] = set(terrain), enemies
    m["law"] = MotionLearner.law(m)

    geometry = Geometry(m)
    kernel = Kernel(geometry, m["law"])
    uniform_kernel = Kernel(geometry, UNIFORM)
    candidates = geometry.hidden_cells()
    hidden_count = max(0, 3 - len(enemies))
    belief = m["belief"]

    # Observation conditioning always runs: it is state estimation, whereas
    # the learn flag controls inference of the attempted-movement law.
    belief["hidden"] = EnemyFilter.condition(
        belief["predicted"], candidates, hidden_count
    )
    belief["uniform_hidden"] = EnemyFilter.condition(
        belief["uniform_predicted"], candidates, hidden_count
    )
    risk, hazards, predicted = EnemyFilter.forecast(
        m, kernel, belief["hidden"], HORIZON
    )
    belief["predicted"] = predicted
    m["risk"], m["hazards"] = risk, hazards

    if local_obs.get("predictive_planning", True):
        belief["uniform_predicted"] = uniform_kernel.spread(
            EnemyFilter.aggregate(belief["uniform_hidden"], enemies)
        )
        m["control_risk"], m["control_hazards"] = risk, hazards
    else:
        # The uniform shadow filter prevents learned historical propagation
        # from leaking into this prediction-use intervention.
        uniform_risk, uniform_hazards, uniform_prediction = (
            EnemyFilter.forecast(
                m, uniform_kernel, belief["uniform_hidden"], HORIZON
            )
        )
        belief["uniform_predicted"] = uniform_prediction
        m["control_risk"] = uniform_risk
        m["control_hazards"] = uniform_hazards

    # The forecast covers every currently plausible interior coordinate.
    # Coordinates outside this support cannot contain an enemy.
    m["default_enemy"] = 0.0
    m["previous"] = (
        terrain, enemies, m["pos"], m["step"], m["door_open"]
    )
    return m


def planner(memory, local_obs):
    navigator = Navigation(memory)
    return navigator.action(
        memory["control_risk"], memory["control_hazards"]
    )


def export_model(memory, local_obs):
    m = memory
    motion = m["motion"]
    hidden = m["belief"]["hidden"]
    return {
        "enemy": [
            [x, y, float(_clip(probability))]
            for (x, y), probability in m["risk"].items()
        ],
        "default_enemy": float(m["default_enemy"]),
        "terrain": [
            [x, y, cell, m["seen"][(x, y)]]
            for (x, y), cell in m["map"].items()
        ],
        "position": list(m["pos"]),
        "learning": {
            "updates": motion["updates"],
            "effective_updates": motion["effective_updates"],
            "attempted_movement_law": list(m["law"]),
            "directions": [list(d) for d in DIRECTIONS],
            "expert_weights": list(motion["weights"]),
            "slow_law": list(_mean(motion["slow"])),
            "fast_law": list(_mean(motion["fast"])),
            "hidden_expected_count": float(sum(hidden.values())),
            "forecast_horizon": HORIZON,
        },
    }
# EVOLVE-BLOCK-END
