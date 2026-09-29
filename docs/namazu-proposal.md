Here is a fully fleshed-out **ShinkaDreamer** project. It evolves an agent that must survive and escape a partially observable, dynamic maze. The twist is that the evolved program contains **both** a `world_model_step()` function (which maintains an internal map and tracks moving enemies from limited observations) and a `planner()` function (which uses that internal model to choose actions).

---

## Conceptual Overview

**The Problem:** The agent sees only a 5x5 local window. Walls shift periodically. Enemies wander randomly. Keys are scattered behind corners, a locked door blocks the exit, and the global map is never revealed. An agent that simply reacts to its immediate view will get cornered or lost.

**The Evolved Solution:** A single Python module containing two functions:
1. `world_model_step(memory, local_obs, last_action)` -- integrates observations into a persistent cognitive map.
2. `planner(memory, local_obs)` -- decides where to move based on the believed map.

**The Fitness Signal:** ShinkaEvolve does not just reward escaping the maze. It also rewards *accuracy*: after each step, the evaluator compares the agent's internal `believed_map` against the hidden ground truth. This creates explicit selection pressure for agents that genuinely *understand* their world, not just ones that get lucky.

This connects directly to the *Neuroevolution* book's themes of evolving robust behavior under partial observability and the value of internal models for flexible agent design (Chapters 6 and 12). <source-chip title="Neuroevolution Book" url="https://neuroevolutionbook.com/" />

---

## `initial.py` -- The Evolved Mind

```python
"""
ShinkaDreamer: Evolving a World Model and Planner
"""

# EVOLVE-BLOCK-START
def world_model_step(memory: dict, local_obs: dict, last_action: dict) -> dict:
    """
    Update the agent's internal model of the world.
    
    Parameters
    ----------
    memory : dict
        Persistent memory across timesteps. Expected keys:
        - 'believed_map': dict mapping (x, y) -> cell_type
        - 'agent_pos': (x, y) in the believed_map coordinate frame
        - 'enemy_tracks': dict of enemy_id -> (x, y, last_seen_step)
        - 'keys_collected': set of (x, y)
        - 'door_open': bool
        - 'step': int
    local_obs : dict
        - 'grid': 5x5 list-of-lists. Agent is at [2][2].
          Values: 0=empty, 1=wall, 2=key, 3=door, 4=exit, 5=enemy, -1=unknown
        - 'step': current timestep
        - 'health': always 1 in this domain
    last_action : dict
        - 'move': [dx, dy] taken in the previous step
        - 'interact': bool
    
    Returns
    -------
    updated_memory : dict
    """
    if memory is None:
        memory = {
            "believed_map": {},
            "agent_pos": (0, 0),
            "enemy_tracks": {},
            "keys_collected": set(),
            "door_open": False,
            "step": 0,
        }

    memory["step"] = memory.get("step", 0) + 1
    grid = local_obs["grid"]
    ax, ay = memory["agent_pos"]

    # On first step, anchor the coordinate system at (0,0)
    if memory["step"] == 1:
        memory["agent_pos"] = (0, 0)
        ax, ay = 0, 0

    # -----------------------------------------------------------------
    # 1. Spatial integration: map local view into global believed_map
    # -----------------------------------------------------------------
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cell = grid[dy + 2][dx + 2]
            if cell == -1:
                continue  # out of bounds / fog of war
            gx = ax + dx
            gy = ay + dy
            memory["believed_map"][(gx, gy)] = cell

    # -----------------------------------------------------------------
    # 2. Enemy tracking: note positions and timestamps of sightings
    # -----------------------------------------------------------------
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if grid[dy + 2][dx + 2] == 5:
                gx = ax + dx
                gy = ay + dy
                eid = f"e_{gx}_{gy}"
                memory["enemy_tracks"][eid] = (gx, gy, memory["step"])

    # -----------------------------------------------------------------
    # 3. Dead reckoning: update believed agent position from last action
    # -----------------------------------------------------------------
    if last_action is not None:
        dx, dy = last_action.get("move", [0, 0])
        memory["agent_pos"] = (ax + dx, ay + dy)

    # -----------------------------------------------------------------
    # 4. Reasoning about dynamic walls (placeholder for evolution)
    # -----------------------------------------------------------------
    # A more advanced model might mark cells near dynamic_interval boundaries
    # as uncertain, or predict toggle patterns. The LLM can evolve this.

    return memory


def planner(memory: dict, local_obs: dict) -> dict:
    """
    Choose an action based on the internal world model and current view.
    
    Returns
    -------
    action : dict with keys:
        - 'move': [dx, dy] where dx, dy in {-1, 0, 1}
        - 'interact': bool
    """
    import random

    believed_map = memory.get("believed_map", {})
    agent_pos = memory.get("agent_pos", (0, 0))
    enemy_tracks = memory.get("enemy_tracks", {})
    current_step = memory.get("step", 0)

    # -----------------------------------------------------------------
    # Default: cautious random walk
    # -----------------------------------------------------------------
    action = {
        "move": [random.choice([-1, 0, 1]), random.choice([-1, 0, 1])],
        "interact": False,
    }

    # -----------------------------------------------------------------
    # Goal selection: prioritize keys, then exit, while avoiding enemies
    # -----------------------------------------------------------------
    targets = []
    avoid = set()

    # Things to seek
    for pos, cell in believed_map.items():
        if cell == 2 and pos not in memory.get("keys_collected", set()):
            targets.append((pos, "key"))
        if cell == 4:
            targets.append((pos, "exit"))
        if cell == 3 and len(memory.get("keys_collected", set())) >= 2:
            targets.append((pos, "door"))

    # Things to avoid (believed enemy positions, decayed by time)
    for eid, (ex, ey, seen_step) in enemy_tracks.items():
        age = current_step - seen_step
        if age <= 5:  # believe enemy is still roughly there
            avoid.add((ex, ey))

    if targets:
        # Pick nearest target by Manhattan distance
        best_target = min(
            targets,
            key=lambda item: abs(item[0][0] - agent_pos[0]) + abs(item[0][1] - agent_pos[1])
        )
        tx, ty = best_target[0]

        # Simple gradient step toward target
        dx = max(-1, min(1, tx - agent_pos[0]))
        dy = max(-1, min(1, ty - agent_pos[1]))
        action["move"] = [dx, dy]

        # If adjacent to target, interact
        if abs(tx - agent_pos[0]) + abs(ty - agent_pos[1]) == 1:
            action["interact"] = True

    # -----------------------------------------------------------------
    # Collision avoidance: if planned move goes toward a believed enemy,
    # prefer a safer direction (very naive; LLM can improve this)
    # -----------------------------------------------------------------
    nx = agent_pos[0] + action["move"][0]
    ny = agent_pos[1] + action["move"][1]
    if (nx, ny) in avoid:
        # Try a random safe move
        safe_moves = [
            [mx, my] for mx in [-1, 0, 1] for my in [-1, 0, 1]
            if (agent_pos[0] + mx, agent_pos[1] + my) not in avoid
        ]
        if safe_moves:
            action["move"] = random.choice(safe_moves)

    return action
# EVOLVE-BLOCK-END


# =====================================================================
# FIXED WRAPPER: ShinkaEvolve entry point
# =====================================================================
def run_evaluation(env_config: dict):
    """
    env_config is injected by evaluate.py's get_experiment_kwargs.
    """
    maze = env_config["maze_world"]
    results = maze.run_episode(world_model_step, planner, seed=env_config.get("seed", 0))
    return results
```

---

## `evaluate.py` -- The Hidden World and the Model Auditor

```python
import os
import copy
import numpy as np
from typing import Any, Dict, List, Tuple
from shinka.core import run_shinka_eval


# =============================================================================
# 1. DYNAMIC MAZE WORLD
# =============================================================================

class DynamicMaze:
    """
    A 15x15 gridworld with:
      - Partial observability (5x5 local window)
      - Wandering enemies
      - Dynamic walls that toggle every DYNAMIC_INTERVAL steps
      - Keys, a locked door, and an exit
    """
    EMPTY = 0
    WALL = 1
    KEY = 2
    DOOR = 3
    EXIT = 4
    ENEMY = 5

    DYNAMIC_INTERVAL = 25

    def __init__(self, seed: int = 0, max_steps: int = 200):
        self.rng = np.random.RandomState(seed)
        self.max_steps = max_steps
        self.size = 15
        self._build_maze()

    def _random_empty_cell(self):
        while True:
            x = self.rng.randint(1, self.size - 1)
            y = self.rng.randint(1, self.size - 1)
            if self.grid[y, x] == self.EMPTY:
                # Ensure not on agent or enemies
                if [x, y] == getattr(self, 'agent_pos', [-1, -1]):
                    continue
                if [x, y] in getattr(self, 'enemies', []):
                    continue
                return x, y

    def _build_maze(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)

        # Border walls
        self.grid[0, :] = self.WALL
        self.grid[-1, :] = self.WALL
        self.grid[:, 0] = self.WALL
        self.grid[:, -1] = self.WALL

        # Random interior walls
        for _ in range(self.rng.randint(20, 35)):
            x, y = self.rng.randint(1, self.size - 1, size=2)
            self.grid[y, x] = self.WALL

        # Dynamic subset of walls
        wall_coords = list(zip(*np.where(self.grid == self.WALL)))
        self.dynamic_walls = set()
        if len(wall_coords) > 5:
            idxs = self.rng.choice(len(wall_coords), size=len(wall_coords) // 4, replace=False)
            for idx in idxs:
                y, x = wall_coords[idx]
                self.dynamic_walls.add((x, y))

        # Keys
        self.keys = set()
        for _ in range(2):
            x, y = self._random_empty_cell()
            self.grid[y, x] = self.KEY
            self.keys.add((x, y))

        # Door and exit
        dx, dy = self._random_empty_cell()
        self.grid[dy, dx] = self.DOOR
        self.door_pos = (dx, dy)

        ex, ey = self._random_empty_cell()
        self.grid[ey, ex] = self.EXIT
        self.exit_pos = (ex, ey)

        # Agent
        ax, ay = self._random_empty_cell()
        self.agent_pos = [ax, ay]

        # Enemies
        self.enemies = []
        for _ in range(3):
            x, y = self._random_empty_cell()
            self.enemies.append([x, y])

        self.keys_collected = set()
        self.door_open = False
        self.steps = 0
        self.done = False
        self.reason = "max_steps"

    def _toggle_dynamic_walls(self):
        for x, y in self.dynamic_walls:
            if self.grid[y, x] == self.WALL:
                self.grid[y, x] = self.EMPTY
            else:
                self.grid[y, x] = self.WALL

    def get_local_obs(self) -> dict:
        ax, ay = self.agent_pos
        local_grid = [[-1 for _ in range(5)] for _ in range(5)]
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                gx, gy = ax + dx, ay + dy
                if 0 <= gx < self.size and 0 <= gy < self.size:
                    cell = int(self.grid[gy, gx])
                    if [gx, gy] in self.enemies:
                        cell = self.ENEMY
                    local_grid[dy + 2][dx + 2] = cell
                else:
                    local_grid[dy + 2][dx + 2] = -1
        return {"grid": local_grid, "step": self.steps, "health": 1}

    def step(self, action: dict):
        if self.done:
            return

        self.steps += 1

        # Agent movement
        dx, dy = action.get("move", [0, 0])
        nx = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        ny = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        if self.grid[ny, nx] != self.WALL:
            self.agent_pos = [nx, ny]

        # Interaction
        if action.get("interact", False):
            cx, cy = self.agent_pos
            pos = (cx, cy)
            if pos in self.keys:
                self.keys_collected.add(pos)
                self.grid[cy, cx] = self.EMPTY
                self.keys.discard(pos)
            if pos == self.door_pos and len(self.keys_collected) >= 2:
                self.door_open = True
                self.grid[cy, cx] = self.EMPTY

        # Enemy movement (random walk)
        for e in self.enemies:
            edx = self.rng.randint(-1, 2)
            edy = self.rng.randint(-1, 2)
            ex = max(1, min(self.size - 2, e[0] + edx))
            ey = max(1, min(self.size - 2, e[1] + edy))
            if self.grid[ey, ex] != self.WALL:
                e[0], e[1] = ex, ey

        # Dynamic environment
        if self.steps % self.DYNAMIC_INTERVAL == 0:
            self._toggle_dynamic_walls()

        # Terminal checks
        agent_tuple = tuple(self.agent_pos)
        enemy_tuples = [tuple(e) for e in self.enemies]
        if agent_tuple in enemy_tuples:
            self.done = True
            self.reason = "caught"
            return

        if agent_tuple == self.exit_pos and self.door_open:
            self.done = True
            self.reason = "escaped"
            return

        if self.steps >= self.max_steps:
            self.done = True
            self.reason = "timeout"

    def run_episode(self, world_model_fn, planner_fn, seed: int = 0) -> dict:
        self.rng = np.random.RandomState(seed)
        self._build_maze()

        memory = None
        last_action = {"move": [0, 0], "interact": False}
        memory_snapshots = []
        true_states = []

        while not self.done:
            local_obs = self.get_local_obs()

            # Agent cognition
            memory = world_model_fn(memory, local_obs, last_action)
            action = planner_fn(memory, local_obs)

            # Audit trail for model accuracy
            memory_snapshots.append(copy.deepcopy(memory))
            true_states.append({
                "agent_pos": tuple(self.agent_pos),
                "enemies": [tuple(e) for e in self.enemies],
                "grid": self.grid.copy(),
                "keys_remaining": set(self.keys),
                "door_open": self.door_open,
                "step": self.steps,
            })

            self.step(action)
            last_action = action

        accuracy = self._compute_model_accuracy(memory_snapshots, true_states)
        score = self._compute_task_score()

        return {
            "score": score,
            "model_accuracy": accuracy,
            "steps": self.steps,
            "reason": self.reason,
            "keys_collected": len(self.keys_collected),
        }

    def _compute_task_score(self) -> float:
        s = self.steps * 0.1
        s += len(self.keys_collected) * 20.0
        if self.door_open:
            s += 30.0
        if self.reason == "escaped":
            s += 100.0
        if self.reason == "caught":
            s -= 50.0
        # Normalize to roughly [0, 1]
        return float(np.clip((s + 50.0) / 250.0, 0.0, 1.0))

    def _compute_model_accuracy(self, memory_snapshots, true_states) -> float:
        if not memory_snapshots:
            return 0.0

        correct = 0
        total = 0

        for mem, true in zip(memory_snapshots, true_states):
            believed_map = mem.get("believed_map", {})
            for (x, y), believed_cell in believed_map.items():
                if 0 <= x < self.size and 0 <= y < self.size:
                    true_cell = int(true["grid"][y, x])
                    if (x, y) in true["enemies"]:
                        true_cell = self.ENEMY
                    if believed_cell == true_cell:
                        correct += 1
                    total += 1

        return correct / total if total > 0 else 0.0


# =============================================================================
# 2. SHINKAEVOLVE INTEGRATION
# =============================================================================

def get_experiment_kwargs(run_idx: int) -> dict:
    maze = DynamicMaze(seed=run_idx, max_steps=200)
    return {"maze_world": maze, "seed": run_idx}


def validate_results(run_output: Any) -> Tuple[bool, str]:
    if not isinstance(run_output, dict):
        return False, "Output must be a dict"
    for key in ("score", "model_accuracy"):
        if key not in run_output:
            return False, f"Missing '{key}'"
        if not (0.0 <= run_output[key] <= 1.0):
            return False, f"'{key}' {run_output[key]} out of [0,1]"
    return True, None


def aggregate_metrics(results: List[dict], results_dir: str) -> dict:
    scores = [r["score"] for r in results]
    accuracies = [r["model_accuracy"] for r in results]
    steps = [r["steps"] for r in results]
    keys = [r["keys_collected"] for r in results]

    mean_score = float(np.mean(scores))
    mean_accuracy = float(np.mean(accuracies))

    # KEY DESIGN CHOICE: explicitly select for understanding, not just luck
    combined = 0.6 * mean_score + 0.4 * mean_accuracy

    return {
        "combined_score": float(np.clip(combined, 0.0, 1.0)),
        "public": {
            "mean_task_score": round(mean_score, 4),
            "mean_model_accuracy": round(mean_accuracy, 4),
            "avg_steps": round(float(np.mean(steps)), 1),
            "avg_keys": round(float(np.mean(keys)), 2),
            "num_episodes": len(results),
        },
        "private": {
            "all_scores": scores,
            "all_accuracies": accuracies,
            "termination_reasons": [r["reason"] for r in results],
        }
    }


# =============================================================================
# 3. ENTRY POINT
# =============================================================================

def main(program_path: str, results_dir: str):
    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_evaluation",
        num_runs=5,
        run_workers=1,
        get_experiment_kwargs=get_experiment_kwargs,
        validate_fn=validate_results,
        aggregate_metrics_fn=aggregate_metrics,
    )
    return metrics, correct, error_msg


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])
```

---

## Why This Fitness Design Matters

The most important line in the entire project is in `aggregate_metrics`:

```python
combined = 0.6 * mean_score + 0.4 * mean_accuracy
```

Without the `mean_accuracy` term, ShinkaEvolve would happily evolve agents that exploit statistical regularities of the random seed distribution, or agents that gamble on enemy spawns. With it, the LLM mutator is pressured to evolve **structured cognition**. You are explicitly selecting for programs that build a coherent internal representation of the world.

Over generations, you should see the LLM propose edits like:
- "Add a decay factor to enemy tracks so old beliefs fade."
- "Mark cells observed before a wall-toggle event as uncertain."
- "Use pathfinding over the believed_map instead of greedy Manhattan distance."
- "If the exit is behind the door and we lack keys, replan to nearest unexplored region."

These are **interpretable cognitive upgrades**, not opaque weight changes.

---

## LLM Mutation Strategy

To make this work well, your ShinkaEvolve task system message should explicitly frame the problem as cognitive architecture evolution:

> "You are evolving an agent mind for a partially observable dynamic maze. The agent has two functions: `world_model_step` maintains an internal map from limited 5x5 views, and `planner` chooses actions using that map. Improve the code so the agent escapes more mazes while keeping its internal believed_map accurate. Consider: spatial memory, enemy tracking, uncertainty handling, pathfinding, and dynamic wall prediction."

This prompt encourages the LLM to make *semantic* mutations (e.g., adding an A* search or a belief decay mechanism) rather than just shuffling syntax.

---

## Launch Configuration

```bash
shinka_run \
    --task-dir ./shinkadreamer \
    --results_dir ./results/shinkadreamer_run1 \
    --num_generations 100 \
    --set evo.llm_models='["gpt-5-mini","gemini-3-flash-preview"]' \
    --set db.num_islands=4
```

Because the evaluation is cheap (a pure Python gridworld), you can run many episodes per candidate and many islands in parallel, letting ShinkaEvolve's sample efficiency shine.

---

## Connection to the Book

This project sits at the intersection of several *Neuroevolution* themes:

- **Partial observability and memory** (Chapter 6): The evolved world model is essentially a recurrent internal state that allows the agent to act on information no longer visible.
- **Modeling the environment** (Chapter 6, Context+Skill Networks): The agent must learn to predict and represent structure rather than just map observations to actions.
- **Open-endedness** (Chapter 9): The space of possible world-modeling strategies is vast. The LLM can keep inventing new representational schemes (graphs, occupancy grids, enemy velocity estimates) without hitting the representational ceiling of a fixed neural topology.

If you run this, I would be very curious to see whether generation 5 produces simple map-remembering agents, while generation 50 produces agents with explicit path replanning and enemy-avoidance heuristics. That trajectory would be a beautiful demonstration of ShinkaEvolve evolving *minds*, not just policies.