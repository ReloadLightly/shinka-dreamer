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
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            cell = grid[dy + 2][dx + 2]
            if cell == -1:
                continue
            pos = (ax + dx, ay + dy)
            memory["believed_map"][pos] = cell
            if cell == 5:
                # A location and timestamp are a sighting, not an enemy track.
                memory["enemy_sightings"][pos] = memory["step"]

    # Dynamic-wall reasoning remains the original placeholder for evolution.
    return memory


def planner(memory: dict, local_obs: dict) -> dict:
    """Use the proposal's cautious random walk and nearest-target gradient."""
    import random

    believed_map = memory.get("believed_map", {})
    agent_pos = memory.get("agent_pos", (0, 0))
    current_step = memory.get("step", 0)
    action = {
        "move": [random.choice([-1, 0, 1]), random.choice([-1, 0, 1])],
        "interact": False,
    }

    targets = []
    avoid = set()
    for pos, cell in believed_map.items():
        if cell == 2 and pos not in memory.get("keys_collected", set()):
            targets.append((pos, "key"))
        if cell == 4:
            targets.append((pos, "exit"))
        if cell == 3 and memory.get("keys", 0) >= 2:
            targets.append((pos, "door"))

    # Preserve the original five-step persistence heuristic for sightings.
    for pos, seen_step in memory.get("enemy_sightings", {}).items():
        if current_step - seen_step <= 5:
            avoid.add(pos)

    if targets:
        best_target = min(
            targets,
            key=lambda item: abs(item[0][0] - agent_pos[0])
            + abs(item[0][1] - agent_pos[1]),
        )
        tx, ty = best_target[0]
        dx = max(-1, min(1, tx - agent_pos[0]))
        dy = max(-1, min(1, ty - agent_pos[1]))
        action["move"] = [dx, dy]
        if abs(tx - agent_pos[0]) + abs(ty - agent_pos[1]) <= 1:
            action["interact"] = True

    nx = agent_pos[0] + action["move"][0]
    ny = agent_pos[1] + action["move"][1]
    if (nx, ny) in avoid:
        safe_moves = [
            [mx, my]
            for mx in [-1, 0, 1]
            for my in [-1, 0, 1]
            if (agent_pos[0] + mx, agent_pos[1] + my) not in avoid
        ]
        if safe_moves:
            action["move"] = random.choice(safe_moves)
    return action
# EVOLVE-BLOCK-END
