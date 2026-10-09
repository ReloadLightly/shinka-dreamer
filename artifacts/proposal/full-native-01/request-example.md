# System Instructions

You are evolving an agent mind for a partially observable dynamic maze. The
agent's world_model_step maintains an internal map from limited 5x5 views, and
planner chooses actions using that map. Improve escape and current believed-map
accuracy. Consider spatial memory, anonymous enemy sightings, uncertainty,
pathfinding and dynamic walls. Both functions, representations and helpers may
evolve freely inside the EVOLVE block.

The maze is 15x15, with two keys, a gating door, exit, three anonymous enemies,
walls toggling every25steps and a200step horizon. Nine attempted displacements
include waiting; blocked moves stay and diagonal corner cutting is forbidden.
Interaction opens an adjacent cardinal door before movement when both keys are
held; interaction also collects a key at the destination. Contact with an enemy
before or after enemy movement is fatal. Enemies use uniform attempted moves.

world_model_step(memory,local_obs,last_action) returns memory; planner returns
{'move':[dx,dy], 'interact':bool}. The fixed wrapper reads memory['believed_map'],
a dictionary {(x,y):cell_type} in coordinates relative to the initial agent
position. Other internal representations are unrestricted. Observations provide
grid,terrain,step,health,keys,door_open and feedback with actual displacement,
collected and opened. Enemy identities, hidden world and seeds are not supplied.

The original task reward is clip((.1*steps+20*keys+30*door_open+
100*escaped-50*caught+50)/250,0,1). Model accuracy is the fraction of correct
reported in-bounds map entries accumulated after observation and before action
outcomes; enemy cells use code5. Episode fitness=.6*task+.4*accuracy; invalid
executions score0. A population's score is its episode mean. Missing maps score0.
Coverage and escape/death/timeout are reported separately. This is current-state
reconstruction, not future prediction. No learned transition predictor is required.

Use Python3.10 standard library,192MiB,10CPU-seconds/episode,3s/reply. Do not access
the environment, files, processes or network. Return the native requested patch
or full module. Do not call tools or inspect the repository.

You MUST respond using an edit name, description, and the exact SEARCH/REPLACE diff format shown below to indicate changes:

<NAME>
A shortened name summarizing the edit you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
A description and argumentation process of the edit you are proposing.
</DESCRIPTION>

<DIFF>
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE

</DIFF>


Example of a valid diff format:
<DIFF>
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

</DIFF>

* You may only modify text that lies below a line containing "EVOLVE-BLOCK-START" and above the next "EVOLVE-BLOCK-END". Everything outside those markers is read-only.
* Do not repeat the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the SEARCH/REPLACE blocks.  
* Every block’s SEARCH section must be copied **verbatim** from the current file, including indentation.
* You can propose multiple independent edits. SEARCH/REPLACE blocks follow one after another. DO NOT ADD ANY OTHER TEXT BETWEEN THESE BLOCKS.
* Make sure the file still runs after your changes.

NON-EVOLVING ORIGINAL-PROPOSAL BOUNDARY: Jointly evolve world_model_step and planner, including helpers and arbitrary internal representations. Export current memory["believed_map"] in initial-relative coordinates. The fixed 15x15 maze, local observations, 200-step horizon, evaluator, isolation and resource rules cannot be edited. Selection is the original 0.6 task reward + 0.4 current-map accuracy, not a forecast objective. Do not inspect files, tools, seeds, hidden state or assessment pools. All candidate network/process/model calls are forbidden.

# Previous Messages

[]

# User Request


# Current program

Here is the current program we are trying to improve (you will need to propose a modification to it below):

```python
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

```

Here are the performance metrics of the program:

Combined score to maximize: 0.54
episodes: 5; mean_task_score: 0.24; mean_model_accuracy: 0.99; avg_steps: 150.00; avg_keys: 1.20; mean_map_coverage: 0.34; escaped: 0; caught: 3; timeout: 2; invalid: 0

Here is additional text feedback about the current program:

namazu-proposal-reconstruction-v1. Original .6 task + .4 current-map accuracy. {"episodes": 5, "mean_task_score": 0.23600000000000004, "mean_model_accuracy": 0.9869041845779242, "avg_steps": 150.0, "avg_keys": 1.2, "mean_map_coverage": 0.3431111111111111, "escaped": 0, "caught": 3, "timeout": 2, "invalid": 0}. Improve both map updating and planning. This is reconstruction after observation, not future prediction. Accuracy uses reported in-bounds map cells; coverage is diagnostic. Inspect escapes and failures separately from the weighted score.


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
