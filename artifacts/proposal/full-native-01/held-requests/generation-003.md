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

Redesign the program with a different structural approach while potentially using similar core concepts.
Focus on changing the overall architecture, data flow, or program organization.
You MUST respond using a short summary name, description and the full code:

<NAME>
A shortened name summarizing the code you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
Describe the structural changes you are making and how they improve the program's performance, maintainability, or efficiency.
</DESCRIPTION>

<CODE>
```{language}
# The structurally redesigned program here.
```
</CODE>

* Keep the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the code.
* Focus on changing the program's structure: modularization, data flow, control flow, or architectural patterns.
* The core problem-solving approach may be similar but organized differently.
* Ensure the same inputs and outputs are maintained.
* Use the <NAME>, <DESCRIPTION>, and <CODE> delimiters to structure your response. It will be parsed afterwards.

NON-EVOLVING ORIGINAL-PROPOSAL BOUNDARY: Jointly evolve world_model_step and planner, including helpers and arbitrary internal representations. Export current memory["believed_map"] in initial-relative coordinates. The fixed 15x15 maze, local observations, 200-step horizon, evaluator, isolation and resource rules cannot be edited. Selection is the original 0.6 task reward + 0.4 current-map accuracy, not a forecast objective. Do not inspect files, tools, seeds, hidden state or assessment pools. All candidate network/process/model calls are forbidden.

# Previous Messages

[]

# User Request

The previous attempt failed. Try again.
