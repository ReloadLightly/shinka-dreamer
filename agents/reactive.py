"""Local-only legal random movement with immediate enemy avoidance."""
import random


def world_model_step(memory, local_obs, last_action):
    return None


def planner(memory, local_obs):
    grid = local_obs["grid"]
    moves = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if grid[2+dy][2+dx] in (0, 2, 4) and not (dx and dy and (grid[2][2+dx] in (1, 3) or grid[2+dy][2] in (1, 3))):
                moves.append([dx, dy])
    return {"move": random.choice(moves or [[0, 0]]), "interact": True}
