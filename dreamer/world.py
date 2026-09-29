"""Namazu repair v1: local-square observation, gated exit, independent RNGs."""
import hashlib
import random
from collections import deque

EMPTY, WALL, KEY, DOOR, EXIT, ENEMY = range(6)
MOVES = [(x, y) for y in (-1, 0, 1) for x in (-1, 0, 1)]


def stream_seed(seed, stream):
    return int.from_bytes(hashlib.sha256(f"{seed}:{stream}".encode()).digest()[:8], "big")


def valid_action(action):
    if not isinstance(action, dict):
        return False
    move = action.get("move")
    return (isinstance(move, (list, tuple)) and len(move) == 2
            and all(type(v) is int and -1 <= v <= 1 for v in move)
            and type(action.get("interact")) is bool)


class DynamicMaze:
    size = 15

    def __init__(self, seed=0, max_steps=200, interval=25):
        self.seed, self.max_steps, self.interval = seed, max_steps, interval
        self.reset()

    def reset(self):
        self.layout_rng = random.Random(stream_seed(self.seed, "layout"))
        self.enemy_rng = random.Random(stream_seed(self.seed, "enemies"))
        self.steps, self.keys_collected, self.door_open = 0, 0, False
        self.done, self.reason = False, "timeout"
        self.feedback = {"displacement": [0, 0], "collected": False, "opened": False}
        rng = self.layout_rng
        # Exit chamber at a randomly rotated corner. Its only entrance is the door.
        # All dynamic cells begin closed. Connected closed-state routes are thus
        # feasible for EVERY wall phase (opening only adds traversable cells).
        for _ in range(1000):
            self.grid = [[WALL if x in (0, 14) or y in (0, 14) else EMPTY
                          for x in range(15)] for y in range(15)]
            exit_pos, door_pos = (13, 13), (12, 13)
            self.grid[12][13] = self.grid[12][12] = WALL
            reserve = {(13, 13), (12, 13), (11, 13), (13, 12), (12, 12)}
            cells = [(x, y) for y in range(1, 14) for x in range(1, 14)
                     if (x, y) not in reserve]
            walls = rng.sample(cells, rng.randint(20, 34))
            for x, y in walls:
                self.grid[y][x] = WALL
            self.grid[13][13], self.grid[13][12] = EXIT, DOOR
            free = [p for p in cells if self.grid[p[1]][p[0]] == EMPTY]
            seen, queue = {free[0]}, deque([free[0]])
            while queue:
                x, y = queue.popleft()
                for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    q = x + dx, y + dy
                    if q not in seen and self.grid[q[1]][q[0]] == EMPTY:
                        seen.add(q)
                        queue.append(q)
            if len(seen) == len(free) + 1:  # includes reserved door approach
                break
        else:
            raise RuntimeError("Unable to build connected closed-state maze")
        self.dynamic_walls = set(rng.sample(walls, max(1, len(walls) // 4)))
        self.agent_pos, *keys = rng.sample(free, 3)
        for x, y in keys:
            self.grid[y][x] = KEY
        enemy_cells = [p for p in free if p != self.agent_pos and p not in keys
                       and max(abs(p[0] - self.agent_pos[0]), abs(p[1] - self.agent_pos[1])) > 2]
        self.enemies = rng.sample(enemy_cells, 3)
        self.exit_pos, self.door_pos = exit_pos, door_pos
        # Rotation is a layout choice, not information exposed to candidates.
        for _ in range(rng.randrange(4)):
            self.grid = [list(row) for row in zip(*self.grid[::-1])]
            rotate = lambda p: (14 - p[1], p[0])
            self.agent_pos = rotate(self.agent_pos)
            self.enemies = list(map(rotate, self.enemies))
            self.dynamic_walls = set(map(rotate, self.dynamic_walls))
            self.exit_pos, self.door_pos = rotate(self.exit_pos), rotate(self.door_pos)
        self.origin = self.agent_pos
        return self.observe()

    def blocked(self, p):
        x, y = p
        return (not (0 <= x < 15 and 0 <= y < 15)
                or self.grid[y][x] == WALL
                or (p == self.door_pos and not self.door_open))

    def can_move(self, pos, move):
        x, y = pos
        dx, dy = move
        return (not self.blocked((x + dx, y + dy))
                and not (dx and dy and (self.blocked((x + dx, y)) or self.blocked((x, y + dy)))))

    def observe(self):
        ax, ay = self.agent_pos
        terrain, grid = [], []
        for dy in range(-2, 3):
            tr, gr = [], []
            for dx in range(-2, 3):
                p = ax + dx, ay + dy
                cell = self.grid[p[1]][p[0]] if 0 <= p[0] < 15 and 0 <= p[1] < 15 else -1
                tr.append(cell)
                gr.append(ENEMY if p in self.enemies else cell)
            terrain.append(tr)
            grid.append(gr)
        return {"grid": grid, "terrain": terrain, "step": self.steps,
                "health": 1, "keys": self.keys_collected, "door_open": self.door_open,
                "feedback": dict(self.feedback)}

    def step(self, action):
        if self.done:
            raise RuntimeError("Terminal episode")
        if not valid_action(action):
            raise ValueError("Invalid action")
        old = self.agent_pos
        self.feedback = {"displacement": [0, 0], "collected": False, "opened": False}
        # Interact before movement: adjacent cardinal door can open this turn.
        if action["interact"] and not self.door_open and self.keys_collected == 2 and sum(abs(a-b) for a,b in zip(old, self.door_pos)) == 1:
            self.door_open = True
            self.grid[self.door_pos[1]][self.door_pos[0]] = EMPTY
            self.feedback["opened"] = True
        if self.can_move(old, action["move"]):
            self.agent_pos = tuple(a + b for a, b in zip(old, action["move"]))
        self.feedback["displacement"] = [a - b for a, b in zip(self.agent_pos, old)]
        x, y = self.agent_pos
        if action["interact"] and self.grid[y][x] == KEY:
            self.grid[y][x] = EMPTY
            self.keys_collected += 1
            self.feedback["collected"] = True
        self.steps += 1
        caught = self.agent_pos in self.enemies
        # Three independent random displacement draws every turn, even if blocked.
        # Occupancy can overlap; observations reveal occupancy, never enemy IDs.
        new_enemies = []
        for enemy in self.enemies:
            move = self.enemy_rng.choice(MOVES)
            new_enemies.append(tuple(a+b for a,b in zip(enemy, move)) if self.can_move(enemy, move) else enemy)
        self.enemies = new_enemies
        if self.steps % self.interval == 0:
            for wx, wy in sorted(self.dynamic_walls):
                if self.grid[wy][wx] == WALL:
                    self.grid[wy][wx] = EMPTY
                elif (wx, wy) not in self.enemies and (wx, wy) != self.agent_pos:
                    self.grid[wy][wx] = WALL
                # Occupied closure skipped, retried only at NEXT scheduled toggle.
        if caught or self.agent_pos in self.enemies:
            self.done, self.reason = True, "caught"
        elif self.agent_pos == self.exit_pos and self.door_open:
            self.done, self.reason = True, "escaped"
        elif self.steps >= self.max_steps:
            self.done = True
        return self.observe()

    def snapshot(self):
        return {"grid": self.grid, "enemies": self.enemies, "agent": self.agent_pos,
                "origin": self.origin, "step": self.steps, "keys": self.keys_collected,
                "door_open": self.door_open, "reason": self.reason if self.done else "running"}
