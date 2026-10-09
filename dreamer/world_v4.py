"""Declared persistence/information factors; unchanged v3 physical maze semantics."""
import bisect
import random

from .world import stream_seed
from .world_v3 import UnknownDynamicsMaze, draw_law

CONDITIONS = ("uniform-full", "stationary-full", "switch-full",
              "stationary-late", "repeat25-full", "repeat25-late")


class FactorMaze(UnknownDynamicsMaze):
    def __init__(self, seed=0, max_steps=200, interval=25, condition="stationary-full"):
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown v4 condition: {condition}")
        self.condition = condition
        self.persistence, visibility = condition.split("-")
        self.enemy_visibility_radius = 1 if visibility == "late" else 2
        super().__init__(seed, max_steps, interval,
                         "stationary" if self.persistence == "repeat25" else self.persistence)

    def reset(self):
        obs = super().reset()
        self.laws = [self.initial_law]
        self.switch_transitions = []
        if self.persistence == "repeat25":
            # Private dwell times average 25 ticks. A fixed public schedule would
            # reveal future change times from the ordinary step observation.
            transition = 1
            segment = 0
            while True:
                timing = random.Random(stream_seed(self.seed, f"v4:switch-time:segment:{segment}"))
                transition += timing.randint(20, 30)
                if transition > self.max_steps:
                    break
                segment += 1
                rng = random.Random(stream_seed(self.seed, f"v4:law:segment:{segment}"))
                law = draw_law(rng)
                while sum(abs(a-b) for a, b in zip(self.laws[-1], law))/2 < .3:
                    law = draw_law(rng)
                self.laws.append(law)
                self.switch_transitions.append(transition)
            self.switch_step = self.switch_transitions[0] if self.switch_transitions else None
        elif self.switch_step is not None:
            self.switch_transitions = [self.switch_step]
        return obs

    def law_for_transition(self, transition):
        if self.persistence == "repeat25":
            return self.laws[bisect.bisect_right(self.switch_transitions, transition)]
        return super().law_for_transition(transition)

    def observe(self):
        obs = super().observe()
        radius = self.enemy_visibility_radius
        for y in range(5):
            for x in range(5):
                if max(abs(x-2), abs(y-2)) > radius:
                    obs["grid"][y][x] = obs["terrain"][y][x]
        obs["enemy_visibility_radius"] = radius
        return obs
