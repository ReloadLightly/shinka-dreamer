"""Unknown enemy dynamics extension; v2 geometry and action order are unchanged.

The switch_step is the one-indexed transition first using the replacement law.
Uniform v3 has the original uniform attempted-movement distribution, using one
inverse-CDF innovation per enemy rather than v2's random.choice implementation.
This intentional RNG implementation difference does not overwrite v2 results.
"""
import math
import random

from .world import DynamicMaze, MOVES, stream_seed

REGIMES = ("uniform", "stationary", "switch")
UNIFORM = (1 / 9,) * 9


def draw_law(rng):
    """Dirichlet(1,...,1), rejection truncated to the recorded initial proposal."""
    while True:
        values = [rng.expovariate(1) for _ in MOVES]
        total = sum(values)
        law = tuple(v / total for v in values)
        if max(law) <= .6 and -sum(p * math.log(p) for p in law) >= 1.2:
            return law


class _AttemptStream:
    def __init__(self, owner, seed):
        self.owner = owner
        self.rng = random.Random(stream_seed(seed, "v3:enemy-innovations"))
        self.draws = 0

    def choice(self, moves):
        assert moves == MOVES
        value = self.rng.random()
        self.draws += 1
        cumulative = 0.
        # DynamicMaze.step increments steps before requesting each enemy draw.
        for move, probability in zip(MOVES, self.owner.law_for_transition(self.owner.steps)):
            cumulative += probability
            if value < cumulative:
                return move
        return MOVES[-1]


class UnknownDynamicsMaze(DynamicMaze):
    def __init__(self, seed=0, max_steps=200, interval=25, regime="stationary"):
        if regime not in REGIMES:
            raise ValueError(f"Unknown v3 dynamics regime: {regime}")
        self.regime = regime
        super().__init__(seed, max_steps, interval)

    def reset(self):
        observation = super().reset()
        initial_rng = random.Random(stream_seed(self.seed, "v3:law:initial"))
        replacement_rng = random.Random(stream_seed(self.seed, "v3:law:replacement"))
        initial = draw_law(initial_rng)
        replacement = draw_law(replacement_rng)
        while sum(abs(a - b) for a, b in zip(initial, replacement)) / 2 < .3:
            replacement = draw_law(replacement_rng)
        self.initial_law = UNIFORM if self.regime == "uniform" else initial
        self.replacement_law = replacement
        self.switch_step = (random.Random(stream_seed(self.seed, "v3:switch-time")).randint(25, 75)
                            if self.regime == "switch" else None)
        self.enemy_rng = _AttemptStream(self, self.seed)
        return observation

    def law_for_transition(self, transition):
        if self.switch_step is not None and transition >= self.switch_step:
            return self.replacement_law
        return self.initial_law

    @property
    def current_law(self):
        """Evaluator-only probabilities governing the NEXT enemy transition."""
        return self.law_for_transition(self.steps + 1)
