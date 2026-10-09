"""Focused checks for the proposal's coordinate, objective and time semantics."""
import unittest
from proposal.evaluation import audit_map, task_score, aggregate
from proposal.initial import world_model_step
from dreamer.world import DynamicMaze


class ProposalTests(unittest.TestCase):
    def test_original_task_reward(self):
        self.assertAlmostEqual(task_score(100, 2, True, 'escaped'), .92)
        self.assertAlmostEqual(task_score(200, 0, False, 'timeout'), .28)
        self.assertAlmostEqual(task_score(10, 0, False, 'caught'), .004)

    def test_relative_coordinates_and_observation_reconstruction(self):
        env = DynamicMaze(0)
        obs = env.observe()
        memory = world_model_step(None, obs, None)
        belief = [[x, y, cell] for (x, y), cell in memory['believed_map'].items()]
        correct, total = audit_map(belief, env)
        self.assertGreater(total, 0)
        self.assertEqual(correct, total)
        self.assertEqual(audit_map([], env), (0, 0))

    def test_localization_uses_outcome_not_attempt(self):
        env = DynamicMaze(0)
        obs = env.observe()
        memory = world_model_step(None, obs, None)
        obs['feedback']['displacement'] = [0, 0]
        memory = world_model_step(memory, obs, {'move': [1, 1], 'interact': False})
        self.assertEqual(memory['agent_pos'], (0, 0))
        obs['feedback']['displacement'] = [1, 0]
        memory = world_model_step(memory, obs, {'move': [1, 1], 'interact': False})
        self.assertEqual(memory['agent_pos'], (1, 0))

    def test_bad_map_does_not_enter_accuracy(self):
        with self.assertRaises(ValueError):
            audit_map([[0, 0, '5']], DynamicMaze(0))

    def test_invalid_stays_in_fitness_denominator(self):
        good = dict(task=.8, model_accuracy=.9, combined_score=.84, steps=100,
                    keys=2, final_coverage=.5, reason='escaped')
        bad = {**good, 'combined_score': 0., 'reason': 'invalid'}
        result = aggregate([good, bad])
        self.assertAlmostEqual(result['combined_score'], .42)
        self.assertEqual(result['public']['invalid'], 1)


if __name__ == '__main__':
    unittest.main()
