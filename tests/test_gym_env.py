import importlib.util
import unittest

from test_env import make_episode


@unittest.skipUnless(importlib.util.find_spec("gymnasium"), "install requirements-env.txt")
class GymAdapterTests(unittest.TestCase):
    def test_official_gymnasium_checker(self):
        from gymnasium.utils.env_checker import check_env
        from gym_env import GymSellExecutionEnv

        env = GymSellExecutionEnv([make_episode()], 1.0)
        check_env(env, skip_render_check=True)

    def test_all_actions_terminate_with_valid_observations(self):
        from gym_env import GymSellExecutionEnv

        for action in range(4):
            env = GymSellExecutionEnv([make_episode()], 1.0)
            obs, _ = env.reset(seed=42)
            self.assertTrue(env.observation_space.contains(obs))
            for _ in range(60):
                obs, reward, terminated, truncated, info = env.step(action)
                self.assertTrue(env.observation_space.contains(obs))
                self.assertFalse(truncated)
                if terminated:
                    break
            self.assertTrue(terminated)
            self.assertTrue(info["summary"]["completed"])
            self.assertEqual(obs[1], 0)

    def test_episode_selection_is_seeded_and_can_be_explicit(self):
        from gym_env import GymSellExecutionEnv

        env = GymSellExecutionEnv([make_episode(), make_episode()], 1.0)
        first, first_info = env.reset(seed=18)
        second, second_info = env.reset(seed=18)
        self.assertEqual(first_info, second_info)
        self.assertTrue((first == second).all())
        _, info = env.reset(options={"episode_index": 1})
        self.assertEqual(info["episode_index"], 1)

    def test_incomplete_horizon_terminates_and_reports_residual(self):
        from gym_env import GymSellExecutionEnv

        env = GymSellExecutionEnv([make_episode([0.0] * 25)], 1.0)
        env.reset(seed=0)
        for _ in range(60):
            obs, reward, terminated, truncated, info = env.step(0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(reward, -10_000.0)
        self.assertEqual(obs[0], 1.0)
        self.assertFalse(info["summary"]["completed"])

    def test_reset_and_action_errors_are_explicit(self):
        from gym_env import GymSellExecutionEnv

        env = GymSellExecutionEnv([make_episode()], 1.0)
        with self.assertRaises(RuntimeError):
            env.step(0)
        with self.assertRaises(ValueError):
            env.reset(options={"episode_index": 2})
        env.reset()
        for action in (True, 1.0, -1, 4):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)


if __name__ == "__main__":
    unittest.main()
