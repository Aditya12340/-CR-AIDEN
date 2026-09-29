import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from train import evenly_spaced_indices, training_settings


class TrainingConfigurationTests(unittest.TestCase):
    def test_validation_subset_is_fixed_evenly_spaced_and_unique(self):
        indices = evenly_spaced_indices(285, 32)
        self.assertEqual(len(indices), 32)
        self.assertEqual(len(set(indices)), 32)
        self.assertEqual((indices[0], indices[-1]), (0, 284))
        self.assertTrue(all(right - left in (9, 10) for left, right in zip(indices, indices[1:])))
        self.assertEqual(evenly_spaced_indices(3, 32), (0, 1, 2))
        self.assertEqual(evenly_spaced_indices(5, 1), (2,))

    def test_nested_training_budget_overrides_top_level(self):
        self.assertEqual(training_settings({"total_timesteps": 100})["total_timesteps"], 100)
        config = {"total_timesteps": 100, "training": {"total_timesteps": 200}}
        self.assertEqual(training_settings(config)["total_timesteps"], 200)
        self.assertEqual(training_settings({})["ppo"]["gamma"], 1.0)

    def test_invalid_or_misspelled_options_are_rejected(self):
        for config in (
            {"training": {"ppo": {"n_steps": 0}}},
            {"training": {"ppo": {"gamma": 1.1}}},
            {"training": {"ppo": {"batch_size": 100}}},
            {"training": {"ppo": {"learning_rate": float("nan")}}},
            {"training": {"ppo": {"lr": 0.01}}},
            {"total_timesteps": True},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                training_settings(config)


@unittest.skipUnless(importlib.util.find_spec("stable_baselines3"), "install requirements-train.txt")
class TrainingSmokeTests(unittest.TestCase):
    def test_real_ppo_updates_saves_reloads_and_selects_only_trained_checkpoints(self):
        from gym_env import GymSellExecutionEnv
        from test_env import make_episode
        from train import evaluate_checkpoint, load_trained_policy, train_seed

        config = {
            "sizes": {"small": 0.5, "medium": 1.0, "large": 2.0},
            "primary_size": "medium", "fee_bps": 10.0,
            "training": {"total_timesteps": 64, "validation_interval": 32,
                         "validation_episodes": 1, "torch_threads": 1,
                         "ppo": {"n_steps": 32, "batch_size": 16, "n_epochs": 1}},
        }
        episodes = [make_episode()]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seed_7"
            manifest = train_seed(episodes, episodes, config, output, 7)
            self.assertEqual(manifest["actual_timesteps"], 64)
            self.assertIn(manifest["best_checkpoint_timesteps"], (32, 64))
            self.assertNotEqual(manifest["initial_parameter_sha256"], manifest["final_parameter_sha256"])
            for filename in ("model.zip", "best_model.zip", "last_model.zip", "manifest.json",
                             "selection.json", "progress.csv", "validation.csv", "episodes.monitor.csv"):
                self.assertTrue((output / filename).is_file(), filename)
            with (output / "validation.csv").open(newline="") as handle:
                records = list(csv.DictReader(handle))
            self.assertEqual([int(record["num_timesteps"]) for record in records], [0, 32, 64])
            self.assertEqual(records[0]["eligible_for_selection"], "False")
            model = load_trained_policy(output / "model.zip")
            env = GymSellExecutionEnv(episodes, 1.0, fee_bps=10.0)
            metrics = evaluate_checkpoint(model, env, (0,))
            self.assertAlmostEqual(metrics["mean_completion_score_bps"], manifest["best_validation_score_bps"])
            self.assertEqual(model.policy.mlp_extractor.policy_net[0].out_features, 64)
            self.assertEqual(model.policy.mlp_extractor.value_net[0].out_features, 64)
            self.assertIsNot(model.policy.mlp_extractor.policy_net[0].weight,
                             model.policy.mlp_extractor.value_net[0].weight)
            saved = json.loads((output / "manifest.json").read_text())
            self.assertEqual(saved["seed"], 7)
            self.assertEqual(saved["validation_episode_indices"], [0])
            with self.assertRaises(FileExistsError):
                train_seed(episodes, episodes, config, output, 7)
            env.close()


if __name__ == "__main__":
    unittest.main()
