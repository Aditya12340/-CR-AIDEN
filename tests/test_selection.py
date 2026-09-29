"""Selection bookkeeping tests with synthetic metadata, not research results.

Dummy checkpoint bytes exercise artifact integrity without training or evaluating
on generated market data. Reported research outcomes come from the real splits.
"""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from select_model import choose_model, freeze_selection, sha256


def selection_fixture():
    config = {
        "seeds": [7, 19, 42], "primary_size": "medium", "fee_bps": 10.0,
        "sizes": {"small": 0.1, "medium": 0.5, "large": 2.0},
        "splits": {"validation": ["Feb2021"]},
    }
    groups = []
    for seed, score, filled_only_cost, completion in (
        (7, 12.0, -50.0, 0.9), (19, 8.0, 30.0, 1.0), (42, 9.0, 0.0, 1.0)
    ):
        groups.append({
            "policy": f"ppo_seed_{seed}", "size": "medium",
            "parent_quantity_btc": 0.5, "fee_bps": 10.0, "episodes": 285,
            "completion_score_bps": {"mean": score, "n": 285},
            "net_shortfall_bps": {"mean": filled_only_cost},
            "completion_rate": completion,
        })
    summary = {
        "metadata": {"split": "validation", "diagnostic_subset": False,
                     "max_episodes": None, "fee_bps": 10.0,
                     "episode_count": 285, "day_count": 1,
                     "sizes": config["sizes"].copy(),
                     "load_stats": {"files_loaded": 1, "complete_episodes": 285}},
        "groups": groups,
    }
    return config, summary


def write_artifact_fixture(directory):
    root = Path(directory)
    config, summary = selection_fixture()
    config_path = root / "experiment.json"
    config_path.write_text(json.dumps(config))
    validation_dir = root / "validation"
    validation_dir.mkdir()
    # Freeze hashes these bytes; no synthetic execution result is calculated.
    (validation_dir / "per_episode.csv").write_text("fixture_metadata_only\n")
    models_dir = root / "models"
    models = {}
    for seed in config["seeds"]:
        path = models_dir / f"seed_{seed}" / "model.zip"
        path.parent.mkdir(parents=True)
        path.write_bytes(f"dummy checkpoint bytes, seed {seed}".encode())
        models[f"ppo_seed_{seed}"] = {"path": str(path), "sha256": sha256(path)}
    summary["metadata"]["models"] = models
    summary["metadata"]["config_sha256"] = sha256(config_path)
    (validation_dir / "summary.json").write_text(json.dumps(summary))
    return config_path, validation_dir, models_dir, root / "selection.json"


class SelectionCriterionTests(unittest.TestCase):
    def test_selects_full_order_completion_score_at_primary_size(self):
        config, summary = selection_fixture()
        # Another size and a baseline have tempting costs but are ineligible.
        other_size = deepcopy(summary["groups"][0])
        other_size.update(size="small", parent_quantity_btc=0.1)
        other_size["completion_score_bps"]["mean"] = -1_000.0
        baseline = deepcopy(summary["groups"][0])
        baseline["policy"] = "twap"
        baseline["completion_score_bps"]["mean"] = -2_000.0
        summary["groups"].extend([other_size, baseline])

        selected, candidates = choose_model(summary, config)

        self.assertEqual(selected["seed"], 19)
        self.assertEqual(selected["mean_completion_score_bps"], 8.0)
        self.assertEqual([row["seed"] for row in candidates], [19, 42, 7])

    def test_exact_score_ties_choose_lower_seed_not_config_order(self):
        config, summary = selection_fixture()
        config["seeds"] = [42, 19, 7]
        for group in summary["groups"]:
            group["completion_score_bps"]["mean"] = 8.0

        selected, candidates = choose_model(summary, config)

        self.assertEqual(selected["seed"], 7)
        self.assertEqual([row["seed"] for row in candidates], [7, 19, 42])

    def test_rejects_testing_and_diagnostic_results(self):
        for changes in (
            {"split": "testing"}, {"split": "training"},
            {"diagnostic_subset": True}, {"max_episodes": 32},
        ):
            config, summary = selection_fixture()
            summary["metadata"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                choose_model(summary, config)

    def test_rejects_missing_duplicate_or_partial_candidate(self):
        for defect in ("missing", "duplicate", "partial"):
            config, summary = selection_fixture()
            if defect == "missing":
                summary["groups"].pop()
            elif defect == "duplicate":
                summary["groups"].append(deepcopy(summary["groups"][0]))
            else:
                summary["groups"][0]["episodes"] = 284
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                choose_model(summary, config)

    def test_rejects_changed_fee_and_primary_quantity(self):
        for scope, field, value in (
            ("metadata", "fee_bps", 0.0),
            ("group", "fee_bps", 20.0),
            ("group", "parent_quantity_btc", 0.25),
        ):
            config, summary = selection_fixture()
            target = summary["metadata"] if scope == "metadata" else summary["groups"][0]
            target[field] = value
            with self.subTest(scope=scope, field=field), self.assertRaises(ValueError):
                choose_model(summary, config)

    def test_rejects_survivor_only_nonfinite_or_empty_score_aggregates(self):
        for defect in ("survivors", "nan", "infinite", "empty", "completion_rate"):
            config, summary = selection_fixture()
            if defect == "survivors":
                summary["groups"][0]["completion_score_bps"]["n"] = 280
            elif defect in ("nan", "infinite"):
                summary["groups"][0]["completion_score_bps"]["mean"] = (
                    float("nan") if defect == "nan" else float("inf")
                )
            elif defect == "completion_rate":
                summary["groups"][0]["completion_rate"] = 1.1
            else:
                summary["metadata"]["episode_count"] = 0
                for group in summary["groups"]:
                    group["episodes"] = 0
                    group["completion_score_bps"]["n"] = 0
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                choose_model(summary, config)


class SelectionArtifactTests(unittest.TestCase):
    def test_freeze_records_all_checkpoint_hashes_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_artifact_fixture(directory)
            record = freeze_selection(*paths)
            self.assertEqual(record["selected"]["seed"], 19)
            self.assertEqual(record["selection_split"], "validation")
            self.assertEqual(record["selection_episodes"], 285)
            self.assertEqual(set(record["checkpoints"]), {"ppo_seed_7", "ppo_seed_19", "ppo_seed_42"})
            self.assertEqual(record["config_sha256"], sha256(paths[0]))
            for checkpoint in record["checkpoints"].values():
                self.assertEqual(checkpoint["sha256"], sha256(checkpoint["path"]))
            saved = paths[-1].read_bytes()
            with self.assertRaises(FileExistsError):
                freeze_selection(*paths)
            self.assertEqual(paths[-1].read_bytes(), saved)

    def test_rejects_config_changed_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_artifact_fixture(directory)
            config = json.loads(paths[0].read_text())
            config["fee_bps"] = 20.0
            paths[0].write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                freeze_selection(*paths)
            self.assertFalse(paths[-1].exists())

    def test_rejects_changed_checkpoint_even_if_it_was_not_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = write_artifact_fixture(directory)
            checkpoint = paths[2] / "seed_7" / "model.zip"
            checkpoint.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                freeze_selection(*paths)
            self.assertFalse(paths[-1].exists())


if __name__ == "__main__":
    unittest.main()
