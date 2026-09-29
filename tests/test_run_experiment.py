"""Integrity checks use tiny temporary inputs; no training or evaluation runs."""

from copy import deepcopy
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from data import SPLIT_DAYS
from run_experiment import verify_training_inputs


class TrainingIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name) / "project"
        self.default_data = self.project / "data"
        self.custom_data = Path(self.temp.name) / "custom_data"
        self.config = {"training_manifest": []}
        for index, day in enumerate(SPLIT_DAYS["training"]):
            payload = f"synthetic training day {index}\n".encode()
            relative = Path("training") / f"{day}_5s.csv.gz"
            for root in (self.default_data, self.custom_data):
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                (root / relative).write_bytes(payload)
            self.config["training_manifest"].append({
                "path": str(Path("data") / relative),
                "sha256": hashlib.sha256(payload).hexdigest(),
            })

    def test_default_data_root_covers_all_training_days(self):
        self.assertEqual(verify_training_inputs(self.config, project=self.project), self.default_data)

    def test_absolute_override_checks_actual_inputs_instead_of_default_files(self):
        self.config["data_dir"] = str(self.custom_data)
        self.assertEqual(verify_training_inputs(self.config, project=self.project), self.custom_data)
        first = self.custom_data / "training" / "Sep2020_5s.csv.gz"
        first.write_bytes(b"changed custom data while default data still matches")
        with self.assertRaisesRegex(ValueError, "differs from calibration"):
            verify_training_inputs(self.config, project=self.project)

    def test_relative_override_resolves_against_project(self):
        self.config["data_dir"] = "../custom_data"
        # The default data is deliberately unavailable, so accidental fallback fails.
        (self.default_data / "training" / "Sep2020_5s.csv.gz").unlink()
        resolved = verify_training_inputs(self.config, project=self.project)
        self.assertEqual(resolved.resolve(), self.custom_data.resolve())

    def test_missing_duplicate_or_unexpected_manifest_coverage_is_rejected(self):
        incomplete = deepcopy(self.config)
        incomplete["training_manifest"].pop()
        duplicate = deepcopy(self.config)
        duplicate["training_manifest"].append(duplicate["training_manifest"][0].copy())
        unexpected = deepcopy(self.config)
        unexpected["training_manifest"][0]["path"] = "data/testing/Mar2021_5s.csv.gz"
        for config, message in ((incomplete, "missing"), (duplicate, "duplicate"), (unexpected, "unexpected")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    verify_training_inputs(config, project=self.project)

    def test_absolute_and_traversing_manifest_paths_cannot_escape_data_root(self):
        for label in (str(self.custom_data / "training" / "Sep2020_5s.csv.gz"),
                      "data/../custom_data/training/Sep2020_5s.csv.gz"):
            with self.subTest(label=label):
                config = deepcopy(self.config)
                config["training_manifest"][0]["path"] = label
                with self.assertRaisesRegex(ValueError, "unexpected"):
                    verify_training_inputs(config, project=self.project)

    def test_missing_actual_input_fails_without_default_fallback(self):
        self.config["data_dir"] = str(self.custom_data)
        (self.custom_data / "training" / "Sep2020_5s.csv.gz").unlink()
        with self.assertRaises(FileNotFoundError):
            verify_training_inputs(self.config, project=self.project)


if __name__ == "__main__":
    unittest.main()
