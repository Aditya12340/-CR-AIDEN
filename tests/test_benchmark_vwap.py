"""Software regressions for archived obligations and shared VWAP execution.

The small hand-built books here test accounting contracts; published research
results are generated separately from the actual processed market-data files.
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from benchmark_vwap import archived_records, run_vwap
from evaluate import evaluate_episode, write_csv
from test_env import make_episode


def make_config():
    return {
        "fee_bps": 10.0,
        "sizes": {"medium": 0.1},
        "market_regimes": {
            "arrival_spread_bps": {"lower": 1.0, "upper": 2.0},
            "episode_realized_volatility_bps": {"lower": 1.0, "upper": 2.0},
        },
    }


class ArchivedVwapBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.config = make_config()
        twap, _ = evaluate_episode(
            make_episode(), "twap", 0.1, size="medium", split="validation",
            fee_bps=self.config["fee_bps"], market_regimes=self.config["market_regimes"],
        )
        # A named fixture record exercises lookup; it is not a trained model.
        selected = {**twap, "policy": "ppo_fixture"}
        self.records = [twap, selected]

    def read_archive(self, records):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            (study / "validation").mkdir()
            write_csv(study / "validation" / "per_episode.csv", records)
            return archived_records(study, "validation", "ppo_fixture", self.config)

    def test_exact_episode_size_policy_keys_preserve_saved_accounting(self):
        indexed = self.read_archive(self.records)
        episode_id = self.records[0]["episode_id"]
        self.assertEqual(set(indexed), {
            (episode_id, "medium", "twap"),
            (episode_id, "medium", "ppo_fixture"),
        })
        for expected in self.records:
            row = indexed[(episode_id, "medium", expected["policy"])]
            self.assertEqual(row, expected)
            self.assertIs(row["completed"], True)

    def test_rejects_changed_fee_quantity_size_and_split(self):
        for field, value in (("fee_bps", 0.0), ("parent_quantity_btc", 0.2),
                             ("size", "unknown"), ("split", "testing")):
            with self.subTest(field=field):
                rows = deepcopy(self.records)
                rows[0][field] = value
                with self.assertRaisesRegex(ValueError, "configured obligation"):
                    self.read_archive(rows)

    def test_rejects_duplicate_archived_policy_obligation(self):
        with self.assertRaisesRegex(ValueError, "duplicate archived result"):
            self.read_archive(self.records + [deepcopy(self.records[0])])

    def assert_uniform_schedule_matches_twap(self, episode, quantity):
        config = make_config()
        original, original_trace = evaluate_episode(
            episode, "twap", quantity, size="medium", split="validation",
            fee_bps=config["fee_bps"], market_regimes=config["market_regimes"],
        )
        actual, actual_trace = run_vwap(
            episode, quantity, "medium", "validation", config, [1.0] * 60,
        )
        self.assertEqual(set(actual), set(original))
        for field, expected in original.items():
            if field == "policy":
                self.assertEqual(actual[field], "vwap")
            elif isinstance(expected, float):
                self.assertAlmostEqual(actual[field], expected, places=10, msg=field)
            else:
                self.assertEqual(actual[field], expected, field)
        self.assertEqual(len(actual_trace), len(original_trace))
        for step, expected in zip(actual_trace, original_trace):
            self.assertEqual(step["forecast_weight"], 1.0)
            for field in ("filled_btc", "remaining_btc", "received_usdt",
                          "fee_usdt", "net_received_usdt", "reward"):
                self.assertAlmostEqual(step[field], expected[field], places=10, msg=field)
        return actual

    def test_uniform_forecast_matches_twap_on_price_moving_books(self):
        episode = make_episode()
        for index, row in enumerate(episode):
            for side in ("bids", "asks"):
                for level in range(25):
                    row[f"{side}[{level}].price"] += index / 10.0
        result = self.assert_uniform_schedule_matches_twap(episode, 0.1)
        self.assertTrue(result["completed"])

    def test_uniform_forecast_matches_twap_with_depth_shortfall_and_residual(self):
        episode = make_episode([0.0] * 25)
        episode[-1]["bids[0].amount"] = 0.25
        result = self.assert_uniform_schedule_matches_twap(episode, 1.0)
        self.assertFalse(result["completed"])
        self.assertEqual(result["remaining_btc"], 0.75)
        self.assertIsNone(result["net_implementation_shortfall_bps"])
        self.assertAlmostEqual(result["fees_usdt"], 0.025)


if __name__ == "__main__":
    unittest.main()
