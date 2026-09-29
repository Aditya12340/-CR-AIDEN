"""Saved-result repricing preserves fills and handles incomplete obligations."""

from copy import deepcopy
import csv
from pathlib import Path
import tempfile
import unittest

from robustness import fee_sensitivity, hourly_comparisons, read_results, reprice_fees


def accounting_row(*, completed=True):
    return {
        "episode_id": "2021-03-01T00:05:00+00:00",
        "episode_start": "2021-03-01T00:05:00+00:00", "day": "2021-03-01",
        "split": "testing", "policy": "ppo_seed_7", "size": "medium",
        "parent_quantity_btc": 2.0, "arrival_midprice": 100.0, "fee_bps": 10.0,
        "gross_received_usdt": 180.0 if completed else 90.0,
        "filled_btc": 2.0 if completed else 1.0,
        "remaining_btc": 0.0 if completed else 1.0,
        "completion_fraction": 1.0 if completed else 0.5,
        "residual_mark_usdt": 0.0 if completed else 110.0,
        "completed": completed, "forced_terminal_liquidation": False,
        "net_implementation_shortfall_bps": 1009.0 if completed else None,
        "completion_score_bps": 1009.0 if completed else 5504.5,
    }


class RobustnessTests(unittest.TestCase):
    def test_fee_repricing_matches_hand_calculation(self):
        source = accounting_row()
        original = deepcopy(source)
        result = reprice_fees(source, 10)
        self.assertAlmostEqual(result["fees_usdt"], 0.18)
        self.assertAlmostEqual(result["net_received_usdt"], 179.82)
        self.assertAlmostEqual(result["net_average_fill_price"], 89.91)
        self.assertAlmostEqual(result["net_implementation_shortfall_bps"], 1009)
        self.assertAlmostEqual(result["completion_score_bps"], 1009)
        self.assertAlmostEqual(result["total_reward"], -1009)
        self.assertEqual(result["filled_btc"], 2)
        self.assertEqual(source, original)
        self.assertAlmostEqual(reprice_fees(source, 0)["net_implementation_shortfall_bps"], 1000)
        self.assertAlmostEqual(reprice_fees(source, 20)["net_implementation_shortfall_bps"], 1018)

    def test_incomplete_inventory_is_not_cash_or_completed_shortfall(self):
        result = reprice_fees(accounting_row(completed=False), 10)
        self.assertFalse(result["completed"])
        self.assertIsNone(result["net_implementation_shortfall_bps"])
        self.assertAlmostEqual(result["fees_usdt"], 0.09)
        self.assertAlmostEqual(result["net_received_usdt"], 89.91)
        self.assertAlmostEqual(result["completion_score_bps"], 5504.5)
        self.assertAlmostEqual(result["marked_shortfall_bps"], 4.5)
        self.assertEqual(result["residual_mark_usdt"], 110)
        self.assertEqual(result["remaining_btc"], 1)

    def test_no_fills_never_accrue_fees(self):
        row = accounting_row(completed=False)
        row.update(gross_received_usdt=0, filled_btc=0, residual_mark_usdt=220,
                   remaining_btc=2, completion_fraction=0)
        result = reprice_fees(row, 20)
        self.assertEqual(result["fees_usdt"], 0)
        self.assertEqual(result["net_received_usdt"], 0)
        self.assertIsNone(result["net_average_fill_price"])
        self.assertIsNone(result["net_implementation_shortfall_bps"])
        self.assertEqual(result["completion_score_bps"], 10_000)
        self.assertAlmostEqual(result["marked_shortfall_bps"], -1000)

    def test_fee_grid_preserves_obligations_and_rejects_ambiguous_source(self):
        row = accounting_row()
        results = fee_sensitivity([row], [0, 10, 20])
        self.assertEqual(len(results), 3)
        self.assertEqual([result["fee_bps"] for result in results], [0, 10, 20])
        self.assertTrue(all(result["filled_btc"] == row["filled_btc"] for result in results))
        with self.assertRaises(ValueError):
            fee_sensitivity([row], [10, 10])
        with self.assertRaises(ValueError):
            fee_sensitivity([row, {**row, "fee_bps": 20}], [0, 10, 20])
        for bad in (-1, float("nan"), float("inf"), 10_000):
            with self.subTest(fee=bad), self.assertRaises(ValueError):
                reprice_fees(row, bad)

    def test_hourly_pairs_exclude_incomplete_cost_but_keep_completion_score(self):
        records = []
        for stamp, delta, complete in (
            ("2021-03-01T00:05:00+00:00", -2, True),
            ("2021-03-01T00:10:00+00:00", -4, True),
            ("2021-03-01T01:05:00+00:00", None, False),
        ):
            for policy in ("ppo_seed_7", "twap", "immediate", "wait"):
                row = accounting_row(completed=(complete if policy == "ppo_seed_7" else True))
                row.update(episode_id=stamp, episode_start=stamp, policy=policy)
                row["net_implementation_shortfall_bps"] = (10 + delta if policy == "ppo_seed_7" and complete
                                                           else 10 if policy != "ppo_seed_7" else None)
                row["completion_score_bps"] = (row["net_implementation_shortfall_bps"]
                                               if row["completed"] else 5000)
                records.append(row)
        hours, summary = hourly_comparisons(records, "ppo_seed_7")
        self.assertEqual(len(hours), 6)
        first = next(row for row in hours if row["benchmark"] == "twap"
                     and "T00:" in row["hour_utc"])
        self.assertEqual(first["episodes"], 2)
        self.assertEqual(first["complete_pairs"], 2)
        self.assertEqual(first["mean_net_shortfall_delta_bps"], -3)
        second = next(row for row in hours if row["benchmark"] == "twap"
                      and "T01:" in row["hour_utc"])
        self.assertEqual(second["complete_pairs"], 0)
        self.assertIsNone(second["mean_net_shortfall_delta_bps"])
        self.assertEqual(second["mean_completion_score_delta_bps"], 4990)
        group = next(row for row in summary if row["benchmark"] == "twap")
        self.assertEqual(group["hours_observed"], 2)
        self.assertEqual(group["hourly_mean_net_shortfall_delta_bps"]["n"], 1)
        self.assertEqual(group["fraction_hours_with_lower_mean_cost"], 1)

    def test_hourly_analysis_rejects_unpaired_results(self):
        with self.assertRaisesRegex(ValueError, "benchmark"):
            hourly_comparisons([accounting_row()], "ppo_seed_7")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            hourly_comparisons([accounting_row(), accounting_row()], "ppo_seed_7")

    def test_csv_false_does_not_become_truthy_and_empty_cost_is_none(self):
        row = accounting_row(completed=False)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            loaded = read_results(path)[0]
        self.assertIs(loaded["completed"], False)
        self.assertIs(loaded["forced_terminal_liquidation"], False)
        self.assertIsNone(loaded["net_implementation_shortfall_bps"])
        self.assertEqual(loaded["gross_received_usdt"], 90)
        self.assertIsNone(reprice_fees(loaded, 0)["net_implementation_shortfall_bps"])


if __name__ == "__main__":
    unittest.main()
