"""VWAP metrics must preserve units, fee effects, pairing, and completion."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from vwap_reporting import add_market_metrics, compare_and_summarize, write_report


MARKET = {"market_vwap_usdt_per_btc": 100.0, "market_volume_btc": 10.0,
          "market_notional_usdt": 1000.0, "bins": 59}


def result(policy="ppo_seed_19", price=99.0, completed=True):
    net_price = price * 0.999  # 10-bp fee on actual execution notional.
    quantity = 1.0
    filled = quantity if completed else 0.5
    return {
        "split": "testing", "episode_id": "2021-03-01T00:05:00+00:00",
        "day": "2021-03-01", "size": "medium", "parent_quantity_btc": quantity,
        "fee_bps": 10.0, "policy": policy, "completed": completed,
        "completion_fraction": filled / quantity,
        "average_fill_price": price, "net_average_fill_price": net_price,
        "net_implementation_shortfall_bps": 10_000 * (1 - net_price / 100) if completed else None,
        "completion_score_bps": 10_000 * (1 - filled * net_price / 100),
    }


def matched_records():
    return [add_market_metrics(result(policy, price), MARKET)
            for policy, price in (("ppo_seed_19", 102), ("vwap", 100), ("twap", 101))]


class VWAPReportingTests(unittest.TestCase):
    def test_sell_slippage_signs_and_fee_effect_hand_calculation(self):
        original = result(price=99)
        saved = deepcopy(original)
        row = add_market_metrics(original, MARKET)
        self.assertAlmostEqual(row["gross_vwap_slippage_bps"], 100)
        self.assertAlmostEqual(row["net_vwap_slippage_bps"], 109.9)
        self.assertEqual(row["market_bins"], 59)
        self.assertEqual(original, saved)
        better = add_market_metrics(result(price=101), MARKET)
        self.assertAlmostEqual(better["gross_vwap_slippage_bps"], -100)
        self.assertAlmostEqual(better["net_vwap_slippage_bps"], -89.9)

    def test_incomplete_order_has_no_market_vwap_cost(self):
        row = add_market_metrics(result(completed=False), MARKET)
        self.assertIsNone(row["gross_vwap_slippage_bps"])
        self.assertIsNone(row["net_vwap_slippage_bps"])
        self.assertGreater(row["completion_score_bps"], 5000)

    def test_no_market_volume_leaves_arrival_metrics_intact(self):
        market = {**MARKET, "market_volume_btc": 0, "market_notional_usdt": 0,
                  "market_vwap_usdt_per_btc": None}
        rows = [add_market_metrics(result(policy), market)
                for policy in ("ppo_seed_19", "vwap", "twap")]
        pairs, summary = compare_and_summarize(rows, "ppo_seed_19", {})
        self.assertTrue(all(row["both_completed"] for row in pairs))
        self.assertTrue(all(row["net_vwap_slippage_delta_bps"] is None for row in pairs))
        self.assertTrue(all(row["arrival_shortfall_delta_bps"] == 0 for row in pairs))
        for group in summary["groups"]:
            self.assertEqual(group["zero_market_volume_episodes"], 1)
            self.assertEqual(group["net_arrival_shortfall_bps"]["n"], 1)
            self.assertEqual(group["net_vwap_slippage_bps"]["n"], 0)

    def test_selected_policy_is_paired_against_both_schedules(self):
        pairs, summary = compare_and_summarize(matched_records(), "ppo_seed_19", {"config": {"primary_size": "medium"}})
        self.assertEqual({row["benchmark"] for row in pairs}, {"vwap", "twap"})
        vwap = next(row for row in pairs if row["benchmark"] == "vwap")
        self.assertAlmostEqual(vwap["arrival_shortfall_delta_bps"], -199.8)
        self.assertAlmostEqual(vwap["net_vwap_slippage_delta_bps"], -199.8)
        self.assertTrue(vwap["arrival_cost_win"])
        self.assertTrue(vwap["vwap_cost_win"])
        self.assertEqual(len(summary["primary_test"]["groups"]), 3)
        self.assertEqual(len(summary["primary_test"]["comparisons"]), 2)

    def test_incomplete_pairs_do_not_enter_cost_win_denominator(self):
        rows = matched_records()
        incomplete = add_market_metrics(result("vwap", completed=False), MARKET)
        rows[1] = incomplete
        pairs, summary = compare_and_summarize(rows, "ppo_seed_19", {})
        pair = next(row for row in pairs if row["benchmark"] == "vwap")
        self.assertFalse(pair["both_completed"])
        self.assertIsNone(pair["arrival_shortfall_delta_bps"])
        self.assertIsNone(pair["arrival_cost_win"])
        self.assertLess(pair["completion_score_delta_bps"], -5000)
        group = next(row for row in summary["benchmark_comparisons"] if row["benchmark"] == "vwap")
        self.assertEqual(group["excluded_incomplete_pairs"], 1)
        self.assertIsNone(group["arrival_cost_win_rate"])

    def test_duplicate_unmatched_and_inconsistent_market_are_rejected(self):
        rows = matched_records()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare_and_summarize(rows + [rows[0]], "ppo_seed_19", {})
        with self.assertRaisesRegex(ValueError, "unmatched"):
            compare_and_summarize(rows[:-1], "ppo_seed_19", {})
        for field, value in (("fee_bps", 20), ("parent_quantity_btc", 2)):
            changed = deepcopy(rows)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unmatched"):
                compare_and_summarize(changed, "ppo_seed_19", {})
        changed = deepcopy(rows)
        changed[1]["market_volume_btc"] = 11
        with self.assertRaisesRegex(ValueError, "inconsistent market"):
            compare_and_summarize(changed, "ppo_seed_19", {})

    def test_market_reference_must_match_its_volume_and_notional(self):
        with self.assertRaises(ValueError):
            add_market_metrics(result(), {**MARKET, "market_vwap_usdt_per_btc": 90})
        with self.assertRaises(ValueError):
            add_market_metrics(result(), {**MARKET, "market_volume_btc": 0})

    def test_report_distinguishes_forecast_from_ex_post_price(self):
        _, summary = compare_and_summarize(matched_records(), "ppo_seed_19", {})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "REPORT.md"
            write_report(summary, path)
            report = path.read_text()
        self.assertIn("training-only", report)
        self.assertIn("295-second", report)
        self.assertIn("post-hoc", report)
        self.assertIn("Completed-order metrics exclude incomplete", report)


if __name__ == "__main__":
    unittest.main()
