"""Hand-computable baseline schedules and shared execution constraints."""

from datetime import datetime, timedelta, timezone
import math
import unittest

from baselines import BASELINE_NAMES, baseline_request
from calibrate import episode_market_context, quantile
from env import SellExecutionEnv


def make_episode(depth=1.0):
    start = datetime(2020, 9, 1, 0, 5, tzinfo=timezone.utc)
    rows = []
    for step in range(60):
        decision = start + timedelta(seconds=5 * step)
        row = {
            "decision_time": decision,
            "book_local_timestamp": decision - timedelta(milliseconds=10),
            "trade_volume_btc_5s": 0.0,
            "signed_trade_volume_btc_5s": 0.0,
        }
        for level in range(25):
            row[f"bids[{level}].price"] = 100.0 - level
            row[f"bids[{level}].amount"] = depth if level == 0 else 0.0
            row[f"asks[{level}].price"] = 101.0 + level
            row[f"asks[{level}].amount"] = 1.0
        rows.append(row)
    return rows


class BaselineTests(unittest.TestCase):
    def test_immediate_retries_only_remaining_inventory_after_partial_fill(self):
        env = SellExecutionEnv(make_episode(0.4), 1.0)
        self.assertEqual(baseline_request("immediate", env), 1.0)
        env.step(baseline_request("immediate", env))
        self.assertAlmostEqual(baseline_request("immediate", env), 0.6)
        while not env.done:
            env.step(baseline_request("immediate", env))
        self.assertAlmostEqual(env.summary().filled_btc, 1.0)
        self.assertEqual(len(env.history), 3)

    def test_twap_catches_up_from_actual_fills(self):
        env = SellExecutionEnv(make_episode(0.4), 60.0)
        self.assertEqual(baseline_request("twap", env), 1.0)
        env.step(baseline_request("twap", env))
        self.assertAlmostEqual(baseline_request("twap", env), 59.6 / 59)
        self.assertGreater(baseline_request("twap", env), 1.0)

    def test_liquidity_multiplier_clips_and_reads_no_future_book(self):
        for next_depth, expected_multiplier in ((0.1, 0.5), (1.5, 1.5), (5.0, 2.0)):
            with self.subTest(depth=next_depth):
                rows = make_episode(1.0)
                rows[1]["bids[0].amount"] = next_depth
                env = SellExecutionEnv(rows, 60.0)
                self.assertEqual(baseline_request("liquidity", env), 1.0)
                env.step(0.0)
                self.assertAlmostEqual(baseline_request("liquidity", env), 60 / 59 * expected_multiplier)

    def test_zero_arrival_depth_uses_neutral_schedule(self):
        rows = make_episode(0.0)
        rows[1]["bids[0].amount"] = 5.0
        env = SellExecutionEnv(rows, 60.0)
        env.step(0.0)
        self.assertAlmostEqual(baseline_request("liquidity", env), 60 / 59)

    def test_wait_control_executes_only_at_the_common_deadline(self):
        env = SellExecutionEnv(make_episode(), 0.5)
        for _ in range(59):
            self.assertEqual(baseline_request("wait", env), 0.0)
            result = env.step(baseline_request("wait", env))
            self.assertEqual(result.filled_btc, 0.0)
            self.assertFalse(result.done)
        final = env.step(baseline_request("wait", env))
        self.assertEqual(final.strategy_requested_btc, 0.0)
        self.assertEqual(final.submitted_btc, 0.5)
        self.assertEqual(final.filled_btc, 0.5)
        self.assertTrue(final.forced_terminal_liquidation)
        self.assertTrue(env.done)

    def test_above_schedule_control_caps_inventory_and_preserves_partial_fills(self):
        env = SellExecutionEnv(make_episode(0.4), 1.0)
        self.assertAlmostEqual(baseline_request("above_schedule", env), 2 / 60)
        for _ in range(58):
            env.step(0.0)
        self.assertEqual(baseline_request("above_schedule", env), 1.0)
        env.step(baseline_request("above_schedule", env))
        self.assertAlmostEqual(env.remaining_btc, 0.6)
        self.assertAlmostEqual(baseline_request("above_schedule", env), 0.6)
        env.step(baseline_request("above_schedule", env))
        self.assertAlmostEqual(env.summary().remaining_btc, 0.2)
        self.assertAlmostEqual(env.summary().filled_btc, 0.8)

    def test_shared_deadline_depth_and_fees_for_every_baseline(self):
        rows = make_episode(0.0)
        rows[-1]["bids[0].amount"] = 0.4
        for name in BASELINE_NAMES:
            with self.subTest(name=name):
                env = SellExecutionEnv(rows, 1.0, fee_bps=10)
                while not env.done:
                    env.step(baseline_request(name, env))
                summary = env.summary()
                self.assertAlmostEqual(summary.filled_btc, 0.4)
                self.assertAlmostEqual(summary.remaining_btc, 0.6)
                self.assertAlmostEqual(summary.fees_usdt, 0.04)
                self.assertEqual(env.history[-1].submitted_btc, 1.0)

    def test_request_is_pure_and_invalid_or_finished_calls_fail(self):
        env = SellExecutionEnv(make_episode(), 0.5)
        for name in BASELINE_NAMES:
            request = baseline_request(name, env)
            self.assertGreaterEqual(request, 0)
            self.assertLessEqual(request, env.remaining_btc)
            self.assertEqual(env.history, [])
            self.assertEqual(env.step_index, 0)
        with self.assertRaises(ValueError):
            baseline_request("unknown", env)
        env.step(0.5)
        with self.assertRaises(RuntimeError):
            baseline_request("twap", env)


class CalibrationMathTests(unittest.TestCase):
    def test_quantile_uses_linear_interpolation(self):
        self.assertEqual(quantile([10.0, 0.0], 0.25), 2.5)
        self.assertEqual(quantile([1.0], 0.9), 1.0)
        with self.assertRaises(ValueError):
            quantile([], 0.5)

    def test_posthoc_context_has_correct_units(self):
        rows = make_episode()
        self.assertEqual(episode_market_context(rows)["episode_realized_volatility_bps"], 0.0)
        self.assertAlmostEqual(episode_market_context(rows)["arrival_spread_bps"], 10000 / 100.5)
        for row in rows[1:]:
            row["bids[0].price"] *= 1.01
            row["asks[0].price"] *= 1.01
        context = episode_market_context(rows)
        self.assertAlmostEqual(context["episode_realized_volatility_bps"], 10000 * math.log(1.01))


if __name__ == "__main__":
    unittest.main()
