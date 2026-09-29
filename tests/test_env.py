import math
import random
import unittest
from datetime import datetime, timedelta, timezone

from env import (
    Action,
    SellExecutionEnv,
    available_bid_depth,
    book_midprice,
    execute_sell,
    compile_action,
    validate_episode,
)


def make_book(amounts=None):
    if amounts is None:
        amounts = [1.0] * 25

    row = {"asks[0].price": 101.0}
    for level in range(25):
        row[f"bids[{level}].price"] = 100.0 - level
        row[f"bids[{level}].amount"] = amounts[level]
        row[f"asks[{level}].price"] = 101.0 + level
        row[f"asks[{level}].amount"] = 1.0
    return row


def make_episode(amounts=None):
    start = datetime(2020, 9, 1, 0, 5, tzinfo=timezone.utc)
    rows = []
    for step in range(60):
        row = make_book(amounts)
        row["decision_time"] = start + timedelta(seconds=5 * step)
        row["book_local_timestamp"] = row["decision_time"] - timedelta(milliseconds=10)
        row["trade_volume_btc_5s"] = 1.0
        row["signed_trade_volume_btc_5s"] = 0.5
        rows.append(row)
    return rows


class ExecuteSellTests(unittest.TestCase):
    def test_tiny_quantity_without_liquidity_is_never_a_fill(self):
        self.assertEqual(execute_sell(make_book([0.0] * 25), 1e-13),
                         (0.0, 0.0, 1e-13))

    def test_exact_depth_does_not_leave_roundoff_inventory(self):
        book = make_book([0.04] * 25)
        filled, proceeds, residual = execute_sell(book, 1.0)
        self.assertEqual(filled, 1.0)
        self.assertEqual(residual, 0.0)
        self.assertAlmostEqual(proceeds, 88.0)
        env = SellExecutionEnv(make_episode([0.04] * 25), 1.0)
        for _ in range(60):
            env.step_action(Action.WAIT)
        self.assertTrue(env.summary().completed)

    def test_execution_preserves_input(self):
        book = make_book()
        before = book.copy()
        execute_sell(book, 3.0)
        self.assertEqual(book, before)

    def test_walks_multiple_levels(self):
        amounts = [0.4, 0.7, 0.5] + [0.0] * 22
        book = make_book(amounts)

        filled, proceeds, unfilled = execute_sell(book, 0.9)

        self.assertTrue(math.isclose(filled, 0.9))
        self.assertTrue(math.isclose(proceeds, 89.5))
        self.assertEqual(unfilled, 0.0)

    def test_leaves_quantity_unfilled_beyond_displayed_depth(self):
        amounts = [0.4, 0.7, 0.5] + [0.0] * 22

        filled, proceeds, unfilled = execute_sell(make_book(amounts), 2.0)

        self.assertTrue(math.isclose(filled, 1.6))
        self.assertTrue(math.isclose(proceeds, 158.3))
        self.assertTrue(math.isclose(unfilled, 0.4))
        self.assertTrue(math.isclose(filled + unfilled, 2.0))

    def test_zero_request_has_zero_accounting(self):
        self.assertEqual(execute_sell(make_book(), 0.0), (0.0, 0.0, 0.0))

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            execute_sell(make_book(), -1.0)
        with self.assertRaises(ValueError):
            execute_sell(make_book(), float("nan"))

        missing_level = make_book()
        del missing_level["bids[10].price"]
        with self.assertRaises(ValueError):
            execute_sell(missing_level, 1.0)

    def test_book_helpers(self):
        book = make_book([0.25] * 25)
        self.assertTrue(math.isclose(available_bid_depth(book), 6.25))
        self.assertEqual(book_midprice(book), 100.5)


class SellExecutionEnvTests(unittest.TestCase):
    def test_tiny_parent_does_not_disappear(self):
        env = SellExecutionEnv(make_episode([0.0] * 25), 1e-13)
        for _ in range(60):
            env.step_action(Action.WAIT)
        summary = env.summary()
        self.assertEqual(summary.filled_btc, 0.0)
        self.assertEqual(summary.remaining_btc, 1e-13)
        self.assertFalse(summary.completed)
        self.assertAlmostEqual(summary.total_reward, -10_000.0)

    def test_causal_book_and_full_ask_validation(self):
        for field, value in (
            ("asks[12].amount", float("nan")),
            ("asks[24].price", 1.0),
            ("bids[20].amount", -1.0),
            ("signed_trade_volume_btc_5s", 2.0),
        ):
            with self.subTest(field=field):
                rows = make_episode()
                rows[5][field] = value
                with self.assertRaises(ValueError):
                    validate_episode(rows)
        for delta in (timedelta(microseconds=1), -timedelta(seconds=5, microseconds=1)):
            rows = make_episode()
            rows[5]["book_local_timestamp"] = rows[5]["decision_time"] + delta
            with self.assertRaises(ValueError):
                validate_episode(rows)

    def test_missing_receipt_time_is_rejected(self):
        rows = make_episode()
        del rows[0]["book_local_timestamp"]
        with self.assertRaises(ValueError):
            validate_episode(rows)

    def test_submicrosecond_times_are_not_silently_rounded(self):
        rows = make_episode()
        rows[0]["decision_time"] = "2020-09-01T00:05:00.000000001+00:00"
        with self.assertRaises(ValueError):
            validate_episode(rows)

    def test_completed_fee_and_reward_accounting(self):
        env = SellExecutionEnv(make_episode(), 0.75, fee_bps=10.0)
        result = env.step(0.75)
        summary = env.summary()
        self.assertAlmostEqual(result.fee_usdt, 0.075)
        self.assertAlmostEqual(summary.net_received_usdt, 74.925)
        self.assertAlmostEqual(summary.net_average_fill_price, 99.9)
        self.assertAlmostEqual(summary.total_reward,
                               -summary.net_implementation_shortfall_bps)
        self.assertAlmostEqual(summary.total_reward, -summary.completion_score_bps)

    def test_unsold_mark_is_separate_from_cash_and_reward(self):
        env = SellExecutionEnv(make_episode(), 1.0, force_terminal_liquidation=False)
        for _ in range(60):
            env.step_action(Action.WAIT)
        summary = env.summary()
        self.assertEqual(summary.net_received_usdt, 0.0)
        self.assertEqual(summary.residual_mark_usdt, 100.5)
        self.assertEqual(summary.marked_shortfall_bps, 0.0)
        self.assertEqual(summary.completion_score_bps, 10_000.0)
        self.assertEqual(summary.total_reward, -10_000.0)
        self.assertIsNone(summary.net_implementation_shortfall_bps)

    def test_withholding_inventory_cannot_erase_spread_and_fee_costs(self):
        # Same flat market and fees: a completed sale must beat avoiding fills.
        sell = SellExecutionEnv(make_episode(), 1.0, fee_bps=10.0)
        sell.step(1.0)
        wait = SellExecutionEnv(make_episode(), 1.0, fee_bps=10.0,
                                force_terminal_liquidation=False)
        for _ in range(60):
            wait.step_action(Action.WAIT)
        self.assertGreater(sell.summary().total_reward, wait.summary().total_reward)

    def test_higher_proceeds_improve_reward_for_same_obligation(self):
        rows = make_episode()
        # Both environments arrive at the same price. Only the next book differs.
        higher_rows = make_episode()
        for row in higher_rows[1:]:
            for name in row:
                if name.endswith(".price"):
                    row[name] += 1.0
        rewards = []
        for episode in (rows, higher_rows):
            env = SellExecutionEnv(episode, 1.0)
            env.step_action(Action.WAIT)
            env.step(1.0)
            rewards.append(env.summary().total_reward)
        self.assertAlmostEqual(rewards[1] - rewards[0], 10_000.0 / 100.5)

    def test_terminal_observation_does_not_read_future_book(self):
        rows = make_episode()
        for row in rows[1:]:
            for name in row:
                if name.endswith(".price"):
                    row[name] *= 2
        env = SellExecutionEnv(rows, 0.5)
        env.step(0.5)
        obs = env.observation()
        self.assertEqual(obs["midprice_return_from_arrival_bps"], 0.0)
        self.assertEqual(obs["time_remaining_fraction"], 0.0)
        with self.assertRaises(RuntimeError):
            env.step_action(Action.WAIT)

    def test_seeded_mixed_actions_preserve_conservation(self):
        rng = random.Random(7)
        env = SellExecutionEnv(make_episode([0.002] * 25), 1.0, fee_bps=8.0)
        while not env.done:
            result = env.step_action(rng.randrange(4))
            self.assertGreaterEqual(result.remaining_btc, 0.0)
            self.assertLessEqual(result.filled_btc, 0.05 + 1e-14)
            self.assertAlmostEqual(result.filled_btc + result.unfilled_btc,
                                   result.submitted_btc)
            self.assertAlmostEqual(result.cumulative_filled_btc + result.remaining_btc, 1.0)
        summary = env.summary()
        self.assertAlmostEqual(summary.total_reward, -summary.completion_score_bps)
        self.assertAlmostEqual(summary.gross_received_usdt - summary.fees_usdt,
                               summary.net_received_usdt)

    def test_caps_request_at_parent_inventory(self):
        env = SellExecutionEnv(make_episode(), parent_quantity_btc=0.75)

        result = env.step(10.0)
        summary = env.summary()

        self.assertEqual(result.submitted_btc, 0.75)
        self.assertEqual(result.filled_btc, 0.75)
        self.assertTrue(result.done)
        self.assertTrue(summary.completed)
        self.assertEqual(summary.remaining_btc, 0.0)
        self.assertEqual(summary.gross_received_usdt, 75.0)
        self.assertTrue(
            math.isclose(
                summary.implementation_shortfall_bps,
                10_000.0 * (100.5 - 100.0) / 100.5,
            )
        )

    def test_unfilled_request_is_not_automatically_retried(self):
        amounts = [0.4] + [0.0] * 24
        env = SellExecutionEnv(make_episode(amounts), parent_quantity_btc=1.0)

        first = env.step(1.0)
        second = env.step(0.0)

        self.assertTrue(math.isclose(first.filled_btc, 0.4))
        self.assertTrue(math.isclose(first.unfilled_btc, 0.6))
        self.assertTrue(math.isclose(first.remaining_btc, 0.6))
        self.assertEqual(second.filled_btc, 0.0)
        self.assertTrue(math.isclose(second.remaining_btc, 0.6))

    def test_final_step_forces_remaining_request_but_respects_depth(self):
        amounts = [0.04] * 25
        env = SellExecutionEnv(make_episode(amounts), parent_quantity_btc=2.0)

        for _ in range(59):
            env.step(0.0)
        final = env.step(0.0)
        summary = env.summary()

        self.assertTrue(final.forced_terminal_liquidation)
        self.assertEqual(final.submitted_btc, 2.0)
        self.assertTrue(math.isclose(final.filled_btc, 1.0))
        self.assertTrue(math.isclose(final.unfilled_btc, 1.0))
        self.assertTrue(math.isclose(summary.remaining_btc, 1.0))
        self.assertFalse(summary.completed)
        self.assertIsNone(summary.implementation_shortfall_bps)

    def test_reset_restores_initial_state(self):
        env = SellExecutionEnv(make_episode(), parent_quantity_btc=1.0)
        env.step(0.25)

        observation = env.reset()

        self.assertEqual(observation["inventory_remaining_fraction"], 1.0)
        self.assertEqual(observation["time_remaining_seconds"], 300.0)
        self.assertEqual(observation["midprice_return_from_arrival_bps"], 0.0)
        self.assertEqual(len(env.history), 0)

    def test_episode_grid_must_be_complete(self):
        with self.assertRaises(ValueError):
            validate_episode(make_episode()[:-1])

        bad_grid = make_episode()
        bad_grid[20]["decision_time"] += timedelta(seconds=1)
        with self.assertRaises(ValueError):
            validate_episode(bad_grid)


class ActionTests(unittest.TestCase):
    def test_four_actions_have_hand_calculated_quantities(self):
        # With 6 BTC and 60 decisions, adaptive TWAP requests 0.1 BTC.
        for action, expected in zip(Action, (0.0, 0.05, 0.1, 0.2)):
            with self.subTest(action=action):
                request = compile_action(action, 6.0, 0, make_book())
                self.assertAlmostEqual(request.strategy_requested_btc, expected)
                self.assertAlmostEqual(request.submitted_btc, expected)

    def test_final_actions_all_attempt_remaining_once(self):
        for action in Action:
            request = compile_action(action, 30.0, 59, make_book())
            self.assertEqual(request.submitted_btc, 30.0)
            self.assertEqual(request.executable_btc, 25.0)

    def test_catch_up_after_no_fill_and_after_wait(self):
        rows = make_episode()
        for level in range(25):
            rows[0][f"bids[{level}].amount"] = 0.0
        env = SellExecutionEnv(rows, 6.0)
        result = env.step_action(Action.ON_SCHEDULE)
        self.assertAlmostEqual(result.unfilled_btc, 0.1)
        self.assertAlmostEqual(env.action_request(Action.ON_SCHEDULE).submitted_btc, 6.0 / 59)

    def test_intention_and_depth_limit_are_separate(self):
        request = compile_action(Action.ABOVE_SCHEDULE, 6.0, 0,
                                 make_book([0.001] * 25))
        self.assertAlmostEqual(request.submitted_btc, 0.2)
        self.assertAlmostEqual(request.executable_btc, 0.025)

    def test_invalid_actions_cannot_mutate_state(self):
        env = SellExecutionEnv(make_episode(), 1.0)
        for action in (-1, 4, True, 1.0, "WAIT", None):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step_action(action)
        self.assertEqual(env.step_index, 0)
        self.assertEqual(env.remaining_btc, 1.0)


if __name__ == "__main__":
    unittest.main()
