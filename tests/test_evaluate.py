"""Evaluation regressions: match obligations, cost signs, and incomplete orders."""

from copy import deepcopy
from datetime import timedelta
from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest

from baselines import BASELINE_NAMES
from evaluate import (evaluate_episode, evaluate_episodes, prepare_output_directory,
                      regime_label)
from reporting import distribution, paired_results, summarize_results
from test_env import make_episode


class EvaluationTests(unittest.TestCase):
    def test_paired_delta_sign_and_net_fee_accounting(self):
        episode = make_episode()
        # Rising executable prices make the immediate sell worse than TWAP.
        for index, row in enumerate(episode):
            for side in ("bids", "asks"):
                for level in range(25):
                    row[f"{side}[{level}].price"] += index
        records, _ = evaluate_episodes([episode], {"medium": 0.1}, fee_bps=10)
        immediate = next(row for row in records if row["policy"] == "immediate")
        self.assertAlmostEqual(immediate["gross_received_usdt"], 10)
        self.assertAlmostEqual(immediate["fees_usdt"], 0.01)
        self.assertAlmostEqual(immediate["net_received_usdt"], 9.99)
        pairs = paired_results(records)
        pair = next(row for row in pairs if row["policy"] == "immediate")
        self.assertTrue(pair["both_completed"])
        self.assertGreater(pair["net_shortfall_delta_bps"], 0)
        self.assertLess(pair["net_proceeds_delta_usdt"], 0)
        self.assertFalse(pair["cost_win"])
        twap = next(row for row in pairs if row["policy"] == "twap")
        self.assertEqual(twap["net_shortfall_delta_bps"], 0)
        self.assertTrue(twap["cost_tie"])

    def test_incomplete_orders_remain_in_score_but_not_shortfall(self):
        records, _ = evaluate_episodes([make_episode([0.0] * 25)], {"medium": 1.0})
        pairs = paired_results(records)
        self.assertTrue(all(not row["both_completed"] for row in pairs))
        self.assertTrue(all(row["net_shortfall_delta_bps"] is None for row in pairs))
        self.assertTrue(all(row["cost_win"] is None for row in pairs))
        result = summarize_results(records, pairs)
        for group in result["groups"]:
            self.assertEqual(group["completion_rate"], 0)
            self.assertEqual(group["net_shortfall_bps"]["n"], 0)
            self.assertIsNone(group["net_shortfall_bps"]["mean"])
            self.assertEqual(group["completion_score_bps"]["mean"], 10_000)
            self.assertEqual(group["paired_vs_twap"]["excluded_incomplete_pairs"], 1)
            self.assertIsNone(group["paired_vs_twap"]["cost_win_rate"])
            self.assertIsNone(group["average_execution_time_seconds"]["mean"])

    def test_pairing_rejects_mismatched_obligation_or_duplicate(self):
        records, _ = evaluate_episodes([make_episode()], {"medium": 0.1})
        for field, value in (("fee_bps", 20), ("parent_quantity_btc", 0.2),
                             ("episode_id", "another-episode"), ("split", "testing")):
            with self.subTest(field=field):
                changed = deepcopy(records)
                changed[0][field] = value
                with self.assertRaisesRegex(ValueError, "missing matched TWAP"):
                    paired_results(changed)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            paired_results(records + [records[0]])

    def test_only_complete_pairs_contribute_to_cost_win_rate(self):
        records, _ = evaluate_episodes([make_episode()], {"medium": 0.1})
        additional = deepcopy(records)
        for row in additional:
            row["episode_id"] = "second-episode"
        # The TWAP obligation fails in episode two; a completed competitor must
        # not gain a cheap cost win against an incomparable partial execution.
        benchmark = next(row for row in additional if row["policy"] == "twap")
        benchmark["completed"] = False
        benchmark["net_implementation_shortfall_bps"] = None
        combined = records + additional
        summary = summarize_results(combined, paired_results(combined))
        immediate = next(group for group in summary["groups"] if group["policy"] == "immediate")
        self.assertEqual(immediate["paired_vs_twap"]["paired_episodes"], 2)
        self.assertEqual(immediate["paired_vs_twap"]["complete_pairs"], 1)
        self.assertEqual(immediate["paired_vs_twap"]["excluded_incomplete_pairs"], 1)

    def test_first_episode_traces_are_fixed_and_sizing_is_paired(self):
        first, second = make_episode(), make_episode()
        for row in second:
            row["decision_time"] += timedelta(minutes=5)
            row["book_local_timestamp"] += timedelta(minutes=5)
        records, traces = evaluate_episodes([first, second], {"small": 0.1, "large": 0.5})
        self.assertEqual(len(records), 2 * 2 * len(BASELINE_NAMES))
        self.assertEqual({row["episode_id"] for row in traces}, {first[0]["decision_time"].isoformat()})
        self.assertEqual({(row["size"], row["policy"]) for row in traces},
                         {(size, policy) for size in ("small", "large")
                          for policy in BASELINE_NAMES})
        for row in records:
            if row["policy"] == "immediate":
                self.assertEqual(row["average_execution_time_seconds"], 0)
            elif row["policy"] == "twap":
                self.assertAlmostEqual(row["average_execution_time_seconds"], 147.5)

    def test_training_cutoffs_are_applied_without_refitting(self):
        cutoffs = {"lower": 2.0, "upper": 4.0}
        self.assertEqual([regime_label(value, cutoffs) for value in (1, 2, 3, 4, 5)],
                         ["low", "low", "medium", "medium", "high"])
        config = {"arrival_spread_bps": {"lower": 0.01, "upper": 0.02},
                  "episode_realized_volatility_bps": {"lower": 1, "upper": 2}}
        records, _ = evaluate_episodes([make_episode()], {"medium": 0.1}, market_regimes=config)
        self.assertTrue(all(row["spread_regime"] == "high" for row in records))
        self.assertTrue(all(row["volatility_regime"] == "low" for row in records))
        self.assertEqual(config["arrival_spread_bps"], {"lower": 0.01, "upper": 0.02})

    @unittest.skipUnless(find_spec("numpy"), "model inference requires NumPy")
    def test_model_interface_tracks_actual_actions_and_terminal_residual(self):
        class WaitPolicy:
            calls = 0

            def predict(self, observation, deterministic):
                self.calls += 1
                assert deterministic
                assert observation.shape == (10,)
                assert str(observation.dtype) == "float32"
                return 0, None

        policy = WaitPolicy()
        record, trace = evaluate_episode(make_episode([0.01] * 25), "ppo_wait", 1.0,
                                         size="medium", split="validation", model=policy)
        self.assertEqual(policy.calls, 60)
        self.assertEqual(record["action_0_count"], 60)
        self.assertEqual(record["trade_decisions"], 1)
        self.assertAlmostEqual(record["completion_fraction"], 0.25)
        self.assertIsNone(record["net_implementation_shortfall_bps"])
        self.assertTrue(trace[-1]["forced_terminal_liquidation"])
        self.assertGreater(record["completion_score_bps"], 7500)

    def test_distribution_tail_and_missing_values(self):
        values = distribution([None, -10, 0, 10, 20])
        self.assertEqual(values["n"], 4)
        self.assertEqual(values["mean"], 5)
        self.assertEqual(values["median"], 5)
        self.assertAlmostEqual(values["p05"], -8.5)
        self.assertAlmostEqual(values["p95"], 18.5)
        self.assertEqual(values["worst_5pct_mean"], 20)
        self.assertIsNone(distribution([1])["std"])
        with self.assertRaises(ValueError):
            distribution([float("nan")])

    def test_run_directory_is_never_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = prepare_output_directory(Path(temporary) / "run")
            sentinel = output / "existing.txt"
            sentinel.write_text("retain me")
            with self.assertRaises(FileExistsError):
                prepare_output_directory(output)
            prepare_output_directory(output, overwrite=True)
            self.assertEqual(sentinel.read_text(), "retain me")


if __name__ == "__main__":
    unittest.main()
