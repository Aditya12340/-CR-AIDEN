import copy
import math
import unittest

from features import FEATURE_NAMES, build_observation, observation_vector


def feature_book(mid=101.0, bid_amount=0.2, ask_amount=0.1):
    row = {
        "trade_volume_btc_5s": 5.0,
        "signed_trade_volume_btc_5s": -1.0,
    }
    for level in range(25):
        row[f"bids[{level}].price"] = mid - 1.0 - level
        row[f"asks[{level}].price"] = mid + 1.0 + level
        row[f"bids[{level}].amount"] = bid_amount
        row[f"asks[{level}].amount"] = ask_amount
    return row


class ObservationTests(unittest.TestCase):
    def test_hand_calculated_features_and_order(self):
        observation = build_observation([feature_book()], 0, 2.5, 5.0)

        self.assertEqual(tuple(observation), FEATURE_NAMES)
        self.assertEqual(len(observation), 10)
        expected = (0.5, 1.0, 20_000.0 / 101.0, 1.0 / 3.0,
                    math.log(2), 10_000.0 / 303.0, 0.0, 0.0,
                    math.log(2), -0.2)
        for name, value in zip(FEATURE_NAMES, expected):
            with self.subTest(name=name):
                self.assertAlmostEqual(observation[name], value)

    def test_zero_depth_and_no_trades_are_finite(self):
        row = feature_book(bid_amount=0.0, ask_amount=0.0)
        row["trade_volume_btc_5s"] = 0.0
        row["signed_trade_volume_btc_5s"] = 0.0

        observation = build_observation([row], 0, 1.0, 1.0)

        for name in ("depth_imbalance", "relative_bid_depth",
                     "weighted_mid_deviation_bps", "recent_trade_volume",
                     "signed_trade_flow"):
            self.assertEqual(observation[name], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in observation.values()))

    def test_future_changes_cannot_change_current_observation(self):
        rows = [feature_book(mid=100.0 + index) for index in range(60)]
        before = build_observation(rows, 4, 0.5, 1.0)
        changed = copy.deepcopy(rows)
        for index in range(5, 60):
            # Deliberately unreadable: even validating a future feature here
            # would introduce a dependency on data beyond the current decision.
            changed[index] = {}

        self.assertEqual(build_observation(changed, 4, 0.5, 1.0), before)

    def test_volatility_warms_up_using_available_intervals(self):
        rows = [feature_book(mid=100.0 * math.exp(0.001 * index)) for index in range(4)]

        for index in range(4):
            observation = build_observation(rows, index, 1.0, 1.0)
            self.assertAlmostEqual(observation["trailing_volatility_bps"],
                                   10.0 * math.sqrt(index), places=8)
            self.assertAlmostEqual(observation["recent_mid_return_bps"],
                                   10.0 if index else 0.0, places=8)

    def test_volatility_discards_returns_older_than_twelve_intervals(self):
        mids = [100.0] + [100.0 * math.exp(0.5 + 0.001 * index) for index in range(13)]
        rows = [feature_book(mid=mid) for mid in mids]
        before_cutoff = build_observation(rows, 12, 1.0, 1.0)
        after_cutoff = build_observation(rows, 13, 1.0, 1.0)

        self.assertAlmostEqual(before_cutoff["trailing_volatility_bps"],
                               10_000.0 * math.sqrt(0.5 ** 2 + 11 * 0.001 ** 2))
        self.assertAlmostEqual(after_cutoff["trailing_volatility_bps"],
                               10.0 * math.sqrt(12), places=8)

    def test_last_decision_has_one_execution_opportunity_remaining(self):
        rows = [feature_book() for _ in range(60)]
        observation = build_observation(rows, 59, 1.0, 1.0)
        self.assertEqual(observation["time_remaining_fraction"], 1.0 / 60.0)

    def test_vector_has_fixed_order_and_only_scales_bps(self):
        # Insertion order of the input mapping must not affect model inputs.
        observation = {name: 250.0 for name in reversed(FEATURE_NAMES)}
        vector = observation_vector(observation)
        self.assertEqual(vector, (250.0, 250.0, 2.5, 250.0, 250.0,
                                  2.5, 2.5, 2.5, 250.0, 250.0))

    def test_price_unit_and_quantity_scale_invariance(self):
        rows = [feature_book(100.0), feature_book(101.0)]
        original_rows = copy.deepcopy(rows)
        expected = build_observation(rows, 1, 2.5, 5.0)
        scaled = copy.deepcopy(rows)
        for row in scaled:
            for name in row:
                row[name] *= 7.0 if name.endswith(".price") else 3.0

        actual = build_observation(scaled, 1, 7.5, 15.0)

        for name in FEATURE_NAMES:
            self.assertAlmostEqual(actual[name], expected[name], places=9)
        self.assertEqual(rows, original_rows)

    def test_rejects_invalid_state_and_vector_inputs(self):
        rows = [feature_book()]
        for index in (-1, 1, 0.5, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                build_observation(rows, index, 1.0, 1.0)
        for quantity in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(quantity=quantity), self.assertRaises(ValueError):
                build_observation(rows, 0, 0.0, quantity)
        for remaining in (-1.0, 2.0, float("nan")):
            with self.subTest(remaining=remaining), self.assertRaises(ValueError):
                build_observation(rows, 0, remaining, 1.0)
        with self.assertRaises(ValueError):
            observation_vector({})
        invalid = build_observation(rows, 0, 1.0, 1.0)
        invalid[FEATURE_NAMES[0]] = float("inf")
        with self.assertRaises(ValueError):
            observation_vector(invalid)


if __name__ == "__main__":
    unittest.main()
