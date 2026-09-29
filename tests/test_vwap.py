"""Hand-computable VWAP software checks; these are not research performance data."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import unittest

from env import SellExecutionEnv
from test_baselines import make_episode
from vwap import fit_volume_profile, market_vwap, schedule_weights, vwap_request


def volume_episode(day=1, month=9, year=2020, volume=1.0):
    rows = make_episode()
    start = datetime(year, month, day, 0, 5, tzinfo=timezone.utc)
    for index, row in enumerate(rows):
        row["decision_time"] = start + timedelta(seconds=5 * index)
        row["book_local_timestamp"] = row["decision_time"] - timedelta(milliseconds=10)
        row["trade_volume_btc_5s"] = volume
        row["trade_notional_usdt_5s"] = 100.0 * volume
    return rows


class VolumeProfileTests(unittest.TestCase):
    def test_minute_means_missing_bucket_fallback_and_coverage(self):
        september = volume_episode(volume=1.0)
        october = volume_episode(month=10, volume=3.0)
        for row in september[:12]:
            row["trade_volume_btc_5s"] = 5.0
        profile = fit_volume_profile([september, october])

        self.assertEqual(profile["episodes"], 2)
        self.assertEqual(profile["rows"], 120)
        self.assertEqual(profile["training_days"], ["2020-09-01", "2020-10-01"])
        self.assertEqual(profile["observations_by_utc_minute"][5:10], [24] * 5)
        self.assertEqual(profile["observations_by_utc_minute"][0], 0)
        self.assertEqual(profile["volume_btc_per_5s_by_utc_minute"][5], 4.0)
        self.assertEqual(profile["volume_btc_per_5s_by_utc_minute"][6], 2.0)
        self.assertAlmostEqual(profile["fallback_volume_btc_per_5s"], 2.4)
        self.assertAlmostEqual(profile["volume_btc_per_5s_by_utc_minute"][0], 2.4)

    def test_profile_rejects_validation_testing_wrong_day_and_duplicates(self):
        for rows in (volume_episode(month=2, year=2021), volume_episode(month=3, year=2021),
                     volume_episode(day=2), volume_episode()[:-1]):
            with self.subTest(start=rows[0]["decision_time"]), self.assertRaises(ValueError):
                fit_volume_profile([rows])
        rows = volume_episode()
        with self.assertRaises(ValueError):
            fit_volume_profile([rows, rows])
        with self.assertRaises(ValueError):
            fit_volume_profile([])

    def test_schedule_uses_utc_clock_and_wraps_midnight(self):
        profile = {"volume_btc_per_5s_by_utc_minute": [float(i) for i in range(1440)]}
        start = datetime(2021, 3, 1, 23, 59, 55, tzinfo=timezone.utc)
        weights = schedule_weights(profile, start)
        self.assertEqual(len(weights), 60)
        self.assertEqual(weights[0], 1439.0)
        self.assertEqual(weights[1:13], (0.0,) * 12)
        self.assertEqual(weights[13], 1.0)


class VwapRequestTests(unittest.TestCase):
    def test_uneven_weights_and_partial_fill_catchup(self):
        env = SellExecutionEnv(make_episode(0.4), 1.0)
        weights = (2.0, 1.0, 1.0) + (0.0,) * 57
        request = vwap_request(env, weights)
        self.assertEqual(request, 0.5)
        self.assertEqual(env.step_index, 0)
        self.assertEqual(env.history, [])
        env.step(request)
        self.assertAlmostEqual(env.remaining_btc, 0.6)
        self.assertAlmostEqual(vwap_request(env, weights), 0.3)
        env.step(vwap_request(env, weights))
        self.assertAlmostEqual(vwap_request(env, weights), 0.3)
        env.step(vwap_request(env, weights))
        self.assertTrue(env.done)
        with self.assertRaises(RuntimeError):
            vwap_request(env, weights)

    def test_zero_weights_fallback_and_zero_current_weight(self):
        env = SellExecutionEnv(make_episode(), 60.0)
        self.assertEqual(vwap_request(env, (0.0,) * 60), 1.0)
        self.assertEqual(vwap_request(env, (0.0,) + (1.0,) * 59), 0.0)
        env.step(0.0)
        self.assertAlmostEqual(vwap_request(env, (0.0,) * 60), 60.0 / 59)

    def test_future_market_changes_do_not_change_forecast_requests(self):
        profile = fit_volume_profile([volume_episode()])
        episode = volume_episode(year=2021, month=3)
        altered = deepcopy(episode)
        for row in altered[1:]:
            row["trade_volume_btc_5s"] *= 100
            row["bids[0].amount"] = 1000.0
        first = SellExecutionEnv(episode, 1.0)
        second = SellExecutionEnv(altered, 1.0)
        weights = schedule_weights(profile, episode[0]["decision_time"])
        self.assertEqual(vwap_request(first, weights), vwap_request(second, weights))
        self.assertEqual(weights, schedule_weights(profile, altered[0]["decision_time"]))

    def test_final_request_respects_same_depth_and_fee_rules(self):
        episode = make_episode(0.0)
        episode[-1]["bids[0].amount"] = 0.4
        env = SellExecutionEnv(episode, 1.0, fee_bps=10.0)
        weights = (0.0,) * 59 + (1.0,)
        while not env.done:
            env.step(vwap_request(env, weights))
        self.assertAlmostEqual(env.summary().filled_btc, 0.4)
        self.assertAlmostEqual(env.summary().remaining_btc, 0.6)
        self.assertAlmostEqual(env.summary().fees_usdt, 0.04)

    def test_invalid_schedule_values_fail(self):
        env = SellExecutionEnv(make_episode(), 1.0)
        for weights in ((1.0,) * 59, (-1.0,) * 60, (float("nan"),) * 60):
            with self.subTest(weights=weights[:1]), self.assertRaises(ValueError):
                vwap_request(env, weights)


class MarketVwapTests(unittest.TestCase):
    def test_volume_weighted_value_excludes_prearrival_bin(self):
        episode = volume_episode(volume=0.0)
        episode[0]["trade_volume_btc_5s"] = 1000.0
        episode[0]["trade_notional_usdt_5s"] = 1_000_000.0
        episode[1]["trade_volume_btc_5s"] = 1.0
        episode[1]["trade_notional_usdt_5s"] = 100.0
        episode[2]["trade_volume_btc_5s"] = 3.0
        episode[2]["trade_notional_usdt_5s"] = 600.0
        result = market_vwap(episode)
        self.assertEqual(result, {"market_vwap_usdt_per_btc": 175.0,
                                  "market_volume_btc": 4.0,
                                  "market_notional_usdt": 700.0, "bins": 59})

    def test_zero_volume_has_no_benchmark_price(self):
        result = market_vwap(volume_episode(volume=0.0))
        self.assertIsNone(result["market_vwap_usdt_per_btc"])
        self.assertEqual(result["market_volume_btc"], 0.0)
        self.assertEqual(result["market_notional_usdt"], 0.0)

    def test_missing_invalid_or_inconsistent_aggregates_fail(self):
        for defect in ("missing", "nan", "negative", "zero_notional"):
            rows = volume_episode()
            if defect == "missing":
                del rows[1]["trade_notional_usdt_5s"]
            elif defect == "nan":
                rows[1]["trade_volume_btc_5s"] = float("nan")
            elif defect == "negative":
                rows[1]["trade_volume_btc_5s"] = -1.0
            else:
                rows[1]["trade_notional_usdt_5s"] = 0.0
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                market_vwap(rows)


if __name__ == "__main__":
    unittest.main()
