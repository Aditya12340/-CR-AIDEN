"""Synthetic raw rows exercise preprocessing boundaries without large data I/O."""

from contextlib import redirect_stdout
import csv
import gzip
from importlib.util import find_spec
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


HAS_PANDAS = find_spec("pandas") is not None and find_spec("numpy") is not None
if HAS_PANDAS:
    import numpy as np
    import pandas as pd
    import preprocess


@unittest.skipUnless(HAS_PANDAS, "preprocessing requires pandas and numpy")
class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "input.csv.gz"
        self.start = 1_598_918_400_000_000  # 2020-09-01 00:00 UTC, microseconds.
        # Silence progress prints; assertions inspect data and explicit errors.
        output = redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)

    def write_rows(self, rows):
        with gzip.open(self.path, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def book_row(self, offset_us, marker=0):
        row = {"local_timestamp": self.start + offset_us}
        for level in range(25):
            row[f"bids[{level}].price"] = 100 + marker - level
            row[f"bids[{level}].amount"] = 1.0
            row[f"asks[{level}].price"] = 101 + marker + level
            row[f"asks[{level}].amount"] = 1.0
        return row

    def grid(self, length=4):
        return self.start + np.arange(length, dtype=np.int64) * preprocess.STEP_US

    def test_duplicate_receipts_across_chunks_choose_last_known_row(self):
        self.write_rows([
            self.book_row(1_000_000, 0),
            self.book_row(5_000_000, 1),
            self.book_row(5_000_000, 2),
            self.book_row(10_000_000, 3),
            self.book_row(10_000_000, 4),
        ])
        for chunk_size in (1, 2, 3):
            with self.subTest(chunk_size=chunk_size), patch.object(preprocess, "CHUNK_SIZE", chunk_size):
                frame = preprocess.sample_book(self.path, self.grid())
                self.assertTrue(pd.isna(frame.loc[0, "book_local_timestamp"]))
                self.assertTrue(pd.isna(frame.loc[0, "bids[0].price"]))
                self.assertEqual(frame.loc[1, "bids[0].price"], 102)
                self.assertEqual(frame.loc[2, "bids[0].price"], 104)
                self.assertEqual(frame.loc[3, "bids[0].price"], 104)
                self.assertEqual(frame.loc[1, "book_local_timestamp"], pd.to_datetime(self.start + 5_000_000, unit="us", utc=True))

    def test_book_received_one_microsecond_after_decision_is_not_selected(self):
        self.write_rows([
            self.book_row(1_000_000, 0),
            self.book_row(5_000_001, 1),
        ])
        with patch.object(preprocess, "CHUNK_SIZE", 1):
            frame = preprocess.sample_book(self.path, self.grid(3))
        self.assertEqual(frame.loc[1, "bids[0].price"], 100)
        self.assertEqual(frame.loc[2, "bids[0].price"], 101)

    def test_missing_sentinel_is_nat_and_only_missing_stale_decisions_drop(self):
        self.write_rows([self.book_row(5_000_000)])
        grid = self.grid()
        frame = preprocess.sample_book(self.path, grid)
        self.assertTrue(pd.isna(frame.loc[0, "book_local_timestamp"]))
        frame.insert(0, "decision_time", pd.to_datetime(grid, unit="us", utc=True))
        valid = preprocess.validate(frame, "synthetic")
        self.assertEqual(valid.index.tolist(), [1, 2])
        self.assertEqual(valid["book_age_ms"].tolist(), [0.0, 5000.0])

    def test_decreasing_book_receipts_fail_within_or_across_chunks(self):
        self.write_rows([self.book_row(2_000_000), self.book_row(1_000_000)])
        for chunk_size in (1, 2):
            with self.subTest(chunk_size=chunk_size), patch.object(preprocess, "CHUNK_SIZE", chunk_size):
                with self.assertRaisesRegex(ValueError, "decrease"):
                    preprocess.sample_book(self.path, self.grid())

    def test_trades_use_open_left_closed_right_receipt_intervals(self):
        # The t=0 trade belongs to the midnight decision, while 0<t<=5s
        # belongs to the next. Unknown aggressors add volume but zero signed flow.
        rows = [
            {"local_timestamp": self.start + offset, "price": 100.0, "amount": amount, "side": side}
            for offset, amount, side in (
                (-5_000_000, 100.0, "buy"),  # Outside every included interval.
                (0, 1.0, "buy"),
                (1, 2.0, "sell"),
                (5_000_000, 3.0, "buy"),
                (5_000_001, 4.0, "unknown"),
                (10_000_000, 5.0, "sell"),
                (15_000_001, 100.0, "buy"),  # After final decision at 15s.
            )
        ]
        self.write_rows(rows)
        with patch.object(preprocess, "CHUNK_SIZE", 2), patch.object(preprocess, "N_STEPS", 4):
            frame = preprocess.aggregate_trades(self.path, self.start)
        self.assertEqual(frame["trade_count_5s"].tolist(), [1, 2, 2, 0])
        self.assertEqual(frame["trade_volume_btc_5s"].tolist(), [1.0, 5.0, 9.0, 0.0])
        self.assertEqual(frame["signed_trade_volume_btc_5s"].tolist(), [1.0, 1.0, -5.0, 0.0])
        self.assertEqual(frame["trade_notional_usdt_5s"].tolist(), [100.0, 500.0, 900.0, 0.0])
        self.assertEqual(frame["trade_vwap_5s"].iloc[:3].tolist(), [100.0, 100.0, 100.0])
        self.assertTrue(pd.isna(frame.loc[3, "trade_vwap_5s"]))

    def test_invalid_trade_sides_prices_and_amounts_are_rejected(self):
        base = {"local_timestamp": self.start, "price": 100.0, "amount": 1.0, "side": "buy"}
        for field, value, message in (
            ("side", "bid", "Unexpected trade side"),
            ("amount", 0.0, "Non-positive"),
            ("price", -1.0, "Non-positive"),
            ("amount", float("nan"), "Non-finite"),
        ):
            with self.subTest(field=field, value=value):
                self.write_rows([{**base, field: value}])
                with self.assertRaisesRegex(ValueError, message):
                    preprocess.aggregate_trades(self.path, self.start)

    def test_deep_corruption_and_future_books_are_rejected(self):
        self.write_rows([self.book_row(0)])
        frame = preprocess.sample_book(self.path, self.grid(1))
        frame.insert(0, "decision_time", pd.to_datetime(self.grid(1), unit="us", utc=True))
        for field, value, message in (
            ("bids[24].amount", float("nan"), "nonfinite"),
            ("asks[24].price", 1.0, "unordered"),
            ("book_local_timestamp", pd.to_datetime(self.start + 1, unit="us", utc=True), "future book"),
        ):
            with self.subTest(field=field):
                corrupted = frame.copy()
                corrupted.loc[0, field] = value
                with self.assertRaisesRegex(ValueError, message):
                    preprocess.validate(corrupted, "synthetic")


if __name__ == "__main__":
    unittest.main()
