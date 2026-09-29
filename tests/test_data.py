"""Small CSV fixtures test loading rules without reading the research dataset."""

import csv
from datetime import date, datetime, timedelta, timezone
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from data import LoadStats, load_episodes, load_split


def make_rows(count=60, start=None):
    start = start or datetime(2020, 9, 1, 0, 5, tzinfo=timezone.utc)
    rows = []
    for index in range(count):
        stamp = start + timedelta(seconds=5 * index)
        row = {
            "decision_time": stamp.isoformat(),
            "book_local_timestamp": (stamp - timedelta(milliseconds=20)).isoformat(),
            "trade_volume_btc_5s": "0.0",
            "signed_trade_volume_btc_5s": "0.0",
            "trade_vwap_5s": "",
        }
        for level in range(25):
            row[f"bids[{level}].price"] = str(100 - level)
            row[f"bids[{level}].amount"] = "1.0"
            row[f"asks[{level}].price"] = str(101 + level)
            row[f"asks[{level}].amount"] = "1.0"
        rows.append(row)
    return rows


class EpisodeLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "day.csv.gz"

    def write_rows(self, rows):
        with gzip.open(self.path, "wt", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_complete_window_converts_scalars_and_counts(self):
        self.write_rows(make_rows())
        stats = LoadStats()
        episodes = list(load_episodes(self.path, stats=stats))
        self.assertEqual(len(episodes), 1)
        self.assertEqual(len(episodes[0]), 60)
        self.assertEqual(episodes[0][0]["decision_time"].utcoffset(), timedelta(0))
        self.assertEqual(episodes[0][0]["bids[0].price"], 100.0)
        self.assertIsNone(episodes[0][0]["trade_vwap_5s"])
        self.assertEqual(stats.rows_seen, 60)
        self.assertEqual(stats.files_loaded, 1)
        self.assertEqual(stats.complete_episodes, 1)
        self.assertEqual(stats.discarded_rows, 0)

    def test_missing_decision_discards_whole_window(self):
        rows = make_rows(120)
        del rows[20]
        self.write_rows(rows)
        stats = LoadStats()
        episodes = list(load_episodes(self.path, stats=stats))
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0][0]["decision_time"].minute, 10)
        self.assertEqual(stats.incomplete_windows, 1)
        self.assertEqual(stats.discarded_rows, 59)

    def test_sixty_rows_straddling_windows_are_not_an_episode(self):
        start = datetime(2020, 9, 1, 0, 5, 5, tzinfo=timezone.utc)
        self.write_rows(make_rows(start=start))
        stats = LoadStats()
        self.assertEqual(list(load_episodes(self.path, stats=stats)), [])
        self.assertEqual(stats.incomplete_windows, 2)
        self.assertEqual(stats.discarded_rows, 60)

    def test_rejects_duplicates_and_reversed_times(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                rows = make_rows()
                if reverse:
                    rows[2], rows[3] = rows[3], rows[2]
                else:
                    rows.insert(2, dict(rows[1]))
                self.write_rows(rows)
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    list(load_episodes(self.path))

    def test_rejects_off_grid_naive_future_and_stale_timestamps(self):
        mutations = (
            ("decision_time", "2020-09-01T00:05:01+00:00", "five-second"),
            ("decision_time", "2020-09-01T00:05:00", "timezone-aware"),
            ("book_local_timestamp", "2020-09-01T00:05:00.000001+00:00", "book receipt"),
            ("book_local_timestamp", "2020-09-01T00:04:54+00:00", "book receipt"),
        )
        for field, value, message in mutations:
            with self.subTest(field=field, value=value):
                rows = make_rows()
                rows[0][field] = value
                self.write_rows(rows)
                with self.assertRaisesRegex(ValueError, message):
                    list(load_episodes(self.path))

    def test_rejects_missing_level_and_nonfinite_numeric_data(self):
        rows = make_rows()
        for row in rows:
            del row["asks[24].amount"]
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "missing fields"):
            list(load_episodes(self.path))
        rows = make_rows()
        rows[0]["bids[3].amount"] = "nan"
        self.write_rows(rows)
        with self.assertRaisesRegex(ValueError, "must be finite"):
            list(load_episodes(self.path))

    def test_rejects_precision_that_would_hide_future_data_or_off_grid_time(self):
        for field in ("decision_time", "book_local_timestamp"):
            for separator in (".", ","):
                with self.subTest(field=field, separator=separator):
                    rows = make_rows()
                    rows[0][field] = f"2020-09-01T00:05:00{separator}000000001+00:00"
                    self.write_rows(rows)
                    with self.assertRaisesRegex(ValueError, "sub-microsecond precision"):
                        list(load_episodes(self.path))

    def test_rejects_unexpected_split_date(self):
        self.write_rows(make_rows())
        with self.assertRaisesRegex(ValueError, "expected split day"):
            list(load_episodes(self.path, expected_date=date(2021, 3, 1)))

    def test_split_must_be_explicit_valid_name(self):
        with self.assertRaisesRegex(ValueError, "split must be"):
            list(load_split("all", data_dir=self.temp.name))

    def test_loading_validation_does_not_open_other_splits(self):
        destination = Path(self.temp.name) / "validation"
        destination.mkdir()
        self.path = destination / "Feb2021_5s.csv.gz"
        start = datetime(2021, 2, 1, 0, 5, tzinfo=timezone.utc)
        self.write_rows(make_rows(start=start))
        self.assertEqual(len(list(load_split("validation", self.temp.name))), 1)


if __name__ == "__main__":
    unittest.main()
