"""Stream processed CSV days into complete, causal five-minute episodes.

Importing this module performs no I/O. The original top-of-book sampling demo
has been superseded by ``dataprocess.ipynb``; this module only reads its output.
Book validity and execution rules are checked by ``env.validate_episode``.
"""

from collections.abc import Iterator
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import gzip
from math import isfinite
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any


DATA_DIR = Path(__file__).resolve().parent / "data"
SPLIT_DAYS = MappingProxyType({
    "training": ("Sep2020", "Oct2020", "Nov2020", "Dec2020", "Jan2021"),
    "validation": ("Feb2021",),
    "testing": ("Mar2021",),
})
_DAY_DATES = {
    "Sep2020": date(2020, 9, 1),
    "Oct2020": date(2020, 10, 1),
    "Nov2020": date(2020, 11, 1),
    "Dec2020": date(2020, 12, 1),
    "Jan2021": date(2021, 1, 1),
    "Feb2021": date(2021, 2, 1),
    "Mar2021": date(2021, 3, 1),
}
_TIMESTAMP_FIELDS = {"decision_time", "book_local_timestamp", "episode_start"}
_REQUIRED_FIELDS = {
    "decision_time", "book_local_timestamp",
    "trade_volume_btc_5s", "signed_trade_volume_btc_5s",
    *(f"{side}[{level}].{field}"
      for side in ("bids", "asks")
      for level in range(25)
      for field in ("price", "amount")),
}
_STEP = timedelta(seconds=5)
_EPISODE_STEPS = 60


@dataclass
class LoadStats:
    """Optional running counters, final after the episode iterator is exhausted.

    ``incomplete_windows`` counts windows containing at least one input row;
    a completely absent window cannot be counted without a requested date range.
    """

    files_loaded: int = 0
    rows_seen: int = 0
    complete_episodes: int = 0
    incomplete_windows: int = 0
    discarded_rows: int = 0


def _utc_timestamp(value: str, field: str) -> datetime:
    # datetime stores microseconds. Reject finer precision instead of truncating
    # a future receipt onto the decision boundary and admitting future data.
    if isinstance(value, str) and re.search(r"[.,]\d{7,}", value):
        raise ValueError(f"{field} must not contain sub-microsecond precision")
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO timestamp") from exc
    if stamp.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be timezone-aware UTC")
    return stamp


def _parse_row(raw: dict[str, str]) -> dict[str, Any]:
    """Convert CSV strings without silently replacing missing market data."""
    if None in raw or any(value is None for value in raw.values()):
        raise ValueError("CSV row has a different width from its header")

    row: dict[str, Any] = {}
    for field, value in raw.items():
        if field in _TIMESTAMP_FIELDS:
            row[field] = _utc_timestamp(value, field)
            continue
        # A no-trade interval has no VWAP. It is not an observation feature.
        if field == "trade_vwap_5s" and value.strip().lower() in ("", "nan"):
            row[field] = None
            continue
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"{field} must be numeric") from exc
        if not isfinite(number):
            raise ValueError(f"{field} must be finite")
        row[field] = number

    decision = row["decision_time"]
    if decision.microsecond or decision.second % 5:
        raise ValueError("decision_time must lie on the five-second UTC grid")
    age = decision - row["book_local_timestamp"]
    if not timedelta(0) <= age <= _STEP:
        raise ValueError("book receipt must be at or before the decision and at most five seconds old")
    return row


def load_episodes(
    path: str | Path,
    *,
    stats: LoadStats | None = None,
    expected_date: date | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield complete UTC-aligned 60-row windows from one CSV or CSV.gz file.

    Rows stay in file order: duplicate, reversed, off-grid, future-book, or stale
    timestamps raise ``ValueError``. Missing decisions cause the whole window
    to be discarded; no rows or liquidity are interpolated. Only one window is
    buffered, so callers can evaluate an entire day without holding it in RAM.
    Each returned row contains floats and aware UTC ``datetime`` timestamps.

    ``expected_date`` verifies split membership when called by ``load_split``.
    Book ordering/amount constraints are validated when constructing the env.
    """
    path = Path(path)
    stats = stats if stats is not None else LoadStats()
    opener = gzip.open if path.suffix == ".gz" else open
    previous_time: datetime | None = None
    current_start: datetime | None = None
    window: list[dict[str, Any]] = []

    def complete_window() -> bool:
        if len(window) == _EPISODE_STEPS and all(
            row["decision_time"] == current_start + index * _STEP
            for index, row in enumerate(window)
        ):
            stats.complete_episodes += 1
            return True
        stats.incomplete_windows += 1
        stats.discarded_rows += len(window)
        return False

    with opener(path, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        if len(header) != len(set(header)):
            raise ValueError(f"{path}: duplicate CSV header fields")
        missing = _REQUIRED_FIELDS.difference(header)
        if missing:
            raise ValueError(f"{path}: missing fields: {', '.join(sorted(missing))}")

        for line_number, raw in enumerate(reader, start=2):
            stats.rows_seen += 1
            try:
                row = _parse_row(raw)
                decision = row["decision_time"]
                if expected_date is not None and decision.date() != expected_date:
                    raise ValueError(f"decision date does not match expected split day {expected_date}")
                if previous_time is not None and decision <= previous_time:
                    raise ValueError("decision timestamps must be strictly increasing; duplicates are invalid")
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc

            start = decision.replace(minute=decision.minute // 5 * 5, second=0, microsecond=0)
            if current_start is not None and start != current_start:
                if complete_window():
                    yield window
                window = []
            current_start = start
            window.append(row)
            previous_time = decision

        if window and complete_window():
            yield window
    stats.files_loaded += 1


def load_split(
    split: str = "training",
    data_dir: str | Path = DATA_DIR,
    *,
    stats: LoadStats | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Load only the named day-based split, defaulting to training.

    Explicit filenames prevent accidental split mixing through a broad glob.
    Validation and testing must be requested by name; this function runs no
    policies, trains no models, and performs no evaluation.
    """
    if split not in SPLIT_DAYS:
        raise ValueError(f"split must be one of {tuple(SPLIT_DAYS)}, got {split!r}")
    for day in SPLIT_DAYS[split]:
        yield from load_episodes(
            Path(data_dir) / split / f"{day}_5s.csv.gz",
            stats=stats,
            expected_date=_DAY_DATES[day],
        )
