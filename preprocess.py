"""Chunked conversion of the original Tardis files; never runs on import."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"
STEP_US = 5_000_000
N_STEPS = 24 * 60 * 60 // 5
CHUNK_SIZE = 100_000

SPLITS = {
    "training": ["Sep2020", "Oct2020", "Nov2020", "Dec2020", "Jan2021"],
    "validation": ["Feb2021"],
    "testing": ["Mar2021"],
}

BOOK_COLUMNS = [
    f"{side}[{level}].{field}"
    for side in ("bids", "asks")
    for level in range(25)
    for field in ("price", "amount")
]

TRADE_COLUMNS = ["local_timestamp", "price", "amount", "side"]


def make_grid(month: str):
    """Return the day's five-second decision grid in UTC microseconds."""
    day = pd.to_datetime(month, format="%b%Y", utc=True)
    start_us = day.value // 1_000

    grid_us = start_us + np.arange(N_STEPS, dtype=np.int64) * STEP_US
    return grid_us, start_us


def sample_book(path: Path, grid_us: np.ndarray) -> pd.DataFrame:
    """
    For each grid time, take the last book row whose local_timestamp
    is <= that grid time. Keep all 25 bid and 25 ask levels.
    """
    selected_values = np.full(
        (len(grid_us), len(BOOK_COLUMNS)),
        np.nan,
        dtype=np.float64,
    )
    selected_times = np.full(len(grid_us), -1, dtype=np.int64)

    cursor = 0
    previous_row = None
    previous_time = None
    rows_read = 0

    reader = pd.read_csv(
        path,
        usecols=["local_timestamp", *BOOK_COLUMNS],
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        times = chunk["local_timestamp"].to_numpy(dtype=np.int64)
        values = chunk[BOOK_COLUMNS].to_numpy(dtype=np.float64)

        if len(times) == 0:
            continue

        if np.any(np.diff(times) < 0):
            raise ValueError(f"Book times decrease within a chunk: {path}")

        if previous_time is not None and times[0] < previous_time:
            raise ValueError(f"Book times decrease between chunks: {path}")

        rows_read += len(times)

        # Carry the previous chunk's final book across the boundary.
        if previous_row is not None:
            times = np.concatenate(([previous_time], times))
            values = np.vstack((previous_row, values))

        # Leave decisions exactly at this chunk's last timestamp for the
        # next chunk, in case that timestamp has more than one book row.
        end = np.searchsorted(grid_us, times[-1], side="left")
        end = min(end, len(grid_us))

        if end > cursor:
            positions = np.searchsorted(
                times,
                grid_us[cursor:end],
                side="right",
            ) - 1

            valid = positions >= 0
            target = np.arange(cursor, end)[valid]

            selected_values[target] = values[positions[valid]]
            selected_times[target] = times[positions[valid]]

            cursor = end

        previous_time = times[-1]
        previous_row = values[-1].copy()

    if previous_time is None:
        raise ValueError(f"Empty book file: {path}")

    # Finish decisions after the last observed book row.
    if cursor < len(grid_us):
        remaining_times = grid_us[cursor:]
        valid = remaining_times >= previous_time
        target = np.arange(cursor, len(grid_us))[valid]

        selected_values[target] = previous_row
        selected_times[target] = previous_time

    book = pd.DataFrame(selected_values, columns=BOOK_COLUMNS)

    book_times = pd.Series(pd.to_datetime(selected_times, unit="us", utc=True))
    book_times = book_times.mask(selected_times < 0, pd.NaT)
    book.insert(
        0,
        "book_local_timestamp",
        book_times,
    )

    print(f"  Book rows read: {rows_read:,}")
    return book


def aggregate_trades(path: Path, start_us: int) -> pd.DataFrame:
    """
    Aggregate trades by arrival time into (t - 5 seconds, t].

    Each output row describes trades known by its decision time.
    'buy' and 'sell' refer to the trade's aggressor side.
    """
    count = np.zeros(N_STEPS, dtype=np.int64)
    volume = np.zeros(N_STEPS, dtype=np.float64)
    notional = np.zeros(N_STEPS, dtype=np.float64)
    signed_volume = np.zeros(N_STEPS, dtype=np.float64)

    rows_read = 0

    reader = pd.read_csv(
        path,
        usecols=TRADE_COLUMNS,
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        rows_read += len(chunk)

        times = chunk["local_timestamp"].to_numpy(dtype=np.int64)
        prices = chunk["price"].to_numpy(dtype=np.float64)
        amounts = chunk["amount"].to_numpy(dtype=np.float64)

        sides = chunk["side"].astype(str).str.lower().to_numpy()
        known_sides = np.isin(sides, ["buy", "sell", "unknown"])

        if not known_sides.all():
            unexpected = np.unique(sides[~known_sides])
            raise ValueError(f"Unexpected trade side in {path}: {unexpected}")

        if not np.isfinite(prices).all() or not np.isfinite(amounts).all():
            raise ValueError(f"Non-finite trade price/amount in {path}")

        if (prices <= 0).any() or (amounts <= 0).any():
            raise ValueError(f"Non-positive trade price/amount in {path}")

        # Ceiling division puts a trade at t into (t-5s, t].
        slots = (times - start_us + STEP_US - 1) // STEP_US

        valid = (slots >= 0) & (slots < N_STEPS)
        slots = slots[valid]
        prices = prices[valid]
        amounts = amounts[valid]
        sides = sides[valid]

        signs = np.where(sides == "buy", 1.0, 0.0)
        signs = np.where(sides == "sell", -1.0, signs)

        count += np.bincount(slots, minlength=N_STEPS)
        volume += np.bincount(
            slots, weights=amounts, minlength=N_STEPS
        )
        notional += np.bincount(
            slots, weights=prices * amounts, minlength=N_STEPS
        )
        signed_volume += np.bincount(
            slots, weights=signs * amounts, minlength=N_STEPS
        )

    trade_vwap = np.full(N_STEPS, np.nan, dtype=np.float64)
    np.divide(notional, volume, out=trade_vwap, where=volume > 0)

    print(f"  Trade rows read: {rows_read:,}")

    return pd.DataFrame({
        "trade_count_5s": count,
        "trade_volume_btc_5s": volume,
        "trade_notional_usdt_5s": notional,
        "signed_trade_volume_btc_5s": signed_volume,
        "trade_vwap_5s": trade_vwap,
    })


def validate(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    """Drop unavailable books; reject invalid depth, ordering, and crossed quotes."""
    frame["book_age_ms"] = (
        frame["decision_time"] - frame["book_local_timestamp"]
    ).dt.total_seconds() * 1_000

    missing = frame["book_local_timestamp"].isna()
    stale = frame["book_age_ms"] > 5_000
    future = frame["book_age_ms"] < 0

    if future.any():
        raise ValueError(f"{month}: future book received after its decision")
    # Only missing or stale decisions are dropped. Corrupt depth raises below.
    usable = frame.loc[~(missing | stale)].copy()
    for side in ("bids", "asks"):
        prices = usable[[f"{side}[{i}].price" for i in range(25)]].to_numpy()
        amounts = usable[[f"{side}[{i}].amount" for i in range(25)]].to_numpy()
        if not np.isfinite(prices).all() or not np.isfinite(amounts).all():
            raise ValueError(f"{month}: nonfinite {side} depth")
        if (prices <= 0).any() or (amounts < 0).any():
            raise ValueError(f"{month}: invalid {side} prices/amounts")
        differences = np.diff(prices, axis=1)
        if ((differences >= 0) if side == "bids" else (differences <= 0)).any():
            raise ValueError(f"{month}: unordered {side} levels")

    bad_top = usable["bids[0].price"] >= usable["asks[0].price"]

    if bad_top.any():
        raise ValueError(
            f"{month}: {bad_top.sum()} invalid top-of-book rows. "
            "Inspect these before deciding how to handle them."
        )

    usable["mid_price"] = (
        usable["bids[0].price"] + usable["asks[0].price"]
    ) / 2
    usable["spread"] = (
        usable["asks[0].price"] - usable["bids[0].price"]
    )

    if usable["decision_time"].duplicated().any():
        raise ValueError(f"{month}: duplicate decision times")

    print(
        f"  Decisions: {len(frame):,} total, "
        f"{len(usable):,} usable, "
        f"{missing.sum():,} missing, "
        f"{stale.sum():,} stale"
    )

    if not usable.empty:
        print(f"  Maximum usable book age: {usable['book_age_ms'].max():.1f} ms")

    return usable


def process_day(split: str, month: str, *, data_dir: Path = DATA_DIR,
                overwrite: bool = False) -> None:
    if split not in SPLITS or month not in SPLITS[split]:
        raise ValueError("Month does not belong to the requested split")
    folder = data_dir / split
    book_path = folder / f"{month}_T25.gz"
    trade_path = folder / f"{month}_Trade.gz"
    output_path = folder / f"{month}_5s.csv.gz"

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path}: pass --overwrite to replace it")

    for path in (book_path, trade_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"\nProcessing {split}/{month}")
    grid_us, start_us = make_grid(month)

    book = sample_book(book_path, grid_us)
    trades = aggregate_trades(trade_path, start_us)

    frame = pd.DataFrame({
        "decision_time": pd.to_datetime(grid_us, unit="us", utc=True)
    })
    frame = pd.concat([frame, book, trades], axis=1)
    frame = validate(frame, month)

    frame.to_csv(output_path, index=False, compression="gzip")
    print(f"  Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLITS), default="training")
    parser.add_argument("--month", help="One day label, e.g. Sep2020; omitted processes the split")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for month in ([args.month] if args.month else SPLITS[args.split]):
        process_day(args.split, month, data_dir=args.data_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
